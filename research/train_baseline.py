"""Chronological energy baseline benchmark and explicitly simulated credit study.
Run: python train_baseline.py --emission-factor 0.5
Factor units: kg CO2e/kWh. 0.5 is an illustrative assumption, NOT an official factor.
"""
from pathlib import Path
import argparse, json, math, platform
import numpy as np
import pandas as pd
import sklearn, catboost
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def metrics(y, p):
    return dict(MAE_kWh=float(mean_absolute_error(y,p)),
                RMSE_kWh=float(np.sqrt(mean_squared_error(y,p))),
                R2=float(r2_score(y,p)))

def finite_quantile(scores, alpha):
    """Split-conformal order statistic; temporal dependence limits guarantees."""
    scores = np.sort(np.asarray(scores, dtype=float))
    k = min(len(scores), math.ceil((len(scores)+1)*(1-alpha)))
    return float(scores[k-1])

def features(d):
    t = d.timestamp
    minute = t.dt.hour*60+t.dt.minute
    dow = t.dt.dayofweek
    # Exclude CO2, reactive power, power factors, current usage and Load_Type.
    # Their contemporaneous/derived nature can inflate baseline performance.
    return pd.DataFrame({
        'minute':minute, 'weekday':dow, 'weekend':(dow>=5).astype(int),
        'hour_sin':np.sin(2*np.pi*minute/1440),
        'hour_cos':np.cos(2*np.pi*minute/1440),
        'week_sin':np.sin(2*np.pi*dow/7),
        'week_cos':np.cos(2*np.pi*dow/7),
    }, index=d.index)

def daily(d, pred):
    return pd.DataFrame({'date':d.day.to_numpy(), 'actual_kWh':d.Usage_kWh.to_numpy(),
                         'predicted_kWh':np.maximum(pred,0)}).groupby('date').sum()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/Steel_industry_data.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--emission-factor',type=float,default=None,
                        help='kg CO2e/kWh; required for simulated carbon results')
    args=parser.parse_args()
    if args.emission_factor is not None and args.emission_factor<=0:
        parser.error('Emission factor must be positive')
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.data)
    d['timestamp']=pd.to_datetime(d['date'],format='%d/%m/%Y %H:%M',errors='raise')
    d=d.sort_values('timestamp').reset_index(drop=True)
    if d.timestamp.duplicated().any() or d.Usage_kWh.isna().any() or (d.Usage_kWh<0).any():
        raise ValueError('Duplicate timestamps, missing target, or negative energy')
    d['day']=d.timestamp.dt.floor('D')
    counts=d.groupby('day').size()
    if not (counts==96).all():
        raise ValueError('Expected 96 readings per source calendar day; inspect incomplete days')
    days=np.sort(d.day.unique()); n=len(days)
    cuts=[0,int(.55*n),int(.70*n),int(.85*n),n]
    parts=[d[d.day.isin(days[cuts[i]:cuts[i+1]])].copy() for i in range(4)]
    train,val,cal,test=parts
    Xtr,Xv,Xc,Xt=[features(p) for p in parts]
    ytr=train.Usage_kWh; yv=val.Usage_kWh
    tuning=[]
    # Selection uses validation only. Calibration and test remain untouched.
    for depth in [4,6,8]:
        for iterations in [400,800]:
            m=CatBoostRegressor(depth=depth,iterations=iterations,learning_rate=.05,
                loss_function='RMSE',random_seed=42,thread_count=4,verbose=False,
                allow_writing_files=False)
            m.fit(Xtr,ytr)
            tuning.append({'depth':depth,'iterations':iterations,
                           'validation_MAE':mean_absolute_error(yv,m.predict(Xv))})
    best=min(tuning,key=lambda x:x['validation_MAE'])
    pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
    fit=pd.concat([train,val]); Xfit=features(fit); yf=fit.Usage_kWh
    models={
        'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)),
        'RandomForest':RandomForestRegressor(n_estimators=300,max_depth=12,
            min_samples_leaf=10,random_state=42,n_jobs=4),
        'CatBoost':CatBoostRegressor(depth=best['depth'],iterations=best['iterations'],
            learning_rate=.05,loss_function='RMSE',random_seed=42,thread_count=4,
            verbose=False,allow_writing_files=False),
    }
    rows=[]; daily_predictions={}
    group_means=fit.groupby([fit.timestamp.dt.dayofweek,
                             fit.timestamp.dt.hour*60+fit.timestamp.dt.minute]).Usage_kWh.mean()
    keys=list(zip(test.timestamp.dt.dayofweek,test.timestamp.dt.hour*60+test.timestamp.dt.minute))
    schedule=np.array([group_means.get(k,yf.mean()) for k in keys])
    predictions={'FixedMean':np.full(len(test),yf.mean()),'WeeklySchedule':schedule}
    for name,m in models.items():
        m.fit(Xfit,yf); predictions[name]=np.maximum(m.predict(Xt),0)
    for name,p in predictions.items():
        rows.append({'model':name,'resolution':'15-minute',**metrics(test.Usage_kWh,p)})
        dd=daily(test,p); daily_predictions[name]=dd
        rows.append({'model':name,'resolution':'daily',**metrics(dd.actual_kWh,dd.predicted_kWh)})
    result=pd.DataFrame(rows); result.to_csv(out/'model_metrics.csv',index=False)
    model=models['CatBoost']; model.save_model(str(out/'catboost_baseline.cbm'))
    pd.DataFrame({'feature':Xfit.columns,'importance':model.feature_importances_}).sort_values(
        'importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    cd=daily(cal,model.predict(Xc)); td=daily_predictions['CatBoost'].copy()
    # Calibrate on DAY-level residuals, not a sum of interval-level bounds.
    q=finite_quantile(np.abs(cd.actual_kWh-cd.predicted_kWh),alpha=.10)
    td['lower_kWh']=np.maximum(td.predicted_kWh-q,0)
    td['upper_kWh']=td.predicted_kWh+q
    td.to_csv(out/'daily_predictions.csv')
    covered=(td.actual_kWh>=td.lower_kWh)&(td.actual_kWh<=td.upper_kWh)
    uncertainty={'nominal_daily_coverage':.9,'observed_test_daily_coverage':float(covered.mean()),
        'mean_width_kWh':float((td.upper_kWh-td.lower_kWh).mean()),'calibration_days':len(cd),
        'absolute_residual_radius_kWh':q,
        'limitation':'Empirical temporal calibration; no exchangeability guarantee, causal identification, or period-total coverage guarantee.'}
    (out/'uncertainty_metrics.json').write_text(json.dumps(uncertainty,indent=2))
    # Held-out ORIGINAL observations serve as empirical counterfactuals only
    # inside explicitly synthetic efficiency scenarios. No intervention is observed.
    if args.emission_factor is not None:
        factor=args.emission_factor/1000 # kg -> tonnes
        summaries=[]; details=[]
        for fraction in [0,.10,.20]:
            actual=td.actual_kWh*(1-fraction)
            truth=(td.actual_kWh-actual)*factor
            estimates={
                'FixedMean':np.maximum(96*yf.mean()-actual,0)*factor,
                'WeeklySchedule':np.maximum(daily_predictions['WeeklySchedule'].predicted_kWh-actual,0)*factor,
                'CatBoostPoint':np.maximum(td.predicted_kWh-actual,0)*factor,
                'CatBoostConservative':np.maximum(td.lower_kWh-actual,0)*factor,
            }
            for method,estimate in estimates.items():
                summaries.append({'synthetic_reduction_fraction':fraction,'method':method,
                    'daily_MAE_tCO2e':float(np.abs(estimate-truth).mean()),
                    'sum_daily_overestimation_tCO2e':float(np.maximum(estimate-truth,0).sum()),
                    'sum_daily_underestimation_tCO2e':float(np.maximum(truth-estimate,0).sum()),
                    'fraction_days_overcredited':float((estimate>truth+1e-9).mean())})
                details.append(pd.DataFrame({'date':td.index,'synthetic_reduction_fraction':fraction,
                    'method':method,'simulated_actual_kWh':actual.to_numpy(),
                    'simulated_true_reduction_tCO2e':truth.to_numpy(),
                    'simulated_estimate_tCO2e':estimate.to_numpy()}))
        pd.DataFrame(summaries).to_csv(out/'simulation_metrics.csv',index=False)
        pd.concat(details).to_csv(out/'simulated_daily_credits.csv',index=False)
    splits={name:{'first':str(p.day.min().date()),'last':str(p.day.max().date()),
                   'rows':len(p),'days':p.day.nunique()} for name,p in zip(
                       ['train','validation','calibration','test'],parts)}
    manifest={'dataset':'UCI Steel Industry Energy Consumption DOI:10.24432/C52G8C',
        'splits':splits,'selected_catboost':best,'emission_factor_kgCO2e_per_kWh':args.emission_factor,
        'factor_status':'User-supplied assumption; must document authoritative regional/year factor for real accounting',
        'excluded_columns':['CO2(tCO2)','Load_Type','reactive power','power factors'],
        'scope':'Calendar-conditioned electricity baseline. No measured production or intervention labels.',
        'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'catboost':catboost.__version__}}
    (out/'run_metadata.json').write_text(json.dumps(manifest,indent=2))
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(td.index,td.actual_kWh,label='Observed daily energy',linewidth=1)
    ax.plot(td.index,td.predicted_kWh,label='CatBoost baseline',linewidth=1)
    ax.fill_between(td.index,td.lower_kWh,td.upper_kWh,alpha=.2,label='Empirically calibrated 90% interval')
    ax.set_ylabel('Daily electricity (kWh)');ax.legend();fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(out/'baseline_plot.png',dpi=160);plt.close(fig)
    print(result.to_string(index=False));print(json.dumps(uncertainty,indent=2))
    print('Outputs:',out)

if __name__=='__main__':
    main()
