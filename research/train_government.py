"""NYC government data: property-disjoint annual GHG peer benchmarking.
This is NOT a causal counterfactual or certified credit issuance model.
"""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train_baseline import finite_quantile

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=ROOT/'data/NYC_building_emissions_government.csv')
    p.add_argument('--year',type=int,default=None)
    args=p.parse_args(); out=ROOT/'government_results';out.mkdir(exist_ok=True)
    raw=pd.read_csv(args.data,dtype=str)
    year=pd.to_numeric(raw.report_year,errors='coerce')
    # Select latest sufficiently populated year; report counts, do not assert completeness.
    counts=year.value_counts(); chosen=args.year or int(counts[counts>=1000].index.max())
    d=raw[year==chosen].copy()
    original=len(d)
    numeric=['largest_property_use_type_1','_2nd_largest_property_use_1',
             '_3rd_largest_property_use_1','year_built','number_of_buildings','occupancy']
    for col in numeric+['total_location_based_ghg']:
        d[col]=pd.to_numeric(d[col].str.replace(',','',regex=False),errors='coerce')
    d=d[d.total_location_based_ghg.notna()&(d.total_location_based_ghg>=0)&d.property_id.notna()]
    d=d[d.largest_property_use_type_1>0]
    # Remove parent/child campuses to reduce overlapping reporting boundaries.
    d=d[d.parent_property_id.fillna('Not Available').isin(['Not Available','Not Applicable: Standalone Property','0',''])]
    for col in ['alert_energy_meter_has_less','alert_energy_meter_has_gaps']:
        d=d[~d[col].fillna('').str.lower().isin(['yes','true'])]
    # Keep only fully metered properties. Do not silently interpret partial coverage.
    d=d[d.metered_areas_energy.fillna('').str.lower().isin(['whole building','whole property'])]
    d=d[~d.estimated_values_energy.fillna('').str.lower().isin(['yes','true'])]
    d=d.drop_duplicates().copy()
    # Conflicting repeated property/year records are excluded, not averaged.
    d=d[~d.property_id.duplicated(keep=False)].reset_index(drop=True)
    for c in numeric:
        d.loc[d[c]<0,c]=np.nan
    d.loc[(d.occupancy>100), 'occupancy']=np.nan
    d.loc[(d.year_built<1600)|(d.year_built>chosen),'year_built']=np.nan
    nums=numeric
    cats=['primary_property_type_self','borough']
    X=d[nums+cats].copy()
    X[cats]=X[cats].fillna('Unknown').astype(str)
    y=np.log1p(d.total_location_based_ghg.to_numpy())
    # Whole properties are assigned to one partition only. Same year, cross-sectional.
    idx=np.arange(len(d));groups=d.property_id
    def split(ids, fraction, seed):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(
            ids,groups=groups.iloc[ids]))
        return ids[a],ids[b]
    remain,test=split(idx,.15,42); remain,cal=split(remain,.10/.85,43)
    train,val=split(remain,.15/.75,44)
    assert not set(groups.iloc[train])&set(groups.iloc[test])
    med=X.iloc[train][nums].median().fillna(0)
    X[nums]=X[nums].fillna(med)
    tuning=[]
    for depth in [4,6,8]:
        for it in [400,800]:
            m=CatBoostRegressor(iterations=it,depth=depth,learning_rate=.05,loss_function='RMSE',
                random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)
            m.fit(X.iloc[train],y[train],cat_features=cats)
            tuning.append({'depth':depth,'iterations':it,
                           'validation_log_MAE':mean_absolute_error(y[val],m.predict(X.iloc[val]))})
    best=min(tuning,key=lambda r:r['validation_log_MAE'])
    fit=np.concatenate([train,val])
    # The same train-derived feature imputation is retained in all partitions.
    m=CatBoostRegressor(iterations=best['iterations'],depth=best['depth'],learning_rate=.05,
        loss_function='RMSE',random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)
    m.fit(X.iloc[fit],y[fit],cat_features=cats)
    rf=make_pipeline(ColumnTransformer([
        ('nums',SimpleImputer(strategy='median'),nums),
        ('cats',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)]),
        RandomForestRegressor(n_estimators=300,min_samples_leaf=5,max_depth=18,n_jobs=4,random_state=42))
    rf.fit(X.iloc[fit],y[fit])
    preds={'Median':np.full(len(test),np.median(y[fit])),
           'RandomForest':rf.predict(X.iloc[test]),'CatBoost':m.predict(X.iloc[test])}
    rows=[]; actual=np.expm1(y[test])
    for name,logpred in preds.items():
        pred=np.maximum(np.expm1(logpred),0)
        rows.append({'model':name,'MAE_tCO2e':mean_absolute_error(actual,pred),
          'RMSE_tCO2e':np.sqrt(mean_squared_error(actual,pred)),
          'R2':r2_score(actual,pred),'MAE_log1p':mean_absolute_error(y[test],logpred)})
    q=finite_quantile(np.abs(y[cal]-m.predict(X.iloc[cal])),.1)
    lp=preds['CatBoost'];lo=np.maximum(np.expm1(lp-q),0);hi=np.maximum(np.expm1(lp+q),0)
    result=d.iloc[test][['property_id','primary_property_type_self','total_location_based_ghg']].copy()
    result['peer_baseline_tCO2e']=np.maximum(np.expm1(lp),0)
    result['lower_tCO2e']=lo;result['upper_tCO2e']=hi
    result['gap_to_peer_baseline_tCO2e']=result.peer_baseline_tCO2e-actual
    result.to_csv(out/'government_test_predictions.csv',index=False)
    simulations=[]
    for fraction in [0,.1,.2]:
        simulated_actual=actual*(1-fraction)
        truth=actual-simulated_actual
        for name,baseline in {'Median':np.full(len(test),np.expm1(np.median(y[fit]))),
                             'CatBoostPoint':result.peer_baseline_tCO2e.to_numpy(),
                             'CatBoostConservative':lo}.items():
            estimate=np.maximum(baseline-simulated_actual,0)
            simulations.append({'synthetic_reduction_fraction':fraction,'method':name,
                'MAE_tCO2e':float(np.abs(estimate-truth).mean()),
                'sum_overestimation_tCO2e':float(np.maximum(estimate-truth,0).sum()),
                'sum_underestimation_tCO2e':float(np.maximum(truth-estimate,0).sum()),
                'fraction_properties_overcredited':float((estimate>truth+1e-9).mean())})
    pd.DataFrame(simulations).to_csv(out/'government_simulation_metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'government_model_metrics.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'government_tuning.csv',index=False)
    pd.DataFrame({'feature':X.columns,'importance':m.feature_importances_}).sort_values(
        'importance',ascending=False).to_csv(out/'government_feature_importance.csv',index=False)
    m.save_model(str(out/'government_catboost.cbm'))
    metadata={'source':'City of New York NYC Open Data dataset 5zyy-y8am',
      'selected_report_year':chosen,'downloaded_rows':len(raw),'year_rows_before_cleaning':original,
      'clean_rows':len(d),'partition_rows':{k:len(v) for k,v in zip(
          ['train','validation','calibration','test'],[train,val,cal,test])},
      'nominal_coverage':.9,'test_coverage':float(((actual>=lo)&(actual<=hi)).mean()),
      'mean_interval_width_tCO2e':float(np.mean(hi-lo)),'selected_model':best,
      'features':list(X.columns),'imputation_values':med.to_dict(),
      'scope':'Property-disjoint, same-year peer emissions benchmark; NOT a verified causal baseline or carbon credits.',
      'limitations':['Self-reported government-published records','Incomplete activity/weather controls',
                    'US building results do not validate Indian establishments',
                    'No future-year validation; latest available year is not claimed complete',
                    'Log-space intervals assume sufficiently comparable calibration/test properties']}
    (out/'government_run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
