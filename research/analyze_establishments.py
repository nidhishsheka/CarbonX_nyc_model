"""Post-model comparisons for held-out properties; no regulatory credit claims."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def main():
    source=ROOT/'government_results'
    out=ROOT/'establishment_analysis';out.mkdir(exist_ok=True)
    d=pd.read_csv(source/'government_test_predictions.csv')
    d=d.rename(columns={'total_location_based_ghg':'reported_emissions_tCO2e'})
    actual=d.reported_emissions_tCO2e; benchmark=d.peer_baseline_tCO2e
    d['excess_over_benchmark_tCO2e']=actual-benchmark
    d['excess_over_benchmark_percent']=np.where(benchmark>0,100*(actual-benchmark)/benchmark,np.nan)
    d['point_comparison']=np.where(actual>benchmark,'Above peer benchmark','At or below peer benchmark')
    d['interval_comparison']=np.select([actual>d.upper_tCO2e,actual<d.lower_tCO2e],
        ['Above prediction interval','Below prediction interval'],default='Within prediction interval')
    d['review_action']=np.select([actual>d.upper_tCO2e,actual<d.lower_tCO2e],
        ['Investigate reporting, energy use, and unmeasured operating conditions',
         'Check reporting completeness and documented efficiency measures'],
        default='No interval-based exception; continue routine monitoring')
    # No credit column is computed from real peer gaps. Simulation is separate.
    d.to_csv(out/'establishment_comparison.csv',index=False)
    flagged=d[d.interval_comparison=='Above prediction interval'].sort_values(
        'excess_over_benchmark_tCO2e',ascending=False)
    flagged.to_csv(out/'properties_for_review.csv',index=False)
    counts=d.interval_comparison.value_counts()
    simulations=pd.read_csv(source/'government_simulation_metrics.csv')
    simulations.to_csv(out/'simulated_credit_evaluation.csv',index=False)
    zero=simulations[simulations.synthetic_reduction_fraction==0].set_index('method')
    point=float(zero.loc['CatBoostPoint','fraction_properties_overcredited'])*100
    conservative=float(zero.loc['CatBoostConservative','fraction_properties_overcredited'])*100
    model=pd.read_csv(source/'government_model_metrics.csv').set_index('model').loc['CatBoost']
    meta=json.loads((source/'government_run_metadata.json').read_text())
    summary={'test_properties':len(d),'above_point_benchmark':int((actual>benchmark).sum()),
       'above_interval':int(counts.get('Above prediction interval',0)),
       'within_interval':int(counts.get('Within prediction interval',0)),
       'below_interval':int(counts.get('Below prediction interval',0)),
       'zero_reduction_point_false_credit_percent':point,
       'zero_reduction_conservative_false_credit_percent':conservative}
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    first=flagged.head(5)
    table='| Property ID | Type | Reported emissions | Predicted benchmark | Upper bound |\n|---|---|---:|---:|---:|\n'
    for _,r in first.iterrows():
        table+=f'| {r.property_id} | {r.primary_property_type_self} | {r.reported_emissions_tCO2e:,.2f} | {r.peer_baseline_tCO2e:,.2f} | {r.upper_tCO2e:,.2f} |\n'
    report=f'''# Establishment emissions: benchmark comparison and simulation

CatBoost was used to compare reported annual emissions with a peer benchmark for {len(d):,} held-out NYC properties in {meta['selected_report_year']}. All emission quantities below use metric tonnes CO2e.

## Prediction results
Test MAE: {model.MAE_tCO2e:.2f} tonnes CO2e. Test RMSE: {model.RMSE_tCO2e:.2f}. Test R2: {model.R2:.3f}.
Nominal 90% interval coverage was {meta['test_coverage']*100:.2f}%, below nominal. These flags are exploratory model exceptions, not findings of excessive legal emissions or inefficiency.

## Property comparison
{summary['above_point_benchmark']:,} properties reported emissions above the point benchmark. Comparing against the wider prediction intervals gives:

| Result | Properties |
|---|---:|
| Above upper prediction bound | {summary['above_interval']} |
| Within prediction interval | {summary['within_interval']} |
| Below lower prediction bound | {summary['below_interval']} |

Properties above the upper bound merit investigation of missing operating conditions, reporting problems, and energy consumption. They are not automatically violating a limit. Properties below the lower bound also merit checks: incomplete reporting and efficiency improvements can both produce low emissions.

## Illustrative properties for investigation
Sorted by excess over the point benchmark; this is not a verified inefficiency ranking.

{table}
## Synthetic credit experiment
Original held-out emissions serve as empirical counterfactuals only inside simulation. We inject 0%, 10%, and 20% reductions, without changing property characteristics. No observed intervention or certified credits are present.

Point estimate = max(predicted peer benchmark - simulated actual emissions, 0).
Conservative estimate = max(lower prediction bound - simulated actual emissions, 0).

Under the 0% reduction scenario, the true synthetic reduction is zero. Nevertheless, point estimates generated positive simulated credits for {point:.2f}% of properties. Lower-bound estimates reduced this proportion to {conservative:.2f}%.

This demonstrates the danger of interpreting peer-benchmark gaps as credits. Conservative estimates reduce false credits at the cost of underestimating genuine synthetic reductions. They do not remove the need for a causal baseline or establish actual credit eligibility. The full overestimation and underestimation tradeoff is in simulated_credit_evaluation.csv.

## Files and reproduction
- establishment_comparison.csv: all held-out property comparisons, intervals and review actions.
- properties_for_review.csv: properties above upper bounds.
- simulated_credit_evaluation.csv: synthetic reduction results.
- analysis_summary.json: numeric summary.

Run `python analyze_establishments.py` after `python train_government.py`. No model retraining is required if the existing government prediction files are present.

## Research conclusion
The current evidence supports emission peer benchmarking and an uncertainty-aware simulation study. Before claiming actual reductions, add pre-intervention emissions, intervention dates, measured activity and weather, and an approved accounting methodology. Validate across different splits and future years before generalizing these results.

Source: City of New York, NYC Open Data, dataset 5zyy-y8am. https://data.cityofnewyork.us/Environment/NYC-Building-Energy-and-Water-Data-Disclosure-for-/5zyy-y8am
'''
    (out/'Establishment_Emission_Analysis.md').write_text(report)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
