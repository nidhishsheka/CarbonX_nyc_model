# Establishment emissions: benchmark comparison and simulation

CatBoost was used to compare reported annual emissions with a peer benchmark for 1,828 held-out NYC properties in 2024. All emission quantities below use metric tonnes CO2e.

## Prediction results
Test MAE: 194.36 tonnes CO2e. Test RMSE: 797.52. Test R2: 0.677.
Nominal 90% interval coverage was 87.69%, below nominal. These flags are exploratory model exceptions, not findings of excessive legal emissions or inefficiency.

## Property comparison
994 properties reported emissions above the point benchmark. Comparing against the wider prediction intervals gives:

| Result | Properties |
|---|---:|
| Above upper prediction bound | 91 |
| Within prediction interval | 1603 |
| Below lower prediction bound | 134 |

Properties above the upper bound merit investigation of missing operating conditions, reporting problems, and energy consumption. They are not automatically violating a limit. Properties below the lower bound also merit checks: incomplete reporting and efficiency improvements can both produce low emissions.

## Illustrative properties for investigation
Sorted by excess over the point benchmark; this is not a verified inefficiency ranking.

| Property ID | Type | Reported emissions | Predicted benchmark | Upper bound |
|---|---|---:|---:|---:|
| 65345892 | Office | 25,965.14 | 181.56 | 372.39 |
| 2869789 | Hospital (General Medical & Surgical) | 10,967.78 | 3,177.72 | 6,500.24 |
| 2831001 | Multifamily Housing | 8,669.25 | 1,204.83 | 2,465.22 |
| 2831078 | Multifamily Housing | 7,232.06 | 2,769.06 | 5,664.43 |
| 2612621 | Residential Care Facility | 6,360.39 | 2,222.90 | 4,547.41 |

## Synthetic credit experiment
Original held-out emissions serve as empirical counterfactuals only inside simulation. We inject 0%, 10%, and 20% reductions, without changing property characteristics. No observed intervention or certified credits are present.

Point estimate = max(predicted peer benchmark - simulated actual emissions, 0).
Conservative estimate = max(lower prediction bound - simulated actual emissions, 0).

Under the 0% reduction scenario, the true synthetic reduction is zero. Nevertheless, point estimates generated positive simulated credits for 45.62% of properties. Lower-bound estimates reduced this proportion to 7.33%.

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
