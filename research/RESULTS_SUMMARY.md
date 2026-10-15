# Executed benchmark results

Government: 103,259 raw selected-column building-year rows (2022–2024); 12,183 properties retained in the 2024 modelling sample after documented filters. Test properties: 1,828.

| Model | MAE (tonnes CO2e) | RMSE (tonnes CO2e) | R2 |
|---|---:|---:|---:|
| Median |447.28|1438.85|-0.050|
| Random Forest |202.80|828.16|0.652|
| CatBoost |194.36|797.52|0.677|

Nominal 90% government intervals achieved 87.69% empirical test coverage; average width 842.62 tonnes CO2e. This is under nominal coverage and must be reported as a limitation, not a guaranteed 90% result. Evaluate subgroups and multiple splits before paper submission.

Industrial CatBoost: 15-minute MAE 11.14 kWh, RMSE 18.66 kWh, R2 0.641; daily MAE 776.10 kWh, RMSE 989.63 kWh, R2 0.416. Weekly schedule baseline is almost as good; calendar variables alone provide limited additional ML advantage. Daily nominal 90% intervals: observed coverage 94.55% on 55 test days, mean width 3120.50 kWh.

Both simulation outputs use invented 0%,10%,20% efficiency reductions, clearly marked synthetic. Government emissions use reported tonnes directly; industrial emission conversion uses illustrative 0.5 kgCO2e/kWh. Neither experiment demonstrates real credit eligibility, additionality, carbon neutrality, or trading compliance.
