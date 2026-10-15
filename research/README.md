# Carbon baseline research: datasets, CatBoost code and reproducible results

Two distinct experiments are provided. Do not pool their rows or compare their error units directly.

## 1. Government dataset — recommended main experiment
City of New York, NYC Open Data: NYC Building Energy and Water Data Disclosure for Local Law 84 2023 to Present (calendar year 2022 onward).
Official page: https://data.cityofnewyork.us/Environment/NYC-Building-Energy-and-Water-Data-Disclosure-for-/5zyy-y8am
Downloaded 5 October 2026: 103,259 rows, 16 selected original columns, report years 2022–2024. This is a documented column subset, not the full original schema. Original values are retained in the downloaded CSV. Source metadata and download query are included.

Target: total location-based GHG emissions, metric tonnes CO2e. Predictors: main and secondary use floor areas, occupancy percentage, construction year, number of buildings, self-selected property type, and borough. Floor areas refer to the listed use types, not a verified sum of all property area. There are no measured production volumes or operating-hour controls in this selected subset.

The code selects the latest year with at least 1,000 raw rows (2024 in this snapshot), filters incomplete/partial metering and reported estimates, removes parent/child reporting and conflicting duplicate property records, and creates property-disjoint train/validation/calibration/test partitions of approximately 60/15/10/15 percent. It tunes CatBoost depth and iteration count on validation log-MAE, compares median and Random Forest models, and calibrates 90% log-residual intervals using calibration properties only. All reported emissions, energy totals, EUI, ENERGY STAR scores, and emission intensities are excluded from predictors. No target-based trimming is done on the test set.

This is SAME-YEAR PEER BENCHMARKING, not future-year forecasting and not a causal no-intervention baseline. A negative or positive peer gap is not proof of additional emission reduction. The data are government-published building-owner submissions, not claimed independently verified sensor measurements. Building performance and carbon policy in New York do not automatically transfer to India.

Suggested provisional paper framing: “Uncertainty-Aware Machine Learning for Building Emission Benchmarking and Simulated Carbon Credit Assessment.” CatBoost and conformal residual calibration are existing methods: novelty needs a focused literature review and comparative research, not merely their inclusion.

## 2. Industrial time-series dataset — supporting experiment
UCI Steel Industry Energy Consumption, 35,040 records, 365 source calendar days in 2018, 96 readings per day. A single South Korean establishment; not multiple establishments.
Official page: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
Citation: V E, S., Shin, C., & Cho, Y. (2021). Steel Industry Energy Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C
License: CC BY 4.0. Retain attribution when sharing.

Target: Usage_kWh. Only calendar/schedule variables are used. Load_Type may be a label derived from electricity use; its provenance does not establish an independent production measurement, so it is excluded. CO2(tCO2) has inconsistent unit labelling in UCI documentation (column name suggests tonnes, units say ppm), so it is excluded. Contemporaneous reactive power and power factors are also excluded to avoid presenting an electrical reconstruction task as a causal baseline. No energy target lags are used, preventing adaptation to a simulated efficiency intervention through those lags.

Train/validation/calibration/test comprise consecutive whole days in approximately 55/15/15/15 proportions. CatBoost depth/iterations are tuned on validation only, then fitted to train+validation. Daily intervals use DAILY calibration residuals; interval-level bounds are not summed to claim daily coverage. Temporal dependence/distribution shift mean classical exchangeability coverage guarantees do not automatically apply.

Credit experiments inject 0%, 10%, and 20% synthetic reductions into held-out observations. Original observations are empirical counterfactuals ONLY inside this simulation. An assumed emission factor of 0.5 kgCO2e/kWh is used in the supplied demonstration; it is NOT an official Indian or Korean factor. Replace it with a documented regional/year factor for real accounting. Credits are illustrative research outputs, not certified tradable units. The 0% scenario quantifies false credit estimation. Daily over/underestimation sums are descriptive metrics, not a guaranteed interval for total annual credits.

## Run locally
```bash
pip install -r requirements.txt
python train_government.py
python train_baseline.py --emission-factor 0.5
```
For energy-only results, omit --emission-factor. Run the scripts from the extracted package or open Carbon_Baseline_Research.ipynb in Colab/Jupyter. The notebook contains the program sources and can download both datasets itself when they are absent. Downloads require internet access. GPU is not required.

Outputs include test MAE/RMSE/R2, tuning tables, feature importance, predictions, calibrated intervals, simulation errors and saved CatBoost models. Feature importance is predictive association, not causal explanation. The government model predicts log1p emissions, so raw-scale predictions approximate a conditional geometric central tendency rather than an unbiased mean. Always report both log and raw-scale evaluation.

## Further work needed for the paper
1. Add pre-intervention energy data and dates of actual efficiency changes; freeze baseline models before intervention.
2. Include independently measured operating hours, occupancy, production, and weather.
3. Evaluate multiple random property splits and future-year temporal tests on government data; the supplied benchmark is one fixed split.
4. Check residual interval coverage by property type and occupancy; overall coverage alone is insufficient.
5. Evaluate missingness, reporting errors, activity changes, and simulated interventions separately.
6. Keep a separate regulatory credit-verification and trading ledger layer. Peer predictions are not emission caps or issued credits.

No results in this package are fabricated; saved results came from executing the accompanying scripts. Exact numeric results may vary with library versions. run_metadata.json records industrial runtime versions and split dates.
