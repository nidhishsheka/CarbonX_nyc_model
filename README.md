# CarbonX

Carbon accounting, **peer emission benchmarking with uncertainty**, and a **simulated** carbon-credit
marketplace. This version restructures the project around the student research workflow in
`docs/student_workflow.jpg` and `docs/Carbon_Project_Student_Description.pdf`.

## What changed in this version

| Area | Before | Now |
|---|---|---|
| Expected emissions | none (manual baseline number) | CatBoost peer benchmark + calibrated 90% interval (`benchmark.py`, `/benchmark`) |
| Credits | presented as real | labelled **simulated**; never derived from the peer benchmark |
| Research evidence | separate zip | `research/` (code, data, results) and `/benchmark/model` (metrics, coverage, 0/10/20% simulation) |
| Grid factor | CEA v22.0 (separate repo) | merged: `import_cea.py`, `migrate_carbonx.py`, `import_cea_credits.py` live here |

The central finding of the research drives the design: with **no real reduction**, the point benchmark
still produced a positive "credit" for 45.6% of properties (7.3% with the lower bound). A peer gap is not
evidence of a reduction, so the benchmark informs review and never mints credits.

## Run

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5000   (FLASK_DEBUG=1 for debug mode)
python -m unittest discover -s tests -v
```

The shipped `cc.db` already contains the CEA data and the benchmark tables. To rebuild from an older
database: `python migrate_carbonx.py && python import_cea.py && python migrate_benchmark.py`
(each writes a timestamped backup first; set `CARBONX_DB` or pass a path to use another file).

## Layout

```
app.py                 Flask routes (trading fixes, benchmark pages)
db.py                  connection + exclusive write transactions
benchmark.py           peer benchmark: validation, CatBoost prediction, interval, comparison
migrate_benchmark.py   adds Benchmark_* / Credit_Simulation / Establishment_Profile tables
ml_model.py            next-day credit price forecast (market model, unrelated to emissions)
models/                government_catboost.cbm + government_benchmark.json (interval, imputation, categories)
research/              original analysis package (notebook, train_*.py, results, NYC and steel data)
tests/                 25 tests, incl. exact reproduction of the research predictions
```

## Scope and limits (read before relying on any number)

* The benchmark is trained on **New York City** buildings (LL84, report year 2024, self-reported). It has
  **not** been validated for Indian establishments. Its nominal 90% interval covered 87.7% of test
  properties. One fixed split, one year; repeated splits and future-year tests are still to do.
* Credits here come from `baseline x sector reduction - recorded emissions` and are simulated. Seven seeded
  sectors (Mining, Food, Electronics, Petrochemical, Glass, Shipping, Agriculture) have **no reduction
  policy** and are treated as 0%; the report flags them.
* Only grid **electricity** has a sourced factor (CEA v22.0). Coal, diesel, natural gas and petrol factors
  are still unsourced seed values.
* The steel time-series experiment (`research/`) uses an illustrative 0.5 kgCO2e/kWh factor and is not
  connected to the application.
* A verification layer (evidence, additionality, verifier sign-off) before credits become tradable is
  future work; the research README lists the data it would need.
