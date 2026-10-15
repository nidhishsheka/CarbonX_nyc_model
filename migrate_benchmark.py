"""Add the peer-benchmark layer to cc.db (idempotent; writes a backup first).

New tables:
  Benchmark_Model         one row per trained model (metrics, interval, scope, limits)
  Benchmark_Model_Metric  Median / Random Forest / CatBoost comparison on held-out test data
  Credit_Simulation       the 0% / 10% / 20% synthetic-reduction experiment
  Establishment_Profile   the eight model inputs for an establishment (optional)
  Benchmark_Prediction    stored benchmark + interval (+ comparison with reported emissions)

Nothing here touches Carbon_Credit, Credit_Wallet, Credit_Ledger or any credit trigger:
a peer benchmark is not an emission limit and never mints credits.
"""
import csv
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
db_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "cc.db")
cfg = json.loads((ROOT / "models" / "government_benchmark.json").read_text())
sim_csv = ROOT / "research" / "government_results" / "government_simulation_metrics.csv"

backup = f"{db_path}.bak_{datetime.now():%Y%m%d_%H%M%S}"
shutil.copy2(db_path, backup)
print(f"Backup written: {backup}")

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='Data_Provenance'").fetchone():
    sys.exit("Data_Provenance not found - run migrate_carbonx.py first.")

conn.executescript("""
CREATE TABLE IF NOT EXISTS Benchmark_Model (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    source TEXT,
    report_year INTEGER,
    algorithm TEXT NOT NULL DEFAULT 'CatBoost regression on log(1 + tCO2e)',
    interval_method TEXT NOT NULL DEFAULT 'split-conformal absolute log-residual, nominal 90%',
    interval_log_quantile REAL NOT NULL,
    nominal_coverage REAL NOT NULL,
    test_coverage REAL NOT NULL,
    mean_interval_width_tco2e REAL,
    train_rows INTEGER, validation_rows INTEGER, calibration_rows INTEGER, test_rows INTEGER,
    model_sha256 TEXT,
    scope TEXT NOT NULL,
    limitations TEXT NOT NULL,
    provenance_id INTEGER REFERENCES Data_Provenance(provenance_id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS Benchmark_Model_Metric (
    model_id INTEGER NOT NULL REFERENCES Benchmark_Model(model_id),
    method TEXT NOT NULL,
    mae_tco2e REAL, rmse_tco2e REAL, r2 REAL, mae_log1p REAL,
    PRIMARY KEY (model_id, method)
);
CREATE TABLE IF NOT EXISTS Credit_Simulation (
    model_id INTEGER NOT NULL REFERENCES Benchmark_Model(model_id),
    reduction_fraction REAL NOT NULL,
    method TEXT NOT NULL,
    mae_tco2e REAL,
    sum_overestimation_tco2e REAL,
    sum_underestimation_tco2e REAL,
    fraction_overcredited REAL,
    is_synthetic INTEGER NOT NULL DEFAULT 1 CHECK (is_synthetic = 1),
    PRIMARY KEY (model_id, reduction_fraction, method)
);
CREATE TABLE IF NOT EXISTS Establishment_Profile (
    est_id INTEGER PRIMARY KEY REFERENCES Establishment(est_id),
    property_type TEXT,
    borough TEXT,
    largest_use_area_sqft REAL,
    second_use_area_sqft REAL,
    third_use_area_sqft REAL,
    year_built INTEGER,
    number_of_buildings INTEGER,
    occupancy_pct REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS Benchmark_Prediction (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    est_id INTEGER REFERENCES Establishment(est_id),
    model_id INTEGER NOT NULL REFERENCES Benchmark_Model(model_id),
    period TEXT,
    expected_tco2e REAL NOT NULL,
    lower_tco2e REAL NOT NULL,
    upper_tco2e REAL NOT NULL,
    reported_tco2e REAL,
    comparison TEXT CHECK (comparison IN
        ('Above prediction interval','Within prediction interval','Below prediction interval')),
    review_action TEXT,
    inputs_json TEXT NOT NULL,
    imputed_fields TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_bench_pred_est ON Benchmark_Prediction(est_id, created_at);
""")

prov = conn.execute("SELECT provenance_id FROM Data_Provenance WHERE dataset_name = ?",
                    ("NYC Building Energy and Water Data Disclosure (Local Law 84)",)).fetchone()
if prov is None:
    prov_id = conn.execute("""
        INSERT INTO Data_Provenance (dataset_name, organization, dataset_version, source_url,
            license, retrieval_date, methodology, is_synthetic)
        VALUES (?,?,?,?,?,?,?,0)""", (
        "NYC Building Energy and Water Data Disclosure (Local Law 84)",
        "City of New York (NYC Open Data, dataset 5zyy-y8am)",
        f"report year {cfg['report_year']} (3 years downloaded)",
        "https://data.cityofnewyork.us/Environment/NYC-Building-Energy-and-Water-Data-Disclosure-for-/5zyy-y8am",
        "See NYC Open Data terms of use",
        "2026-10-05",
        "Self-reported building-owner submissions; filtered for estimated values, meter gaps, "
        "partial metering, parent/child reporting and duplicates. Used only for peer benchmarking.",
    )).lastrowid
else:
    prov_id = prov["provenance_id"]

row = conn.execute("SELECT model_id FROM Benchmark_Model WHERE version = ?", (cfg["version"],)).fetchone()
p = cfg["partition_rows"]
vals = (cfg["name"], cfg["source"], cfg["report_year"], cfg["interval_log_quantile"],
        cfg["nominal_coverage"], cfg["test_coverage"], cfg["mean_interval_width_tCO2e"],
        p["train"], p["validation"], p["calibration"], p["test"], cfg["model_sha256"],
        cfg["scope"], "\n".join(cfg["limitations"]), prov_id)
if row is None:
    model_id = conn.execute("""INSERT INTO Benchmark_Model
        (version, name, source, report_year, interval_log_quantile, nominal_coverage, test_coverage,
         mean_interval_width_tco2e, train_rows, validation_rows, calibration_rows, test_rows,
         model_sha256, scope, limitations, provenance_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (cfg["version"],) + vals).lastrowid
else:
    model_id = row["model_id"]

for m in cfg["metrics"]:
    conn.execute("""INSERT OR REPLACE INTO Benchmark_Model_Metric
        (model_id, method, mae_tco2e, rmse_tco2e, r2, mae_log1p) VALUES (?,?,?,?,?,?)""",
        (model_id, m["model"], m["MAE_tCO2e"], m["RMSE_tCO2e"], m["R2"], m["MAE_log1p"]))

with open(sim_csv, newline="") as fh:
    for r in csv.DictReader(fh):
        conn.execute("""INSERT OR REPLACE INTO Credit_Simulation
            (model_id, reduction_fraction, method, mae_tco2e, sum_overestimation_tco2e,
             sum_underestimation_tco2e, fraction_overcredited)
            VALUES (?,?,?,?,?,?,?)""", (
            model_id, float(r["synthetic_reduction_fraction"]), r["method"],
            float(r["MAE_tCO2e"]), float(r["sum_overestimation_tCO2e"]),
            float(r["sum_underestimation_tCO2e"]), float(r["fraction_properties_overcredited"])))
conn.commit()

print(f"Benchmark model {cfg['version']} registered (model_id {model_id})")
for t in ("Benchmark_Model", "Benchmark_Model_Metric", "Credit_Simulation",
          "Establishment_Profile", "Benchmark_Prediction"):
    print(f"  {t}: {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows")
credits_touched = conn.execute("SELECT COUNT(*) FROM Carbon_Credit").fetchone()[0]
print(f"Carbon_Credit untouched ({credits_touched} rows).")
conn.close()
