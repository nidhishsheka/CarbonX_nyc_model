import json
import math
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

NUMERIC_FIELDS = {
    # model feature name: (label, min, max)
    "largest_property_use_type_1": ("Largest-use floor area (sq ft)", 1, 5e7),
    "_2nd_largest_property_use_1": ("2nd-largest-use floor area (sq ft)", 0, 5e7),
    "_3rd_largest_property_use_1": ("3rd-largest-use floor area (sq ft)", 0, 5e7),
    "year_built": ("Year built", 1600, None),
    "number_of_buildings": ("Number of buildings", 1, 10000),
    "occupancy": ("Occupancy (% of floor area)", 0, 100),
}
REQUIRED = ("largest_property_use_type_1", "primary_property_type_self")

INTERVAL_ABOVE = "Above prediction interval"
INTERVAL_WITHIN = "Within prediction interval"
INTERVAL_BELOW = "Below prediction interval"
REVIEW_ACTION = {
    INTERVAL_ABOVE: "Investigate reporting, energy use, and unmeasured operating conditions",
    INTERVAL_BELOW: "Check reporting completeness and documented efficiency measures",
    INTERVAL_WITHIN: "No interval-based exception; continue routine monitoring",
}


class BenchmarkError(ValueError):
    """Raised for invalid benchmark inputs."""


_cache = {}


def load_config():
    if "cfg" not in _cache:
        _cache["cfg"] = json.loads((MODEL_DIR / "government_benchmark.json").read_text())
    return _cache["cfg"]


def _model():
    if "model" not in _cache:
        from catboost import CatBoostRegressor
        cfg = load_config()
        m = CatBoostRegressor()
        m.load_model(str(MODEL_DIR / cfg["model_file"]))
        if list(m.feature_names_) != cfg["features"]:
            raise RuntimeError("Model features do not match government_benchmark.json")
        _cache["model"] = m
    return _cache["model"]


def _num(value, name):
    if value is None or str(value).strip() == "":
        return None
    try:
        v = float(str(value).replace(",", ""))
    except ValueError:
        raise BenchmarkError(f"{NUMERIC_FIELDS[name][0]} must be a number.")
    if not math.isfinite(v):
        raise BenchmarkError(f"{NUMERIC_FIELDS[name][0]} must be a finite number.")
    return v


def clean_inputs(raw):
   
    cfg = load_config()
    out, imputed = {}, []
    for name, (label, lo, hi) in NUMERIC_FIELDS.items():
        v = _num(raw.get(name), name)
        if v is None:
            if name in REQUIRED:
                raise BenchmarkError(f"{label} is required.")
            v = float(cfg["imputation_values"][name])
            imputed.append(name)
        if v < lo or (hi is not None and v > hi):
            raise BenchmarkError(f"{label} is outside the supported range.")
        out[name] = v
    if out["year_built"] > date.today().year:
        raise BenchmarkError("Year built cannot be in the future.")

    ptype = (raw.get("primary_property_type_self") or "").strip()
    if not ptype:
        raise BenchmarkError("Property type is required.")
    if ptype not in cfg["property_types"]:
        raise BenchmarkError("Unsupported property type (not seen in the training data).")
    borough = (raw.get("borough") or "").strip() or "Unknown"
    if borough != "Unknown" and borough not in cfg["boroughs"]:
        raise BenchmarkError("Unsupported borough.")
    out["primary_property_type_self"] = ptype
    out["borough"] = borough
    return out, imputed


def interval_comparison(reported, lower, upper):
    if reported is None:
        return None
    if reported > upper:
        return INTERVAL_ABOVE
    if reported < lower:
        return INTERVAL_BELOW
    return INTERVAL_WITHIN


def predict(raw_inputs, reported_tco2e=None):
    """Return the peer benchmark, 90% interval and (optional) comparison."""
    import pandas as pd
    cfg = load_config()
    inputs, imputed = clean_inputs(raw_inputs)
    if reported_tco2e is not None and (not math.isfinite(reported_tco2e) or reported_tco2e < 0):
        raise BenchmarkError("Reported emissions must be a non-negative number.")
    X = pd.DataFrame([inputs])[cfg["features"]]
    log_pred = float(_model().predict(X)[0])
    q = cfg["interval_log_quantile"]
    expected = max(float(np.expm1(log_pred)), 0.0)
    lower = max(float(np.expm1(log_pred - q)), 0.0)
    upper = max(float(np.expm1(log_pred + q)), 0.0)
    comparison = interval_comparison(reported_tco2e, lower, upper)
    return {
        "expected_tco2e": expected,
        "lower_tco2e": lower,
        "upper_tco2e": upper,
        "nominal_coverage": cfg["nominal_coverage"],
        "observed_test_coverage": cfg["test_coverage"],
        "reported_tco2e": reported_tco2e,
        "gap_tco2e": None if reported_tco2e is None else expected - reported_tco2e,
        "comparison": comparison,
        "review_action": REVIEW_ACTION.get(comparison),
        "inputs": inputs,
        "imputed_fields": imputed,
        "model_version": cfg["version"],
    }
