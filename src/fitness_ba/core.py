"""Standalone implementation of the frozen FAI -> BA calculation.

This module contains prediction only. It does not need or expose the original
participant-level training data.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PACKAGE_ROOT / "models"

ITEMS = {
    "grip": {"source": "Grip-score", "unit": "kg", "direction": 1, "transform": "identity", "bounds": [0.1, 150], "domain": "strength"},
    "jump": {"source": "Vertical-jump-score", "unit": "cm", "direction": 1, "transform": "identity", "bounds": [0, 150], "domain": "power"},
    "vital": {"source": "Vital-capacity-score", "unit": "mL", "direction": 1, "transform": "identity", "bounds": [300, 10000], "domain": "pulmonary"},
    "step": {"source": "Step-index-score", "unit": "index", "direction": 1, "transform": "identity", "bounds": [10, 200], "domain": "cardiovascular"},
    "reach": {"source": "Sit-reach-score", "unit": "cm", "direction": 1, "transform": "identity", "bounds": [-40, 60], "domain": "flexibility"},
    "balance": {"source": "One-leg-stand-score", "unit": "s", "direction": 1, "transform": "log1p", "bounds": [0, 600], "domain": "sensorimotor"},
    "reaction": {"source": "Reaction-time-score", "unit": "s", "direction": -1, "transform": "log", "bounds": [0.1, 3], "domain": "sensorimotor"},
    "pushup": {"source": "Pushup-score", "unit": "count", "direction": 1, "transform": "log1p", "bounds": [0, 200], "domain": "endurance", "sex": "M"},
    "situp": {"source": "Situp-score", "unit": "count", "direction": 1, "transform": "log1p", "bounds": [0, 150], "domain": "endurance", "sex": "F"},
}
COMMON = list(ITEMS)[:7]
DOMAINS = ["strength", "power", "pulmonary", "cardiovascular", "flexibility", "sensorimotor"]
KNOTS = np.array([20] * 4 + [25, 30, 35, 40] + [44.999] * 4, dtype=float)


@lru_cache(maxsize=1)
def model() -> dict:
    return json.loads((MODEL_DIR / "pfa_v1.json").read_text())


@lru_cache(maxsize=1)
def scale() -> dict:
    return json.loads((MODEL_DIR / "m0_ba_scale.json").read_text())


@lru_cache(maxsize=1)
def completion() -> dict:
    return json.loads((MODEL_DIR / "annual_bridge_models.json").read_text())


def _number(value: Any) -> float:
    if value is None or str(value).strip() in {"", "nan", "None", "NA", "N/A"}:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _sex(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return {"M": "M", "MALE": "M", "男": "M", "男性": "M", "F": "F", "FEMALE": "F", "女": "F", "女性": "F"}.get(text)


def _transform(value: np.ndarray | list[float], name: str) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    kind = ITEMS[name]["transform"]
    if kind == "log1p":
        return np.log1p(x)
    if kind == "log":
        return np.log(x)
    return x


def _basis(age: np.ndarray | list[float]) -> np.ndarray:
    x = np.clip(np.asarray(age, dtype=float).reshape(-1), 20, 44.999)
    return BSpline.design_matrix(x, KNOTS, 3).toarray()


def _reference(parameters: dict, age: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    basis = _basis(age)
    mu = (basis @ np.asarray(parameters["beta"])) * parameters["unit"] + parameters["center"]
    c = np.asarray(parameters["scale_beta"])
    log_sd = basis @ c if len(c) > 1 else np.repeat(c[0], len(basis))
    sd = np.exp(log_sd) * parameters["unit"]
    mu = mu + sd * parameters.get("cal_location", 0)
    sd = sd * parameters.get("cal_scale", 1)
    return mu, sd


def _clean_record(record: Mapping[str, Any]) -> tuple[dict, list[str]]:
    row: dict[str, Any] = {"sex": _sex(record.get("sex"))}
    flags: list[str] = []
    age = _number(record.get("exact_age", record.get("age")))
    dob = record.get("birthday", record.get("dob"))
    date = record.get("exam_date", record.get("exam_time", record.get("date")))
    if dob is not None and date is not None and str(dob).strip() and str(date).strip():
        b = pd.to_datetime(dob, errors="coerce")
        d = pd.to_datetime(date, errors="coerce")
        if pd.notna(b) and pd.notna(d) and d >= b:
            derived = (d - b).days / 365.2425
            if np.isfinite(age) and abs(derived - age) > 0.6:
                flags.append("supplied_age_disagrees_with_dates_dates_used")
            age = derived
        else:
            flags.append("invalid_birthday_or_exam_date")
    if not np.isfinite(age) or not 0 < age < 120:
        age = np.nan
    row["exact_age"] = age
    anchor = _number(record.get("RA_anchor", record.get("RA", record.get("age"))))
    row["RA_anchor"] = anchor
    for name, meta in ITEMS.items():
        value = _number(record.get(name, record.get(meta["source"])))
        if np.isfinite(value) and not meta["bounds"][0] <= value <= meta["bounds"][1]:
            flags.append(f"{name}:outside_QC_bounds")
            value = np.nan
        row[name] = value
    return row, flags


def _deficits(row: Mapping[str, Any], sex: str, names: list[str], reference_age: float) -> np.ndarray:
    values = []
    for name in names:
        mu, sd = _reference(model()["models"][sex][name], [reference_age])
        z = (_transform([row[name]], name)[0] - mu[0]) / sd[0]
        values.append(-ITEMS[name]["direction"] * z)
    return np.asarray(values, dtype=float)


def _complete_fai(row: Mapping[str, Any], sex: str, reference_age: float) -> tuple[float, dict, float]:
    deficits = _deficits(row, sex, COMMON, reference_age)
    domain_scores = {}
    for domain in DOMAINS:
        indexes = [i for i, name in enumerate(COMMON) if ITEMS[name]["domain"] == domain]
        domain_scores[domain] = float(np.mean(deficits[indexes]))
    raw = float(np.mean(list(domain_scores.values())))
    panel = model()["panels"][f"{sex}7"]
    fai = 50 + 10 * (raw - panel["fai_center"]) / panel["fai_sd"]
    empirical = np.asarray(panel.get("fai_empirical_raw", []))
    percentile = float(np.searchsorted(empirical, raw, side="right") / len(empirical) * 100) if len(empirical) else np.nan
    return float(fai), domain_scores, percentile


def _partial_fai(row: Mapping[str, Any], sex: str, names: list[str], reference_age: float):
    key = sex + "|" + ",".join(names)
    bridge = completion()["bridge_models"].get(key)
    if bridge is None:
        return None
    deficits = _deficits(row, sex, names, reference_age)
    fai = float(bridge["intercept"] + np.dot(np.asarray(bridge["coef"]), deficits))
    interval = fai + np.asarray(bridge["validation_residual_q025_q975"], dtype=float)
    return fai, interval, bridge


def score_record(record: Mapping[str, Any]) -> dict:
    """Score one assessment row using the frozen reference and BA scale.

    A complete seven-common-test row uses the main FAI definition. Supported
    incomplete patterns use the frozen ridge bridge and are low confidence.
    """
    row, flags = _clean_record(record)
    known_age = np.isfinite(row["exact_age"])
    assumed_age = float(completion()["assumed_reference_age"])
    reference_age_raw = float(row["exact_age"]) if known_age else assumed_age
    reference_age = float(np.clip(reference_age_raw, 20, 44.999))
    anchor = float(row["RA_anchor"]) if np.isfinite(row["RA_anchor"]) else reference_age_raw
    age_basis = "provided_RA_anchor" if np.isfinite(row["RA_anchor"]) else ("date_or_age" if known_age else "training_mean_age_assumption")
    common_names = [name for name in COMMON if np.isfinite(row[name])]
    recorded_sex = row["sex"]
    if recorded_sex:
        candidates = [(recorded_sex, 1.0)]
        sex_used, sex_basis = recorded_sex, "recorded"
    elif np.isfinite(row["pushup"]) and not np.isfinite(row["situp"]):
        candidates = [("M", 1.0)]
        sex_used, sex_basis = "M", "inferred_from_pushup_protocol"
    elif np.isfinite(row["situp"]) and not np.isfinite(row["pushup"]):
        candidates = [("F", 1.0)]
        sex_used, sex_basis = "F", "inferred_from_situp_protocol"
    else:
        prior = completion()["sex_prior"]
        candidates = [("F", prior["F"]), ("M", prior["M"])]
        sex_used, sex_basis = "pooled", "training_sex_weighted_average"

    predictions = []
    intervals = []
    domain_weighted: dict[str, list[tuple[float, float]]] = {}
    percentiles = []
    methods = []
    unsupported = []
    for sex, weight in candidates:
        if len(common_names) == 7:
            fai, domains, percentile = _complete_fai(row, sex, reference_age)
            methods.append("complete_7_common")
            percentiles.append((weight, percentile))
            for name, value in domains.items():
                domain_weighted.setdefault(name, []).append((weight, value))
            interval = None
        else:
            partial = _partial_fai(row, sex, common_names, reference_age)
            if partial is None:
                unsupported.append(sex + "|" + ",".join(common_names))
                continue
            fai, interval, _ = partial
            methods.append(f"partial_bridge_{len(common_names)}")
        predictions.append((weight, fai))
        if interval is not None:
            intervals.append((weight, interval))

    base = {
        "status": "evaluated" if predictions else "unsupported_missingness_pattern",
        "model_version": model()["version"],
        "BA_method_version": scale()["version"],
        "absolute_age_validated": False,
        "sex_used": sex_used,
        "sex_basis": sex_basis,
        "reference_age": reference_age,
        "reference_age_clamped": bool(reference_age != reference_age_raw),
        "BA_age_anchor": anchor,
        "BA_age_basis": age_basis,
        "observed_common_items": len(common_names),
        "observed_item_names": common_names,
        "input_flags": flags,
        "unsupported_bridge_keys": unsupported,
    }
    if not predictions:
        base.update(FAI=None, BA=None, BA_gap=None, BA_low_confidence=True)
        return base
    weight_sum = sum(weight for weight, _ in predictions)
    fai = sum(weight * value for weight, value in predictions) / weight_sum
    factor = float(scale()["years_per_fai_point"])
    ba = anchor + factor * (fai - 50)
    low_confidence = not (len(common_names) == 7 and recorded_sex and known_age and 20 <= reference_age_raw < 40)
    base.update(
        FAI=float(fai),
        BA=float(ba),
        BA_gap=float(ba - anchor),
        BA_method=";".join(sorted(set(methods))),
        BA_low_confidence=bool(low_confidence),
        BA_validated=False,
        reliability_note=("exploratory BA; partial/demographic/out-of-reference extension" if low_confidence else "internal-reference complete panel; absolute BA not clinically validated"),
    )
    if domain_weighted:
        base["domain_scores"] = {name: float(sum(w * v for w, v in pairs) / sum(w for w, _ in pairs)) for name, pairs in domain_weighted.items()}
    if percentiles:
        base["deficit_percentile"] = float(sum(w * v for w, v in percentiles) / sum(w for w, _ in percentiles))
    if intervals:
        iw = sum(weight for weight, _ in intervals)
        fai_bounds = sum(weight * bounds for weight, bounds in intervals) / iw
        base["BA_imputation_interval_low"] = float(anchor + factor * (fai_bounds[0] - 50))
        base["BA_imputation_interval_high"] = float(anchor + factor * (fai_bounds[1] - 50))
        base["BA_interval_scope"] = "partial-score bridge validation residual only"
    return base


def score_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, record in data.iterrows():
        result = score_record(record.to_dict())
        result["input_row"] = int(index) + 2
        if "oneid" in record:
            result["oneid"] = str(record["oneid"])
        rows.append(result)
    return pd.DataFrame(rows)
