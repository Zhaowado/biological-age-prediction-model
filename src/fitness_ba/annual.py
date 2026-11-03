"""Annual aggregation helpers for wide visits and item-value long records."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .core import COMMON, ITEMS, score_record


def _last_nonnull(series: pd.Series):
    value = series.dropna()
    return value.iloc[-1] if len(value) else np.nan


def item_long_to_visits(data: pd.DataFrame, basic: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convert one-row-per-test source data into one-row-per-visit data."""
    required = {"oneid", "exam_item", "value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"item-long input missing columns: {sorted(missing)}")
    x = data.copy()
    x["oneid"] = x["oneid"].astype(str)
    date_col = "exam_time" if "exam_time" in x else "exam_date"
    if date_col not in x:
        raise ValueError("item-long input requires exam_time or exam_date")
    x["exam_date"] = pd.to_datetime(x[date_col], errors="coerce")
    source_to_name = {meta["source"]: name for name, meta in ITEMS.items()}
    x["item"] = x["exam_item"].map(source_to_name)
    x["numeric_value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x[x["item"].notna() & x["exam_date"].notna()].copy()
    if "QC" in x:
        x = x[pd.to_numeric(x["QC"], errors="coerce").eq(1)]
    valid = np.zeros(len(x), dtype=bool)
    for name, meta in ITEMS.items():
        mask = x["item"].eq(name)
        valid |= mask & x["numeric_value"].between(*meta["bounds"])
    x = x[valid].copy()
    x["source_order"] = np.arange(len(x))
    x = x.sort_values("source_order").drop_duplicates(["oneid", "exam_date", "item"], keep="last")
    visits = x.pivot(index=["oneid", "exam_date"], columns="item", values="numeric_value").reset_index()
    metadata = [c for c in ["age", "year", "sex", "birthday", "dob"] if c in x]
    if metadata:
        meta = x.groupby(["oneid", "exam_date"], as_index=False)[metadata].agg(_last_nonnull)
        visits = visits.merge(meta, on=["oneid", "exam_date"], how="left")
    if basic is not None:
        b = basic.copy()
        b["oneid"] = b["oneid"].astype(str)
        keep = [c for c in ["oneid", "sex", "birthday", "dob"] if c in b]
        visits = visits.merge(b[keep].drop_duplicates("oneid"), on="oneid", how="left", suffixes=("", "_basic"))
        for name in ["sex", "birthday", "dob"]:
            other = name + "_basic"
            if other in visits:
                visits[name] = visits.get(name, pd.Series(index=visits.index, dtype=object)).fillna(visits[other])
                visits = visits.drop(columns=other)
    return visits


def annualize_visits(data: pd.DataFrame) -> pd.DataFrame:
    """Use the last valid value of each test within each person-year."""
    if "oneid" not in data:
        raise ValueError("input requires oneid")
    x = data.copy()
    x["oneid"] = x["oneid"].astype(str)
    date_col = next((c for c in ["exam_date", "exam_time", "date"] if c in x), None)
    if date_col:
        x["exam_date"] = pd.to_datetime(x[date_col], errors="coerce")
    else:
        x["exam_date"] = pd.NaT
    if "year" not in x:
        x["year"] = x["exam_date"].dt.year
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    if x["year"].isna().any():
        raise ValueError("every visit requires year or a valid exam date")
    # Apply physical QC before taking the last value, so a later invalid score
    # cannot hide an earlier valid measurement in the same year.
    for name, meta in ITEMS.items():
        source = name if name in x else meta["source"] if meta["source"] in x else None
        values = pd.to_numeric(x[source], errors="coerce") if source else pd.Series(np.nan, index=x.index)
        x[name] = values.where(values.between(*meta["bounds"]))
    x = x.sort_values(["oneid", "year", "exam_date"], na_position="first")
    records = []
    for (oneid, year), group in x.groupby(["oneid", "year"], sort=True):
        latest_date = group["exam_date"].dropna().max()
        row = {"oneid": oneid, "year": int(year)}
        if pd.notna(latest_date):
            row["exam_date"] = latest_date.strftime("%Y-%m-%d")
        for name in ["sex", "age", "exact_age", "RA", "RA_anchor", "birthday", "dob"]:
            if name in group:
                row[name] = _last_nonnull(group[name])
        for name in ITEMS:
            row[name] = _last_nonnull(group[name])
        row["visit_count"] = int(len(group))
        records.append(row)
    return pd.DataFrame(records)


def score_annual(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = annualize_visits(data)
    results = []
    for _, row in annual.iterrows():
        result = score_record(row.to_dict())
        result.update(oneid=row["oneid"], year=int(row["year"]), visit_count=int(row["visit_count"]))
        results.append(result)
    long = pd.DataFrame(results).sort_values(["oneid", "year"])
    wide = long.pivot(index="oneid", columns="year", values="BA").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return long, wide


def read_and_score(input_path: str | Path, basic_path: str | Path | None = None):
    data = pd.read_csv(input_path, dtype={"oneid": str})
    if {"exam_item", "value"}.issubset(data.columns):
        basic = pd.read_csv(basic_path, dtype={"oneid": str}) if basic_path else None
        data = item_long_to_visits(data, basic)
    return score_annual(data)
