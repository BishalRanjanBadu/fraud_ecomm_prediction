"""One transform for training parity and serving: 02 derivations -> 03 imputation -> 06 encoding.

Input is a *source frame*: one row per payment at stage-02 grain (see constants.SOURCE_COLUMNS), exactly what
notebook 02 wrote before notebook 03 imputed it. Nothing here is fitted: the 03 medians and the 06 encoder maps
come from the promoted artifacts. tests/test_transform_parity.py proves this reproduces the notebooks' saved
stage files cell for cell.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import constants as C


class ContractError(ValueError):
    """A request that the model cannot score faithfully (maps to HTTP 422)."""

    def __init__(self, field: str, detail: str):
        super().__init__(f"{field}: {detail}")
        self.field, self.detail = field, detail


# ----------------------------------------------------------------------------- 02: derivations
def derive(src: pd.DataFrame) -> pd.DataFrame:
    """Notebook 02, cell 'Deterministic feature engineering' (row-wise part), verbatim."""
    missing = [c for c in C.SOURCE_COLUMNS if c not in src.columns]
    if missing:
        raise ValueError(f"source frame is missing columns: {missing}")
    df = src.copy()
    df["amount_vs_merchant_ticket"] = df.payment_amount / df.avg_ticket_size
    df["ip_country_mismatch"] = np.where(df.ip_country.isna(), np.nan, (df.ip_country != "IN").astype(float))
    df["ip_country_missing"] = df.ip_country.isna().astype(int)
    df["ip_region_mismatch"] = np.where(df.ip_region_code.isna() | df.home_region_code.isna(), np.nan,
                                        (df.ip_region_code != df.home_region_code).astype(float))
    df["issuer_foreign"] = np.where(df.issuing_country.isna(), np.nan, (df.issuing_country != "IN").astype(float))
    df["hour_of_day"] = df.payment_ts.dt.hour
    df["day_of_week"] = df.payment_ts.dt.dayofweek
    df["is_weekend"] = df.day_of_week.isin([5, 6]).astype(int)
    df["is_night"] = df.hour_of_day.between(0, 5).astype(int)
    return df


# ----------------------------------------------------------------------------- 03: imputation
def _group_key(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def impute(df: pd.DataFrame, p03: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Notebook 03 with its train-fitted values. Returns (frame, mask of cells that were filled)."""
    out = df.copy()
    filled = pd.DataFrame(index=out.index)
    gm, cf = p03.get("global_medians", {}), p03.get("cat_fill", {})

    def _fill(col, values):
        m = out[col].isna()
        if m.any():
            filled[col] = m if col not in filled else (filled[col] | m)
            out[col] = out[col].where(~m, values)

    for key in C.GROUP_KEYS:                                  # keys first, exactly as notebook 03
        if key in out.columns and out[key].isna().any():
            if key in gm:
                _fill(key, gm[key])
            elif key in cf:
                _fill(key, cf[key])
                out[key] = out[key].astype(object)
    for col, g in p03.get("group_medians", {}).items():
        if col not in out.columns:
            continue
        by_group = out[g["by"]].map(lambda v: g["map"].get(_group_key(v)))
        by_group = pd.to_numeric(by_group, errors="raise").astype("float64").fillna(g["fallback"])
        _fill(col, by_group)
    for col, v in gm.items():
        if col in out.columns:
            _fill(col, v)
    for col, v in p03.get("sentinel_fill", {}).items():
        if col in out.columns:
            _fill(col, v)
    for col, v in cf.items():
        if col in out.columns:
            m = out[col].isna()
            if m.any():
                out[col] = out[col].astype(object).where(~m, v)
                filled[col] = m
    # notebook 03 cell 'Structural indicators' and the log columns (computed after imputation)
    for base in C.LOG_BASES:
        out["log_" + base] = np.log1p(out[base].clip(lower=0))
    return out, filled.reindex(columns=sorted(filled.columns)).fillna(False).astype(bool)


# ----------------------------------------------------------------------------- 06: encoding (verbatim)
def feature_contract_hash(p: dict) -> str:
    core = {"tree": [p["tree"]["feature_order"], p["tree"]["dtypes"], p["tree"]["categorical"]],
            "linear": [p["linear"]["feature_order"], p["linear"]["categorical"]]}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def linear_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in C.LINEAR_NUMERIC:
        cols[c] = frame[c].astype("float64")
    for c in C.LINEAR_LOG1P:
        x = frame[c].astype("float64")
        if (x < 0).any():
            raise ContractError(c, "must be >= 0")
        cols[f"log1p_{c}"] = np.log1p(x)
    for c, period in C.LINEAR_CYCLIC.items():
        ang = 2 * np.pi * frame[c].astype("float64") / period
        cols[f"{c}_sin"], cols[f"{c}_cos"] = np.sin(ang), np.cos(ang)
    return pd.DataFrame(cols, index=frame.index)


def encode_tree(frame: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Categories always come from the encoder map, never from the rows in hand."""
    cols = {c: frame[c].astype("float64") for c in p["tree"]["numeric"]}
    for c, levels in p["tree"]["categorical"].items():
        cols[c] = pd.Categorical(frame[c].astype(object).where(frame[c].notna(), None), categories=levels)
    return pd.DataFrame(cols, index=frame.index)[p["tree"]["feature_order"]]


def encode_linear(frame: pd.DataFrame, p: dict) -> pd.DataFrame:
    lp = p["linear"]
    num = linear_numeric_frame(frame)
    for c in lp["explicit_indicators"]:
        num[f"{c}__isna"] = num[c].isna().astype("float64")
    num = num.fillna(pd.Series(lp["medians"]))
    num = (num[lp["numeric_order"]] - pd.Series(lp["means"])) / pd.Series(lp["stds"])
    cols = {}
    for c, levels in lp["categorical"].items():
        s = frame[c].astype(object).where(frame[c].notna(), "__MISSING__").astype(str)
        for level in levels:
            cols[f"{c}=={level}"] = s.eq(level).astype("float64")
    out = pd.concat([num, pd.DataFrame(cols, index=frame.index)], axis=1)[lp["feature_order"]].astype("float64")
    if not np.isfinite(out.to_numpy()).all():
        raise ContractError("features", "non-finite values in linear encoding")
    return out


def restore_tree_dtypes(X: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Re-apply trained dtypes to an already-encoded tree frame (e.g. read back from Parquet)."""
    cols = {c: X[c].astype("float64") for c in p["tree"]["numeric"]}
    for c, levels in p["tree"]["categorical"].items():
        cols[c] = pd.Categorical(X[c].astype(object), categories=levels)
    return pd.DataFrame(cols, index=X.index)[p["tree"]["feature_order"]]


# ----------------------------------------------------------------------------- full pipeline
def build_features(src: pd.DataFrame, p06: dict, p03: dict, input_family: str):
    """source frame -> (model-ready X, stage-04-like frame, filled-cell mask)."""
    frame, filled = impute(derive(src), p03)
    if input_family == "tree":
        X = encode_tree(frame, p06)
        allowed = set(p06["tree"]["nan_allowed"])
        bad = [c for c in X.columns if c not in allowed and X[c].isna().any()]
        if bad:
            raise ContractError(bad[0], f"no value after imputation (the model never saw a gap here): {bad}")
    elif input_family == "linear":
        X = encode_linear(frame, p06)
    else:
        raise ValueError(f"unknown input family {input_family!r}")
    return X, frame, filled
