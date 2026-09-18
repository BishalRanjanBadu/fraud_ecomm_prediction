"""Scoring: request -> source frame -> build_features -> model -> calibrator."""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from . import constants as C
from .schema import ScoreRequest
from .transform import ContractError, build_features

# request fields whose values must be one of the trained levels
CATEGORICAL_FIELDS = {
    "payment_method": ("payment_method",), "payment_gateway": ("payment_gateway",),
    "shipping_speed": ("shipping_speed",), "delivery_type": ("delivery_type",),
    "email_domain_class": ("email_domain_class",), "acquisition_channel": ("acquisition_channel",),
    "merchant_category": ("merchant_category",), "asn_type": ("asn_type",), "asn_country": ("asn_country",),
    "network": ("card", "network"), "issuer": ("card", "issuer"), "product_type": ("card", "product_type"),
    "device_type": ("device", "device_type"), "browser_family": ("device", "browser_family"),
}


def _f(v):
    return np.nan if v is None else float(v)


def to_source_row(r: ScoreRequest) -> dict:
    card, dev, bsk, lg = r.card, r.device, r.basket, r.last_login
    return {
        "payment_ts": r.payment_ts,
        "payment_amount": r.payment_amount, "processing_fee": r.processing_fee,
        "discount_amount": _f(r.discount_amount), "item_count": _f(r.item_count),
        "attempt_seq_in_session": _f(r.attempt_seq_in_session),
        "is_guest_checkout": float(r.is_guest_checkout), "is_3ds_attempted": float(r.is_3ds_attempted),
        "is_3ds_success": float(r.is_3ds_success), "has_coupon": int(r.coupon_applied),
        "address_match_flag": _f(None if r.address_match is None else int(r.address_match)),
        "shipping_addr_age_hours": _f(r.shipping_addr_age_hours),
        "kyc_level": _f(r.kyc_level), "city_tier": _f(r.city_tier),
        "prior_return_rate": _f(r.prior_return_rate), "prior_orders_12m": _f(r.prior_orders_12m),
        "account_age_days": _f(r.account_age_days), "avg_ticket_size": r.avg_ticket_size,
        "trailing_chargeback_rate_bps": _f(r.trailing_chargeback_rate_bps),
        "reputation_score": _f(r.reputation_score),
        "ip_region_code": _f(r.ip_region_code), "home_region_code": _f(r.home_region_code),
        "ip_n_customers": float(r.ip_n_customers), "device_n_customers": _f(r.device_n_customers),
        "has_card": int(card is not None),
        "card_token_age_h": _f(card.token_age_h if card else None),
        "network": card.network if card else None, "issuer": card.issuer if card else None,
        "product_type": card.product_type if card else None,
        "issuing_country": card.issuing_country if card else None,
        "has_device_profile": int(dev is not None),
        "device_age_h": _f(dev.device_age_h if dev else None),
        "device_type": dev.device_type if dev else None,
        "browser_family": dev.browser_family if dev else None,
        "is_emulator": _f(None if dev is None else int(dev.is_emulator)),
        "has_basket": int(bsk is not None),
        "n_categories": _f(bsk.n_categories if bsk else None),
        "max_unit_price": _f(bsk.max_unit_price if bsk else None),
        "total_qty": _f(bsk.total_qty if bsk else None),
        "has_prior_login": int(lg is not None),
        "hours_since_last_login": _f(lg.hours_since if lg else None),
        "last_login_new_device": _f(None if lg is None else int(lg.new_device)),
        "last_login_unusual_loc": _f(None if lg is None else int(lg.unusual_location)),
        "last_login_failed_attempts": _f(None if lg is None or lg.failed_attempts in (None, C.FAILED_ATTEMPTS_SENTINEL)
                                         else lg.failed_attempts),
        "last_login_risk": _f(lg.risk_score if lg else None),
        "payment_method": r.payment_method, "payment_gateway": r.payment_gateway,
        "shipping_speed": r.shipping_speed, "delivery_type": r.delivery_type,
        "email_domain_class": r.email_domain_class, "acquisition_channel": r.acquisition_channel,
        "merchant_category": r.merchant_category, "asn_type": r.asn_type, "asn_country": r.asn_country,
        "ip_country": r.ip_country,
    }


def to_source_frame(reqs: list[ScoreRequest]) -> pd.DataFrame:
    df = pd.DataFrame([to_source_row(r) for r in reqs], columns=C.SOURCE_COLUMNS)
    df["payment_ts"] = pd.to_datetime(df["payment_ts"]).astype("datetime64[ns]")
    for c in C.SOURCE_NUMERIC:
        df[c] = pd.to_numeric(df[c]).astype("float64")
    for c in C.SOURCE_CATEGORICAL:
        df[c] = df[c].astype(object)
    return df


def required_categoricals(p06: dict, p03: dict) -> list[str]:
    """Categoricals that may NOT be null for this model version (derived from the artifacts, not hard-coded).

    A null is scoreable only if the model learned a branch for it: either the column carried structural gaps in
    training (nan_allowed), or notebook 03 filled gaps with a value that is itself a trained level (e.g.
    payment_gateway -> 'Unknown'). Otherwise the fill value becomes an unseen category the model never validated.
    """
    maps, nan_ok, fill = p06["tree"]["categorical"], set(p06["tree"]["nan_allowed"]), p03.get("cat_fill", {})
    return sorted(c for c in maps if c not in nan_ok and (c not in fill or fill[c] not in maps[c]))


def check_levels(reqs: list[ScoreRequest], p06: dict, p03: dict | None = None) -> None:
    """Categorical values must be trained levels; nulls only where the model has a trained meaning for them."""
    maps = p06["tree"]["categorical"]
    required = set(required_categoricals(p06, p03)) if p03 is not None else set()
    for r in reqs:
        for name, path in CATEGORICAL_FIELDS.items():
            obj = r
            for part in path:
                obj = getattr(obj, part) if obj is not None else None
            if name not in maps:
                continue
            if obj is None:
                if name in required and len(path) == 1:
                    raise ContractError(name, "required for this model version: training never contained a "
                                              "missing value here, so a gap cannot be scored faithfully")
                continue
            if obj not in maps[name]:
                raise ContractError(".".join(path), f"unknown level {obj!r}; allowed: {maps[name]}")


def score(reqs: list[ScoreRequest], loaded, explain: bool = False) -> list[dict]:
    check_levels(reqs, loaded.p06, loaded.p03)
    src = to_source_frame(reqs)
    X, _, filled = build_features(src, loaded.p06, loaded.p03, loaded.input_family)
    try:
        raw = loaded.model.predict_proba(X)[:, 1]
    except Exception as e:                      # re-raised with the evidence, never swallowed
        dtypes = {c: str(t) for c, t in X.dtypes.items()}
        raise RuntimeError(f"model.predict_proba failed: {type(e).__name__}: {e} | dtypes={dtypes}") from e
    cal = loaded.calibrator.predict(raw)
    contrib = contributions(loaded, X) if explain else None
    out = []
    for i, r in enumerate(reqs):
        applied = sorted(c for c in filled.columns if bool(filled[c].iloc[i]))
        item = {
            "prediction_id": str(uuid.uuid4()), "payment_id": r.payment_id, "source": "model",
            "model_version": loaded.version, "contract_version": C.CONTRACT_VERSION,
            "fraud_probability": float(cal[i]), "score_raw": float(raw[i]),
            "decision": bool(raw[i] >= loaded.threshold_raw), "threshold_raw": loaded.threshold_raw,
            "operating_point_provisional": True, "gate_status": loaded.gate_status,
            "defaults_applied": applied,
        }
        if contrib is not None:
            row, bias = contrib
            top = np.argsort(-np.abs(row[i]))[:10]
            item["explanation"] = [{"feature": X.columns[j], "contribution": float(row[i][j])} for j in top]
            item["explanation_bias"] = float(bias[i])
        out.append(item)
    return out


def contributions(loaded, X: pd.DataFrame):
    """Exact per-feature contributions in log-odds; they sum (with the bias) to the model margin."""
    m = loaded.model
    if loaded.family == "lightgbm":
        c = m.booster_.predict(X, pred_contrib=True, num_iteration=m.best_iteration_ or None)
    elif loaded.family == "xgboost":
        import xgboost as xgb
        rng = (0, m.best_iteration + 1) if getattr(m, "best_iteration", None) is not None else (0, 0)
        c = m.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True, iteration_range=rng)
    else:
        c = np.column_stack([X.to_numpy() * m.coef_[0], np.full(len(X), m.intercept_[0])])
    return c[:, :-1], c[:, -1]


def fallback(reqs: list[ScoreRequest], reason: str) -> list[dict]:
    """Opt-in incumbent rule. Every response is tagged; never used silently."""
    out = []
    for r in reqs:
        if r.payment_risk_score is None:
            raise ContractError("payment_risk_score", f"model unavailable ({reason}); fallback needs this field")
        out.append({
            "prediction_id": str(uuid.uuid4()), "payment_id": r.payment_id,
            "source": "fallback_incumbent_rule", "model_version": None,
            "contract_version": C.CONTRACT_VERSION, "fraud_probability": None, "score_raw": None,
            "decision": bool(r.payment_risk_score >= C.INCUMBENT_ALERT_THRESHOLD),
            "threshold_raw": C.INCUMBENT_ALERT_THRESHOLD, "operating_point_provisional": True,
            "gate_status": None, "defaults_applied": [],
        })
    return out
