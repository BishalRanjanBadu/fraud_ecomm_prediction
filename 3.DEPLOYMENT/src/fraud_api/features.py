"""Reference code for the upstream feature producer (the API does not hold velocity state).

`asof_distinct_count` is notebook 02's definition, verbatim. A caller computing `ip_n_customers` or
`device_n_customers` any other way creates train/serve skew; tests/test_features.py pins the semantics.

`stage02_row_to_request` is the canonical mapping from a notebook-02 row to a v1 request. The fixture builder,
the golden recorder and the parity tests all use it, so the documented contract and the tested contract are one.
"""
from __future__ import annotations

import pandas as pd


def asof_distinct_count(frame: pd.DataFrame, key: str, entity: str = "customer_id",
                        ts: str = "payment_ts") -> pd.Series:
    """Distinct `entity` values seen on `key` at or before each row's `ts` (inclusive). NaN where `key` is null."""
    f = frame.loc[frame[key].notna(), [key, entity, ts]]
    first = (f.groupby([key, entity], as_index=False, sort=True)[ts].min()
               .sort_values(ts, kind="mergesort"))
    first["_n"] = first.groupby(key, sort=False).cumcount() + 1
    rows = f.assign(_row=f.index).sort_values(ts, kind="mergesort")
    m = pd.merge_asof(rows, first[[key, ts, "_n"]], on=ts, by=key,
                      direction="backward", allow_exact_matches=True)
    assert m["_n"].notna().all(), f"as-of join left gaps for {key}"
    return m.set_index("_row")["_n"].reindex(frame.index).astype("float64")


def _v(row, col):
    x = row[col]
    return None if pd.isna(x) else x


def _i(row, col):
    x = _v(row, col)
    return None if x is None else int(x)


def _fl(row, col):
    x = _v(row, col)
    return None if x is None else float(x)


def _b(row, col):
    x = _v(row, col)
    return None if x is None else bool(int(x))


def stage02_row_to_request(row) -> dict:
    card = None
    if pd.notna(row["card_token"]):
        card = {"token_age_h": _fl(row, "card_token_age_h"), "network": _v(row, "network"),
                "issuing_country": _v(row, "issuing_country"), "issuer": _v(row, "issuer"),
                "product_type": _v(row, "product_type")}
    device = None
    if pd.notna(row["device_type"]):
        device = {"device_age_h": _fl(row, "device_age_h"), "device_type": _v(row, "device_type"),
                  "is_emulator": _b(row, "is_emulator"), "browser_family": _v(row, "browser_family")}
    basket = None
    if pd.notna(row["n_lines"]):
        basket = {"n_categories": _i(row, "n_categories"), "max_unit_price": _fl(row, "max_unit_price"),
                  "total_qty": _i(row, "total_qty")}
    last_login = None
    if pd.notna(row["hours_since_last_login"]):
        last_login = {"hours_since": _fl(row, "hours_since_last_login"),
                      "new_device": _b(row, "last_login_new_device"),
                      "unusual_location": _b(row, "last_login_unusual_loc"),
                      "failed_attempts": _i(row, "last_login_failed_attempts"),
                      "risk_score": _fl(row, "last_login_risk")}
    return {
        "payment_id": str(row["payment_id"]),
        "payment_ts": pd.Timestamp(row["payment_ts"]).isoformat(),
        "payment_method": _v(row, "payment_method"), "payment_gateway": _v(row, "payment_gateway"),
        "payment_amount": _fl(row, "payment_amount"), "processing_fee": _fl(row, "processing_fee"),
        "discount_amount": _fl(row, "discount_amount"), "item_count": _i(row, "item_count"),
        "attempt_seq_in_session": _i(row, "attempt_seq_in_session"),
        "is_guest_checkout": _b(row, "is_guest_checkout"), "is_3ds_attempted": _b(row, "is_3ds_attempted"),
        "is_3ds_success": _b(row, "is_3ds_success"), "coupon_applied": bool(int(row["has_coupon"])),
        "address_match": _b(row, "address_match_flag"),
        "shipping_speed": _v(row, "shipping_speed"), "shipping_addr_age_hours": _fl(row, "shipping_addr_age_hours"),
        "delivery_type": _v(row, "delivery_type"),
        "kyc_level": _i(row, "kyc_level"), "city_tier": _i(row, "city_tier"),
        "email_domain_class": _v(row, "email_domain_class"), "acquisition_channel": _v(row, "acquisition_channel"),
        "account_age_days": _fl(row, "account_age_days"), "prior_return_rate": _fl(row, "prior_return_rate"),
        "prior_orders_12m": _i(row, "prior_orders_12m"),
        "merchant_category": _v(row, "merchant_category"), "avg_ticket_size": _fl(row, "avg_ticket_size"),
        "trailing_chargeback_rate_bps": _fl(row, "trailing_chargeback_rate_bps"),
        "ip_country": _v(row, "ip_country"), "ip_region_code": _i(row, "ip_region_code"),
        "home_region_code": _i(row, "home_region_code"),
        "asn_type": _v(row, "asn_type"), "asn_country": _v(row, "asn_country"),
        "reputation_score": _fl(row, "reputation_score"),
        "ip_n_customers": _i(row, "ip_n_customers"), "device_n_customers": _i(row, "device_n_customers"),
        "card": card, "device": device, "basket": basket, "last_login": last_login,
    }


# columns a stage-02 row must carry for the mapping above (the fixture builder keeps exactly these)
STAGE02_COLUMNS = [
    "payment_id", "payment_ts", "payment_method", "payment_gateway", "payment_amount", "processing_fee",
    "discount_amount", "item_count", "attempt_seq_in_session", "is_guest_checkout", "is_3ds_attempted",
    "is_3ds_success", "has_coupon", "address_match_flag", "shipping_speed", "shipping_addr_age_hours",
    "delivery_type", "kyc_level", "city_tier", "email_domain_class", "acquisition_channel", "account_age_days",
    "prior_return_rate", "prior_orders_12m", "merchant_category", "avg_ticket_size", "trailing_chargeback_rate_bps",
    "ip_country", "ip_region_code", "home_region_code", "asn_type", "asn_country", "reputation_score",
    "ip_n_customers", "device_n_customers", "card_token", "card_token_age_h", "network", "issuing_country",
    "issuer", "product_type", "device_type", "device_age_h", "is_emulator", "browser_family", "n_lines",
    "n_categories", "max_unit_price", "total_qty", "hours_since_last_login", "last_login_new_device",
    "last_login_unusual_loc", "last_login_failed_attempts", "last_login_risk",
    # notebook-02 derived columns, kept to prove the service re-derives them identically
    "amount_vs_merchant_ticket", "ip_country_mismatch", "ip_country_missing", "ip_region_mismatch",
    "issuer_foreign", "hour_of_day", "day_of_week", "is_weekend", "is_night",
    # labels, used only to fit the fixture models
    "is_fraud", "sample_weight", "label_source",
]
