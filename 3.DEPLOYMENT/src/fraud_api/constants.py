# MARKER: fraud_phase2_v1
"""Fixed contract values shared by the service, the scripts and the tests.

Everything that was FITTED lives in the model artifacts (preprocessor.json, the 03 params). What lives here is
the code-level recipe notebooks 02, 03 and 06 hard-coded, copied verbatim so serving reproduces training.
"""
MARKER = "fraud_phase2_v1"
TRAINING_MARKER = "fraud_nb01-07_v2"
CONTRACT_VERSION = "v1"
IST_OFFSET_MINUTES = 330                     # payment_ts in training is naive India Standard Time
INCUMBENT_ALERT_THRESHOLD = 14.96            # payment_risk_score >= 14.96 <=> alerted_flag == 'Y' (Phase 1, train)

# notebook 02: tokens that mean "no value"
SENTINELS = {"NA", "N/A", "NAN", "NULL", "NONE", "UNKNOWN", "XX", "-1", "999", "999999", "000000", "AS0",
             "OTHER", "UNKNOWN BANK", "NIL", "?", "TBD"}
REGION_SENTINELS = {-1, 999}
FAILED_ATTEMPTS_SENTINEL = 255
ACCOUNT_AGE_SENTINEL_DAYS = 20000

# notebook 03: group keys are filled before the group medians that use them
GROUP_KEYS = ("city_tier", "shipping_speed")

# notebook 06: linear-family recipes (the fitted statistics are in preprocessor.json)
LINEAR_NUMERIC = [
    "log_payment_amount", "log_max_unit_price", "log_amount_vs_merchant_ticket",
    "item_count", "attempt_seq_in_session", "is_guest_checkout", "is_3ds_attempted", "is_3ds_success",
    "address_match_flag", "shipping_addr_age_hours", "kyc_level", "city_tier", "prior_return_rate",
    "trailing_chargeback_rate_bps", "is_emulator", "reputation_score", "n_categories", "total_qty",
    "last_login_new_device", "last_login_unusual_loc", "last_login_failed_attempts", "last_login_risk",
    "account_age_days", "ip_country_mismatch", "ip_country_missing", "ip_region_mismatch", "issuer_foreign",
    "has_coupon", "is_weekend", "is_night", "has_card", "has_device_profile", "has_basket", "has_prior_login",
]
LINEAR_LOG1P = ["processing_fee", "discount_amount", "prior_orders_12m", "avg_ticket_size", "hours_since_last_login",
                "card_token_age_h", "device_age_h", "device_n_customers", "ip_n_customers"]
LINEAR_CYCLIC = {"hour_of_day": 24}
LOG_BASES = ["payment_amount", "max_unit_price", "amount_vs_merchant_ticket"]   # notebook 03: log1p(clip(lower=0))

# columns a request provides, at stage-02 grain (before notebook 03 imputation)
SOURCE_NUMERIC = [
    "payment_amount", "processing_fee", "discount_amount", "item_count", "attempt_seq_in_session",
    "is_guest_checkout", "is_3ds_attempted", "is_3ds_success", "address_match_flag", "shipping_addr_age_hours",
    "kyc_level", "city_tier", "prior_return_rate", "prior_orders_12m", "avg_ticket_size",
    "trailing_chargeback_rate_bps", "is_emulator", "reputation_score", "n_categories", "max_unit_price",
    "total_qty", "last_login_new_device", "last_login_unusual_loc", "last_login_failed_attempts",
    "last_login_risk", "hours_since_last_login", "card_token_age_h", "device_age_h", "account_age_days",
    "ip_region_code", "home_region_code", "device_n_customers", "ip_n_customers",
    "has_coupon", "has_card", "has_device_profile", "has_basket", "has_prior_login",
]
SOURCE_CATEGORICAL = [
    "payment_method", "payment_gateway", "shipping_speed", "delivery_type", "email_domain_class",
    "acquisition_channel", "merchant_category", "network", "issuer", "product_type", "issuing_country",
    "device_type", "browser_family", "asn_type", "asn_country", "ip_country",
]
SOURCE_COLUMNS = ["payment_ts"] + SOURCE_NUMERIC + SOURCE_CATEGORICAL

# libraries whose versions are part of the model contract (import name -> manifest key)
STRICT_LIBS = {"pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "sklearn": "sklearn",
               "lightgbm": "lightgbm", "xgboost": "xgboost", "joblib": "joblib"}
FAMILY_MODULE = {"lightgbm": "lightgbm", "xgboost": "xgboost", "logreg": "sklearn"}
