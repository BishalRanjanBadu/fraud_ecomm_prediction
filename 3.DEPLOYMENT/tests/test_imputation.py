"""Notebook-03 values are applied at serving; without them LightGBM silently scores a gap as zero."""
import copy

import numpy as np

from fraud_api.predict import to_source_frame
from fraud_api.schema import ScoreRequest
from fraud_api.transform import build_features, encode_tree, derive, impute


def _one(request_dicts, **overrides):
    d = dict(request_dicts[0])
    d.update(overrides)
    return to_source_frame([ScoreRequest.model_validate(d)])


def test_missing_kyc_gets_the_train_median_and_is_reported(request_dicts, p06, p03):
    src = _one(request_dicts, kyc_level=None)
    _, frame, filled = build_features(src, p06, p03, "tree")
    assert frame.kyc_level.iloc[0] == p03["global_medians"]["kyc_level"]
    assert bool(filled["kyc_level"].iloc[0])


def test_group_median_uses_the_group_key(request_dicts, p06, p03):
    g = p03["group_medians"]["account_age_days"]
    key = sorted(g["map"])[-1]
    src = _one(request_dicts, account_age_days=None, city_tier=int(key))
    _, frame, _ = build_features(src, p06, p03, "tree")
    assert frame.account_age_days.iloc[0] == g["map"][key]


def test_missing_group_key_is_filled_first_then_used(request_dicts, p06, p03):
    tier = p03["global_medians"]["city_tier"]
    g = p03["group_medians"]["account_age_days"]
    src = _one(request_dicts, account_age_days=None, city_tier=None)
    _, frame, filled = build_features(src, p06, p03, "tree")
    assert frame.city_tier.iloc[0] == tier
    assert frame.account_age_days.iloc[0] == g["map"][str(int(tier))]
    assert {"city_tier", "account_age_days"} <= {c for c in filled.columns if filled[c].iloc[0]}


def test_missing_shipping_speed_follows_the_artifact_rule(request_dicts, p06, p03):
    """Option 1: a gap is scored only where the model learned one; otherwise it is a 422."""
    import pytest
    from fraud_api.predict import check_levels, required_categoricals
    from fraud_api.transform import ContractError
    g = p03["group_medians"]["shipping_addr_age_hours"]
    d = dict(request_dicts[0], shipping_speed=None, shipping_addr_age_hours=None)
    r = ScoreRequest.model_validate(d)
    if "shipping_speed" in required_categoricals(p06, p03):
        with pytest.raises(ContractError, match="required for this model version"):
            check_levels([r], p06, p03)
        with pytest.raises(ContractError):                      # backstop inside the transform as well
            build_features(to_source_frame([r]), p06, p03, "tree")
    else:
        _, frame, _ = build_features(to_source_frame([r]), p06, p03, "tree")
        assert frame.shipping_speed.iloc[0] == p03["cat_fill"]["shipping_speed"]
    frame, _ = impute(derive(to_source_frame([r])), p03)       # the numeric fallback is independent of that
    assert frame.shipping_addr_age_hours.iloc[0] == g["fallback"]


def test_required_categoricals_rule(p06, p03):
    from fraud_api.predict import required_categoricals
    req = set(required_categoricals(p06, p03))
    maps, fill = p06["tree"]["categorical"], p03["cat_fill"]
    assert "payment_method" in req                                  # never filled in notebook 03
    for c in maps:
        if c in fill and fill[c] in maps[c]:
            assert c not in req, c                                  # e.g. payment_gateway -> 'Unknown'
    assert not req & set(p06["tree"]["nan_allowed"])


def test_serving_never_fits_anything(source_frame, p06, p03):
    before06, before03 = copy.deepcopy(p06), copy.deepcopy(p03)
    build_features(source_frame, p06, p03, "tree")
    build_features(source_frame, p06, p03, "linear")
    assert p06 == before06 and p03 == before03


def test_why_imputation_is_mandatory_lightgbm_scores_a_gap_as_zero(fitted, source_frame, p06, p03):
    """A feature with no gaps in training: LightGBM maps NaN to 0 at prediction, i.e. a fabricated value."""
    X_train = fitted["lightgbm"]["X"]
    imputed = [c for c in list(p03["global_medians"]) + list(p03["group_medians"])
               if c in X_train.columns and not X_train[c].isna().any()]
    assert imputed, "expected notebook-03 imputed features in the model"
    frame, _ = impute(derive(source_frame.head(200)), p03)
    X = encode_tree(frame, p06)
    m = fitted["lightgbm"]["model"]
    for c in imputed:
        as_nan, as_zero = X.copy(), X.copy()
        as_nan[c], as_zero[c] = np.nan, 0.0
        assert np.array_equal(m.predict_proba(as_nan)[:, 1], m.predict_proba(as_zero)[:, 1]), c
