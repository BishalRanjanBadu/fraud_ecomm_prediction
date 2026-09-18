"""Dtypes are part of the contract: order alone is not enough (Part 0 #4, #15)."""
import pandas as pd
import pytest

from fraud_api.transform import build_features


def test_encoded_dtypes_equal_the_trained_dtypes(source_frame, p06, p03):
    X, _, _ = build_features(source_frame, p06, p03, "tree")
    assert {c: str(t) for c, t in X.dtypes.items()} == p06["tree"]["dtypes"]


def test_categories_come_from_the_encoder_map_not_the_payload(source_frame, p06, p03):
    one = source_frame.iloc[[0]]
    X, _, _ = build_features(one, p06, p03, "tree")
    for c, levels in p06["tree"]["categorical"].items():
        assert list(X[c].cat.categories) == levels, c


def test_fixture_lightgbm_carries_the_production_structural_markers(fitted, p06):
    m = fitted["lightgbm"]["model"]
    cats = [p06["tree"]["categorical"][c] for c in p06["tree"]["feature_order"] if c in p06["tree"]["categorical"]]
    assert m.booster_.pandas_categorical == cats
    assert list(m.feature_names_in_) == p06["tree"]["feature_order"]


def test_lightgbm_predicts_on_the_aligned_frame(fitted, source_frame, p06, p03):
    X, _, _ = build_features(source_frame.head(50), p06, p03, "tree")
    p = fitted["lightgbm"]["model"].predict_proba(X)[:, 1]
    assert p.shape == (50,) and ((p >= 0) & (p <= 1)).all()


def test_lightgbm_rejects_integer_codes_where_categories_were_trained(fitted, source_frame, p06, p03):
    X, _, _ = build_features(source_frame.head(50), p06, p03, "tree")
    bad = X.copy()
    for c in p06["tree"]["categorical"]:
        bad[c] = bad[c].cat.codes.astype("int64")
    with pytest.raises(ValueError):
        fitted["lightgbm"]["model"].predict_proba(bad)


def test_xgboost_and_logreg_fixture_models_use_their_input_family(fitted, p06):
    assert list(fitted["xgboost"]["model"].feature_names_in_) == p06["tree"]["feature_order"]
    assert list(fitted["logreg"]["model"].feature_names_in_) == p06["linear"]["feature_order"]
    assert all(str(t) == "float64" for t in fitted["logreg"]["X"].dtypes)


def test_parquet_roundtrip_does_not_define_categories(t06, p06):
    # the fixture file's own categories are not trusted: restore_tree_dtypes re-applies the map
    from fraud_api.transform import restore_tree_dtypes
    X = restore_tree_dtypes(t06.drop(columns="payment_id"), p06)
    for c, levels in p06["tree"]["categorical"].items():
        assert list(X[c].cat.categories) == levels
    assert isinstance(X, pd.DataFrame)
