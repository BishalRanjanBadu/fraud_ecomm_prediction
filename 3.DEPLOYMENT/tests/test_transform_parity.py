"""The service transform must reproduce what notebooks 02, 03 and 06 saved, cell for cell."""
import numpy as np
import pandas as pd

from fraud_api import constants as C
from fraud_api.transform import build_features, derive, feature_contract_hash, restore_tree_dtypes

DERIVED_02 = ["amount_vs_merchant_ticket", "ip_country_mismatch", "ip_country_missing", "ip_region_mismatch",
              "issuer_foreign", "hour_of_day", "day_of_week", "is_weekend", "is_night"]


def test_fixture_belongs_to_a_parity_checked_version(info, p06):
    assert info["parity_checked"] is True
    assert info["marker"] == C.MARKER
    assert feature_contract_hash(p06) == info["feature_hash"]
    assert info["rows"] > 1000 and info["positives"] > 50, "fixture too small to behave like production"


def test_tree_encoding_matches_notebook_06_exactly(source_frame, p06, p03, t06):
    X, _, _ = build_features(source_frame, p06, p03, "tree")
    expected = restore_tree_dtypes(t06.drop(columns="payment_id"), p06)
    pd.testing.assert_frame_equal(X.reset_index(drop=True), expected, check_exact=True)


def test_linear_encoding_matches_notebook_06(source_frame, p06, p03, l06):
    X, _, _ = build_features(source_frame, p06, p03, "linear")
    expected = l06.drop(columns="payment_id").astype("float64")
    pd.testing.assert_frame_equal(X.reset_index(drop=True), expected, check_exact=False, rtol=0, atol=1e-12)


def test_notebook_02_derivations_are_reproduced(source_frame, s02):
    d = derive(source_frame)
    for c in DERIVED_02:
        got, exp = d[c].astype("float64").to_numpy(), s02[c].astype("float64").to_numpy()
        assert np.array_equal(got, exp, equal_nan=True), c


def test_request_mapping_covers_every_source_column(source_frame):
    assert list(source_frame.columns) == C.SOURCE_COLUMNS
    assert str(source_frame.payment_ts.dtype) == "datetime64[ns]"


def test_every_model_feature_is_produced(source_frame, p06, p03):
    for inp in ("tree", "linear"):
        X, _, _ = build_features(source_frame, p06, p03, inp)
        assert list(X.columns) == p06[inp]["feature_order"]
