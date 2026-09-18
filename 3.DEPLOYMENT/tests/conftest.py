"""Shared fixtures. Runs with NO cloud credentials: everything comes from tests/fixtures (committed).

Part 0 #15: the fixture models are fitted by the production path (transform.build_features -> train.fit),
so they carry the same dtypes and encoder maps as the promoted model. test_dtypes.py asserts the markers.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
FIX = ROOT / "tests" / "fixtures"
REQUIRED = ["stage02_sample.parquet", "stage06_tree_sample.parquet", "stage06_linear_sample.parquet",
            "preprocessor.json", "fitted_params_03.json", "FIXTURE_INFO.json"]


def pytest_sessionstart(session):
    missing = [f for f in REQUIRED if not (FIX / f).exists()]
    if missing:
        pytest.exit(f"test fixtures missing {missing}. Build them from the real bucket first: "
                    "python scripts/make_fixture.py --version <candidate version>", returncode=2)


from sklearn.isotonic import IsotonicRegression  # noqa: E402

from fakes import FakeS3  # noqa: E402
from fraud_api import constants as C  # noqa: E402
from fraud_api import registry, s3_io, train  # noqa: E402
from fraud_api.features import stage02_row_to_request  # noqa: E402
from fraud_api.predict import to_source_frame  # noqa: E402
from fraud_api.schema import ScoreRequest  # noqa: E402
from fraud_api.transform import build_features, feature_contract_hash  # noqa: E402

BUCKET = "fraud-ecommerce"
INPUT = {"lightgbm": "tree", "xgboost": "tree", "logreg": "linear"}
FAMILY_PARAMS = {
    "lightgbm": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20},
    "xgboost": {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 4},
    "logreg": {"C": 0.1},
}
TEST_VERSION = "v_20260101T000000Z_nbtest01"


@pytest.fixture(scope="session")
def info():
    return json.loads((FIX / "FIXTURE_INFO.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def p06():
    return json.loads((FIX / "preprocessor.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def p03_bytes():
    return (FIX / "fitted_params_03.json").read_bytes()


@pytest.fixture(scope="session")
def p03(p03_bytes):
    return json.loads(p03_bytes)


@pytest.fixture(scope="session")
def s02():
    return pd.read_parquet(FIX / "stage02_sample.parquet")


@pytest.fixture(scope="session")
def t06():
    return pd.read_parquet(FIX / "stage06_tree_sample.parquet")


@pytest.fixture(scope="session")
def l06():
    return pd.read_parquet(FIX / "stage06_linear_sample.parquet")


@pytest.fixture(scope="session")
def request_dicts(s02):
    return [stage02_row_to_request(r) for _, r in s02.iterrows()]


@pytest.fixture(scope="session")
def reqs(request_dicts):
    return [ScoreRequest.model_validate(d) for d in request_dicts]


@pytest.fixture(scope="session")
def source_frame(reqs):
    return to_source_frame(reqs)


@pytest.fixture(scope="session")
def fitted(source_frame, s02, p06, p03):
    out = {}
    y, w = s02.is_fraud.to_numpy(), s02.sample_weight.to_numpy()
    for fam, params in FAMILY_PARAMS.items():
        X, _, _ = build_features(source_frame, p06, p03, INPUT[fam])
        model = train.fit(fam, params, X, y, w, n_jobs=1)
        raw = model.predict_proba(X)[:, 1]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
        out[fam] = {"model": model, "calibrator": iso, "X": X, "raw": raw}
    return out


def _pickle(obj) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def make_store(fam, fitted, p06, p03_bytes, version=TEST_VERSION, gate_passed=True, mutate_manifest=None,
               model=None, feature_names=None):
    """The S3 layout promote.py leaves behind, for one version of `fam`."""
    f = fitted[fam]
    inp = INPUT[fam]
    model = model if model is not None else f["model"]
    manifest = {
        "version": version, "created_utc": "2026-01-01T00:00:00+00:00", "marker": C.TRAINING_MARKER,
        "family": fam, "estimator_class": f"{type(model).__module__}.{type(model).__name__}",
        "params": FAMILY_PARAMS[fam], "n_trees": 60 if inp == "tree" else None, "input_family": inp,
        "feature_order": p06[inp]["feature_order"],
        "feature_dtypes": p06["tree"]["dtypes"] if inp == "tree" else {c: "float64" for c in p06[inp]["feature_order"]},
        "encoder_maps": p06[inp]["categorical"], "feature_hash": feature_contract_hash(p06),
        "contract_version": C.CONTRACT_VERSION, "library_versions": registry.runtime_versions(),
        "version_drift": {}, "git_sha": None,
        "operating_point": {"threshold_raw": float(np.quantile(f["raw"], 0.98))},
    }
    if mutate_manifest:
        mutate_manifest(manifest)
    prefix = f"models/{version}/"
    man_bytes = json.dumps(manifest).encode()
    params_key = f"models/params/03_fitted_params_{s3_io.sha256(p03_bytes)[:12]}.json"
    store = {
        prefix + "manifest.json": man_bytes,
        prefix + "model.pkl": _pickle(model),
        prefix + "calibrator.pkl": _pickle(f["calibrator"]),
        prefix + "preprocessor.json": json.dumps(p06).encode(),
        prefix + "feature_names.json": json.dumps(feature_names or p06[inp]["feature_order"]).encode(),
        prefix + "metrics.json": json.dumps({"version": version, "gate_passed": gate_passed}).encode(),
        params_key: p03_bytes,
    }
    alias = {"alias": "models/CURRENT.json", "version": version, "model_prefix": prefix,
             "manifest_sha256": s3_io.sha256(man_bytes), "imputation_params_key": params_key,
             "imputation_params_sha256": s3_io.sha256(p03_bytes), "gate_passed": gate_passed}
    store["models/CURRENT.json"] = json.dumps(alias).encode()
    return store


@pytest.fixture
def store_for(fitted, p06, p03_bytes):
    def _make(fam="lightgbm", **kw):
        return make_store(fam, fitted, p06, p03_bytes, **kw)
    return _make


@pytest.fixture
def client_for(monkeypatch):
    from fastapi.testclient import TestClient

    from fraud_api.api import create_app
    monkeypatch.setenv("MODEL_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_REGION", "ap-south-2")
    monkeypatch.setenv("MODEL_ALIAS_KEY", "models/CURRENT.json")
    monkeypatch.setenv("LOG_PREDICTIONS", "false")

    def _make(store):
        s3 = FakeS3(store)
        app = create_app(s3_factory=lambda region: s3)
        return TestClient(app), app, s3
    return _make
