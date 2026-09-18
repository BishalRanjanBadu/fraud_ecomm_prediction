"""Readiness is refused for every broken link of the compatibility contract (Part 0 #4)."""
import copy
import json

import pytest

from fakes import FakeS3
from fraud_api.registry import ModelLoadError, load

ALIAS = "models/CURRENT.json"


def _load(store):
    return load(FakeS3(store), "fraud-ecommerce", ALIAS)


def _set(k, v):
    def m(manifest):
        manifest[k] = v
    return m


def _lib(k, v):
    def m(manifest):
        manifest["library_versions"][k] = v
    return m


@pytest.mark.parametrize("mutate,fragment", [
    (_lib("lightgbm", "0.0.0"), "library versions"),
    (_lib("sklearn", "1.5.2"), "library versions"),
    (_lib("python", "3.12.10"), "python minor"),
    (_set("feature_hash", "0" * 64), "feature hash"),
    (_set("marker", "fraud_nb05-07_v1"), "marker"),
    (_set("contract_version", "v2"), "contract"),
    (_set("family", "xgboost"), "estimator"),
    (_set("feature_order", ["x"]), "feature order"),
])
def test_manifest_mismatches_refuse_readiness(store_for, mutate, fragment):
    with pytest.raises(ModelLoadError, match=fragment):
        _load(store_for(mutate_manifest=mutate))


def test_feature_names_file_mismatch(store_for, p06):
    with pytest.raises(ModelLoadError, match="feature order"):
        _load(store_for(feature_names=list(reversed(p06["tree"]["feature_order"]))))


def test_params_bytes_must_match_the_pinned_sha(store_for):
    store = store_for()
    alias = json.loads(store[ALIAS])
    store[alias["imputation_params_key"]] = store[alias["imputation_params_key"]] + b" "
    with pytest.raises(ModelLoadError, match="03 params sha256"):
        _load(store)


def test_manifest_bytes_must_match_the_pinned_sha(store_for):
    store = store_for()
    alias = json.loads(store[ALIAS])
    alias["manifest_sha256"] = "f" * 64
    store[ALIAS] = json.dumps(alias).encode()
    with pytest.raises(ModelLoadError, match="manifest sha256"):
        _load(store)


def test_booster_categories_must_equal_the_encoder_maps(store_for, fitted):
    model = copy.deepcopy(fitted["lightgbm"]["model"])
    model.booster_.pandas_categorical[0] = list(reversed(model.booster_.pandas_categorical[0]))
    with pytest.raises(ModelLoadError, match="pandas_categorical"):
        _load(store_for(model=model))


@pytest.mark.parametrize("breakage,fragment", [
    (lambda s: s.pop(ALIAS), "cannot read"),
    (lambda s: s.__setitem__(ALIAS, b'{"version": "v_x"}'), "missing field"),
])
def test_alias_problems(store_for, breakage, fragment):
    store = store_for()
    breakage(store)
    with pytest.raises(ModelLoadError, match=fragment):
        _load(store)


def test_alias_prefix_must_match_version(store_for):
    store = store_for()
    alias = json.loads(store[ALIAS])
    alias["model_prefix"] = "models/v_other/"
    store[ALIAS] = json.dumps(alias).encode()
    with pytest.raises(ModelLoadError, match="prefix"):
        _load(store)


def test_missing_artifact(store_for):
    store = store_for()
    alias = json.loads(store[ALIAS])
    store.pop(alias["model_prefix"] + "calibrator.pkl")
    with pytest.raises(ModelLoadError, match="artifacts"):
        _load(store)
