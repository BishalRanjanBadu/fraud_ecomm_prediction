"""scripts/promote.py: plan A (pinned notebook-03 params), rollback exactness and refusals."""
import json

import pytest

import promote
from fakes import FakeS3
from fraud_api.registry import load

V1, V2 = "v_20260101T000000Z_nbaaaaaaa", "v_20260201T000000Z_nbbbbbbbb"
ALIAS = "models/CURRENT.json"


def _bucket(store_for, p03_bytes, version, gate_passed=True):
    """A version folder plus the stage/report files of the run that produced it (no alias yet)."""
    store = store_for(version=version, gate_passed=gate_passed)
    for k in [k for k in store if k == ALIAS or k.startswith("models/params/")]:
        store.pop(k)
    created = json.loads(store[f"models/{version}/manifest.json"])["created_utc"]
    store["data/03_fitted_params.json"] = p03_bytes
    store["reports/07_run_record.json"] = json.dumps({"version": version}).encode()
    store["reports/03_run_record.json"] = json.dumps({"marker": "fraud_nb01-07_v2", "created_utc": created}).encode()
    return store


def _promote(s3, version, reason=None):
    return promote.promote(s3, version, ALIAS, reason, "arn:aws:iam::000000000000:root", False, log=lambda *_: None)


def test_first_promotion_pins_params_and_the_pod_can_load_it(store_for, p03_bytes):
    s3 = FakeS3(_bucket(store_for, p03_bytes, V1))
    alias = _promote(s3, V1)
    assert alias["imputation_params_key"].startswith("models/params/03_fitted_params_")
    assert s3.store[alias["imputation_params_key"]] == p03_bytes
    assert f"models/params/by_version/{V1}.json" in s3.store
    assert alias["previous"] is None and alias["gate_deviation"] is None
    assert load(s3, "fraud-ecommerce", ALIAS).version == V1


def test_gate_not_passed_needs_an_explicit_recorded_reason(store_for, p03_bytes):
    s3 = FakeS3(_bucket(store_for, p03_bytes, V1, gate_passed=False))
    with pytest.raises(SystemExit, match="accept-gate-deviation"):
        _promote(s3, V1)
    alias = _promote(s3, V1, reason="Phase-1 sign-off 2026-09-17")
    assert alias["gate_passed"] is False and alias["gate_deviation"] == "Phase-1 sign-off 2026-09-17"
    assert load(s3, "fraud-ecommerce", ALIAS).gate_status == "not_passed_provisional"


def test_rollback_uses_the_pinned_params_not_the_current_stage_file(store_for, p03_bytes):
    s3 = FakeS3(_bucket(store_for, p03_bytes, V1))
    _promote(s3, V1)
    v2 = _bucket(store_for, p03_bytes, V2)
    s3.store.update({k: v for k, v in v2.items() if k.startswith(f"models/{V2}/")})
    s3.store["reports/07_run_record.json"] = v2["reports/07_run_record.json"]
    _promote(s3, V2)
    s3.store["data/03_fitted_params.json"] = b'{"a retrain": "overwrote the stage file"}'
    back = _promote(s3, V1)                                    # rollback
    assert back["version"] == V1 and back["previous"]["version"] == V2
    assert s3.store[back["imputation_params_key"]] == p03_bytes
    assert load(s3, "fraud-ecommerce", ALIAS).version == V1


def test_first_promotion_refuses_stage_files_from_another_run(store_for, p03_bytes):
    store = _bucket(store_for, p03_bytes, V1)
    store["reports/07_run_record.json"] = json.dumps({"version": V2}).encode()
    with pytest.raises(SystemExit, match="stage files of its own run"):
        _promote(FakeS3(store), V1)


def test_incomplete_version_is_refused(store_for, p03_bytes):
    store = _bucket(store_for, p03_bytes, V1)
    store.pop(f"models/{V1}/manifest.json")
    with pytest.raises(SystemExit, match="missing or incomplete"):
        _promote(FakeS3(store), V1)


def test_content_addressed_key_is_write_once(store_for, p03_bytes):
    from fraud_api import s3_io
    store = _bucket(store_for, p03_bytes, V1)
    key = f"models/params/03_fitted_params_{s3_io.sha256(p03_bytes)[:12]}.json"
    store[key] = b"tampered"
    with pytest.raises(SystemExit, match="different content"):
        _promote(FakeS3(store), V1)


def test_alias_is_verified_by_read_back(store_for, p03_bytes):
    s3 = FakeS3(_bucket(store_for, p03_bytes, V1))
    _promote(s3, V1)
    assert s3.puts[-1] == ALIAS
    assert json.loads(s3.store[ALIAS])["version"] == V1
