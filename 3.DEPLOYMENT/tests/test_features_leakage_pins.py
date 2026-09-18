"""Reference velocity semantics, leakage guards, dependency pins and golden files."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import check_pins
from fraud_api.features import asof_distinct_count
from fraud_api.predict import to_source_frame
from fraud_api.schema import ScoreRequest
from fraud_api.transform import build_features

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- velocity reference
def _toy():
    ts = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-02 09:00", "2024-01-02 09:00",
                         "2024-01-03 08:00", "2024-01-04 08:00"])
    return pd.DataFrame({"device_id": ["D1", "D1", "D1", "D1", None, "D2"],
                         "customer_id": ["A", "A", "B", "C", "A", "A"], "payment_ts": ts})


def test_asof_count_is_inclusive_and_point_in_time():
    got = asof_distinct_count(_toy(), "device_id").tolist()
    assert got[:4] == [1.0, 1.0, 3.0, 3.0]            # simultaneous first appearances count together
    assert np.isnan(got[4]) and got[5] == 1.0


def test_strictly_later_rows_never_change_earlier_counts():
    full = _toy()
    whole = asof_distinct_count(full, "device_id")
    for cutoff in sorted(full.payment_ts.unique()):
        past = full[full.payment_ts <= cutoff]
        np.testing.assert_array_equal(asof_distinct_count(past, "device_id").to_numpy(),
                                      whole.loc[past.index].to_numpy())


# ----------------------------------------------------------------------------- leakage guards
def test_features_ignore_identifiers_and_the_incumbent_score(request_dicts, p06, p03):
    base = request_dicts[0]
    variants = [dict(base), dict(base, payment_id="other"), dict(base, payment_risk_score=99.0)]
    frames = [build_features(to_source_frame([ScoreRequest.model_validate(v)]), p06, p03, "tree")[0]
              for v in variants]
    for f in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], f)


def test_only_time_features_move_when_only_the_timestamp_moves(request_dicts, p06, p03):
    a = dict(request_dicts[0], payment_ts="2025-03-03T10:00:00")
    b = dict(request_dicts[0], payment_ts="2025-03-08T02:00:00")
    Xa, Xb = (build_features(to_source_frame([ScoreRequest.model_validate(d)]), p06, p03, "tree")[0]
              for d in (a, b))
    moved = {c for c in Xa.columns if not Xa[c].equals(Xb[c])}
    assert moved <= {"hour_of_day", "day_of_week"}, moved


# ----------------------------------------------------------------------------- pins
def test_requirements_pin_the_trained_library_versions(info):
    pins = check_pins.read_pins(ROOT / "requirements.txt")
    assert check_pins.compare(pins, info["library_versions"]) == []


def test_runtime_python_minor_matches_training(info):
    import platform
    assert platform.python_version().split(".")[:2] == info["library_versions"]["python"].split(".")[:2]


def test_dockerfile_base_image_is_the_training_python(info):
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    m = re.search(r"^ARG PYTHON_IMAGE=python:([0-9.]+)-slim-trixie$", text, re.M)
    assert m, "Dockerfile must pin ARG PYTHON_IMAGE=python:<x.y.z>-slim-trixie"
    assert m.group(1) == info["library_versions"]["python"]


def test_lockfile_agrees_with_requirements():
    req = check_pins.read_pins(ROOT / "requirements.txt")
    lock = check_pins.read_pins(ROOT / "requirements.lock")
    assert {k: lock.get(k) for k in req} == req


# ----------------------------------------------------------------------------- golden files
GOLDEN = ROOT / "tests" / "golden"


def test_golden_files_exist_in_ci_and_match_the_fixture_version(info):
    exp = GOLDEN / "expected.json"
    if not exp.exists():
        if os.environ.get("CI") == "true":
            pytest.fail("tests/golden/expected.json is missing: record it (runbook) before pushing")
        pytest.skip("golden files are recorded in the runbook against the real promoted model")
    expected = json.loads(exp.read_text(encoding="utf-8"))
    payloads = json.loads((GOLDEN / "payloads.json").read_text(encoding="utf-8"))
    assert expected["model_version"] == info["version"]
    assert [c["name"] for c in payloads["cases"]] == [c["name"] for c in expected["cases"]]
    for c in payloads["cases"]:
        ScoreRequest.model_validate(c["request"])
