"""End-to-end through FastAPI with an in-memory S3 holding a promoted version."""
import numpy as np
import pytest

from fraud_api import predict
from fraud_api.registry import load


@pytest.mark.parametrize("fam", ["lightgbm", "xgboost", "logreg"])
def test_every_family_is_served_by_the_same_code(fam, store_for, client_for, request_dicts):
    client, _, _ = client_for(store_for(fam))
    with client:
        h = client.get("/health")
        assert h.status_code == 200 and h.json()["family"] == fam
        r = client.post("/v1/score", json=request_dicts[0])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "model" and 0.0 <= body["fraud_probability"] <= 1.0
        assert body["decision"] == (body["score_raw"] >= body["threshold_raw"])


def test_probes_and_request_id(store_for, client_for):
    client, _, _ = client_for(store_for())
    with client:
        assert client.get("/live").json()["status"] == "alive"
        h = client.get("/health", headers={"x-request-id": "abc-123"})
        assert h.headers["x-request-id"] == "abc-123"
        assert h.json()["checks_passed"] >= 10 and h.json()["gate_status"] == "passed"


def test_api_equals_direct_batch_scoring(store_for, client_for, reqs, request_dicts):
    store = store_for()
    client, app, s3 = client_for(store)
    with client:
        loaded = app.state.model
        direct = predict.score(reqs[:25], loaded)
        for d, body in zip(direct, request_dicts[:25]):
            api = client.post("/v1/score", json=body).json()
            assert api["score_raw"] == pytest.approx(d["score_raw"], abs=1e-12)
            assert api["defaults_applied"] == d["defaults_applied"]


def test_calibrated_probability_is_monotone_in_the_raw_score(store_for, client_for, reqs):
    client, app, _ = client_for(store_for())
    with client:
        out = predict.score(reqs[:400], app.state.model)
    raw = np.array([o["score_raw"] for o in out])
    cal = np.array([o["fraud_probability"] for o in out])
    order = np.argsort(raw, kind="mergesort")
    assert np.all(np.diff(cal[order]) >= -1e-12)


def test_explanations_add_up_to_the_margin(store_for, client_for, reqs, p06, p03):
    from fraud_api.transform import build_features
    client, app, _ = client_for(store_for())
    with client:
        loaded = app.state.model
        X, _, _ = build_features(predict.to_source_frame(reqs[:30]), p06, p03, "tree")
        contrib, bias = predict.contributions(loaded, X)
        margin = loaded.model.booster_.predict(X, raw_score=True)
        assert np.allclose(contrib.sum(axis=1) + bias, margin, atol=1e-6)
        r = client.post("/v1/score?explain=true", json=predict_body(reqs[0]))
        assert r.status_code == 200 and len(r.json()["explanation"]) == 10


def predict_body(req):
    return req.model_dump(mode="json")


def test_unknown_level_returns_typed_422(store_for, client_for, request_dicts):
    client, _, _ = client_for(store_for())
    with client:
        r = client.post("/v1/score", json=dict(request_dicts[0], payment_method="Crypto"))
        assert r.status_code == 422 and r.json()["error"] == "contract_violation"
        assert r.json()["field"] == "payment_method"


def test_model_unavailable_is_503_and_fallback_is_opt_in_and_tagged(client_for, request_dicts):
    client, _, _ = client_for({})                                         # empty bucket: no alias
    with client:
        h = client.get("/health")
        assert h.status_code == 503 and "alias" in h.json()["reason"]
        assert client.post("/v1/score", json=request_dicts[0]).status_code == 503
        body = dict(request_dicts[0], payment_risk_score=20.0)
        assert client.post("/v1/score", json=body).status_code == 503        # not opted in
        r = client.post("/v1/score?allow_fallback=true", json=body)
        assert r.status_code == 200
        assert r.json()["source"] == "fallback_incumbent_rule" and r.json()["decision"] is True
        assert r.json()["fraud_probability"] is None and r.json()["model_version"] is None
        no_score = client.post("/v1/score?allow_fallback=true", json=request_dicts[0])
        assert no_score.status_code == 503


def test_payment_risk_score_never_changes_the_model_score(store_for, client_for, request_dicts):
    client, _, _ = client_for(store_for())
    with client:
        a = client.post("/v1/score", json=dict(request_dicts[0], payment_risk_score=0.0)).json()
        b = client.post("/v1/score", json=dict(request_dicts[0], payment_risk_score=99.0)).json()
        assert a["score_raw"] == b["score_raw"]


def test_failed_reload_clears_the_model(store_for, client_for):
    from fraud_api.api import load_model
    store = store_for()
    client, app, s3 = client_for(store)
    with client:
        assert client.get("/health").status_code == 200
        s3.store["models/CURRENT.json"] = b"{not json"
        load_model(app)
        assert app.state.model is None
        assert client.get("/health").status_code == 503


def test_gate_status_is_surfaced(store_for, client_for, request_dicts):
    client, _, _ = client_for(store_for(gate_passed=False))
    with client:
        assert client.post("/v1/score", json=request_dicts[0]).json()["gate_status"] == "not_passed_provisional"


def test_registry_load_reports_every_check(store_for):
    from fakes import FakeS3
    m = load(FakeS3(store_for()), "fraud-ecommerce", "models/CURRENT.json")
    assert m.version.startswith("v_") and len(m.checks) >= 10
