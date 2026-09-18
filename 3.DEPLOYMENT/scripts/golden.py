"""Golden payloads: record once against the REAL promoted model, then check any endpoint against them.
MARKER: fraud_phase2_v1

    python scripts/golden.py record --base-url http://127.0.0.1:8000     # local service with the real model
    python scripts/golden.py check  --base-url http://127.0.0.1:8000     # point 1: local uvicorn
    python scripts/golden.py check  --base-url http://127.0.0.1:8080     # point 2: docker run
    python scripts/golden.py check  --base-url "http://$LB"             # point 3: EKS (CI does this too)

`check` uses only the standard library so the CI post-deploy job needs no pip install.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
TOL = 1e-6


def _parse(raw: bytes):
    try:
        return json.loads(raw or b"null")
    except ValueError:                      # e.g. an HTML error page from the load balancer
        return {"non_json_body": raw[:200].decode(errors="replace")}


def call(base: str, method: str, path: str, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, _parse(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _parse(e.read())


def wait_ready(base: str, wait_s: int) -> dict:
    deadline, last = time.time() + wait_s, None
    while True:
        try:
            status, body = call(base, "GET", "/health", timeout=10)
            if status == 200:
                return body
            last = f"HTTP {status}: {body}"
        except (urllib.error.URLError, OSError) as e:      # DNS / connection not ready yet: report the last one
            last = f"{type(e).__name__}: {e}"
        if time.time() > deadline:
            raise SystemExit(f"{base} not ready after {wait_s}s; last: {last}")
        time.sleep(5)


def build_payloads():
    """Deterministic selection from the committed fixture (record mode only)."""
    sys.path.insert(0, str(ROOT / "src"))
    import pandas as pd
    from fraud_api.features import stage02_row_to_request
    from fraud_api.predict import required_categoricals
    fix = ROOT / "tests" / "fixtures"
    required = set(required_categoricals(json.loads((fix / "preprocessor.json").read_text(encoding="utf-8")),
                                         json.loads((fix / "fitted_params_03.json").read_text(encoding="utf-8"))))
    s02 = pd.read_parquet(fix / "stage02_sample.parquet").sort_values("payment_id")
    groups = [("baseline", s02.head(12)),
              ("fraud_label", s02[s02.is_fraud.eq(1)].head(4)),
              ("no_card", s02[s02.card_token.isna()].head(2)),
              ("no_device_profile", s02[s02.device_type.isna()].head(2)),
              ("no_prior_login", s02[s02.hours_since_last_login.isna()].head(2)),
              ("no_basket", s02[s02.n_lines.isna()].head(1)),
              ("cash_on_delivery", s02[s02.payment_method.eq("Cash on Delivery")].head(2)),
              ("kyc_missing", s02[s02.kyc_level.isna()].head(1)),
              ("account_age_missing", s02[s02.account_age_days.isna()].head(1))]
    cases, seen = [], set()
    for name, g in groups:
        for _, row in g.iterrows():
            if row.payment_id not in seen:
                seen.add(row.payment_id)
                cases.append({"name": f"{name}:{row.payment_id}", "request": stage02_row_to_request(row)})
    base = dict(cases[0]["request"])
    gaps = dict(base, payment_id="synthetic-all-optional-null")
    for k in ["discount_amount", "item_count", "attempt_seq_in_session", "address_match", "shipping_speed",
              "shipping_addr_age_hours", "delivery_type", "kyc_level", "city_tier", "email_domain_class",
              "acquisition_channel", "account_age_days", "prior_return_rate", "prior_orders_12m",
              "merchant_category", "trailing_chargeback_rate_bps", "ip_country", "ip_region_code",
              "home_region_code", "asn_type", "asn_country", "reputation_score"]:
        if k not in required:                    # categoricals without a trained gap stay filled
            gaps[k] = None
    cases.append({"name": "all_optional_null", "request": gaps})
    probes = [
        {"name": "unknown_field", "request": dict(base, is_fraud=1), "expect_status": 422},
        {"name": "incumbent_field", "request": dict(base, alerted_flag="Y"), "expect_status": 422},
        {"name": "missing_required", "request": {k: v for k, v in base.items() if k != "payment_amount"},
         "expect_status": 422},
        {"name": "negative_amount", "request": dict(base, payment_amount=-5.0), "expect_status": 422},
        {"name": "unknown_level", "request": dict(base, payment_method="Crypto"), "expect_status": 422},
        {"name": "null_required_bool", "request": dict(base, is_3ds_success=None), "expect_status": 422},
    ]
    for k in sorted(required - {"payment_method"}):
        if k in base:
            probes.append({"name": f"null_required_categorical:{k}", "request": dict(base, **{k: None}),
                           "expect_status": 422})
    return cases, probes


def record(base: str, wait_s: int):
    health = wait_ready(base, wait_s)
    cases, probes = build_payloads()
    results = []
    for c in cases:
        status, body = call(base, "POST", "/v1/score", c["request"])
        if status != 200:
            raise SystemExit(f"record: case {c['name']} returned {status}: {body}")
        results.append({"name": c["name"], "status": status, "fraud_probability": body["fraud_probability"],
                        "score_raw": body["score_raw"], "decision": body["decision"],
                        "defaults_applied": body["defaults_applied"], "model_version": body["model_version"]})
    for p in probes:
        status, body = call(base, "POST", "/v1/score", p["request"])
        if status != p["expect_status"]:
            raise SystemExit(f"record: probe {p['name']} expected {p['expect_status']}, got {status}: {body}")
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "payloads.json").write_text(json.dumps({"cases": cases, "probes": probes}, indent=1), encoding="utf-8")
    (GOLDEN / "expected.json").write_text(json.dumps({
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recorded_against": base,
        "model_version": health["model_version"], "family": health["family"],
        "service_version": health["service_version"], "tolerance": TOL, "cases": results}, indent=1),
        encoding="utf-8")
    print(f"recorded {len(results)} cases + {len(probes)} probes against {health['model_version']}")


def check(base: str, wait_s: int, tol: float) -> int:
    health = wait_ready(base, wait_s)
    payloads = json.loads((GOLDEN / "payloads.json").read_text(encoding="utf-8"))
    expected = json.loads((GOLDEN / "expected.json").read_text(encoding="utf-8"))
    fails = []
    if health["model_version"] != expected["model_version"]:
        fails.append(f"model_version: live {health['model_version']} != golden {expected['model_version']}")
    exp = {e["name"]: e for e in expected["cases"]}
    worst = 0.0
    for c in payloads["cases"]:
        status, body = call(base, "POST", "/v1/score", c["request"])
        e = exp[c["name"]]
        if status != e["status"]:
            fails.append(f"{c['name']}: status {status} != {e['status']} ({body})")
            continue
        for k in ("fraud_probability", "score_raw"):
            d = abs(body[k] - e[k])
            worst = max(worst, d)
            if d > tol:
                fails.append(f"{c['name']}: {k} {body[k]!r} vs golden {e[k]!r} (|diff| {d:.3g} > {tol})")
        for k in ("decision", "defaults_applied"):
            if body[k] != e[k]:
                fails.append(f"{c['name']}: {k} {body[k]!r} != golden {e[k]!r}")
    for p in payloads["probes"]:
        status, body = call(base, "POST", "/v1/score", p["request"])
        if status != p["expect_status"]:
            fails.append(f"probe {p['name']}: {status} != {p['expect_status']} ({body})")
    print(f"{base}: {len(payloads['cases'])} cases, {len(payloads['probes'])} probes, "
          f"model {health['model_version']}, worst |diff| {worst:.3g}")
    for f in fails:
        print("  FAIL", f)
    print("GOLDEN CHECK", "PASSED" if not fails else f"FAILED ({len(fails)})")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--wait", type=int, default=60, help="seconds to wait for /health to return 200")
    ap.add_argument("--tolerance", type=float, default=TOL)
    a = ap.parse_args()
    if a.mode == "record":
        record(a.base_url, a.wait)
        return 0
    return check(a.base_url, a.wait, a.tolerance)


if __name__ == "__main__":
    sys.exit(main())
