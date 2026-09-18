"""Build the committed test fixture from the REAL stage files and artifacts. MARKER: fraud_phase2_v1

    python scripts/make_fixture.py --version v_20260916T181427Z_nb93c753a

Reads (read-only): the version's manifest.json + preprocessor.json, data/06_encoding_params.json,
data/03_fitted_params.json, data/02_cleaned.parquet, data/06_encoded_tree.parquet, data/06_encoded_linear.parquet,
reports/07_run_record.json.
Writes tests/fixtures/: stage02_sample.parquet, stage06_tree_sample.parquet, stage06_linear_sample.parquet,
preprocessor.json, fitted_params_03.json, FIXTURE_INFO.json. CI tests then run with NO cloud credentials.

The fixture is refused unless the service transform reproduces the notebooks' stage-06 rows cell for cell.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError  # noqa: E402

from fraud_api import constants as C  # noqa: E402
from fraud_api import s3_io  # noqa: E402
from fraud_api.features import STAGE02_COLUMNS, stage02_row_to_request  # noqa: E402
from fraud_api.predict import check_levels, to_source_frame  # noqa: E402
from fraud_api.schema import ScoreRequest  # noqa: E402
from fraud_api.transform import (ContractError, build_features, feature_contract_hash,  # noqa: E402
                                 restore_tree_dtypes)

BUCKET, REGION, SEED = "fraud-ecommerce", "ap-south-2", 42
OUT = ROOT / "tests" / "fixtures"
NULLABLE_SOURCES = ["discount_amount", "item_count", "attempt_seq_in_session", "address_match_flag",
                    "shipping_speed", "shipping_addr_age_hours", "delivery_type", "kyc_level", "city_tier",
                    "email_domain_class", "acquisition_channel", "account_age_days", "prior_return_rate",
                    "prior_orders_12m", "merchant_category", "trailing_chargeback_rate_bps", "ip_country",
                    "ip_region_code", "home_region_code", "asn_type", "reputation_score", "payment_gateway",
                    "card_token", "device_type", "device_id", "n_lines", "hours_since_last_login",
                    "last_login_failed_attempts", "last_login_risk", "issuer", "product_type", "browser_family"]


def validate_row(row, p06, p03):
    try:
        req = ScoreRequest.model_validate(stage02_row_to_request(row))
        check_levels([req], p06, p03)
        return req, None
    except ValidationError as e:
        err = e.errors()[0]
        return None, f"schema:{'.'.join(str(x) for x in err['loc'])}:{err['type']}"
    except ContractError as e:
        return None, f"level:{e.field}"


def build(s3, version: str, out_dir: Path, n_target: int = 4000, log=print) -> dict:
    get = lambda k: s3_io.get_bytes(s3, BUCKET, k)  # noqa: E731
    prefix = f"models/{version}/"
    manifest = json.loads(get(prefix + "manifest.json"))
    p06_bytes = get(prefix + "preprocessor.json")
    p06 = json.loads(p06_bytes)
    if manifest.get("marker") != C.TRAINING_MARKER or manifest.get("version") != version:
        raise SystemExit("manifest marker/version mismatch")
    if feature_contract_hash(p06) != manifest["feature_hash"]:
        raise SystemExit("preprocessor.json does not reproduce the manifest feature hash")
    if json.loads(get("data/06_encoding_params.json")) != p06:
        raise SystemExit("data/06_encoding_params.json differs from the version's preprocessor.json: "
                         "the stage files belong to a different pipeline run")
    rec07 = json.loads(get("reports/07_run_record.json"))
    if rec07.get("version") != version:
        raise SystemExit(f"reports/07_run_record.json is for {rec07.get('version')}, not {version}")
    p03_bytes = get("data/03_fitted_params.json")
    p03 = json.loads(p03_bytes)
    raw = {k: get(k) for k in ["data/02_cleaned.parquet", "data/06_encoded_tree.parquet",
                               "data/06_encoded_linear.parquet"]}
    s02 = pd.read_parquet(io.BytesIO(raw["data/02_cleaned.parquet"]))
    log(f"stage 02: {s02.shape}")

    # candidate pool: every stratum, every null pattern, the structural edge cases
    rng_state = SEED
    s02 = s02.assign(_card=s02.card_token.notna().astype(int)).sort_values("payment_id", kind="mergesort")
    picks = []
    for _, g in s02.groupby(["label_source", "is_fraud", "_card"], sort=True):
        picks.append(g.sample(n=min(len(g), 700), random_state=rng_state))
    for col in NULLABLE_SOURCES:
        m = s02[col].isna()
        if m.any():
            picks.append(s02[m].sample(n=min(int(m.sum()), 40), random_state=rng_state))
    for mask in [s02.payment_method.eq("Cash on Delivery"), s02.device_id.notna() & s02.device_type.isna(),
                 s02.ip_region_code.eq(99), s02.is_fraud.eq(1)]:
        if mask.any():
            picks.append(s02[mask].sample(n=min(int(mask.sum()), 150), random_state=rng_state))
    pool = pd.concat(picks).drop_duplicates("payment_id").sort_values("payment_id", kind="mergesort")
    log(f"candidate pool: {len(pool):,} rows")

    excluded, keep = collections.Counter(), []
    for idx, row in pool.iterrows():
        _, why = validate_row(row, p06, p03)
        if why:
            excluded[why] += 1
        else:
            keep.append(idx)
    valid = pool.loc[keep]
    # each edge case is capped, so no single group (e.g. card-less payments, ~62% of traffic) crowds out the rest
    edge = {"fraud": (valid.is_fraud.eq(1), 800), "cod": (valid.payment_method.eq("Cash on Delivery"), 150),
            "no_card": (valid.card_token.isna(), 150), "card": (valid.card_token.notna(), 150),
            "no_device_profile": (valid.device_type.isna(), 150), "no_basket": (valid.n_lines.isna(), 150),
            "no_prior_login": (valid.hours_since_last_login.isna(), 150)}
    must = pd.concat([valid[m].sample(n=min(int(m.sum()), cap), random_state=SEED)
                      for m, cap in edge.values() if m.any()]).drop_duplicates("payment_id")
    must = must.head(n_target)
    rest = valid.drop(must.index)
    n_rest = max(0, n_target - len(must))
    final = pd.concat([must, rest.sample(n=min(len(rest), n_rest), random_state=SEED)])
    final = final.drop_duplicates("payment_id").sort_values("payment_id", kind="mergesort")
    final = final[STAGE02_COLUMNS].reset_index(drop=True)
    coverage = {k: int(m.loc[valid.payment_id.isin(set(final.payment_id))].sum()) for k, (m, _) in edge.items()}
    log(f"fixture rows: {len(final):,} (positives {int(final.is_fraud.sum())}); coverage {coverage}; "
        f"excluded: {dict(excluded)}")

    ids = set(final.payment_id)
    t06 = pd.read_parquet(io.BytesIO(raw["data/06_encoded_tree.parquet"]))
    t06 = t06[t06.payment_id.isin(ids)].set_index("payment_id").loc[final.payment_id].reset_index()
    l06 = pd.read_parquet(io.BytesIO(raw["data/06_encoded_linear.parquet"]))
    l06 = l06[l06.payment_id.isin(ids)].set_index("payment_id").loc[final.payment_id].reset_index()
    tree_cols = ["payment_id"] + p06["tree"]["feature_order"]
    lin_cols = ["payment_id"] + p06["linear"]["feature_order"]
    t06, l06 = t06[tree_cols], l06[lin_cols]

    # parity gate: refuse to write a fixture the service cannot reproduce
    reqs = [ScoreRequest.model_validate(stage02_row_to_request(r)) for _, r in final.iterrows()]
    src = to_source_frame(reqs)
    Xt, _, _ = build_features(src, p06, p03, "tree")
    Xl, _, _ = build_features(src, p06, p03, "linear")
    pd.testing.assert_frame_equal(Xt.reset_index(drop=True),
                                  restore_tree_dtypes(t06.drop(columns="payment_id"), p06), check_exact=True)
    pd.testing.assert_frame_equal(Xl.reset_index(drop=True), l06.drop(columns="payment_id").astype("float64"),
                                  check_exact=False, rtol=0, atol=1e-12)
    log("parity: service transform reproduces stage-06 tree (exact) and linear (1e-12) rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_dir / "stage02_sample.parquet", index=False)
    t06.to_parquet(out_dir / "stage06_tree_sample.parquet", index=False)
    l06.to_parquet(out_dir / "stage06_linear_sample.parquet", index=False)
    (out_dir / "preprocessor.json").write_bytes(p06_bytes)
    (out_dir / "fitted_params_03.json").write_bytes(p03_bytes)
    info = {
        "marker": C.MARKER, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bucket": BUCKET, "version": version, "family": manifest["family"], "params": manifest["params"],
        "n_trees": manifest["n_trees"], "input_family": manifest["input_family"],
        "feature_hash": manifest["feature_hash"], "library_versions": manifest["library_versions"],
        "p06_sha256": s3_io.sha256(p06_bytes), "p03_sha256": s3_io.sha256(p03_bytes),
        "data_sha256": {k: s3_io.sha256(v) for k, v in raw.items()},
        "rows": len(final), "positives": int(final.is_fraud.sum()), "pool_rows": len(pool),
        "excluded": dict(excluded), "coverage": coverage, "parity_checked": True,
    }
    (out_dir / "FIXTURE_INFO.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"wrote {out_dir}")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--rows", type=int, default=4000)
    a = ap.parse_args()
    build(s3_io.client(REGION), a.version, OUT, a.rows)


if __name__ == "__main__":
    main()
