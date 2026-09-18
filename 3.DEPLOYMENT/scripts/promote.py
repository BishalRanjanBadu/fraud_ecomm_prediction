"""Promote (or roll back to) a model version by rewriting the alias. MARKER: fraud_phase2_v1

    python scripts/promote.py --version v_20260916T181427Z_nb93c753a --accept-gate-deviation "Phase-1 sign-off 2026-09-17"
    python scripts/promote.py --version "$PREVIOUS_VERSION"      # rollback: same command, older version

What it does, in order (nothing is overwritten except the alias itself):
  1. reads the immutable version folder; manifest.json must exist (it is written last = completion marker)
  2. refuses a version whose gate did not pass unless --accept-gate-deviation gives the recorded reason
  3. pins the notebook-03 imputation params:
       first promotion  -> copies data/03_fitted_params.json to models/params/03_fitted_params_<sha12>.json
                           and writes models/params/by_version/<version>.json (both write-once), after checking
                           that the stage files belong to this version's pipeline run
       later promotions -> reuses the pinned pointer (this is what makes rollback exact)
  4. validates the whole compatibility contract with THIS environment's libraries (same code the pod runs)
  5. writes the alias with the previous alias embedded, then reads it back and compares bytes
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402

from fraud_api import constants as C  # noqa: E402
from fraud_api import registry, s3_io  # noqa: E402

BUCKET, REGION = "fraud-ecommerce", "ap-south-2"


def _json(s3, key):
    return json.loads(s3_io.get_bytes(s3, BUCKET, key))


def pin_params(s3, version: str, manifest: dict, log) -> dict:
    pointer_key = f"models/params/by_version/{version}.json"
    if s3_io.exists(s3, BUCKET, pointer_key):
        ptr = _json(s3, pointer_key)
        body = s3_io.get_bytes(s3, BUCKET, ptr["imputation_params_key"])
        if s3_io.sha256(body) != ptr["imputation_params_sha256"]:
            raise SystemExit(f"pinned params {ptr['imputation_params_key']} no longer match their recorded sha256")
        log(f"params: reusing pin {ptr['imputation_params_key']}")
        return ptr
    rec07 = _json(s3, "reports/07_run_record.json")
    rec03 = _json(s3, "reports/03_run_record.json")
    if rec07.get("version") != version:
        raise SystemExit(f"first promotion of {version} needs the stage files of its own run, but "
                         f"reports/07_run_record.json is for {rec07.get('version')}. Pin params for that run instead.")
    if rec03.get("marker") != C.TRAINING_MARKER or rec03["created_utc"] > manifest["created_utc"]:
        raise SystemExit("reports/03_run_record.json is not from this model's pipeline run "
                         f"(marker {rec03.get('marker')}, created {rec03.get('created_utc')})")
    body = s3_io.get_bytes(s3, BUCKET, "data/03_fitted_params.json")
    sha = s3_io.sha256(body)
    key = f"models/params/03_fitted_params_{sha[:12]}.json"
    if s3_io.exists(s3, BUCKET, key):
        if s3_io.sha256(s3_io.get_bytes(s3, BUCKET, key)) != sha:
            raise SystemExit(f"{key} exists with different content")
    else:
        s3_io.put_bytes(s3, BUCKET, key, body)
    ptr = {"version": version, "imputation_params_key": key, "imputation_params_sha256": sha,
           "source_key": "data/03_fitted_params.json", "source_run_created_utc": rec03["created_utc"],
           "pinned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": C.MARKER}
    s3_io.put_bytes(s3, BUCKET, pointer_key, json.dumps(ptr, indent=1).encode())
    log(f"params: pinned {key} (sha256 {sha[:16]}...)")
    return ptr


def promote(s3, version: str, alias_key: str, accept_reason: str | None, promoted_by: str, dry_run: bool,
            log=print) -> dict:
    prefix = f"models/{version}/"
    if not s3_io.exists(s3, BUCKET, prefix + "manifest.json"):
        raise SystemExit(f"{prefix}manifest.json not found - the version is missing or incomplete")
    blobs = {name: s3_io.get_bytes(s3, BUCKET, prefix + name) for name in registry.ARTIFACTS}
    manifest = json.loads(blobs["manifest.json"])
    metrics = _json(s3, prefix + "metrics.json")
    if manifest.get("version") != version or manifest.get("marker") != C.TRAINING_MARKER:
        raise SystemExit(f"manifest says version={manifest.get('version')} marker={manifest.get('marker')}")
    gate_passed = bool(metrics.get("gate_passed"))
    if not gate_passed and not accept_reason:
        raise SystemExit("the gate did not pass; re-run with --accept-gate-deviation \"reason text recorded in the sign-off\"")
    ptr = pin_params(s3, version, manifest, log) if not dry_run else {
        "imputation_params_key": "(dry-run)", "imputation_params_sha256": "(dry-run)"}
    previous = None
    if s3_io.exists(s3, BUCKET, alias_key):
        cur = _json(s3, alias_key)
        previous = {k: cur.get(k) for k in ("version", "model_prefix", "promoted_utc")}
    alias = {
        "alias": alias_key, "version": version, "model_prefix": prefix,
        "manifest_sha256": s3_io.sha256(blobs["manifest.json"]),
        "imputation_params_key": ptr["imputation_params_key"],
        "imputation_params_sha256": ptr["imputation_params_sha256"],
        "gate_passed": gate_passed, "gate_deviation": None if gate_passed else accept_reason,
        "family": manifest["family"], "feature_hash": manifest["feature_hash"],
        "promoted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "promoted_by": promoted_by, "previous": previous, "tool": C.MARKER,
    }
    if dry_run:
        log(json.dumps(alias, indent=1))
        return alias
    blobs["params03"] = s3_io.get_bytes(s3, BUCKET, ptr["imputation_params_key"])
    registry.validate(manifest, json.loads(blobs["preprocessor.json"]), json.loads(blobs["params03"]),
                      json.loads(blobs["feature_names.json"]), joblib.load(io.BytesIO(blobs["model.pkl"])),
                      joblib.load(io.BytesIO(blobs["calibrator.pkl"])), alias, blobs, [])
    body = json.dumps(alias, indent=1).encode()
    s3_io.put_bytes(s3, BUCKET, alias_key, body)
    if s3_io.get_bytes(s3, BUCKET, alias_key) != body:
        raise SystemExit(f"read-back of {alias_key} differs from what was written")
    log(f"alias {alias_key} -> {version} (previous: {previous['version'] if previous else 'none'}); read-back OK")
    return alias


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--alias", default="models/CURRENT.json",
                    help="models/CURRENT.json for prod, models/CURRENT_UAT.json for UAT")
    ap.add_argument("--accept-gate-deviation", default=None, metavar="REASON")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.alias not in {"models/CURRENT.json", "models/CURRENT_UAT.json"}:
        raise SystemExit("--alias must be models/CURRENT.json or models/CURRENT_UAT.json")
    import boto3
    who = boto3.client("sts", region_name=REGION).get_caller_identity()["Arn"]
    promote(s3_io.client(REGION), a.version, a.alias, a.accept_gate_deviation, who, a.dry_run)


if __name__ == "__main__":
    main()
