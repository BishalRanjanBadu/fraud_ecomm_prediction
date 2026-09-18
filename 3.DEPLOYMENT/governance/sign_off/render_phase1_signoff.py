"""MARKER: phase1_signoff_kit_v1

Renders PHASE1_SIGNOFF.md next to this file from the AS-RUN artifacts in S3. Read-only against S3.
Standard library only; talks to S3 through the AWS CLI already configured on this machine.

Run from Git Bash:  python render_phase1_signoff.py
"""
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BUCKET = "fraud-ecommerce"
REGION = "ap-south-2"
PACKAGE = "fraud_nb01-07_v2"
SIGNER = "Bishal Ranjan Badu"
SIGN_DATE = "2026-09-17"
OUT = Path(__file__).resolve().with_name("PHASE1_SIGNOFF.md")
ARTIFACTS = ["model.pkl", "calibrator.pkl", "preprocessor.json", "feature_names.json", "reference.parquet",
             "metrics.json", "model_card.md", "manifest.json"]
REPORTS = [f"reports/0{i}_run_record.json" for i in range(1, 8)]


# ----------------------------------------------------------------------------- S3 access (AWS CLI)
def _aws(args):
    return subprocess.run(["aws", *args, "--region", REGION], capture_output=True)


def fetch_aws(key):
    """Download one object to a temp file; return (bytes, metadata). Integrity: MD5 == ETag for single-part SSE-S3."""
    with tempfile.TemporaryDirectory() as d:
        target = str(Path(d) / "obj")
        r = _aws(["s3api", "get-object", "--bucket", BUCKET, "--key", key, target])
        if r.returncode != 0:
            raise RuntimeError(f"get-object failed for s3://{BUCKET}/{key}: {r.stderr.decode(errors='replace').strip()}")
        meta = json.loads(r.stdout.decode())
        body = Path(target).read_bytes()
    return body, {"etag": meta.get("ETag", "").strip('"'), "sse": meta.get("ServerSideEncryption", "none")}


def exists_aws(key):
    r = _aws(["s3api", "head-object", "--bucket", BUCKET, "--key", key])
    if r.returncode == 0:
        return True
    err = r.stderr.decode(errors="replace")
    if "404" in err or "Not Found" in err:
        return False
    raise RuntimeError(f"head-object failed for s3://{BUCKET}/{key}: {err.strip()}")


# ----------------------------------------------------------------------------- formatting
def num(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def pct(x, nd=3):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- render
def render(fetch, exists):
    problems, integrity = [], []

    def get(key):
        body, meta = fetch(key)
        sha = hashlib.sha256(body).hexdigest()
        md5 = hashlib.md5(body).hexdigest()
        if meta["sse"] in ("AES256", "none") and "-" not in meta["etag"] and meta["etag"] and meta["etag"] != md5:
            problems.append(f"download of {key} does not match its ETag")
        integrity.append((key, len(body), sha, meta["sse"]))
        return body

    cand = json.loads(get("models/CANDIDATE.json"))
    version, prefix = cand["version"], cand["prefix"]
    if prefix != f"models/{version}/":
        problems.append(f"CANDIDATE prefix {prefix} does not match version {version}")
    art = {name: get(prefix + name) for name in ARTIFACTS}
    man = json.loads(art["manifest.json"])
    met = json.loads(art["metrics.json"])
    feats = json.loads(art["feature_names.json"])
    rec = {k[8:10]: json.loads(get(k)) for k in REPORTS}
    current_exists = exists("models/CURRENT.json")

    # --- structural cross-checks (any failure aborts: the record would describe inconsistent artifacts)
    for label, v in [("manifest", man["version"]), ("metrics", met["version"]), ("07 run record", rec["07"]["version"])]:
        if v != version:
            problems.append(f"{label} version {v} != CANDIDATE {version}")
    for nb, r in rec.items():
        if r.get("marker") != PACKAGE:
            problems.append(f"reports/{nb}_run_record.json marker {r.get('marker')} != {PACKAGE}")
    if man.get("marker") != PACKAGE:
        problems.append(f"manifest marker {man.get('marker')} != {PACKAGE}")
    if rec["06"]["feature_hash"] != man["feature_hash"]:
        problems.append("06 run record feature_hash differs from the model manifest")
    if feats != man["feature_order"]:
        problems.append("feature_names.json differs from manifest feature_order")
    flagged = sorted(k for k, v in rec["05"]["sentinels"].items() if v["flag"])
    unack = [k for k in flagged if k not in rec["06"].get("acknowledged_sentinels", {})]
    if unack:
        problems.append(f"unacknowledged leakage sentinels: {unack}")
    if not (cand["gate_passed"] == met["gate_passed"] == rec["07"]["gate_passed"]):
        problems.append("gate_passed differs between CANDIDATE, metrics and the 07 run record")
    if problems:
        raise SystemExit("REFUSING TO WRITE THE SIGN-OFF - artifacts are inconsistent:\n  - " + "\n  - ".join(problems))

    g, gr = met["gate"], met["gate"]["result"]
    gate_passed = met["gate_passed"]
    val, rep, tst, cal = met["validation"], met["test_report_all_candidates"], met["test"], met["calibration"]
    fam = man["family"]
    op = man["operating_point"]
    fair = met["fairness"]
    r1, r2, r3, r5, r6 = rec["01"], rec["02"], rec["03"], rec["05"], rec["06"]
    drift = man.get("version_drift") or {}

    deviations = [
        ("Gate is provisional", "Cost figures are pending. The absolute floor and operating threshold are not yet derived "
         "from a cost matrix. Review trigger: " + g["review_trigger"] + "."),
        ("Gate result: " + ("PASSED" if gate_passed else "NOT PASSED"),
         "Recorded as rendered from metrics.json." + ("" if gate_passed else
         " The signer accepts this candidate for Phase 2 deployment work with the gate not passed, as a known "
         "deviation. The provisional gate stays open and must be re-evaluated when cost figures arrive.")),
        ("Credentials", "Local work, Colab and CI use the account root access key (signer's decision; IAM principals "
         "are not used in this project). Pods still use IRSA in-cluster."),
        ("No git SHA in the manifest", "The notebooks ran in Colab before the repo existed. Record the first commit SHA "
         "that contains 2.RUNNING_CODE_FILES/ in this file's 'Code reference' line when pushing."),
        ("Immature test labels", f"{tst['label_maturity']['pending']} of {tst['label_maturity']['test_chargebacks']} "
         "test-period chargebacks were still pending at extraction. All-row test PR-AUC is understated."),
        ("Small audit stratum", f"{gr['audit_positives']} audit positives in TEST, so the small-sample rule applied: "
         + num(gr["small_sample_rule_applied"]) + "."),
        ("Velocity features are lifetime as-of counts", "They grow with accumulated history. A trailing-window count "
         "is a Phase-3 retrain candidate. Serving needs per-key first-seen state (Phase-2 design: the caller supplies "
         "the features and `src/features.py` is the reference implementation)."),
        ("Snapshot attributes", "`prior_orders_12m`, `prior_return_rate` and the merchant fields are not point-in-time. "
         "They are accepted per the data contract (post_decision = false)."),
        ("`ip_region_code == 99`", "Semantics unresolved (foreign IP or unknown). Values are left unchanged; notebook 02 "
         "prints a diagnostic crosstab. The column is excluded as a feature and affects only `ip_region_mismatch`."),
        ("Retry rows", f"{r1['retries_retained']:,} retries keep their artefactual label 0 (locked decision); "
         "`is_retry` is excluded as a feature."),
        ("Shared validation window", "Early stopping, isotonic calibration and the operating point all use VAL. "
         "TEST is the independent read."),
    ]
    if drift:
        deviations.append(("Library version drift at training time", json.dumps(drift)))
    for k, v in met["flags"].items():
        if v:
            deviations.append((f"Flag `{k}`", "raised in metrics.json; see the fairness and explanation sections"))

    cv_rows = [(r["family"], num(r["cv_ap_mean"]), num(r["cv_ap_std"]), num(r["cv_ipw_ap_mean"]),
                ("**selected**" if r["family"] == fam else "")) for r in met["cv"]]
    val_rows = [(name, num(v.get("val_ap")), num(v.get("val_ipw_ap")), num(v.get("val_audit_ap")),
                 num(v.get("val_roc_auc")), num(v.get("n_trees")) if "n_trees" in v else "")
                for name, v in val.items()]
    test_rows = [(name, num(v["test_ap_ipw"]), num(v["test_ap_all"]), num(v["test_ap_audit_only"]),
                  ("**selected**" if name == fam else "")) for name, v in rep.items()]
    pins = [(k, v) for k, v in man["library_versions"].items()]
    integ_rows = [(k, f"{n:,}", sha[:16] + "...", sse) for k, n, sha, sse in integrity]
    removed = "; ".join(f"{x['reason']}: {x['rows']:,}" for x in r2["rows_removed"])

    md = f"""# Phase 1 sign-off — FRAUD_ECOMMERCE

| | |
|---|---|
| **Decision** | **Phase 1 accepted.** Proceed to Phase 2 (deployment). |
| **Signed off by** | {SIGNER} |
| **Date** | {SIGN_DATE} |
| **Notebook package** | `{PACKAGE}` (notebooks 01–07, outputs stripped in the repo) |
| **Candidate model** | `{version}` |
| **Promotion state** | `models/CURRENT.json` {"EXISTS" if current_exists else "not yet written"}. Promotion is a Phase-2 runbook step, done after this record is committed. |
| **Code reference** | git SHA: *to be recorded at first push* |
| **Rendered** | {datetime.now(timezone.utc).isoformat(timespec="seconds")} by `render_phase1_signoff.py` from S3 (as-run values only) |

## 1. What was reviewed

**Notebooks.** `01_Data_Loading_and_First_Look` through `07_Model_Building_and_Evaluation`, from package `{PACKAGE}`. They were run top to bottom in Colab (Python {man['library_versions'].get('python')}) against `s3://{BUCKET}/`, flat layout, region `{REGION}`.

**Artifacts read for this record.** Every object below was downloaded and its size and SHA-256 recorded. Where S3 exposes an MD5 ETag, the download was verified against it.

{table(["object", "bytes", "sha256", "encryption"], integ_rows)}

**Cross-checks that passed before this file was written:**
* The version matches across `CANDIDATE.json`, `manifest.json`, `metrics.json` and the 07 run record.
* All seven run records and the manifest carry marker `{PACKAGE}`.
* The 06 feature hash equals the manifest feature hash.
* `feature_names.json` equals the manifest feature order.
* There are no unacknowledged leakage sentinels.
* `gate_passed` is consistent in all three places.

## 2. Data and labels (as run)

| Item | Value |
|---|---|
| Payments in → out (01) | {num(r1['rows_in'])} → {num(r1['rows_out'])} ({num(r1['exact_duplicates_dropped'])} exact duplicates, {num(r1['pk_conflicts'])} PK-conflict rows, {num(r1['retries_retained'])} retries kept) |
| Rows removed in cleaning (02) | {removed} |
| Rows out of 02 | {num(r2['rows_out'])} |
| Split | chronological at {r1['split_date']}; train {num(r1['train_rows'])}, test {num(r1['test_rows'])} |
| Observed label rate | {pct(r1['observed_positive_rate'])} |
| Estimated true prevalence (01) | {pct(r1['estimated_true_prevalence'])} |
| Label weights | alert review 1.0, random audit 1.0, chargeback 0.60, unadjudicated 0.35 |
| Review design (train) | audit rows that were alerted: {r5['review_design_train']['audit_rows_alerted']} (the audit samples the non-alerted stratum only); design-weighted prevalence {pct(r5['review_design_train']['ipw_prevalence_train'])} |
| Velocity fix (02) | as-of counts; rows that differ from the old full-period counts: {json.dumps(r2['velocity_fix']['rows_differing_from_full_period'])} |
| Winsor caps (03, train-fitted) | {json.dumps({k: round(v, 2) for k, v in r3['winsor_caps'].items()})} |
| Leakage sentinels (05) | {json.dumps({k: v['flag'] for k, v in r5['sentinels'].items()})} |
| Leakage self-checks (06) | {json.dumps(r6['leakage_checks'])} |
| Structural identities asserted (06) | {len(r6['identities_asserted'])} |
| Features | {r6['tree_features']} tree / {r6['linear_features']} linear; {len(r6['excluded'])} columns excluded by name with reasons |

## 3. Model selection (CV on DEV, chargeback-maturity masked)

**Rule.** Candidates are compared on unweighted CV PR-AUC. LightGBM is the pre-registered default, and another family replaces it only with a relative CV margin of at least 5%.

**Outcome.** {man['selection_reason']}.

{table(["family", "CV PR-AUC", "std", "CV design-weighted PR-AUC", ""], cv_rows)}

## 4. Validation quarter (2025-04-01 → 2025-06-30)

{table(["model", "PR-AUC all rows", "design-weighted", "audit only", "ROC-AUC", "trees"], val_rows)}

**Calibration** (isotonic, fitted on VAL). Brier score {num(cal['val_brier_raw'])} → {num(cal['val_brier_cal'])}; ECE {num(cal['val_ece_raw'])} → {num(cal['val_ece_cal'])}.
This calibrates to the observed union-label rate, not the true fraud rate.

**Operating point (provisional).** {op['rule']}. The raw threshold is {num(op['threshold_raw'], 6)}, which is {num(op['threshold_calibrated'], 6)} on the calibrated scale.

## 5. Test set — scored once

{table(["model", "design-weighted PR-AUC", "all rows", "audit only", ""], test_rows)}

**Gate definition** (fixed before the test set was scored).
* **Metric.** {g['metric']}.
* **Population.** {g['population']}.
* **Why not audit only.** {g['why_not_audit_only']}.
* **Absolute floor.** {num(g['absolute_floor'])}, derived as follows: {g['absolute_floor_derivation']}.
* **Relative requirement.** {g['relative_ratio']}× the incumbent ({g['relative_derivation']}).
* **Small-sample rule.** {g['small_sample_rule']}.

**Gate result.**

| | value |
|---|---|
| Review sample | {num(gr['sample_rows_alerted_stratum'])} alerted-stratum rows ({num(gr['positives_alerted_stratum'])} positives) + {num(gr['sample_rows_audit_stratum'])} audit rows ({num(gr['audit_positives'])} positives) |
| Stratum weights | alerted {num(gr['stratum_weights']['alerted'])}, non-alerted {num(gr['stratum_weights']['non_alerted'])} |
| Design-weighted prevalence (test) | {pct(gr['ipw_prevalence_test'])} |
| Model PR-AUC | {num(gr['model_ap'])} (90% CI {num(gr['model_ap_ci90'][0])}–{num(gr['model_ap_ci90'][1])}) |
| Incumbent PR-AUC | {num(gr['incumbent_ap'])} |
| Ratio | {num(gr['ratio_point'], 3)} (90% CI {num(gr['ratio_ci90'][0], 3)}–{num(gr['ratio_ci90'][1], 3)}) |
| Small-sample rule applied | {num(gr['small_sample_rule_applied'])} |
| Absolute half | {num(gr['absolute']['value'])} vs floor {num(gr['absolute']['floor'])} → **{"pass" if gr['absolute']['passed'] else "fail"}** |
| Relative half | {num(gr['relative']['value'], 3)} vs {gr['relative']['required']} → **{"pass" if gr['relative']['passed'] else "fail"}** |
| Audit-only (supporting) | model {num(gr['audit_only_supporting']['model'])} vs incumbent {num(gr['audit_only_supporting']['incumbent'])} |
| **Gate** | **{"PASSED" if gate_passed else "NOT PASSED"} (provisional)** |

**Other test metrics.**
* ROC-AUC {num(tst['roc_auc_all'])}; PR-AUC on adjudicated rows only {num(tst['ap_adjudicated_only'])}.
* Calibrated Brier score {num(tst['brier_cal'])}; ECE {num(tst['ece_cal'])}.
* Mean calibrated score {num(tst['mean_cal_vs_observed'][0])} vs observed rate {num(tst['mean_cal_vs_observed'][1])}.

**Operating point on TEST:**
* Model: alert rate {pct(tst['operating_model']['alert_rate'])}, precision {num(tst['operating_model']['precision'], 3)}, recall {num(tst['operating_model']['recall'], 3)}.
* Incumbent: alert rate {pct(tst['operating_incumbent']['alert_rate'])}, precision {num(tst['operating_incumbent']['precision'], 3)}, recall {num(tst['operating_incumbent']['recall'], 3)}.

## 6. Fairness and personal data

* **Regime.** India DPDP Act 2023, lawful basis fraud prevention. No direct protected attributes are present in the data.
* **Geographic disparity** (selection-rate ratio, lowest group, VAL):
  * city_tier {num(fair['city_tier']['min_selection_rate_ratio'], 3)} (four-fifths flag: {num(fair['city_tier']['four_fifths_flag'])});
  * city {num(fair['city']['min_selection_rate_ratio'], 3)} (four-fifths flag: {num(fair['city']['four_fifths_flag'])}).
* **Screening only.** The four-fifths rule is a screening flag, not a pass condition.
* **Excluded as inputs.** Identifiers, IP addresses and pincodes.

## 7. Carried into Phase 2 (binding)

**Model to promote:** `{version}`

| Item | Value |
|---|---|
| Family | `{fam}` (`{man['estimator_class']}`) |
| Input family | `{man['input_family']}` |
| Parameters | `{json.dumps(man['params'])}` |
| Trees | {num(man['n_trees'])} |
| Features | {len(man['feature_order'])} |
| Feature hash | `{man['feature_hash'][:16]}...` |
| Data hash (06) | `{man['data_hash_06'][:16]}...` |
| Contract | `{man['contract_version']}` |

**Serving must restore** feature order, dtypes and encoder maps from `manifest.json` / `preprocessor.json`, and apply the calibrator from `calibrator.pkl`.

**Library pins.** Train/serve parity is a hard contract, so every library the image installs uses exactly the version below:

{table(["library", "version"], pins)}

**Approved Phase-2 decisions:**
* **API input.** The caller sends enriched raw features; `src/features.py` is the reference implementation of the as-of counts.
* **Model delivery.** `CURRENT.json` alias.
* **Shape.** 1 replica, API only; canary deferred (documented deviation).
* **Node.** `t4g.small`, ARM64 (free-tier eligible in {REGION}).
* **Names.** `fraud-ecomm-eks`, `fraud-ecomm-api`, `fraud-prod` / `fraud-uat`, `fraud-api-sa`, `fraud-api`.
* **Repo layout.** Root `FRAUD_ECOMMERCE`, application in `3.DEPLOYMENT/`, raw CSVs gitignored.
* **CI credentials.** Root access key in GitHub Actions secrets (signer's decision).
* **Fallback.** Opt-in incumbent rule, tagged as fallback.
* **Explanations.** Opt-in.
* **Prediction logging and S3 write access.** Off until Phase 3.
* **Base image.** `python:3.13-slim`.

## 8. Accepted deviations and known limitations

{table(["item", "detail"], deviations)}

## 9. Statement

Phase 1 is accepted on the evidence above.
* **Not promoted yet.** The candidate `{version}` is approved for Phase-2 deployment work, but it is not promoted until the Phase-2 runbook writes `models/CURRENT.json` and verifies the write by reading it back.
* **Gate stays provisional.** It remains open until cost figures arrive.
* **Every deviation above is known and accepted.**

— {SIGNER}, {SIGN_DATE}
"""
    return md, gate_passed, version


def main():
    md, gate_passed, version = render(fetch_aws, exists_aws)
    OUT.write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(md):,} chars)")
    print(f"candidate {version} | gate {'PASSED' if gate_passed else 'NOT PASSED'} (provisional)")
    if not gate_passed:
        print("NOTE: the record states the signer accepts deployment work with the gate not passed. "
              "If that is not your decision, do not commit this file.")


if __name__ == "__main__":
    sys.exit(main())
