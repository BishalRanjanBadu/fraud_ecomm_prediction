# Phase 1 sign-off — FRAUD_ECOMMERCE

| | |
|---|---|
| **Decision** | **Phase 1 accepted.** Proceed to Phase 2 (deployment). |
| **Signed off by** | Bishal Ranjan Badu |
| **Date** | 2026-09-17 |
| **Notebook package** | `fraud_nb01-07_v2` (notebooks 01–07, outputs stripped in the repo) |
| **Candidate model** | `v_20260916T181427Z_nb93c753a` |
| **Promotion state** | `models/CURRENT.json` not yet written. Promotion is a Phase-2 runbook step, done after this record is committed. |
| **Code reference** | git SHA: 596b00c4e45aa5d851672ff030a9c9cf9d5de152 (first commit containing 2.RUNNING_CODE_FILES/) |
| **Rendered** | 2026-09-17T02:24:13+00:00 by `render_phase1_signoff.py` from S3 (as-run values only) |

## 1. What was reviewed

**Notebooks.** `01_Data_Loading_and_First_Look` through `07_Model_Building_and_Evaluation`, from package `fraud_nb01-07_v2`. They were run top to bottom in Colab (Python 3.13.15) against `s3://fraud-ecommerce/`, flat layout, region `ap-south-2`.

**Artifacts read for this record.** Every object below was downloaded and its size and SHA-256 recorded. Where S3 exposes an MD5 ETag, the download was verified against it.

| object | bytes | sha256 | encryption |
|---|---|---|---|
| models/CANDIDATE.json | 275 | 0cdeed9cba5551d2... | AES256 |
| models/v_20260916T181427Z_nb93c753a/model.pkl | 1,548,340 | 934e26a3a0d09933... | AES256 |
| models/v_20260916T181427Z_nb93c753a/calibrator.pkl | 1,463 | 6da47e0c4a38b153... | AES256 |
| models/v_20260916T181427Z_nb93c753a/preprocessor.json | 20,096 | 3efbdc4bc07b733e... | AES256 |
| models/v_20260916T181427Z_nb93c753a/feature_names.json | 1,144 | 0fc120e4adcdd735... | AES256 |
| models/v_20260916T181427Z_nb93c753a/reference.parquet | 26,098,007 | 2988bae0b94cdc8b... | AES256 |
| models/v_20260916T181427Z_nb93c753a/metrics.json | 13,440 | 75a074a93e9bb1c1... | AES256 |
| models/v_20260916T181427Z_nb93c753a/model_card.md | 3,423 | b59eeea920dbfbe3... | AES256 |
| models/v_20260916T181427Z_nb93c753a/manifest.json | 8,712 | f5f7cbbb28ca10c0... | AES256 |
| reports/01_run_record.json | 2,279 | 1986d5d1b4e356fb... | AES256 |
| reports/02_run_record.json | 1,412 | 262d84fcbfdd258b... | AES256 |
| reports/03_run_record.json | 2,183 | c3965203b61f1321... | AES256 |
| reports/04_run_record.json | 1,345 | dac18369031d117b... | AES256 |
| reports/05_run_record.json | 7,414 | e38a1a81370c1576... | AES256 |
| reports/06_run_record.json | 7,580 | 34d38eb16bf9f408... | AES256 |
| reports/07_run_record.json | 3,190 | 7d7d870870653264... | AES256 |

**Cross-checks that passed before this file was written:**
* The version matches across `CANDIDATE.json`, `manifest.json`, `metrics.json` and the 07 run record.
* All seven run records and the manifest carry marker `fraud_nb01-07_v2`.
* The 06 feature hash equals the manifest feature hash.
* `feature_names.json` equals the manifest feature order.
* There are no unacknowledged leakage sentinels.
* `gate_passed` is consistent in all three places.

## 2. Data and labels (as run)

| Item | Value |
|---|---|
| Payments in → out (01) | 503,400 → 501,400 (1,994 exact duplicates, 12 PK-conflict rows, 1,400 retries kept) |
| Rows removed in cleaning (02) | non-positive / unparseable amount: 394; internal test transaction: 63 |
| Rows out of 02 | 500,943 |
| Split | chronological at 2025-07-01; train 350,787, test 150,613 |
| Observed label rate | 0.537% |
| Estimated true prevalence (01) | 0.974% |
| Label weights | alert review 1.0, random audit 1.0, chargeback 0.60, unadjudicated 0.35 |
| Review design (train) | audit rows that were alerted: 0 (the audit samples the non-alerted stratum only); design-weighted prevalence 0.966% |
| Velocity fix (02) | as-of counts; rows that differ from the old full-period counts: {"device_n_customers": 8517, "ip_n_customers": 10671} |
| Winsor caps (03, train-fitted) | {"payment_amount": 110883.68, "basket_value": 116057.8, "max_unit_price": 51676.4, "amount_vs_merchant_ticket": 25.71} |
| Leakage sentinels (05) | {"S1_near_perfect_separation": false, "S2_zero_fee_non_cod": false, "S3_velocity_not_point_in_time": false} |
| Leakage self-checks (06) | {"fit_ignores_test_rows": true, "fit_ignores_corrupted_test_rows": true, "features_independent_of_label_and_incumbent_columns": true} |
| Structural identities asserted (06) | 25 |
| Features | 57 tree / 146 linear; 41 columns excluded by name with reasons |

## 3. Model selection (CV on DEV, chargeback-maturity masked)

**Rule.** Candidates are compared on unweighted CV PR-AUC. LightGBM is the pre-registered default, and another family replaces it only with a relative CV margin of at least 5%.

**Outcome.** lightgbm is the CV leader.

| family | CV PR-AUC | std | CV design-weighted PR-AUC |  |
|---|---|---|---|---|
| lightgbm | 0.4329 | 0.0404 | 0.3426 | **selected** |
| xgboost | 0.4316 | 0.0411 | 0.3295 |  |
| logreg | 0.4028 | 0.0445 | 0.3054 |  |

## 4. Validation quarter (2025-04-01 → 2025-06-30)

| model | PR-AUC all rows | design-weighted | audit only | ROC-AUC | trees |
|---|---|---|---|---|---|
| logreg | 0.3799 | 0.2671 | 0.1300 | 0.9366 | n/a |
| lightgbm | 0.4266 | 0.2987 | 0.1408 | 0.9430 | 276 |
| xgboost | 0.4134 | 0.2842 | 0.1343 | 0.9410 | 250 |
| incumbent | 0.3334 | 0.2154 | 0.0395 | 0.8875 |  |

**Calibration** (isotonic, fitted on VAL). Brier score 0.0040 → 0.0039; ECE 0.0018 → 0.0001.
This calibrates to the observed union-label rate, not the true fraud rate.

**Operating point (provisional).** top-k by raw score, k = incumbent alert rate on VAL (provisional until cost figures). The raw threshold is 0.093821, which is 0.075758 on the calibrated scale.

## 5. Test set — scored once

| model | design-weighted PR-AUC | all rows | audit only |  |
|---|---|---|---|---|
| logreg | 0.2185 | 0.4208 | 0.0766 |  |
| lightgbm | 0.2741 | 0.5142 | 0.1212 | **selected** |
| xgboost | 0.2649 | 0.5060 | 0.1150 |  |
| incumbent | 0.1911 | 0.4396 | 0.0546 |  |

**Gate definition** (fixed before the test set was scored).
* **Metric.** design-weighted PR-AUC (average precision), hard union label.
* **Population.** TEST reviewed sample: alert_review rows (alerted stratum) + random_audit rows (non-alerted stratum), each weighted by its stratum's inverse sampling fraction.
* **Why not audit only.** random_audit contains no alerted rows, so audit-only compares models only where the incumbent declined to act.
* **Absolute floor.** 0.2350, derived as follows: incumbent payment_risk_score design-weighted PR-AUC on the train period (audit rows 8314, audit positives 53); fixed before test scoring; note train prevalence differs from test.
* **Relative requirement.** 1.25× the incumbent (model PR-AUC must be >= 1.25x the incumbent's on the same weighted test sample (approved default)).
* **Small-sample rule.** if TEST audit positives < 50: both halves use the 5th percentile of a stratified paired bootstrap (B=1000).

**Gate result.**

| | value |
|---|---|
| Review sample | 1,624 alerted-stratum rows (531 positives) + 3,681 audit rows (41 positives) |
| Stratum weights | alerted 1.0025, non-alerted 40.4363 |
| Design-weighted prevalence (test) | 1.456% |
| Model PR-AUC | 0.2741 (90% CI 0.2305–0.3374) |
| Incumbent PR-AUC | 0.1911 |
| Ratio | 1.434 (90% CI 1.245–1.644) |
| Small-sample rule applied | yes |
| Absolute half | 0.2305 vs floor 0.2350 → **fail** |
| Relative half | 1.245 vs 1.25 → **fail** |
| Audit-only (supporting) | model 0.1212 vs incumbent 0.0546 |
| **Gate** | **NOT PASSED (provisional)** |

**Other test metrics.**
* ROC-AUC 0.9631; PR-AUC on adjudicated rows only 0.6116.
* Calibrated Brier score 0.0029; ECE 0.0013.
* Mean calibrated score 0.0057 vs observed rate 0.0044.

**Operating point on TEST:**
* Model: alert rate 1.087%, precision 0.311, recall 0.762.
* Incumbent: alert rate 1.082%, precision 0.326, recall 0.796.

## 6. Fairness and personal data

* **Regime.** India DPDP Act 2023, lawful basis fraud prevention. No direct protected attributes are present in the data.
* **Geographic disparity** (selection-rate ratio, lowest group, VAL):
  * city_tier 0.938 (four-fifths flag: no);
  * city 0.529 (four-fifths flag: yes).
* **Screening only.** The four-fifths rule is a screening flag, not a pass condition.
* **Excluded as inputs.** Identifiers, IP addresses and pincodes.

## 7. Carried into Phase 2 (binding)

**Model to promote:** `v_20260916T181427Z_nb93c753a`

| Item | Value |
|---|---|
| Family | `lightgbm` (`lightgbm.sklearn.LGBMClassifier`) |
| Input family | `tree` |
| Parameters | `{"colsample_bytree": 0.7475884550556351, "learning_rate": 0.021648036763001914, "min_child_samples": 225, "n_estimators": 280, "num_leaves": 50, "reg_lambda": 0.005357280069601832, "subsample": 0.902144564127061}` |
| Trees | 276 |
| Features | 57 |
| Feature hash | `64bfdc79b8649a15...` |
| Data hash (06) | `39ba36383a149a7c...` |
| Contract | `v1` |

**Serving must restore** feature order, dtypes and encoder maps from `manifest.json` / `preprocessor.json`, and apply the calibrator from `calibrator.pkl`.

**Library pins.** Train/serve parity is a hard contract, so every library the image installs uses exactly the version below:

| library | version |
|---|---|
| python | 3.13.15 |
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| pyarrow | 23.0.1 |
| scipy | 1.16.3 |
| statsmodels | 0.15.0 |
| sklearn | 1.6.1 |
| lightgbm | 4.6.0 |
| xgboost | 3.4.1 |
| joblib | 1.6.0 |
| boto3 | 1.43.95 |

**Approved Phase-2 decisions:**
* **API input.** The caller sends enriched raw features; `src/features.py` is the reference implementation of the as-of counts.
* **Model delivery.** `CURRENT.json` alias.
* **Shape.** 1 replica, API only; canary deferred (documented deviation).
* **Node.** `t4g.small`, ARM64 (free-tier eligible in ap-south-2).
* **Names.** `fraud-ecomm-eks`, `fraud-ecomm-api`, `fraud-prod` / `fraud-uat`, `fraud-api-sa`, `fraud-api`.
* **Repo layout.** Root `FRAUD_ECOMMERCE`, application in `3.DEPLOYMENT/`, raw CSVs gitignored.
* **CI credentials.** Root access key in GitHub Actions secrets (signer's decision).
* **Fallback.** Opt-in incumbent rule, tagged as fallback.
* **Explanations.** Opt-in.
* **Prediction logging and S3 write access.** Off until Phase 3.
* **Base image.** `python:3.13-slim`.

## 8. Accepted deviations and known limitations

| item | detail |
|---|---|
| Gate is provisional | Cost figures are pending. The absolute floor and operating threshold are not yet derived from a cost matrix. Review trigger: cost figures received -> derive operating threshold and floor from the cost matrix. |
| Gate result: NOT PASSED | Recorded as rendered from metrics.json. The signer accepts this candidate for Phase 2 deployment work with the gate not passed, as a known deviation. The provisional gate stays open and must be re-evaluated when cost figures arrive. |
| Credentials | Local work, Colab and CI use the account root access key (signer's decision; IAM principals are not used in this project). Pods still use IRSA in-cluster. |
| No git SHA in the manifest | The notebooks ran in Colab before the repo existed. Record the first commit SHA that contains 2.RUNNING_CODE_FILES/ in this file's 'Code reference' line when pushing. |
| Immature test labels | 256 of 504 test-period chargebacks were still pending at extraction. All-row test PR-AUC is understated. |
| Small audit stratum | 41 audit positives in TEST, so the small-sample rule applied: yes. |
| Velocity features are lifetime as-of counts | They grow with accumulated history. A trailing-window count is a Phase-3 retrain candidate. Serving needs per-key first-seen state (Phase-2 design: the caller supplies the features and `src/features.py` is the reference implementation). |
| Snapshot attributes | `prior_orders_12m`, `prior_return_rate` and the merchant fields are not point-in-time. They are accepted per the data contract (post_decision = false). |
| `ip_region_code == 99` | Semantics unresolved (foreign IP or unknown). Values are left unchanged; notebook 02 prints a diagnostic crosstab. The column is excluded as a feature and affects only `ip_region_mismatch`. |
| Retry rows | 1,400 retries keep their artefactual label 0 (locked decision); `is_retry` is excluded as a feature. |
| Shared validation window | Early stopping, isotonic calibration and the operating point all use VAL. TEST is the independent read. |
| Flag `four_fifths_city` | raised in metrics.json; see the fairness and explanation sections |

## 9. Statement

Phase 1 is accepted on the evidence above.
* **Not promoted yet.** The candidate `v_20260916T181427Z_nb93c753a` is approved for Phase-2 deployment work, but it is not promoted until the Phase-2 runbook writes `models/CURRENT.json` and verifies the write by reading it back.
* **Gate stays provisional.** It remains open until cost figures arrive.
* **Every deviation above is known and accepted.**

— Bishal Ranjan Badu, 2026-09-17
