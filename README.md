# 🛡️ E-commerce Payment Fraud Scoring — Production MLOps

**A leakage-audited, calibrated fraud model for Indian e-commerce payments, served from EKS with an immutable
model registry, a compatibility manifest, hash-pinned imputation values, and three-point train/serve parity —
built on labels that are only partly observed.**

![Python](https://img.shields.io/badge/Python-3.13.15-blue?logo=python)
![Model](https://img.shields.io/badge/Model-LightGBM%204.6.0-brightgreen)
![Features](https://img.shields.io/badge/features-57%20tree-informational)
![PR-AUC](https://img.shields.io/badge/test%20PR--AUC-0.274%20vs%200.191%20incumbent-success)
![Lift](https://img.shields.io/badge/vs%20incumbent-1.43×-success)
![Tests](https://img.shields.io/badge/tests-95%20passing-success)
![Gate](https://img.shields.io/badge/gate-provisional%20·%20not%20passed-yellow)
![Serving](https://img.shields.io/badge/Serving-FastAPI-009688)
![Infra](https://img.shields.io/badge/EKS-ARM64%20t4g-orange?logo=amazonaws)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)

---

## What this repository actually is

Most fraud repos report a ROC-AUC on a label column and stop. Here the label column is the problem: **fraud is
never fully observed.** Only alerted payments get reviewed, audits sample the rest, and chargebacks arrive up to
five months later. Almost every design decision below follows from that, and the deployment carries the same
model through to a running endpoint that refuses to serve if the container cannot reproduce training.

| Phase | Scope | Status |
|---|---|---|
| **1 — Experimentation** | 7 notebooks, S3-chained stages, data contract, leakage audit, model card | **signed off** |
| **2 — Deployment** | `src/` package, 95 tests, ARM64 image, EKS, CI/CD, rollback drill | **signed off** |
| **3 — Operate & Monitor** | drift detection, label collection, automated retraining | not started |

Every phase is gated by a recorded sign-off listing what was reviewed, what was accepted, and which deviations
were taken knowingly — including a gate that **did not pass**.

---

## ✅ What was independently verified, not asserted

The claims below are backed by executable checks in `3.DEPLOYMENT/tests/`, not by prose in this file.

**A live leak was found and removed.** `device_n_customers` and `ip_n_customers` were computed as full-period
`groupby(...).nunique()`. A verification notebook proved the stored columns equalled the full-period count on
**100% of rows**, so a January payment could see a device that was only shared in November, and the statistic
spanned test rows. They are now **as-of** counts — distinct customers seen on that device or IP *at or before*
the payment. Notebook 05 re-verifies this as a sentinel, and notebook 06 **refuses to run** while it fails.

**Contemporaneous-leakage check.** Feature construction is proved independent of the label and of the incumbent
engine's score: identical inputs with `payment_risk_score` swung from 0 to 99 produce bit-identical features.

**The split is stamped once and never recomputed.** Notebook 01 writes `__split` at 2025-07-01; it travels
through all six S3 stage files, and every downstream `.fit()` keys off `__split == 'train'`.

**Deterministic feature exclusions.** 41 columns are excluded **by name** with a reason each, and **25
structural identities** are asserted on train rows to justify them — for example `mcc` ↔ `merchant_category` is
1:1, `os_family` is a function of `device_type`, and `is_hosting` is exactly `asn_type ∈ {Hosting, VPN}`. No
floating-point threshold picks the feature set, because the same notebook then selects different features on a
different library version.

**Selection never touched the test set.** Candidates are compared on 5 expanding-window CV folds; the test set
is scored once, at the end, for reporting and the gate.

**Serving reproduces the notebooks cell for cell.** `src/fraud_api/transform.py` regenerates notebook 06's saved
stage rows **exactly** for the tree encoding and to 1e-12 for the linear one. The fixture builder refuses to
write a fixture that fails this.

---

## Overview

Scores a payment for fraud risk at authorisation time, as a calibrated probability plus a decision, and is
compared against the incumbent rules engine already running in the business.

| | |
|---|---|
| Target | `is_fraud` — confirmed alert ∪ audit fraud ∪ Issuer-Won chargeback |
| Weights | alert review 1.00 · random audit 1.00 · chargeback 0.60 · unadjudicated 0.35 |
| Grain | one payment |
| Split | chronological at **2025-07-01** (no random shuffling of time-ordered rows) |
| Training window | 2024-01-01 … 2025-06-30 — 350,469 payments, 2,022 positives (0.577%) |
| Test window | 2025-07-01 … 2025-12-28 — 150,474 payments, 667 positives (0.443%), scored once |
| Estimated *true* prevalence | **0.974%** — nearly double the observed rate |

**Why the observed rate is not the fraud rate.** Unadjudicated payments are labelled 0 because nobody looked at
them, not because they were clean. The audit stratum gives the unbiased correction, and the gap between 0.577%
and 0.974% is the fraud the labels do not show.

## Business problem

A rules engine already alerts on ~1.1% of payments. Replacing it demands evidence that a model finds more fraud
per review, and the usual failure mode is a backtest that looks excellent because the evaluation used the very
labels the incumbent generated. The second failure mode — quieter — is a model that deploys fine and then
scores a missing field as zero because training and serving disagree about what "missing" means. Both are
addressed explicitly here.

## Dataset

Twelve raw CSVs (a synthetic but deliberately messy landing extract), joined down to payment grain.

| Property | Value |
|---|---|
| Payments after cleaning | **500,943** |
| Stage-04 modelling frame | 500,943 × 110 columns |
| Known defects planted in the raw files | **46**, including mixed datetime formats, sentinel strings, duplicate identities, test transactions |
| Label tables | `fraud_events` (alert decisions), `audit_sample` (12,000 random reviews), `chargebacks` (2,466 disputes) |
| Interchange format | Parquet at every stage boundary; CSV only for the raw landing objects |

### Labels are three biased windows onto the truth

| Source | Covers | Bias | Rows | Weight |
|---|---|---|---|---|
| `alert_review` | payments the incumbent alerted on | only what the incumbent caught | 5,352 | 1.00 |
| `random_audit` | random sample of **non-alerted** payments | unbiased *within that stratum* | 11,995 | 1.00 |
| `chargeback` | Issuer-Won disputes | late, and card payments only | 891 | 0.60 |
| unadjudicated | everything else | under-counts fraud | 482,705 | 0.35 |

**No audited payment was ever alerted** (verified: the overlap is exactly 0). That single fact drives the gate
design below.

### Label latency decides what cross-validation can pretend to know

| Source | Known after |
|---|---|
| Alert decision | ≤ 4 days |
| Audit verdict | ≤ 14 days |
| Chargeback raised | p95 72 days, max 75 |
| Chargeback **resolved** | p50 103 days, p95 **143**, max 162 |

At extraction, **256 of the test-period chargebacks were still pending** — test labels are immature by
construction, and the sign-off records that rather than quietly reporting a metric as if they were complete.

## Methodology

| # | Notebook | Does |
|---|---|---|
| 01 | Data Loading & First Look | Join 12 raw tables; emit the data contract; leakage register; protected-attribute decision; de-duplicate; stamp `__split` |
| 02 | Data Cleaning | Sentinels, canonical spellings, typed columns, dimension joins, **as-of** login and velocity features |
| 03 | Missing Values & Outliers | Group medians, global medians, `Unknown` levels, winsor caps — **fit on train rows only** |
| 04 | Statistics & EDA | Train-rows-only analysis; binary lift table; `is_retry` identified as a label artefact |
| 05 | Hypothesis Testing | Mann-Whitney / chi-square with BH correction, 6 pre-registered hypotheses, 3 leakage sentinels |
| 06 | Correlation, VIF & Encoding | Identity assertions, by-name exclusions, tree and linear encodings, leakage self-checks |
| 07 | Model Building & Evaluation | Logistic regression / LightGBM / XGBoost, maturity-masked CV, calibration, SHAP, fairness, single test scoring |

## Feature engineering

**57 features** after exclusions: 43 numeric and 14 categorical (pandas `category`, levels taken from the
encoder map — never from the payload).

| Type | Features |
|---|---|
| Payment | `payment_amount`, `processing_fee`, `discount_amount`, `item_count`, `attempt_seq_in_session` |
| Authentication | `is_3ds_attempted`, `is_3ds_success`, `is_guest_checkout` |
| Identity age | `account_age_days`, `card_token_age_h`, `device_age_h`, `hours_since_last_login` |
| **Velocity (as-of)** | `device_n_customers`, `ip_n_customers` — distinct customers on that device or IP up to this payment |
| Geo / network | `ip_country_mismatch`, `ip_country_missing`, `ip_region_mismatch`, `reputation_score`, `asn_type`, `asn_country` |
| Merchant | `merchant_category`, `avg_ticket_size`, `trailing_chargeback_rate_bps`, `amount_vs_merchant_ticket` |
| Customer | `kyc_level`, `city_tier`, `prior_return_rate`, `prior_orders_12m`, `email_domain_class`, `acquisition_channel` |
| Card / device | `network`, `issuer`, `product_type`, `issuer_foreign`, `device_type`, `browser_family`, `is_emulator` |
| Basket & login risk | `n_categories`, `max_unit_price`, `total_qty`, `last_login_*` |
| Structural indicators | `has_card`, `has_device_profile`, `has_basket`, `has_prior_login` |
| Calendar | `hour_of_day`, `day_of_week` |

**Feature-order hash: `64bfdc79b8649a15`** — recorded in the manifest and validated at container startup.

### Excluded on purpose

- **Incumbent outputs.** `payment_risk_score` and `alerted_flag` are the baseline to beat, never inputs.
- **Label-join artefacts.** `is_retry` shows a lift of 0.0 — not because retries are safe, but because retries
  got fresh payment IDs and never joined to any label source. A tree would happily learn "retry ⇒ never fraud".
- **Collinear by construction**, asserted then dropped: `primary_category`, `mcc`, `shipping_charge`,
  `os_family`, `bill_ship_mismatch`, `device_orphan`, `is_disposable_email`, `is_hosting`, `basket_value`.
- **Identifiers and geography**: pincodes, IP address, session, card token, device id — also the DPDP proxy
  surface.

## Models & results

Three families were tuned under maturity-masked cross-validation: logistic regression (linear encoding),
LightGBM and XGBoost (tree encoding). **LightGBM was promoted.**

### Selection ran on cross-validation, not on the holdout

| Candidate | CV PR-AUC (all fold rows) | CV design-weighted | Trees | |
|---|---|---|---|---|
| **LightGBM (tuned) — promoted** | **0.4329 ± 0.0404** | **0.3426** | 276 | ✅ |
| XGBoost (tuned) | 0.4316 ± 0.0411 | 0.3295 | 250 | |
| Logistic regression | 0.4028 ± 0.0445 | 0.3054 | — | |

**LightGBM leads XGBoost by 0.3%, well inside the ±0.04 fold spread.** That is a coin flip, not a finding, so
the pre-registered rule applies: the default family is only replaced by a challenger that wins by **≥ 5%**
relative on CV. Without that hysteresis, a monthly retrain would swap the pickle class, the SHAP output and the
serving path on noise.

### Validation quarter (2025-04-01 … 2025-06-30)

| Model | PR-AUC | Design-weighted | Audit-only | ROC-AUC | Fit |
|---|---|---|---|---|---|
| **LightGBM** | **0.4266** | **0.2987** | 0.1408 | **0.9430** | 19.1 s |
| XGBoost | 0.4134 | 0.2842 | 0.1343 | 0.9410 | 34.7 s |
| Logistic regression | 0.3799 | 0.2671 | 0.1300 | 0.9366 | 2.6 s |
| incumbent `payment_risk_score` | 0.3334 | 0.2154 | — | 0.8875 | — |

### Test set — scored once, and the gate

The reviewed test sample is 1,624 alerted rows (531 positives, weight 1.002) plus 3,681 audit rows (41
positives, weight **40.44**). Design-weighted prevalence comes out at **1.456%**, against an observed label rate
of 0.443% — the fraud the raw labels do not show.

| | Model | Incumbent | Ratio |
|---|---|---|---|
| **Design-weighted PR-AUC** | **0.2741** | 0.1911 | **1.434×** |
| 90% CI | 0.2305 – 0.3374 | — | 1.245 – 1.644 |
| Audit-only (supporting) | 0.1212 | 0.0546 | 2.219× |

| Gate half | Rule | Point estimate | Lower bound used | Verdict |
|---|---|---|---|---|
| Absolute | ≥ 0.2350 (incumbent, train period) | 0.2741 ✅ | **0.2305** | ❌ by 0.0045 |
| Relative | ≥ 1.25× incumbent | 1.434 ✅ | **1.2445** | ❌ by 0.0055 |

**Gate: NOT PASSED — and the reason is the sample, not the model.** Both halves clear their thresholds on the
point estimate and miss on the 5th-percentile bootstrap bound — by 0.0045 (1.9% of the floor) and 0.0055
(0.4% of the required ratio). The small-sample
rule fires because the audit stratum holds **41 positives**; that rule was written into the gate *before* the
test set was scored, precisely so this situation could not be argued away afterwards.

> **The timing matters.** Any change now — point estimates instead of bounds, a 1.20 ratio, a wider interval —
> would be a rule rewritten after seeing the result it would flip. The gate stands as written, the failure is
> recorded in the sign-off, and the threshold is re-derived from cost figures at the recorded review trigger.
> What the numbers do support is that the model finds materially more fraud per review than the incumbent
> (+43% by point estimate, with a CI that stays above 1.0 throughout).

### Why the gate is design-weighted, in this project's own numbers

Audit rows never include alerted payments, so an audit-only view measures the model only where the incumbent
declined to act. On this very test set it reports a **2.22×** advantage where the population-weighted estimate
reports **1.43×** — a 55% inflation of the apparent win. That gap is the incumbent's own catch rate, excluded
from the comparison.

### Cross-validation mimics label maturity

In each fold, a training row whose chargeback resolved **after** that fold's retrain moment reverts to
unadjudicated (label 0, weight 0.35). A flat embargo equal to the 143-day resolution p95 was rejected: it
starves the first fold. The final refit uses mature labels, because masking exists to keep the *estimate*
honest, not to weaken the artifact.

## Key insights

- **Missingness is signal.** `ip_country` is absent far more often behind hosting and VPN infrastructure, so the
  indicator stays a feature instead of being imputed away.
- **The incumbent's threshold is exactly 14.96.** Recovered from the data, and reused as the tagged fallback
  rule when the model is unavailable.
- **Geographic proxies were declared before modelling.** No protected attributes exist in this dataset; city and
  city-tier are recorded as proxies under the India DPDP Act 2023, with fairness measured by group.
- **The population ages across the split.** Mean account age rises from 835 to 1,151 days between train and
  test, and test maxima exceed training maxima. Not leakage, but trees extrapolate flat past the training
  range, so it is a monitoring item for Phase 3.

---

## Serving: single payment, enriched by the caller

Unlike a panel model, nothing here is computed across payments **at request time** — so the API is single-row.
The two velocity features are as-of counts over the caller's own history, so the caller supplies them, and
`src/fraud_api/features.py` is the reference implementation they must match. Golden payloads pin the numbers.

| Route | Purpose |
|---|---|
| `GET /live` | process liveness, no I/O — safe for `livenessProbe` |
| `GET /health` | readiness: artifacts loaded **and** the full compatibility contract holds |
| `POST /v1/score` | calibrated probability, decision, `defaults_applied`; `?explain=true`, `?allow_fallback=true` |

- **Unknown fields are rejected** with 422 — which is what keeps label and incumbent columns out of the model.
- **Structural groups** (`card`, `device`, `basket`, `last_login`) are all-or-nothing, mirroring the gaps the
  model actually learned.
- **A categorical null is accepted only where the model learned a meaning for it.** That rule is derived from
  the artifacts, so it follows whichever model is promoted rather than a hard-coded list.
- Every response carries `model_version`, `contract_version`, `gate_status` and `source: model | fallback`. The
  fallback is **opt-in and always tagged** — an untagged fallback is indistinguishable from a real prediction,
  which is worse than an error.

## Model registry, and the third fitted stage

Artifacts are immutable: training writes `models/v_<utc>_nb<hash>/` and never overwrites. `models/CURRENT.json`
is the alias, so **promotion and rollback are both a single JSON write**.

`manifest.json` is the compatibility contract: contract version, feature-order hash, encoder maps, **feature
dtypes**, and the exact library versions the estimator was pickled under. The container validates it at startup
and **fails readiness on mismatch**, so a bad promotion leaves the previous pod serving.

> **Dtypes are in the manifest for a reason.** Notebook 07 fits the 14 categoricals as pandas `category`;
> LightGBM records `pandas_categorical` in the booster and validates it at predict time. An `align()` that
> returns `int64` gives a matching feature hash, a green `/health`, and **HTTP 500** on every request. Feature
> order is necessary and not sufficient.

### Why the imputation values are pinned separately

Notebook 07 persisted the encoding parameters but not notebook 03's imputation values, which live at a mutable
stage path that every retrain overwrites. That matters because LightGBM scores a blank as **0** in any column
that had no blanks during training:

| `account_age_days` sent to the model | Score |
|---|---|
| blank | 0.4116 |
| 0 | 0.4116 |
| 893.5 — the train median notebook 03 fills | 0.0066 |

A missing account age would be scored as a brand-new account, **62× riskier**, with HTTP 200 and nothing in the
logs. So promotion copies `data/03_fitted_params.json` to a **content-addressed** key
(`models/params/03_fitted_params_<sha12>.json`), records its SHA-256 in the alias, and the pod verifies that
hash at startup. Rollback stays exact because each version keeps its own pinned pointer — a later retrain
overwriting the stage file cannot change how an older version scores.

## Train/serve parity

One transform implementation, proved equal to the notebooks' saved stage files (tree: exact; linear: 1e-12),
then proved again at **three points** — local `uvicorn`, `docker run`, and through the LoadBalancer — on
identical payloads including nulls and every structural gap, via `scripts/golden.py`. A score that drifts by
0.01 fails the check.

## Tests

**95 tests, all passing with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` unset.** Data comes from a
committed fixture built from the real bucket, never from S3 at test time — which removes the entire class of
"CI failed on NoSuchKey".

| Category | What it pins |
|---|---|
| Transform parity | `src/` reproduces notebook 06's saved stage rows, cell for cell |
| Dtype parity | category input predicts; int64 input still raises — **both** directions |
| Imputation | train-fitted values applied, and the LightGBM blank-is-zero behaviour pinned as the reason |
| Contemporaneous leakage | features are identical when labels and the incumbent score are corrupted |
| As-of semantics | counts are inclusive; strictly later rows never change earlier values |
| Registry | **16 refusal cases**, one per broken link of the compatibility contract |
| Promotion | first pin, exact rollback, write-once keys, gate deviation must be explicit |
| Serving contract | unknown field, missing required, out-of-range, half-filled group, unknown level → 422 |
| Fallback | tagged, opt-in, equals the incumbent rule at 14.96 |
| Pins | `requirements.txt` equals the trained library versions; the Dockerfile base equals the training Python |

> The fixture models are fitted **through the production code path**, so they carry the same dtypes and encoder
> maps as the promoted model — including LightGBM's `pandas_categorical`, which is asserted. A fixture built
> differently certifies the wrong thing.

## Architecture

```
                    ┌──────────────────────────────────────────┐
   git push ───────►│  GitHub Actions — mlops_pipeline.yml     │
                    │  test (no creds) → build arm64 → ECR      │
                    │  → pin check → render+verify → IRSA       │
                    │  → EKS → post-deploy golden check         │
                    └────────────────┬─────────────────────────┘
                                     ▼
   ┌─────────────────────────────────────────────────────────┐
   │  EKS  fraud-ecomm-eks   ·   t4g.small (ARM64)  ·  k8s 1.35│
   │  ┌───────────────────────────────────────────────────┐  │
   │  │ Pod × 1 — API only, one process                   │  │
   │  │   uvicorn :8000                                   │  │
   │  │   1 validate → 2 derive (nb02) → 3 impute (nb03)  │  │
   │  │   → 4 encode (nb06) → 5 predict + calibrate       │  │
   │  │   startupProbe/readiness /health · liveness /live │  │
   │  └───────────────────────────────────────────────────┘  │
   │  Service (LoadBalancer) → :80 → api                     │
   └────────────────────────────┬────────────────────────────┘
                                │ IRSA, read-only on models/*
                                ▼
        s3://fraud-ecommerce
          raw/  contracts/schema_v1.json  data/01…06  reports/
          models/CURRENT.json → models/v_<utc>_nb<hash>/
          models/params/03_fitted_params_<sha12>.json
```

**One Service, therefore one load balancer, therefore one bill.** Ports that nothing serves are not exposed —
an unserved port leaves a permanently unhealthy target.

**CI does not bootstrap infrastructure.** The cluster, the OIDC provider and the IRSA ServiceAccounts are
one-time human steps that the pipeline *verifies* and never creates.

**Deploys are readiness-gated**: surge 1, unavailable 0. A new pod whose model fails the contract never becomes
Ready, the old pod keeps serving, and CI rolls back after collecting pod logs and events.

## Tech stack

`Python 3.13.15` · `LightGBM 4.6.0` · `scikit-learn 1.6.1` · `pandas 2.2.3` · `numpy 2.1.3` · `scipy 1.16.3` ·
`SHAP (native TreeSHAP)` · `FastAPI` · `pydantic v2` · `pytest` · `Docker (ARM64, multi-stage, non-root)` ·
`AWS S3 · ECR · EKS · IAM/IRSA` · `GitHub Actions`

Serving pins match the artifact's training environment exactly, and every one was verified published on PyPI
with a **cp313 aarch64** wheel, so nothing compiles from source on the ARM node. `xgboost-cpu` replaces
`xgboost` in the image: same library, same import name, without the 252 MB NVIDIA dependency the Linux wheel
would otherwise pull in.

## Repository structure

```
├── 1.RAW/                          generator, data dictionary, 46-defect list (CSVs gitignored)
├── 2.RUNNING_CODE_FILES/           Phase 1 — permanent documentation, committed
│   ├── 01_Data_Loading_and_First_Look.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Missing_Values_and_Outliers.ipynb
│   ├── 04_Statistics_and_EDA.ipynb
│   ├── 05_Hypothesis_Testing.ipynb
│   ├── 06_Correlation_VIF_Encoding.ipynb
│   └── 07_Model_Building_and_Evaluation.ipynb
├── 3.DEPLOYMENT/
│   ├── src/fraud_api/
│   │   ├── constants.py            contract values copied verbatim from the notebooks
│   │   ├── transform.py            THE transform — 02 derive · 03 impute · 06 encode
│   │   ├── features.py             as-of velocity reference + request mapping
│   │   ├── schema.py               pydantic contract, structural groups, sentinel rules
│   │   ├── registry.py             alias resolution + compatibility validation
│   │   ├── predict.py              scoring, calibration, explanations, tagged fallback
│   │   ├── api.py                  FastAPI
│   │   ├── train.py                model constructors, shared with the fixture models
│   │   └── s3_io.py                default credential chain only
│   ├── scripts/                    promote.py · make_fixture.py · golden.py · check_pins.py
│   ├── tests/                      95 tests · fixtures/ (COMMITTED) · golden/
│   ├── k8s/  infra/                manifests · eksctl cluster config · ECR lifecycle
│   ├── governance/sign_off/        phase records, rendered from the artifacts
│   ├── Dockerfile · entrypoint.sh · requirements.txt · requirements.lock
│   └── OPERATOR_RUNBOOK.md         24 numbered steps, each with a verification command
└── .github/
    ├── workflows/mlops_pipeline.yml
    └── scripts/render_and_verify.sh
```

## How to run

**Tests — no cloud credentials required:**

```bash
git clone https://github.com/BishalRanjanBadu/fraud_ecomm_prediction.git
cd fraud_ecomm_prediction/3.DEPLOYMENT
python -m pip install --user uv==0.12.15
python -m uv venv --python 3.13 .venv                 # fetches 3.13.15, the training runtime
python -m uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt   # bin/python on Linux/macOS
source .venv/Scripts/activate

env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY python -m pytest -q tests
```

Install from `requirements.txt`, not `requirements.lock` — the lock is a hashed Linux/arm64 image artifact and
legitimately contains packages with no Windows wheels.

**Serve locally** (requires AWS credentials with read access to the model bucket):

```bash
aws configure                        # keys land in ~/.aws/, never in the repo
export MODEL_BUCKET=fraud-ecommerce AWS_REGION=ap-south-2 MODEL_ALIAS_KEY=models/CURRENT.json ROLE=api
PYTHONPATH=src sh entrypoint.sh      # http://127.0.0.1:8000/docs
```

`.env.example` carries configuration only — never credentials. Environment variables take precedence over
`~/.aws/credentials`, so a stale key in `.env` silently overrides a freshly configured one.

**Promote or roll back:**

```bash
python scripts/promote.py --version v_20260916T181427Z_nb93c753a     # rollback: same command, older version
```

Pins the imputation values, validates the whole contract with this environment's libraries, writes the alias,
and reads it back. A version whose gate did not pass requires an explicit `--accept-gate-deviation "reason"`,
which is stored in the alias.

**Check parity anywhere:**

```bash
python scripts/golden.py check --base-url http://127.0.0.1:8000     # local · docker · load balancer
```

**Deploy:** `3.DEPLOYMENT/OPERATOR_RUNBOOK.md` — 24 numbered steps, each with a verification command, plus a
rollback drill and a verified teardown.

## Known limitations

- **The gate is provisional and did not pass**, by 0.0045 on the absolute half and 0.0055 on the relative half,
  both on bootstrap lower bounds rather than point estimates. Treat the score as a shadow signal beside the
  incumbent until the threshold is re-derived from a cost matrix.
- **Test labels are immature** — 256 test-period chargebacks were still pending at extraction, so any full-test
  metric understates recall.
- **The audit stratum is small**: 41 positives in test, carrying a weight of 40.4 each. That is what makes the
  confidence interval wide enough to decide the gate, and more audit volume is the cheapest way to sharpen it.
- **Velocity counts are lifetime as-of counts** and drift upward as history accumulates. A trailing-window count
  is stationary but changes the feature hash, so it is a Phase-3 retrain candidate, not an edit.
- **Snapshot attributes** (`prior_orders_12m`, `prior_return_rate`, merchant fields) are not point-in-time. They
  are accepted per the data contract and recorded.
- **Serving depends on upstream state.** The caller computes the velocity counts; a caller that computes them
  differently creates skew that only the golden payloads would catch.
- **Canary deferred.** One small node cannot host a canary, so this runs a readiness-gated rolling update with
  automatic rollback. On real infrastructure the canary is not optional.
- **Prediction logging is off** and the serving IRSA policy is read-only. Phase 3 enables both **in the same
  change** — one without the other yields `AccessDenied` per request.
- **Synthetic data**, generated with 46 known defects planted on purpose.

## Costs

| Resource | Cost |
|---|---|
| EKS control plane | ~$0.10/hour (**~$73/month — never free tier**) |
| `t4g.small` node | free-tier eligible in this account |
| Classic ELB | ~$18–20/month |
| S3 + ECR | negligible at this volume |

Teardown is Step 24 of the runbook. **Delete the `type: LoadBalancer` Service before the cluster** or the ELB is
orphaned and keeps billing.

## Author

**Bishal Ranjan Badu**
Data Science · Machine Learning · MLOps
