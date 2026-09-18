# FRAUD_ECOMMERCE — fraud scoring, notebook to EKS

End-to-end, phase-gated MLOps project: a LightGBM fraud model for e-commerce payments, trained in Phase 1
(notebooks), served in Phase 2 (FastAPI on EKS, CI/CD), monitored and retrained in Phase 3.

| Phase | Status | Record |
|---|---|---|
| 1 · Notebooks (`2.RUNNING_CODE_FILES/`) | signed off 2026-09-17 | `3.DEPLOYMENT/governance/sign_off/PHASE1_SIGNOFF.md` |
| 2 · Deployment (`3.DEPLOYMENT/`) | in progress | `3.DEPLOYMENT/governance/sign_off/PHASE2_SIGNOFF_TEMPLATE.md` |
| 3 · Operate & monitor | not started | — |

## Architecture

```
 caller (enriched payment)                    GitHub push ─► Actions
        │  POST /v1/score                         test (arm64, no creds)
        ▼                                         build (native arm64) ─► ECR (scan on push)
 ┌──────────── EKS · fraud-prod ────────────┐     deploy (render ► verify ► apply ► auto-rollback)
 │ Service (LoadBalancer :80)               │     post-deploy golden check through the LB
 │   └─ Deployment fraud-api (1 × arm64)    │
 │        1 validate contract   (422)       │
 │        2 derive     (notebook 02 code)   │
 │        3 impute     (notebook 03 params) │◄── startup: IRSA read-only on s3://fraud-ecommerce/models/*
 │        4 encode     (notebook 06 params) │      models/CURRENT.json ─► models/v_…/ (immutable)
 │        5 predict + isotonic calibration  │                        └─► models/params/03_fitted_params_<sha12>.json
 └──────────────────────────────────────────┘
```

**Why step 3 matters.** LightGBM scores a blank as `0` in any column that had no blanks during training. A
missing `account_age_days` would therefore be scored as a brand-new account. The pod applies notebook 03's
train-fitted values, pinned by hash in `CURRENT.json` and verified at startup.

## Automation model

- **Trigger 1: code.** `git push` to `main` runs test → build → ECR → EKS (`fraud-prod`) → golden check.
  `develop` targets `fraud-uat`.
- **Trigger 2: data (Phase 3).** Drift, when labels are available, triggers a retrain, then evaluation,
  then `scripts/promote.py`, then a pod restart.
- **CI never creates infrastructure.** The EKS cluster, OIDC provider and IRSA service accounts are one-time
  runbook steps (`3.DEPLOYMENT/infra/eksctl-cluster.yaml`). CI only verifies them and fails with an
  actionable message if they are missing.

## Serving contract (v1)

`POST /v1/score` takes one payment. The caller supplies the enriched features, including the as-of velocity
counts defined in `src/fraud_api/features.py`.

- **Rejected fields.** Unknown fields, which covers label and incumbent columns, get a 422.
- **Structural groups.** `card`, `device`, `basket` and `last_login` are all-or-nothing objects.
- **Optional scalars.** A null is filled with the train-fitted value and listed in `defaults_applied`.
- **Categorical values.**
  - A value must be a trained level.
  - A null is accepted only where the model learned a meaning for it. The rule is derived from the artifacts:
    fields whose fill value is not a trained level (for example `shipping_speed`) are required.
- **Opt-in extras.** `?explain=true` adds exact per-feature contributions. `?allow_fallback=true` applies the
  incumbent rule when the model is down; those responses are tagged `source: fallback_incumbent_rule`.
- **Other endpoints.** `GET /live` is the liveness check (process only). `GET /health` is the readiness check:
  the model is loaded and the compatibility contract holds.

## Repository layout

```
1.RAW/                  generator scripts, data dictionary, defect list (CSVs are gitignored)
2.RUNNING_CODE_FILES/   notebooks 01-07 (package fraud_nb01-07_v2)
3.DEPLOYMENT/
  src/fraud_api/        service code (transform, schema, registry, predict, api)
  scripts/              promote.py, make_fixture.py, golden.py, check_pins.py
  tests/                pytest suite + committed fixtures and golden payloads
  k8s/  infra/          manifests (rendered by CI) and the eksctl cluster config
  Dockerfile  entrypoint.sh  requirements.txt  requirements.lock
  OPERATOR_RUNBOOK.md   every command, in order, with a verification after each
  governance/sign_off/  phase sign-off records
.github/                workflow + render-and-verify script
```

## Known deviations (recorded in the sign-offs)

- **Provisional gate, not passed.** The model's gate did not pass and is provisional, so the score is a
  shadow signal until the gate is re-derived from cost figures.
- **No canary.** Canary is deferred on a single `t4g.small` node. Deploys use a readiness-gated rolling
  update with automatic rollback instead.
- **Root credentials.** The account root key is used locally and in CI (owner's decision). Pods use IRSA,
  read-only on `models/*`.
- **Publicly visible account ID.** The AWS account ID appears in CI defaults and manifests. It is an
  identifier, not a secret, but it becomes public along with the repository.
