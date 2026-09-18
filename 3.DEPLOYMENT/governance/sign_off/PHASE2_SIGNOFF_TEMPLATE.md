# Phase 2 sign-off — FRAUD_ECOMMERCE (deployment)

> Fill every field from the evidence the runbook prints; do not estimate. Commit this file with the evidence.

| | |
|---|---|
| **Decision** | Phase 2 accepted / not accepted |
| **Signed off by** | |
| **Date** | |
| **Code reference** | git SHA of the deployed commit |
| **Image** | `117211782845.dkr.ecr.ap-south-2.amazonaws.com/fraud-ecomm-api:` + SHA |
| **Image digest** | from the CI build job summary |
| **Model promoted** | `models/CURRENT.json` → version, manifest SHA-256, params key |
| **Cluster** | `fraud-ecomm-eks`, Kubernetes 1.35, 1 × t4g.small (arm64) |

## 1. Test evidence
- Local test suite (no credentials): passed / skipped counts (runbook step 6)
- CI test job (arm64): run URL
- Pin check against the promoted manifest: output line (runbook step 7 and CI build job)

## 2. Three-point parity (golden payloads, tolerance 1e-6)
| Point | Command | Result | Worst abs. difference |
|---|---|---|---|
| Local uvicorn | `golden.py check --base-url http://127.0.0.1:8000` | | |
| docker run (amd64) | `golden.py check --base-url http://127.0.0.1:8080` | | |
| EKS load balancer (arm64) | CI post-deploy job and runbook step 21 | | |

## 3. Rollback drill (runbook step 22)
- Broken alias rejected by readiness (new pod not Ready, old pod kept serving): yes / no
- Restore by re-promotion: elapsed seconds from restore command to golden check passing
- `kubectl rollout undo` exercised: yes / no

## 4. Security
- ECR scan counts by severity (CI build summary)
- Findings in packages this Dockerfile installs (only `libgomp1` plus the Python venv): action taken
- Findings in the base image with no fix available: listed and accepted, or not

## 5. Deviations carried forward
- Gate NOT PASSED (provisional); served score is a shadow signal until cost figures arrive
- Canary deferred on a single t4g.small node; RollingUpdate with surge 1 and readiness-gated rollback instead
- Root access key used locally and in CI (signer's decision); pods use IRSA read-only on `models/*`
- UAT namespace provisioned but dormant (no Deployment, no LoadBalancer)
- Prediction logging off until Phase 3 (enabled together with its IAM permission)

## 6. Not in Phase 2
Drift detection, label collection, retraining pipeline, alarms/SLOs, prediction logging, HPA, canary.
