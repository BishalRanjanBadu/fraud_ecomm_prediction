# fraud_phase2_v1 — package manifest

**Marker:** `MARKER: fraud_phase2_v1` (in `src/fraud_api/constants.py` and every script/manifest header)
**Unpacks into:** the project root `FRAUD_ECOMMERCE/` (merge copy; nothing existing is deleted or overwritten
except files of the same name — the Phase-1 sign-off files are not in this package).

**Expected file count after extraction: 49** (the 48 files below + this manifest).

Deliberately NOT included (generated from your real bucket by the runbook, then committed):
`3.DEPLOYMENT/tests/fixtures/*.parquet|*.json` (step 5) and `3.DEPLOYMENT/tests/golden/*.json` (step 10).

## Files
```
.gitattributes
.github/scripts/render_and_verify.sh
.github/workflows/mlops_pipeline.yml
.gitignore
3.DEPLOYMENT/.dockerignore
3.DEPLOYMENT/.env.example
3.DEPLOYMENT/Dockerfile
3.DEPLOYMENT/OPERATOR_RUNBOOK.md
3.DEPLOYMENT/entrypoint.sh
3.DEPLOYMENT/governance/sign_off/PHASE2_SIGNOFF_TEMPLATE.md
3.DEPLOYMENT/infra/ecr-lifecycle.json
3.DEPLOYMENT/infra/eksctl-cluster.yaml
3.DEPLOYMENT/k8s/config.yml
3.DEPLOYMENT/k8s/deployment.yml
3.DEPLOYMENT/k8s/pdb.yml
3.DEPLOYMENT/k8s/service.yml
3.DEPLOYMENT/requirements-dev.txt
3.DEPLOYMENT/requirements.lock
3.DEPLOYMENT/requirements.txt
3.DEPLOYMENT/scripts/check_pins.py
3.DEPLOYMENT/scripts/golden.py
3.DEPLOYMENT/scripts/make_fixture.py
3.DEPLOYMENT/scripts/promote.py
3.DEPLOYMENT/src/fraud_api/__init__.py
3.DEPLOYMENT/src/fraud_api/api.py
3.DEPLOYMENT/src/fraud_api/config.py
3.DEPLOYMENT/src/fraud_api/constants.py
3.DEPLOYMENT/src/fraud_api/features.py
3.DEPLOYMENT/src/fraud_api/logs.py
3.DEPLOYMENT/src/fraud_api/predict.py
3.DEPLOYMENT/src/fraud_api/registry.py
3.DEPLOYMENT/src/fraud_api/s3_io.py
3.DEPLOYMENT/src/fraud_api/schema.py
3.DEPLOYMENT/src/fraud_api/train.py
3.DEPLOYMENT/src/fraud_api/transform.py
3.DEPLOYMENT/tests/conftest.py
3.DEPLOYMENT/tests/fakes.py
3.DEPLOYMENT/tests/fixtures/README.md
3.DEPLOYMENT/tests/golden/README.md
3.DEPLOYMENT/tests/test_api.py
3.DEPLOYMENT/tests/test_contract.py
3.DEPLOYMENT/tests/test_dtypes.py
3.DEPLOYMENT/tests/test_features_leakage_pins.py
3.DEPLOYMENT/tests/test_imputation.py
3.DEPLOYMENT/tests/test_promote.py
3.DEPLOYMENT/tests/test_registry.py
3.DEPLOYMENT/tests/test_transform_parity.py
README.md
```

## Start here
`3.DEPLOYMENT/OPERATOR_RUNBOOK.md` — every command in order, with a verification after each step.

## Verified before packaging
- **Tests.** 94 passed and 1 skipped with no cloud credentials, on Python 3.13 with the exact pins; the skipped
  test is the golden-files check, which passes once step 10 records the files. With golden files present and
  `CI=true`: 95 passed, and missing golden files fail CI with an actionable message.
- **Transform parity.** The fixture builder refuses to write unless the service transform reproduces
  notebook 06's stage rows (tree: exact, linear: within 1e-12). This was exercised on a full-schema run of
  notebooks 01–07.
- **End-to-end rehearsal over real HTTP.** The real `promote.py` pinned the notebook-03 params and wrote the
  alias; the real app served it; `golden.py` recorded 21 cases and 10 contract probes, and re-checked them
  with zero difference. A score tampered by 0.01 failed the check.
- **Real entrypoint.** It boots uvicorn with `/live` 200 and `/health` 503 with a reason when no model is
  available; an unknown `ROLE` exits 78.
- **Dependencies.** The lock file downloads with hashes checked on linux/arm64 and linux/x86_64
  (31 wheels, about 100 MB, no NVIDIA packages). Models pickled with full `xgboost` 3.4.1 predict
  bit-identically under `xgboost-cpu` 3.4.1.
- **Lint.**
  - `actionlint` 1.7.12 (with ShellCheck integration) passes on the workflow.
  - `shellcheck` 0.11.0 passes on both shell scripts.
  - `kubeconform` 0.8.0 (strict, Kubernetes 1.35.0 schemas) validates the rendered manifests.
- **Render script.** It succeeds on valid input, and refuses `:latest`, an empty tag, a missing tag,
  an empty image, an unexpected namespace, and a stray `${VAR}`.
- **Git hygiene.** `.gitignore` excludes the raw CSVs (except `KNOWN_DEFECTS.csv`), `.venv`, `.env`, logs
  and `3.DEPLOYMENT/_*`. `.gitattributes` forces LF line endings on the shell scripts and the Dockerfile.
  The step-18 secret scan is clean on the package and catches planted keys.
- **Upstream references**, checked in one pass:
  - PyPI pins, with wheels for linux arm64, linux x86 and Windows;
  - all 9 GitHub Action tags and their input names;
  - the base image `python:3.13.15-slim-trixie`;
  - Kubernetes 1.35 (the highest `eksctl` 0.230.0 supports) and `kubectl` v1.35.8;
  - `ubuntu-24.04-arm` runners.

## Not verified here (verified by the runbook on your machine / in CI)
- The Docker image build itself: this sandbox has no container engine. Runbook step 11 builds it locally
  (amd64), and the CI build job builds it natively for arm64.
- EKS, ECR, IRSA and the load balancer (runbook steps 12–22).
