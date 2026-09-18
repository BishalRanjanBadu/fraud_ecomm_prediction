# Operator runbook — FRAUD_ECOMMERCE Phase 2 (deployment)

**Package:** `fraud_phase2_v1` · **Shell:** Git Bash on Windows 11 · **Account:** 117211782845 · **Region:** ap-south-2

Rules for this runbook:
- **Paths contain spaces.** Every path is quoted. Keep the quotes.
- **Verify before moving on.** Each step ends with a **Verify** block. Do not continue until it matches.
- **On any failure, diagnose first.** Read the real error (see §F) before re-running anything.
- **Use `127.0.0.1`, not `localhost`.** On Windows `localhost` can resolve to IPv6.
- **Credentials.** They live only in `~/.aws/` (outside the repo) and in GitHub Actions secrets. Never in files,
  never on screen. Using the account root key is the owner's recorded decision.
- **Costs.** The first paid resource is created in step 16, and the cost is stated there first.

---

## 0. Session variables (run at the start of every new terminal)

```bash
export PROJECT="/c/Users/HP/Desktop/Bishal_/END TO END PROJECTS/ML PROJECTS/FRAUD_ECOMMERCE"
export DEPLOY="$PROJECT/3.DEPLOYMENT"
export AWS_REGION=ap-south-2
export AWS_ACCOUNT_ID=117211782845
export BUCKET=fraud-ecommerce
export VERSION=v_20260916T181427Z_nb93c753a
export ECR_REPO=fraud-ecomm-api
export CLUSTER=fraud-ecomm-eks
export GATE_REASON="Phase-1 sign-off 2026-09-17: provisional design-weighted gate not passed; accepted as a known deviation"
```

**Verify**
```bash
ls -d "$PROJECT" "$DEPLOY" && echo "paths OK"
```

## 1. Locate and extract the package (nothing is copied or deleted yet)

```bash
ls -la ~/Downloads/fraud_phase2_v1*.zip
```
If the browser renamed the file (for example `fraud_phase2_v1 (1).zip`), use that exact name in the next command.

```bash
mkdir -p "$PROJECT/_incoming"
python -m zipfile -e ~/Downloads/fraud_phase2_v1.zip "$PROJECT/_incoming"
export STAGE="$PROJECT/_incoming/fraud_phase2_v1"
```

**Verify:** the marker is present, and the tree matches `MANIFEST.md`.
```bash
grep -c "MARKER: fraud_phase2_v1" "$STAGE/3.DEPLOYMENT/src/fraud_api/constants.py"
cat "$STAGE/MANIFEST.md" | head -40
( cd "$STAGE" && find . -type f | sort | wc -l )
```
Expected: `1`, and the file count printed in `MANIFEST.md`.

## 2. Copy into the project (merge; existing files such as the Phase-1 sign-off are untouched)

```bash
cp -r "$STAGE"/. "$PROJECT"/ && echo "copy OK"
```
The trailing `/.` is what copies `.github`, `.gitignore` and `.gitattributes`.

**Verify**
```bash
ls -la "$PROJECT" | grep -E "\.github|\.gitignore|\.gitattributes|README.md"
ls "$DEPLOY" | tr '\n' ' '; echo
ls "$DEPLOY/governance/sign_off"
cmp "$STAGE/3.DEPLOYMENT/src/fraud_api/api.py" "$DEPLOY/src/fraud_api/api.py" && echo "api.py identical"
```
Expected: `.github`, `.gitignore`, `.gitattributes`, `README.md` are listed. `3.DEPLOYMENT` holds `src scripts tests
k8s infra Dockerfile …`. The sign-off folder still contains `PHASE1_SIGNOFF.md` and `render_phase1_signoff.py`.

Only after the verification above:
```bash
rm -rf "$PROJECT/_incoming" && ls "$PROJECT"
```

## 3. Python 3.13 environment (training parity: the model was trained on Python 3.13)

```bash
cd "$DEPLOY"
python -m pip install --user uv==0.12.15
python -m uv --version
python -m uv venv --python 3.13 .venv
source .venv/Scripts/activate
python -m uv pip install -r requirements-dev.txt
```

**Verify**
```bash
python --version
python -c "import pandas, numpy, sklearn, lightgbm, xgboost, joblib, fastapi; print(pandas.__version__, numpy.__version__, sklearn.__version__, lightgbm.__version__, xgboost.__version__, joblib.__version__, fastapi.__version__)"
```
Expected: `Python 3.13.x` and `2.2.3 2.1.3 1.6.1 4.6.0 3.4.1 1.6.0 0.141.1`.

> Every later step assumes this venv is active. In a new terminal, run §0, then `cd "$DEPLOY" && source .venv/Scripts/activate`.

## 4. Credentials check (nothing in the environment shadows `~/.aws`)

```bash
env | grep -E '^AWS_(ACCESS|SECRET|SESSION)' | sed 's/=.*/=(set)/'
aws configure list
aws sts get-caller-identity --query Arn --output text
```

**Verify:** the first command prints nothing. `aws configure list` shows the key type `shared-credentials-file`. The ARN ends with `:root`.

## 5. Build the committed test fixture from the real bucket (read-only, downloads about 300 MB)

```bash
cd "$DEPLOY"
python scripts/make_fixture.py --version "$VERSION"
```

**Verify**
```bash
ls -la tests/fixtures
python -c "import json; i = json.load(open('tests/fixtures/FIXTURE_INFO.json')); print(i['version'], i['family'], i['rows'], i['positives'], i['excluded'], i['library_versions']['python'])"
```
Expected:
* The script prints `parity: service transform reproduces stage-06 tree (exact) and linear (1e-12) rows`.
* The summary line shows `v_20260916T181427Z_nb93c753a lightgbm`, then the row count, the positives count, the excluded counts, and `3.13.15`.

## 6. Tests with no cloud credentials (what CI will run)

```bash
cd "$DEPLOY"
env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY AWS_SHARED_CREDENTIALS_FILE="$DEPLOY/_no_such_credentials" \
  AWS_CONFIG_FILE="$DEPLOY/_no_such_config" python -m pytest -q -p no:cacheprovider tests
```

**Verify:** the summary reads `94 passed, 1 skipped`. The skipped test is the golden-files check, which becomes a pass after step 10.

## 7. Pins equal the trained library versions

```bash
python scripts/check_pins.py --version "$VERSION"
```

**Verify:** the output reads `pins vs s3 manifest (v_20260916T181427Z_nb93c753a): OK`.

## 8. Promote the model (plan A: pinned notebook-03 values)

Preview first; nothing is written:
```bash
python scripts/promote.py --version "$VERSION" --accept-gate-deviation "$GATE_REASON" --dry-run
```
Then write it:
```bash
python scripts/promote.py --version "$VERSION" --accept-gate-deviation "$GATE_REASON"
```

**Verify**
```bash
aws s3 cp "s3://$BUCKET/models/CURRENT.json" - | python -m json.tool
aws s3 ls "s3://$BUCKET/models/params/" --recursive
```
Expected:
* The promote output ends with `alias models/CURRENT.json -> v_20260916T181427Z_nb93c753a (previous: none); read-back OK`.
* The alias shows `gate_passed: false` with your reason, `family: lightgbm`, and an `imputation_params_key` under `models/params/`.
* The listing shows `03_fitted_params_…json` and `by_version/v_20260916T181427Z_nb93c753a.json`.

## 9. Serve locally with the real model (parity point 1)

```bash
cd "$DEPLOY"
export MODEL_BUCKET=fraud-ecommerce MODEL_ALIAS_KEY=models/CURRENT.json LOG_PREDICTIONS=false ROLE=api PORT=8000
PYTHONPATH=src sh entrypoint.sh > _local_api.log 2>&1 &
sleep 20
netstat -ano | grep -E "[:.]8000 .*LISTENING"
tail -5 _local_api.log
```

**Verify**
```bash
curl -sS http://127.0.0.1:8000/live
curl -sS http://127.0.0.1:8000/health | python -m json.tool
```
Expected: `/live` returns `alive`. `/health` shows `status: ready`, the model version, `family: lightgbm`,
`gate_status: not_passed_provisional`, and `checks_passed` of 15 or more.

If `/health` is 503, the `reason` field names the failed check. Read it and use §F. Do not restart blindly.

## 10. Record golden payloads, then check them (parity point 1)

```bash
python scripts/golden.py record --base-url http://127.0.0.1:8000
python scripts/golden.py check --base-url http://127.0.0.1:8000
```

**Verify:**
* `recorded … cases + … probes against v_20260916T181427Z_nb93c753a`, then `GOLDEN CHECK PASSED`.
* Re-run the step-6 test command: the result is now `95 passed`.

Stop the local server, then confirm the port is free:
```bash
WINPID=$(netstat -ano | grep -E "[:.]8000 .*LISTENING" | awk '{print $5}' | head -1)
echo "server PID: $WINPID"
taskkill //F //PID "$WINPID"
sleep 2; netstat -ano | grep -E "[:.]8000 .*LISTENING" || echo "port 8000 free"
```

## 11. Container build and container parity (parity point 2)

Docker Desktop must be running. This local build is amd64. CI builds the arm64 image natively.
```bash
docker pull python:3.13.15-slim-trixie
docker build -t fraud-ecomm-api:local "$DEPLOY" 2>&1 | tee "$DEPLOY/_docker_build.log" | tail -25
```

**Verify:** the build log contains the in-image check line
`contract v1 | training marker fraud_nb01-07_v2 | runtime {... 'lightgbm': '4.6.0' ...}`.
```bash
grep -m1 "training marker" "$DEPLOY/_docker_build.log"
docker image ls fraud-ecomm-api:local
```

Run it the way the pod runs: non-root, read-only filesystem, writable `/tmp` only, and your `~/.aws` mounted read-only.
```bash
MSYS_NO_PATHCONV=1 docker run -d --name fraud-api-local -p 127.0.0.1:8080:8000 \
  --read-only --tmpfs /tmp:rw,size=64m \
  -e MODEL_BUCKET=fraud-ecommerce -e AWS_REGION=ap-south-2 -e MODEL_ALIAS_KEY=models/CURRENT.json \
  -e LOG_PREDICTIONS=false \
  -v "$(cygpath -w "$HOME/.aws")":/tmp/.aws:ro \
  fraud-ecomm-api:local
sleep 25
docker logs fraud-api-local --tail 5
```

**Verify**
```bash
curl -sS http://127.0.0.1:8080/health | python -m json.tool
python scripts/golden.py check --base-url http://127.0.0.1:8080
```
Expected: `ready`, then `GOLDEN CHECK PASSED`.

Remove the container, then confirm it is gone:
```bash
docker rm -f fraud-api-local && docker ps -a --filter name=fraud-api-local --format '{{.Names}}' | grep -c . || echo "container removed"
```

## 12. ECR repository (scan on push, lifecycle 15 images)

```bash
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" \
  --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE
aws ecr put-lifecycle-policy --repository-name "$ECR_REPO" --region "$AWS_REGION" \
  --lifecycle-policy-text "$(cat "$DEPLOY/infra/ecr-lifecycle.json")"
```

**Verify**
```bash
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" \
  --query 'repositories[0].[repositoryUri,imageScanningConfiguration.scanOnPush,imageTagMutability]' --output table
```
Expected: `117211782845.dkr.ecr.ap-south-2.amazonaws.com/fraud-ecomm-api | True | MUTABLE`.
Tags stay mutable because CI also pushes `latest`; deploys always use the immutable git-SHA tag.

## 13. kubectl v1.35.8 (your current v1.28 is too old for a 1.35 cluster)

```bash
mkdir -p "$HOME/bin"
curl -sS -fL -o "$HOME/bin/kubectl.exe" "https://dl.k8s.io/release/v1.35.8/bin/windows/amd64/kubectl.exe"
curl -sS -fL -o "$HOME/bin/kubectl.exe.sha256" "https://dl.k8s.io/release/v1.35.8/bin/windows/amd64/kubectl.exe.sha256"
echo "$(cat "$HOME/bin/kubectl.exe.sha256")  $HOME/bin/kubectl.exe" | sha256sum --check
grep -q 'HOME/bin' ~/.bashrc || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/bin:$PATH"; hash -r
```

**Verify**
```bash
command -v kubectl
kubectl version --client
eksctl version
```
Expected: `sha256sum` printed `OK`. `command -v kubectl` is `/c/Users/HP/bin/kubectl`, the client is `v1.35.8`, and eksctl is `0.230.0`.

## 14. Pre-checks for the cluster (read-only)

```bash
aws eks describe-cluster-versions --region "$AWS_REGION" \
  --query 'clusterVersions[?clusterVersion==`1.35`].[clusterVersion,versionStatus,endOfStandardSupportDate]' --output table
aws ec2 describe-instance-type-offerings --region "$AWS_REGION" --location-type availability-zone \
  --filters Name=instance-type,Values=t4g.small --query 'InstanceTypeOfferings[].Location' --output text
aws ec2 describe-instance-types --region "$AWS_REGION" --instance-types t4g.small \
  --query 'InstanceTypes[0].[FreeTierEligible,ProcessorInfo.SupportedArchitectures[0],MemoryInfo.SizeInMiB]' --output text
```

**Verify:**
* Version 1.35 shows standard support.
* `t4g.small` is offered in at least one availability zone.
* The last command prints `True arm64 2048`.

If `describe-cluster-versions` is an unknown command, your AWS CLI is too old; update it before step 16.

## 15. Headroom arithmetic (why one small node is enough)

| | Memory |
|---|---|
| t4g.small allocatable | about 1.4–1.5 GiB (confirmed live in step 17) |
| System pods (coredns ×2, aws-node, kube-proxy, …) | about 0.3–0.5 GiB |
| API pod request | 384 MiB, **×2 during a rolling update** (surge 1) = 768 MiB |

Pods-per-node limit for t4g.small is 11. System pods plus 2 API pods during a surge stays under it.

## 16. Create the cluster — **this starts billing**

> **Cost while the cluster exists (ap-south-2, approximate):**
>
> | Resource | Cost |
> |---|---|
> | EKS control plane | **USD 0.10/hour ≈ USD 73/month**, never free tier |
> | LoadBalancer (created by the first deploy) | ≈ USD 18–20/month |
> | 20 GB gp3 volume | ≈ USD 2/month |
> | t4g.small node | free-tier eligible in this account |
>
> Tear down with §24 when you are not using it.

```bash
eksctl create cluster -f "$DEPLOY/infra/eksctl-cluster.yaml" 2>&1 | tee "$DEPLOY/_eksctl_create.log"
echo "eksctl exit: ${PIPESTATUS[0]}"
```
This takes about 15–25 minutes. If eksctl stops waiting or fails, CloudFormation may still be working. Check the real state before doing anything else:
```bash
aws cloudformation describe-stacks --region "$AWS_REGION" --query 'Stacks[?starts_with(StackName,`eksctl-fraud-ecomm-eks`)].[StackName,StackStatus]' --output table
aws cloudformation describe-stack-events --region "$AWS_REGION" --stack-name eksctl-fraud-ecomm-eks-cluster \
  --query 'StackEvents[?contains(ResourceStatus,`FAILED`)].[LogicalResourceId,ResourceStatusReason]' --output table
```

**Verify**
```bash
aws eks describe-cluster --name "$CLUSTER" --region "$AWS_REGION" --query 'cluster.[status,version,identity.oidc.issuer]' --output table
kubectl get nodes -L kubernetes.io/arch,node.kubernetes.io/instance-type
for ns in fraud-prod fraud-uat; do kubectl -n "$ns" get serviceaccount fraud-api-sa -o jsonpath="{.metadata.annotations.eks\.amazonaws\.com/role-arn}{'\n'}"; done
```
Expected:
* The cluster shows `ACTIVE | 1.35 | https://oidc.eks.ap-south-2.amazonaws.com/id/…`.
* One `Ready` node with `arm64` and `t4g.small`.
* Two role ARNs, one per namespace.

## 17. Node headroom (read the numbers; do not assume them)

```bash
kubectl get nodes -o jsonpath='{.items[0].status.allocatable}{"\n"}'
kubectl describe nodes | grep -A 8 "Allocated resources"
kubectl get pods -A -o wide
```

**Verify:** allocatable memory minus the current memory requests is at least 800 Mi (two API pods during a rollout), and fewer than 9 pods are running.

## 18. Git repository (repo root = FRAUD_ECOMMERCE)

```bash
cd "$PROJECT"
git init -b main
git config user.name || git config user.name "Bishal Ranjan Badu"
git config user.email || git config user.email "BishalRanjanBadu@users.noreply.github.com"
git add -A
git status --short | head -60
```

**Verify (all four must hold before committing):**
```bash
git check-ignore -v "1.RAW/payments.csv" "3.DEPLOYMENT/.venv" "3.DEPLOYMENT/_local_api.log"
git ls-files --cached | grep -E '\.csv$' || echo "no csv staged"
git ls-files --cached | grep -c "3.DEPLOYMENT/tests/fixtures/" ; git ls-files --cached | grep -c "3.DEPLOYMENT/tests/golden/"
git grep --cached -nE 'AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|(aws_secret_access_key|AWS_SECRET_ACCESS_KEY)[[:space:]"]*[=:][[:space:]"]*[A-Za-z0-9/+]{40}|-----BEGIN [A-Z ]*PRIVATE KEY-----'; echo "secret-scan exit: $? (1 = no matches = clean)"
```
Expected:
* All three paths are reported as ignored.
* Only `1.RAW/KNOWN_DEFECTS.csv` is staged among CSVs.
* The fixture count is 7 and the golden count is 3.
* The secret scan exits with `1`, meaning no matches.

```bash
git commit -m "Phase 1 notebooks (v2) + Phase 2 deployment package (fraud_phase2_v1)"
export SHA=$(git rev-parse HEAD); echo "$SHA"
sed -i "s|git SHA: \*to be recorded at first push\*|git SHA: $SHA (first commit containing 2.RUNNING_CODE_FILES/)|" \
  "$DEPLOY/governance/sign_off/PHASE1_SIGNOFF.md"
grep -n "Code reference" "$DEPLOY/governance/sign_off/PHASE1_SIGNOFF.md"
git add "$DEPLOY/governance/sign_off/PHASE1_SIGNOFF.md" && git commit -m "Record Phase-1 code reference"
```

**Verify:** the `grep` shows the SHA, and `git log --oneline | head -3` shows two commits.

## 19. GitHub Actions secrets (the only hand-configured values)

In the browser: **github.com/BishalRanjanBadu/fraud_ecomm_prediction → Settings → Secrets and variables → Actions → New repository secret**
- `AWS_ACCESS_KEY_ID`: the same key as in `~/.aws/credentials`
- `AWS_SECRET_ACCESS_KEY`: its secret

No repository variables are needed; every non-secret identifier has a default in the workflow.

**Verify:** the Secrets page lists exactly these two names.

## 20. Push, which triggers CI (test, then build, then deploy, then golden check)

```bash
cd "$PROJECT"
git remote add origin https://github.com/BishalRanjanBadu/fraud_ecomm_prediction.git
git push -u origin main
```
Watch **Actions → mlops-pipeline**. Expected order:
1. `test (arm64, no cloud credentials)`
2. `build + push (native arm64)`: its summary shows the image reference, the **digest** and the ECR scan counts.
3. `deploy`: the render lines, then `deployed …`.
4. `post-deploy golden check`: ends with `GOLDEN CHECK PASSED`.

**Verify:** all four jobs are green. Copy the image digest and the scan counts into the Phase-2 sign-off.

## 21. Live verification (parity point 3, from your machine)

```bash
kubectl -n fraud-prod get deploy,pods,svc -o wide
export LB=$(kubectl -n fraud-prod get service fraud-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'); echo "$LB"
curl -sS "http://$LB/health" | python -m json.tool
cd "$DEPLOY" && python scripts/golden.py check --base-url "http://$LB" --wait 300
kubectl -n fraud-prod get pods -l app=fraud-api -o jsonpath='{.items[0].status.containerStatuses[0].imageID}{"\n"}'
kubectl -n fraud-prod logs deploy/fraud-api --tail=10
```

**Verify:**
* The pod is `1/1 Running`.
* `/health` is `ready` with the promoted version.
* The golden check shows `PASSED`.
* The image ID ends with the digest from the CI summary.
* The logs are JSON lines, including `model_loaded`.

## 22. Rollback drill (required before sign-off)

**A. A broken promotion must not take the service down.**
```bash
cd "$DEPLOY"
aws s3 cp "s3://$BUCKET/models/CURRENT.json" "_drill_alias_good.json"
python -c "import json; a = json.load(open('_drill_alias_good.json')); a['manifest_sha256'] = '0' * 64; json.dump(a, open('_drill_alias_broken.json', 'w'), indent=1)"
aws s3 cp "_drill_alias_broken.json" "s3://$BUCKET/models/CURRENT.json"
kubectl -n fraud-prod rollout restart deployment/fraud-api
kubectl -n fraud-prod rollout status deployment/fraud-api --timeout=90s; echo "rollout status exit: $? (non-zero is EXPECTED here)"
kubectl -n fraud-prod get pods -l app=fraud-api
export NEWPOD=$(kubectl -n fraud-prod get pods -l app=fraud-api --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
kubectl -n fraud-prod logs "$NEWPOD" | grep -m1 model_load_failed
curl -sS "http://$LB/health" | python -m json.tool
```
**Verify:**
* One pod is `1/1` (the old one) and one is `0/1` (the new one).
* The new pod's log contains `manifest sha256: differs from the value pinned in the alias`.
* The load balancer still answers `ready`.

**B. Restore by re-promotion, and time it.**
```bash
T0=$(date +%s)
python scripts/promote.py --version "$VERSION" --accept-gate-deviation "$GATE_REASON"
kubectl -n fraud-prod rollout restart deployment/fraud-api
kubectl -n fraud-prod rollout status deployment/fraud-api --timeout=420s
python scripts/golden.py check --base-url "http://$LB" --wait 300
echo "restore took $(( $(date +%s) - T0 )) seconds"
```

**C. Exercise `rollout undo`.**
```bash
kubectl -n fraud-prod rollout history deployment/fraud-api
kubectl -n fraud-prod rollout undo deployment/fraud-api
kubectl -n fraud-prod rollout status deployment/fraud-api --timeout=420s
python scripts/golden.py check --base-url "http://$LB" --wait 300
```
The previous pod template also reads the now-good alias, so it becomes Ready.

**Verify:** B and C both end with `GOLDEN CHECK PASSED`. Record the B timing in the sign-off.

## 23. Phase-2 sign-off record

```bash
cp "$DEPLOY/governance/sign_off/PHASE2_SIGNOFF_TEMPLATE.md" "$DEPLOY/governance/sign_off/PHASE2_SIGNOFF.md"
```
Fill it from the evidence in steps 6, 10, 11, 20, 21 and 22, then commit it:
```bash
cd "$PROJECT" && git add "3.DEPLOYMENT/governance/sign_off/PHASE2_SIGNOFF.md" && git commit -m "Phase 2 sign-off" && git push
```
This push re-runs CI, which is a second full pass through the pipeline.

## 24. Teardown (stops the hourly cost; S3 and ECR are kept)

Delete the LoadBalancer **before** the cluster, otherwise the ELB is orphaned and keeps billing:
```bash
kubectl -n fraud-prod delete service fraud-api --wait=true
sleep 60
aws elb describe-load-balancers --region "$AWS_REGION" --query 'LoadBalancerDescriptions[].DNSName' --output text
aws elbv2 describe-load-balancers --region "$AWS_REGION" --query 'LoadBalancers[].DNSName' --output text
```
Continue only when neither command lists this service's load balancer.
```bash
eksctl delete cluster -f "$DEPLOY/infra/eksctl-cluster.yaml" --wait
```

**Verify**
```bash
aws eks list-clusters --region "$AWS_REGION"
aws cloudformation list-stacks --region "$AWS_REGION" \
  --query 'StackSummaries[?starts_with(StackName,`eksctl-fraud-ecomm-eks`) && StackStatus!=`DELETE_COMPLETE`].[StackName,StackStatus]' --output table
```
Expected: `"clusters": []` and an empty table.

After teardown, a push to `main` fails at CI pre-flight with `cluster fraud-ecomm-eks not found`. That is the intended signal. To rebuild, re-run steps 16, 17, 20 and 21. `models/CURRENT.json` is still in place.

---

## §F. Symptom → first command

| Symptom | First command | Usual cause |
|---|---|---|
| `/health` 503 | `curl -sS http://…/health` (read `reason`) | the named contract check, e.g. library versions or params sha |
| Pod `Pending` | `kubectl -n fraud-prod describe pod -l app=fraud-api \| tail -20` | `Insufficient memory` or `Too many pods` (see §15) |
| `ImagePullBackOff` | `kubectl -n fraud-prod describe pod -l app=fraud-api \| grep -A5 Events` | tag not in ECR; check `aws ecr describe-images --repository-name fraud-ecomm-api` |
| `exec format error` | `kubectl get nodes -L kubernetes.io/arch` | image architecture ≠ node architecture (CI builds arm64; do not push the local amd64 image) |
| `CrashLoopBackOff`, exit 78 | `kubectl -n fraud-prod logs deploy/fraud-api --previous` | `ROLE` is not `api` |
| `AccessDenied` in pod logs | `kubectl -n fraud-prod describe serviceaccount fraud-api-sa` | missing IRSA annotation, or the object is outside `models/*` |
| `NoCredentialsError` in pod logs | `kubectl -n fraud-prod get pod -l app=fraud-api -o yaml \| grep -A2 AWS_ROLE_ARN` | pod created before the service account existed; restart the deployment |
| 422 `required for this model` | read `field` in the response | a categorical with no trained meaning for a gap (e.g. `shipping_speed`) |
| CI: `cluster … not found` | this runbook, §16 | infrastructure is created only by hand |
| CI: `Unauthorized` from kubectl | `aws sts get-caller-identity` locally and in the CI log | CI key is not the cluster creator (both must be the same root key) |
| CI: pin mismatch | `python scripts/check_pins.py` | `requirements.txt` differs from the promoted manifest |
| LB hostname empty for >10 min | `kubectl -n fraud-prod describe service fraud-api \| tail -15` | load balancer provisioning events |
| `ERR_CONNECTION_REFUSED` locally | `netstat -ano \| grep 8000` | server not running, or `localhost` resolved to IPv6 |
| `InvalidClientTokenId` | `env \| grep ^AWS_; aws configure list` | a stale key in the environment shadows `~/.aws` |

## Not in Phase 2

- Drift detection, label collection, and the retraining pipeline.
- Alarms, SLOs, and the CloudWatch dashboard.
- Prediction logging and the S3 write permission it needs.
- HPA and canary rollouts, which need more than one small node.
- A UAT deployment: the `fraud-uat` namespace and service account exist, but nothing is deployed there until a `develop` branch is pushed after `python scripts/promote.py --version "$VERSION" --alias models/CURRENT_UAT.json --accept-gate-deviation "$GATE_REASON"`.
- The Phase-3 fix that makes notebook 07 store the 03 values inside the model folder.
