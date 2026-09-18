#!/usr/bin/env bash
# MARKER: fraud_phase2_v1
# Render k8s templates to files, verify them, and only then let the caller apply them.
#   NAMESPACE=fraud-prod MODEL_ALIAS_KEY=models/CURRENT.json \
#   IMAGE=117211782845.dkr.ecr.ap-south-2.amazonaws.com/fraud-ecomm-api:"$GITHUB_SHA" \
#     bash .github/scripts/render_and_verify.sh 3.DEPLOYMENT/k8s _render
# Set SKIP_SERVER_DRY_RUN=1 only where no cluster is reachable (local syntax checks).
set -euo pipefail

SRC="${1:?usage: render_and_verify.sh TEMPLATE_DIR OUTPUT_DIR}"
OUT="${2:?usage: render_and_verify.sh TEMPLATE_DIR OUTPUT_DIR}"

for tool in envsubst grep; do
  command -v "$tool" > /dev/null || { echo "render: required tool not found: $tool" >&2; exit 1; }
done
for v in NAMESPACE IMAGE MODEL_ALIAS_KEY; do
  [ -n "${!v:-}" ] || { echo "render: variable $v is unset or empty" >&2; exit 1; }
done
case "$IMAGE" in
  *:latest) echo "render: refusing ':latest' - deploy the immutable git-SHA tag" >&2; exit 1 ;;
  *:)       echo "render: image tag is empty in '$IMAGE'" >&2; exit 1 ;;
  *:*)      ;;
  *)        echo "render: image has no tag: '$IMAGE'" >&2; exit 1 ;;
esac
case "$NAMESPACE" in
  fraud-prod|fraud-uat) ;;
  *) echo "render: unexpected namespace '$NAMESPACE'" >&2; exit 1 ;;
esac

rm -rf "$OUT"
mkdir -p "$OUT"
shopt -s nullglob
templates=("$SRC"/*.yml)
[ "${#templates[@]}" -gt 0 ] || { echo "render: no templates in $SRC" >&2; exit 1; }
for f in "${templates[@]}"; do
  # only these three variables are substituted; any other $-text stays literal and is caught below
  # shellcheck disable=SC2016,SC2094  # literal ${VAR} list is envsubst's syntax; input and output are different files
  envsubst '${NAMESPACE} ${IMAGE} ${MODEL_ALIAS_KEY}' < "$f" > "$OUT/$(basename "$f")"
done

# unresolved placeholders on NON-comment lines (a ${...} inside a comment is not an error)
if grep -nE '^[[:space:]]*[^#[:space:]].*\$\{[A-Za-z_][A-Za-z0-9_]*\}' "$OUT"/*.yml; then
  echo "render: unresolved placeholders above" >&2
  exit 1
fi
if ! grep -qE "^[[:space:]]+image: ${IMAGE//./\\.}$" "$OUT/deployment.yml"; then
  echo "render: deployment image line does not equal '$IMAGE'" >&2
  exit 1
fi
echo "render: image -> $IMAGE"
echo "render: namespace -> $NAMESPACE, alias -> s3://fraud-ecommerce/$MODEL_ALIAS_KEY"

if [ "${SKIP_SERVER_DRY_RUN:-0}" = "1" ]; then
  echo "render: server-side dry run SKIPPED (SKIP_SERVER_DRY_RUN=1)"
else
  command -v kubectl > /dev/null || { echo "render: kubectl not found" >&2; exit 1; }
  kubectl apply --dry-run=server -f "$OUT" > "$OUT/.dry-run.log"
  echo "render: server-side dry run passed ($(grep -c . "$OUT/.dry-run.log") objects)"
fi
echo "render: OK -> $OUT"
