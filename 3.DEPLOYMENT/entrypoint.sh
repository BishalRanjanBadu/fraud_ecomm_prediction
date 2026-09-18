#!/bin/sh
# MARKER: fraud_phase2_v1 — one process per container; `exec` so SIGTERM reaches uvicorn directly.
set -eu
ROLE="${ROLE:-}"
case "$ROLE" in
  api)
    exec uvicorn fraud_api.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 \
      --no-access-log --timeout-graceful-shutdown 20
    ;;
  *)
    echo "entrypoint: ROLE must be 'api' (got '${ROLE}'). This image only runs the scoring API." >&2
    exit 78
    ;;
esac
