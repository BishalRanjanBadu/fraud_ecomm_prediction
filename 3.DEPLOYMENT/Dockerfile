# syntax=docker/dockerfile:1
# MARKER: fraud_phase2_v1 — scoring API image (linux/arm64 in CI; builds on amd64 for local parity checks)
# Base image = the exact Python the model was trained with (manifest library_versions.python = 3.13.15).
ARG PYTHON_IMAGE=python:3.13.15-slim-trixie

FROM ${PYTHON_IMAGE} AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY requirements.lock /build/requirements.lock
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --require-hashes --only-binary=:all: -r /build/requirements.lock

FROM ${PYTHON_IMAGE} AS runtime
# `apt-get upgrade` applies Debian security fixes at build time. Trade-off: the OS layer is not
# byte-reproducible, so every deploy records the image DIGEST (CI job output), not just the tag.
# libgomp1 is the only added package: the OpenMP runtime that the lightgbm and xgboost wheels load.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
COPY src/ /app/src/
COPY --chmod=0755 entrypoint.sh /app/entrypoint.sh
# HOME=/tmp: the only writable path in the pod (emptyDir), cross-checked with k8s/deployment.yml
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    ROLE=api \
    PORT=8000
WORKDIR /app
# Verification runs in the stage that ships: native libs load, the application imports, versions print.
RUN python -c "import lightgbm, xgboost, sklearn, pandas, fastapi; import fraud_api.api; \
from fraud_api import constants as C; from fraud_api.registry import runtime_versions; \
print('contract', C.CONTRACT_VERSION, '| training marker', C.TRAINING_MARKER, '| runtime', runtime_versions())"
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
