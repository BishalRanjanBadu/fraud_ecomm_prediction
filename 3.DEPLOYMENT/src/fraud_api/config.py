"""Runtime settings (non-secret). Defaults are the project's real identifiers."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: str) -> bool:
    v = os.environ.get(name, default).strip().lower()
    if v not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false', got {v!r}")
    return v == "true"


@dataclass(frozen=True)
class Settings:
    bucket: str
    region: str
    alias_key: str
    log_level: str
    log_predictions: bool

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(bucket=os.environ.get("MODEL_BUCKET", "fraud-ecommerce"),
                region=os.environ.get("AWS_REGION", "ap-south-2"),
                alias_key=os.environ.get("MODEL_ALIAS_KEY", "models/CURRENT.json"),
                log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                log_predictions=_bool("LOG_PREDICTIONS", "false"))
        if s.log_predictions:
            # Phase 3 turns this on together with the IAM permission to write predictions/.
            raise ValueError("LOG_PREDICTIONS=true is a Phase-3 feature (needs s3:PutObject on predictions/); "
                             "keep it false in Phase 2")
        return s
