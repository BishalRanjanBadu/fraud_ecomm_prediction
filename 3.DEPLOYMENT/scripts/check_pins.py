"""Fail if requirements.txt disagrees with the library versions the promoted model was trained with.
MARKER: fraud_phase2_v1

    python scripts/check_pins.py                      # reads models/CURRENT.json -> manifest (needs AWS creds)
    python scripts/check_pins.py --version v_...      # before the first promotion
    python scripts/check_pins.py --fixture-only       # offline: compares with tests/fixtures/FIXTURE_INFO.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUCKET, REGION = "fraud-ecommerce", "ap-south-2"
# requirement name -> manifest key (only libraries that shape the model or its pickles)
CONTRACT = {"pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn",
            "lightgbm": "lightgbm", "xgboost-cpu": "xgboost", "joblib": "joblib", "boto3": "boto3"}


def read_pins(path: Path) -> dict:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def compare(pins: dict, trained: dict) -> list[str]:
    problems = []
    for req, key in CONTRACT.items():
        if req not in pins:
            problems.append(f"{req}: not pinned with == in requirements.txt")
        elif trained.get(key) != pins[req]:
            problems.append(f"{req}: requirements.txt {pins[req]} != trained {trained.get(key)}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version")
    ap.add_argument("--fixture-only", action="store_true")
    a = ap.parse_args()
    pins = read_pins(ROOT / "requirements.txt")
    if a.fixture_only:
        info = json.loads((ROOT / "tests" / "fixtures" / "FIXTURE_INFO.json").read_text(encoding="utf-8"))
        trained, source = info["library_versions"], f"FIXTURE_INFO ({info['version']})"
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from fraud_api import s3_io
        s3 = s3_io.client(REGION)
        version = a.version or json.loads(s3_io.get_bytes(s3, BUCKET, "models/CURRENT.json"))["version"]
        trained = json.loads(s3_io.get_bytes(s3, BUCKET, f"models/{version}/manifest.json"))["library_versions"]
        source = f"s3 manifest ({version})"
    problems = compare(pins, trained)
    print(f"pins vs {source}: {'OK' if not problems else 'MISMATCH'}")
    for p in problems:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
