"""Load the promoted model and refuse readiness on any contract mismatch.

models/CURRENT.json (written by scripts/promote.py) pins:
  - the immutable version prefix and the SHA-256 of its manifest.json
  - the content-addressed notebook-03 params file and its SHA-256
Everything is read into memory; nothing is written to disk (readOnlyRootFilesystem).
"""
from __future__ import annotations

import importlib
import io
import json
import platform
from dataclasses import dataclass, field

import joblib

from . import constants as C
from . import s3_io
from .transform import feature_contract_hash

ARTIFACTS = ["manifest.json", "model.pkl", "calibrator.pkl", "preprocessor.json", "feature_names.json"]


class ModelLoadError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    version: str
    family: str
    input_family: str
    model: object
    calibrator: object
    p06: dict
    p03: dict
    manifest: dict
    alias: dict
    threshold_raw: float
    gate_status: str
    checks: list = field(default_factory=list)


def runtime_versions() -> dict:
    v = {"python": platform.python_version()}
    for mod in C.STRICT_LIBS:
        v[mod] = importlib.import_module(mod).__version__
    return v


def _need(cond: bool, msg: str, checks: list) -> None:
    if not cond:
        raise ModelLoadError(msg)
    checks.append(msg.split(":")[0])


def validate(manifest, p06, p03, feature_names, model, calibrator, alias, blobs, checks) -> None:
    fam, inp = manifest["family"], manifest["input_family"]
    _need(manifest.get("marker") == C.TRAINING_MARKER,
          f"marker: manifest {manifest.get('marker')} != {C.TRAINING_MARKER}", checks)
    _need(manifest.get("contract_version") == C.CONTRACT_VERSION,
          f"contract: manifest {manifest.get('contract_version')} != service {C.CONTRACT_VERSION}", checks)
    _need(s3_io.sha256(blobs["manifest.json"]) == alias["manifest_sha256"],
          "manifest sha256: differs from the value pinned in the alias", checks)
    _need(s3_io.sha256(blobs["params03"]) == alias["imputation_params_sha256"],
          "03 params sha256: differs from the value pinned in the alias", checks)
    rt, want = runtime_versions(), manifest["library_versions"]
    bad = {m: (want.get(k), rt[m]) for m, k in C.STRICT_LIBS.items() if want.get(k) != rt[m]}
    _need(not bad, f"library versions: trained vs runtime mismatch {bad}", checks)
    _need(want.get("python", "").split(".")[:2] == rt["python"].split(".")[:2],
          f"python minor: trained {want.get('python')} vs runtime {rt['python']}", checks)
    _need(feature_contract_hash(p06) == manifest["feature_hash"],
          "feature hash: preprocessor.json does not reproduce the manifest hash", checks)
    _need(feature_names == manifest["feature_order"] == p06[inp]["feature_order"],
          "feature order: feature_names.json / manifest / preprocessor disagree", checks)
    if inp == "tree":
        _need(manifest["feature_dtypes"] == p06["tree"]["dtypes"],
              "feature dtypes: manifest and preprocessor disagree", checks)
        _need(manifest["encoder_maps"] == p06["tree"]["categorical"],
              "encoder maps: manifest and preprocessor disagree", checks)
    module = type(model).__module__.split(".")[0]
    _need(module == C.FAMILY_MODULE.get(fam), f"estimator: family {fam} but pickle is from {module}", checks)
    names = list(getattr(model, "feature_names_in_", getattr(model, "feature_name_", [])))
    _need(names == manifest["feature_order"], "estimator feature names: differ from the manifest order", checks)
    if fam == "lightgbm":
        cats = [p06["tree"]["categorical"][c] for c in p06["tree"]["feature_order"] if c in p06["tree"]["categorical"]]
        _need(model.booster_.pandas_categorical == cats,
              "lightgbm pandas_categorical: booster categories differ from the encoder maps", checks)
    _need(type(calibrator).__name__ == "IsotonicRegression", "calibrator: not an IsotonicRegression", checks)
    for section in ("group_medians", "global_medians", "cat_fill"):
        _need(section in p03, f"03 params: section {section} missing", checks)


def load(s3, bucket: str, alias_key: str) -> LoadedModel:
    try:
        alias = json.loads(s3_io.get_bytes(s3, bucket, alias_key))
    except Exception as e:
        raise ModelLoadError(f"alias {alias_key}: cannot read ({type(e).__name__}: {e})") from e
    for k in ("version", "model_prefix", "manifest_sha256", "imputation_params_key", "imputation_params_sha256"):
        if k not in alias:
            raise ModelLoadError(f"alias {alias_key}: missing field {k}")
    prefix = alias["model_prefix"]
    if prefix != f"models/{alias['version']}/":
        raise ModelLoadError(f"alias: prefix {prefix} does not match version {alias['version']}")
    try:
        blobs = {name: s3_io.get_bytes(s3, bucket, prefix + name) for name in ARTIFACTS}
        blobs["params03"] = s3_io.get_bytes(s3, bucket, alias["imputation_params_key"])
    except Exception as e:
        raise ModelLoadError(f"artifacts: cannot read ({type(e).__name__}: {e})") from e
    manifest = json.loads(blobs["manifest.json"])
    p06 = json.loads(blobs["preprocessor.json"])
    p03 = json.loads(blobs["params03"])
    feature_names = json.loads(blobs["feature_names.json"])
    model = joblib.load(io.BytesIO(blobs["model.pkl"]))
    calibrator = joblib.load(io.BytesIO(blobs["calibrator.pkl"]))
    checks: list = []
    validate(manifest, p06, p03, feature_names, model, calibrator, alias, blobs, checks)
    if manifest["version"] != alias["version"]:
        raise ModelLoadError(f"version: manifest {manifest['version']} != alias {alias['version']}")
    gate = "passed" if alias.get("gate_passed") else "not_passed_provisional"
    return LoadedModel(version=alias["version"], family=manifest["family"], input_family=manifest["input_family"],
                       model=model, calibrator=calibrator, p06=p06, p03=p03, manifest=manifest, alias=alias,
                       threshold_raw=float(manifest["operating_point"]["threshold_raw"]), gate_status=gate,
                       checks=checks)
