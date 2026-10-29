"""Content-verified model artifact and manifest storage in PostgreSQL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pm25_forecast.collector import open_db
from pm25_forecast.storage import db

INSERT_MODEL = """
INSERT INTO models (model_version, sha256, artifact, manifest)
VALUES (%(model_version)s, %(sha256)s, %(artifact)s, %(manifest)s::jsonb)
ON CONFLICT (model_version) DO NOTHING
"""
SELECT_MODEL = """
SELECT sha256, artifact, manifest
FROM models WHERE model_version = %(model_version)s
"""
SELECT_CHAMPION = """
SELECT model_version
FROM model_champion WHERE slot = 'production'
"""
UPSERT_CHAMPION = """
INSERT INTO model_champion (slot, model_version)
VALUES ('production', %(model_version)s)
ON CONFLICT (slot) DO UPDATE
SET model_version = EXCLUDED.model_version, promoted_at = now()
"""


def artifact_sha256(artifact: bytes) -> str:
    return hashlib.sha256(artifact).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require the fields that make a registered model reproducible and interpretable."""
    required = {
        "version",
        "metrics",
        "training_window",
        "feature_list",
        "data_sha256",
        "artifact_sha256",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"model manifest is missing required fields: {', '.join(sorted(missing))}")
    if not isinstance(manifest["metrics"], dict):
        raise ValueError("manifest metrics must be an object")
    if not isinstance(manifest["training_window"], dict):
        raise ValueError("manifest training_window must be an object")
    if not isinstance(manifest["feature_list"], list) or not manifest["feature_list"]:
        raise ValueError("manifest feature_list must be a non-empty ordered list")
    if any(not isinstance(name, str) or not name.strip() for name in manifest["feature_list"]):
        raise ValueError("manifest feature_list entries must be non-empty strings")
    if len(set(manifest["feature_list"])) != len(manifest["feature_list"]):
        raise ValueError("manifest feature_list must not contain duplicates")
    if not isinstance(manifest["version"], str) or not manifest["version"].strip():
        raise ValueError("manifest version must be a non-empty string")
    for field in ("data_sha256", "artifact_sha256"):
        digest = manifest[field]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"manifest {field} must be a SHA-256 hex digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"manifest {field} must be a SHA-256 hex digest") from exc
    return dict(manifest)


def _validate_artifact_features(artifact: bytes, feature_list: list[str]) -> None:
    """Require the manifest's ordered features to match every booster in the bundle."""
    try:
        bundle = json.loads(artifact)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model artifact is not a supported JSON bundle") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != 1:
        raise ValueError("model artifact is not a supported JSON bundle")
    if isinstance(bundle.get("horizons"), dict) and bundle["horizons"]:
        model_texts = list(bundle["horizons"].values())
    elif isinstance(bundle.get("booster"), str):
        model_texts = [bundle["booster"]]
    else:
        raise ValueError("model artifact has no supported LightGBM booster payload")
    if any(not isinstance(model_text, str) for model_text in model_texts):
        raise ValueError("model artifact has an invalid LightGBM booster payload")

    import lightgbm as lgb

    for model_text in model_texts:
        try:
            model_features = lgb.Booster(model_str=model_text).feature_name()
        except (lgb.basic.LightGBMError, TypeError, ValueError) as exc:
            raise ValueError("model artifact contains an invalid LightGBM booster") from exc
        if model_features != feature_list:
            raise ValueError("manifest feature_list does not match model artifact features")


def write_model(con: Any, model_version: str, artifact: bytes, manifest: Mapping[str, Any]) -> str:
    """Insert one immutable version; an identical retry succeeds, a collision fails."""
    if not model_version.strip() or not artifact:
        raise ValueError("model_version and artifact are required")
    validated = validate_manifest(manifest)
    if validated["version"] != model_version:
        raise ValueError("manifest version does not match model_version")
    digest = artifact_sha256(artifact)
    if validated["artifact_sha256"] != digest:
        raise ValueError("manifest artifact_sha256 does not match model artifact")
    _validate_artifact_features(artifact, validated["feature_list"])
    con.execute(
        INSERT_MODEL,
        {
            "model_version": model_version,
            "sha256": digest,
            "artifact": artifact,
            "manifest": json.dumps(validated, separators=(",", ":"), sort_keys=True),
        },
    )
    stored_artifact, stored_manifest = read_model(con, model_version)
    if artifact_sha256(stored_artifact) != digest:
        raise ValueError("model_version already exists with a different artifact")
    if stored_artifact != artifact or stored_manifest != validated:
        raise ValueError("model_version already exists with different artifact or manifest")
    return digest


def read_model(con: Any, model_version: str) -> tuple[bytes, dict[str, Any]]:
    """Fetch and verify one model bundle and its stored checksum."""
    row = con.execute(SELECT_MODEL, {"model_version": model_version}).fetchone()
    if row is None:
        raise LookupError(f"model version not found: {model_version}")
    expected_hash, artifact, manifest = row
    artifact = bytes(artifact)
    if artifact_sha256(artifact) != expected_hash:
        raise ValueError("stored model checksum does not match artifact")
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    validated = validate_manifest(manifest)
    if validated["version"] != model_version:
        raise ValueError("stored model manifest version does not match model_version")
    if validated["artifact_sha256"] != expected_hash:
        raise ValueError("stored model manifest hash does not match artifact")
    _validate_artifact_features(artifact, validated["feature_list"])
    return artifact, validated


def promote_model(con: Any, model_version: str) -> None:
    """Point production at an existing, content-verified model version."""
    # Verify existence and artifact integrity before changing the pointer. The database foreign
    # key is an additional guard against a dangling pointer.
    read_model(con, model_version)
    con.execute(UPSERT_CHAMPION, {"model_version": model_version})
    row = con.execute(SELECT_CHAMPION, {}).fetchone()
    if row is None or row[0] != model_version:
        raise ValueError("production champion pointer does not match requested model version")


def read_champion(con: Any) -> tuple[str, bytes, dict[str, Any]]:
    """Read the current production pointer and return its verified model bundle."""
    row = con.execute(SELECT_CHAMPION, {}).fetchone()
    if row is None:
        raise LookupError("production champion model is not set")
    model_version = row[0]
    artifact, manifest = read_model(con, model_version)
    return model_version, artifact, manifest


def publish_model(
    database_url: str,
    model_version: str,
    artifact: bytes,
    manifest: Mapping[str, Any],
    *,
    connect=db.connect,
) -> str:
    validated = validate_manifest(manifest)
    # The first context commits on exit; the second connection verifies the committed row.
    with open_db(connect, database_url) as con:
        digest = write_model(con, model_version, artifact, manifest)
    with open_db(connect, database_url) as con:
        stored_artifact, stored_manifest = read_model(con, model_version)
    if artifact_sha256(stored_artifact) != digest or stored_artifact != artifact:
        raise ValueError("committed model artifact does not match submitted artifact")
    if stored_manifest != validated:
        raise ValueError("committed model manifest does not match submitted manifest")
    return digest
