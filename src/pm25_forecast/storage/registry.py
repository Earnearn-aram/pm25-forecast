"""Content-verified model artifact and manifest storage in PostgreSQL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pm25_forecast.collector import open_db
from pm25_forecast.storage import db

UPSERT_MODEL = """
INSERT INTO models (model_version, sha256, artifact, manifest)
VALUES (%(model_version)s, %(sha256)s, %(artifact)s, %(manifest)s::jsonb)
ON CONFLICT (model_version) DO UPDATE SET
    sha256 = EXCLUDED.sha256,
    artifact = EXCLUDED.artifact,
    manifest = EXCLUDED.manifest,
    created_at = now()
"""
SELECT_MODEL = """
SELECT sha256, artifact, manifest
FROM models WHERE model_version = %(model_version)s
"""


def artifact_sha256(artifact: bytes) -> str:
    return hashlib.sha256(artifact).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require the fields that make a registered model reproducible and interpretable."""
    required = {"version", "metrics", "training_window", "feature_list"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"model manifest is missing required fields: {', '.join(sorted(missing))}")
    if not isinstance(manifest["metrics"], dict):
        raise ValueError("manifest metrics must be an object")
    if not isinstance(manifest["training_window"], dict):
        raise ValueError("manifest training_window must be an object")
    if not isinstance(manifest["feature_list"], list) or not manifest["feature_list"]:
        raise ValueError("manifest feature_list must be a non-empty ordered list")
    return dict(manifest)


def write_model(con: Any, model_version: str, artifact: bytes, manifest: Mapping[str, Any]) -> str:
    """Write one version, then read it back and verify the stored artifact and metadata."""
    if not model_version.strip() or not artifact:
        raise ValueError("model_version and artifact are required")
    validated = validate_manifest(manifest)
    digest = artifact_sha256(artifact)
    con.execute(
        UPSERT_MODEL,
        {
            "model_version": model_version,
            "sha256": digest,
            "artifact": artifact,
            "manifest": json.dumps(validated, separators=(",", ":"), sort_keys=True),
        },
    )
    return digest


def read_model(con: Any, model_version: str) -> tuple[bytes, dict[str, Any]]:
    """Fetch and verify one model as an hourly inference job would."""
    row = con.execute(SELECT_MODEL, {"model_version": model_version}).fetchone()
    if row is None:
        raise LookupError(f"model version not found: {model_version}")
    expected_hash, artifact, manifest = row
    artifact = bytes(artifact)
    if artifact_sha256(artifact) != expected_hash:
        raise ValueError("stored model checksum does not match artifact")
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    return artifact, validate_manifest(manifest)


def publish_model(
    database_url: str,
    model_version: str,
    artifact: bytes,
    manifest: Mapping[str, Any],
    *,
    connect=db.connect,
) -> str:
    # The first context commits on exit; the second connection proves the hourly reader can
    # retrieve the committed row rather than merely seeing its own uncommitted write.
    with open_db(connect, database_url) as con:
        digest = write_model(con, model_version, artifact, manifest)
    with open_db(connect, database_url) as con:
        read_model(con, model_version)
    return digest
