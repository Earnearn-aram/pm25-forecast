"""Content-verified model artifact and manifest storage in PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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


# Locks the pointer row until the transaction ends, so two concurrent promotes/rollbacks cannot
# both read the same "previous champion". Postgres only (SQLite has no FOR UPDATE).
SELECT_CHAMPION_LOCK = SELECT_CHAMPION.rstrip("\n") + " FOR UPDATE\n"
# `at` is the database clock (column default now()); `id` is the order of events.
INSERT_EVENT = """
INSERT INTO model_events (action, model_version, previous_version, actor, metrics, reason)
VALUES (%(action)s, %(model_version)s, %(previous_version)s, %(actor)s, %(metrics)s, %(reason)s)
"""
# The latest champion change in the log (a recommend/keep decision never moves the pointer).
SELECT_LAST_CHANGE = """
SELECT model_version, previous_version FROM model_events
WHERE action IN ('promote', 'rollback') ORDER BY id DESC LIMIT 1
"""
# When retrain last logged a decision (actor 'auto' is retrain's own auto-promote).
SELECT_LAST_RETRAIN_EVENT = """
SELECT max(at) FROM model_events WHERE actor IN ('retrain', 'auto')
"""
SELECT_ALL_MODELS = "SELECT model_version, created_at FROM models ORDER BY created_at DESC"
SELECT_RECENT_PREDICTION_MODELS = """
SELECT DISTINCT model_version FROM predictions WHERE issue_time >= %(since)s
"""
# Re-checks the protections in the statement itself, so a plan made a moment ago cannot delete a
# model that has become the champion or has just been used for predictions.
DELETE_MODEL = """
DELETE FROM models
WHERE model_version = %(model_version)s
  AND model_version NOT IN (SELECT model_version FROM model_champion)
  AND model_version NOT IN (SELECT model_version FROM predictions WHERE issue_time >= %(since)s)
  AND model_version NOT IN (
      SELECT previous_version FROM model_events
      WHERE id = (SELECT max(id) FROM model_events WHERE action IN ('promote', 'rollback'))
        AND previous_version IS NOT NULL)
"""
DELETE_MODEL_REFERENCE = "DELETE FROM drift_reference WHERE model_version = %(model_version)s"

EVENT_ACTIONS = ("promote", "rollback", "recommend", "keep")
EVENT_ACTORS = ("cli", "retrain", "auto")
PRUNE_REFERENCE_DAYS = 30


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


def _json_safe(value: Any) -> Any:
    """JSON-ready copy: non-finite floats become null, unknown types become strings."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    return str(value)


def record_event(
    con: Any,
    action: str,
    model_version: str,
    *,
    previous_version: str | None = None,
    actor: str = "cli",
    metrics: Mapping[str, Any] | None = None,
    reason: str | None = None,
) -> None:
    """Append one row to ``model_events`` on ``con`` (the caller's transaction).

    The event time is the database clock (``at`` defaults to ``now()``).
    """
    if action not in EVENT_ACTIONS:
        raise ValueError(f"unknown model event action: {action}")
    if actor not in EVENT_ACTORS:
        raise ValueError(f"unknown model event actor: {actor}")
    con.execute(
        INSERT_EVENT,
        {
            "action": action,
            "model_version": model_version,
            "previous_version": previous_version,
            "actor": actor,
            "metrics": None if metrics is None else json.dumps(_json_safe(metrics), sort_keys=True),
            "reason": reason,
        },
    )


def promote_model(
    con: Any,
    model_version: str,
    *,
    actor: str = "cli",
    action: str = "promote",
    reason: str | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> str | None:
    """Point production at an existing, content-verified model version and log the change.

    The pointer change and its ``model_events`` row are written on the same connection, so they
    commit (or roll back) together. The pointer row is locked (``FOR UPDATE``) before the previous
    champion is read, so concurrent changes are serialised. Returns the previous champion, or
    None if there was none.
    """
    if action not in ("promote", "rollback"):
        raise ValueError("a champion change is a promote or a rollback")
    # Verify existence and artifact integrity before changing the pointer. The database foreign
    # key is an additional guard against a dangling pointer.
    read_model(con, model_version)
    before = con.execute(SELECT_CHAMPION_LOCK, {}).fetchone()
    previous = None if before is None else before[0]
    con.execute(UPSERT_CHAMPION, {"model_version": model_version})
    row = con.execute(SELECT_CHAMPION, {}).fetchone()
    if row is None or row[0] != model_version:
        raise ValueError("production champion pointer does not match requested model version")
    record_event(
        con,
        action,
        model_version,
        previous_version=previous,
        actor=actor,
        metrics=metrics,
        reason=reason,
    )
    return previous


def previous_champion(con: Any) -> str:
    """The champion before the latest recorded change, for a rollback.

    Raises LookupError if the log has no change, the latest change had no previous champion, or
    the log disagrees with the live pointer (the pointer was changed outside the log). Call it in
    the transaction that then does the rollback: it locks the pointer row (``FOR UPDATE``).
    """
    current = con.execute(SELECT_CHAMPION_LOCK, {}).fetchone()
    if current is None:
        raise LookupError("production champion model is not set; nothing to roll back")
    last = con.execute(SELECT_LAST_CHANGE, {}).fetchone()
    if last is None or not last[1]:
        raise LookupError("no previous champion is recorded in model_events")
    if last[0] != current[0]:
        raise LookupError(
            f"the latest recorded change ({last[0]}) is not the current champion ({current[0]}); "
            "promote the version you want explicitly"
        )
    return last[1]


@dataclass(frozen=True)
class PrunePlan:
    keep: dict[str, str]  # model_version -> why it is kept
    delete: list[str]


def plan_prune(
    con: Any, keep_newest: int, *, challenger: str | None = None, now: datetime | None = None
) -> PrunePlan:
    """Which ``models`` rows a prune would delete. Nothing is changed.

    Kept: the production champion, ``challenger``, the rollback target (the previous champion of the
    latest logged change), every model with predictions issued in the last ``PRUNE_REFERENCE_DAYS``
    days, and the ``keep_newest`` newest by ``created_at``.
    """
    if keep_newest < 0:
        raise ValueError("--keep must not be negative")
    moment = now or datetime.now(UTC)
    since = moment - timedelta(days=PRUNE_REFERENCE_DAYS)
    models = [row[0] for row in con.execute(SELECT_ALL_MODELS, {}).fetchall()]  # newest first
    keep: dict[str, str] = {}
    champion = con.execute(SELECT_CHAMPION, {}).fetchone()
    if champion is not None:
        keep[champion[0]] = "production champion"
    if challenger:
        keep.setdefault(challenger, "challenger")
    last_change = con.execute(SELECT_LAST_CHANGE, {}).fetchone()
    if last_change is not None and last_change[1]:
        keep.setdefault(last_change[1], "rollback target (previous champion)")
    for (version,) in con.execute(SELECT_RECENT_PREDICTION_MODELS, {"since": since}).fetchall():
        keep.setdefault(version, f"predictions in the last {PRUNE_REFERENCE_DAYS} days")
    for version in models[:keep_newest]:
        keep.setdefault(version, f"one of the {keep_newest} newest")
    return PrunePlan(
        keep={v: why for v, why in keep.items() if v in models},
        delete=[v for v in models if v not in keep],
    )


def prune_models(con: Any, plan: PrunePlan, *, now: datetime | None = None) -> list[str]:
    """Delete the planned models (and their cached drift reference); return those deleted.

    The delete statement re-checks the champion and recent-prediction protections, so a model
    that became referenced after the plan was made is skipped.
    """
    since = (now or datetime.now(UTC)) - timedelta(days=PRUNE_REFERENCE_DAYS)
    deleted: list[str] = []
    for version in plan.delete:
        cursor = con.execute(DELETE_MODEL, {"model_version": version, "since": since})
        if getattr(cursor, "rowcount", 0) == 1:
            con.execute(DELETE_MODEL_REFERENCE, {"model_version": version})
            deleted.append(version)
    return deleted


def last_retrain_event_at(con: Any) -> datetime | None:
    """When retrain last logged a decision (any action by actor retrain or auto), in UTC."""
    row = con.execute(SELECT_LAST_RETRAIN_EVENT, {}).fetchone()
    value = None if row is None else row[0]
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


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
