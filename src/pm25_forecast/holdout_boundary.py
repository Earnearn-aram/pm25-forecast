"""Fixture-stage verification of a reviewed locked-holdout boundary.

This module is not connected to retraining or the live exporter. A caller must supply an anchor
from independently reviewed, protected source control; a digest embedded in the manifest itself
is not a trust anchor. Validation here does not prove that any real export applied the exclusion.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pm25_forecast.experiments import folds

SHA_PATTERN = r"^[0-9a-f]{64}$"
RULES = {
    "observations": ("ts",),
    "forecast_vintages": ("issue_time", "valid_time"),
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("boundary timestamps must be timezone-aware UTC")
    return value.astimezone(UTC)


class BoundaryManifest(BaseModel):
    """Metadata only; the frozen experiment rows must not be loaded to verify this record."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    boundary_id: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=SHA_PATTERN)
    snapshot_manifest_sha256: str = Field(pattern=SHA_PATTERN)
    protocol_sha256: str = Field(pattern=SHA_PATTERN)
    derivation_code_sha256: str = Field(pattern=SHA_PATTERN)
    lock_ref: str = Field(min_length=1)
    s0: datetime
    last_usable_target: datetime
    holdout_start: datetime
    holdout_end: datetime
    exclusion_columns: dict[str, tuple[str, ...]]

    @field_validator("s0", "last_usable_target", "holdout_start", "holdout_end")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        return _utc(value)


@dataclass(frozen=True)
class ReviewedAnchor:
    """Independent reviewed pins, not values copied from an untrusted manifest at runtime."""

    manifest_sha256: str
    snapshot_sha256: str
    snapshot_manifest_sha256: str
    protocol_sha256: str
    derivation_code_sha256: str
    lock_ref: str


def verify_manifest(
    data: bytes,
    anchor: ReviewedAnchor,
    *,
    snapshot_manifest: bytes,
    protocol: bytes,
) -> BoundaryManifest:
    """Validate a pinned metadata record and reproduce its boundary without reading data rows."""
    actual = hashlib.sha256(data).hexdigest()
    if not hmac.compare_digest(actual, anchor.manifest_sha256):
        raise ValueError("boundary manifest does not match the reviewed digest")
    manifest = BoundaryManifest.model_validate_json(data)
    for field in (
        "snapshot_sha256",
        "snapshot_manifest_sha256",
        "protocol_sha256",
        "derivation_code_sha256",
        "lock_ref",
    ):
        if getattr(manifest, field) != getattr(anchor, field):
            raise ValueError(f"boundary {field} does not match the reviewed anchor")
    for content, expected, label in (
        (snapshot_manifest, anchor.snapshot_manifest_sha256, "snapshot manifest"),
        (protocol, anchor.protocol_sha256, "protocol"),
        (Path(folds.__file__).read_bytes(), anchor.derivation_code_sha256, "derivation code"),
    ):
        if not hmac.compare_digest(hashlib.sha256(content).hexdigest(), expected):
            raise ValueError(f"{label} does not match the reviewed digest")
    snapshot_metadata = json.loads(snapshot_manifest)
    if not isinstance(snapshot_metadata, dict) or snapshot_metadata.get("snapshot_sha256") != (
        anchor.snapshot_sha256
    ):
        raise ValueError("snapshot manifest does not identify the reviewed snapshot")
    try:
        last_target = _utc(
            datetime.fromisoformat(snapshot_metadata["time_range"]["last_usable_target"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("snapshot manifest has no valid last usable target") from exc
    if last_target != manifest.last_usable_target:
        raise ValueError("boundary target range does not match the snapshot manifest")
    if manifest.last_usable_target >= folds.TARGET_CUT:
        raise ValueError("snapshot target range exceeds the protocol cutoff")
    if manifest.s0 != folds.S0:
        raise ValueError("boundary start does not match the protocol's fixed S0")
    split = folds.derive_folds(manifest.last_usable_target)
    if (
        split.holdout_start is None
        or split.holdout_start != manifest.holdout_start
        or split.end != manifest.holdout_end
    ):
        raise ValueError("boundary does not match the protocol-derived holdout")
    if manifest.exclusion_columns != RULES:
        raise ValueError("boundary does not specify the required conservative exclusion rules")
    return manifest


def fixture_row_allowed(
    source: str, timestamps: dict[str, datetime], boundary: BoundaryManifest
) -> bool:
    """Apply the conservative half-open exclusion contract to synthetic source rows.

    This is not a database exporter. Exclude any vintage with an issue or valid time in the
    protected interval, even if only one of those timestamps would be used as a target.
    """
    if source not in RULES or set(timestamps) != set(RULES[source]):
        raise ValueError("unknown source or missing/extra timestamp columns")
    return all(
        not boundary.holdout_start <= _utc(value) < boundary.holdout_end
        for value in timestamps.values()
    )
