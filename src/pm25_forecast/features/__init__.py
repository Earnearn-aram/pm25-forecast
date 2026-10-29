"""Leakage-safe feature building shared by the backtest and live inference.

Keep ``FEATURE_NAMES`` in model input order. Model manifests should copy this ordered tuple so
the serialized model's feature contract can be checked at inference time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from pm25_forecast.models.baselines import MAX_OBSERVATION_AGE, require_utc

FEATURE_NAMES: tuple[str, ...] = ("pm25_latest", "pm25_same_hour_24h", "cams_pm25")
FEATURE_ORDER: tuple[str, ...] = FEATURE_NAMES


def _field(row: Any, name: str, *aliases: str) -> Any:
    if isinstance(row, Mapping):
        for key in (name, *aliases):
            if key in row:
                return row[key]
        return None
    for key in (name, *aliases):
        if hasattr(row, key):
            return getattr(row, key)
    return None


def _rows(rows: Iterable[Any], fields: Sequence[tuple[str, ...]]) -> Iterable[dict[str, Any]]:
    """Normalize tuple, mapping, and dataclass records into named values."""
    for row in rows:
        if isinstance(row, Sequence) and not isinstance(row, str | bytes | Mapping):
            yield {
                names[0]: row[index] if len(row) > index else None
                for index, names in enumerate(fields)
            }
        else:
            yield {names[0]: _field(row, *names) for names in fields}


def _usable(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def build_features(
    observations: Iterable[Any],
    vintages: Iterable[Any],
    prediction_time: datetime,
    valid_time: datetime,
) -> dict[str, float | None]:
    """Build the ordered model feature mapping from raw observations and forecast vintages.

    Observation records may be ``(ts, pm25)`` or objects/mappings with ``ts`` and ``pm25``
    (``value`` is also accepted). Vintage records may be ``(issue_time, valid_time, pm25[, model])``
    or objects/mappings with those fields. Records with a future observation timestamp, vintage
    issue time, or optional ``available_at`` timestamp are ignored.
    """
    prediction_time = require_utc(prediction_time)
    valid_time = require_utc(valid_time)
    observation_rows = _rows(
        observations, (("ts",), ("pm25", "value"), ("available_at", "ingested_at"))
    )

    usable_observations: list[tuple[datetime, float]] = []
    history: list[tuple[datetime, float]] = []
    for row in observation_rows:
        ts, value = row["ts"], _usable(row["pm25"])
        if ts is None or value is None:
            continue
        ts = require_utc(ts)
        if ts > prediction_time:
            continue
        available_at = row["available_at"]
        if available_at is not None and require_utc(available_at) > prediction_time:
            continue
        history.append((ts, value))
        if prediction_time - ts <= MAX_OBSERVATION_AGE:
            usable_observations.append((ts, value))
    latest = max(usable_observations, default=None, key=lambda row: row[0])

    daily_time = valid_time - timedelta(hours=24)
    same_hour = next((value for ts, value in history if ts == daily_time), None)

    vintage_fields = (
        ("issue_time",),
        ("valid_time",),
        ("pm25", "pm25_pred", "value"),
        ("model", "model_version"),
        ("available_at", "created_at"),
    )
    vintage_rows = _rows(vintages, vintage_fields)
    eligible: list[tuple[datetime, float]] = []
    for row in vintage_rows:
        issue, valid, value = row["issue_time"], row["valid_time"], _usable(row["pm25"])
        if issue is None or valid is None or value is None:
            continue
        issue, valid = require_utc(issue), require_utc(valid)
        if issue > prediction_time or valid != valid_time:
            continue
        model = row.get("model")
        if model is not None and model not in ("cams", "cams_global"):
            continue
        available_at = row["available_at"]
        if available_at is not None and require_utc(available_at) > prediction_time:
            continue
        eligible.append((issue, value))
    cams = max(eligible, default=None, key=lambda row: row[0])

    return {
        "pm25_latest": None if latest is None else latest[1],
        "pm25_same_hour_24h": same_hour,
        "cams_pm25": None if cams is None else cams[1],
    }
