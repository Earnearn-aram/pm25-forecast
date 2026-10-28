"""Baseline predictors (KAN-28): persistence and CAMS pass-through. Pure, no IO.

Both turn already-loaded data into forecasts for ``valid_time = issue_time + h`` for each horizon
``h``. The backtest (KAN-14) can import them unchanged. All timestamps are timezone-aware UTC;
naive datetimes raise ``ValueError``. Neither predictor ever invents a value: when the input is
missing they return nothing for that horizon and the caller logs it.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

PERSISTENCE_MODEL_VERSION = "persistence-24h"
CAMS_PASSTHROUGH_MODEL_VERSION = "cams"
HORIZONS_H: tuple[int, ...] = (6, 12, 24)  # the headline horizon is +24 h
HEADLINE_HORIZON_H = 24
# An observation older than this (relative to issue_time) is too stale to persist.
MAX_OBSERVATION_AGE = timedelta(hours=3)


@dataclass(frozen=True)
class Forecast:
    horizon_h: int
    valid_time: datetime
    pm25_pred: float


def require_utc(value: datetime) -> datetime:
    """Reject naive datetimes; normalise to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def valid_times(issue_time: datetime, horizons: Sequence[int] = HORIZONS_H) -> dict[int, datetime]:
    """``{h: issue_time + h hours}`` in UTC."""
    issue = require_utc(issue_time)
    return {h: issue + timedelta(hours=h) for h in horizons}


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0


def latest_observation(
    observations: Iterable[tuple[datetime, float | None]],
    issue_time: datetime,
    max_age: timedelta = MAX_OBSERVATION_AGE,
) -> float | None:
    """Value of the newest observation with ``issue_time - max_age <= ts <= issue_time``."""
    issue = require_utc(issue_time)
    best: tuple[datetime, float] | None = None
    for ts, value in observations:
        ts = require_utc(ts)
        if value is None or ts > issue or issue - ts > max_age or not _usable(value):
            continue
        if best is None or ts > best[0]:
            best = (ts, float(value))
    return None if best is None else best[1]


def persistence_forecast(
    observations: Iterable[tuple[datetime, float | None]],
    issue_time: datetime,
    horizons: Sequence[int] = HORIZONS_H,
    max_age: timedelta = MAX_OBSERVATION_AGE,
) -> list[Forecast]:
    """Forecast every horizon as the latest observed value at or before ``issue_time``.

    Empty list if there is no recent usable observation (nothing is invented).
    """
    value = latest_observation(observations, issue_time, max_age)
    if value is None:
        return []
    return [Forecast(h, vt, value) for h, vt in valid_times(issue_time, horizons).items()]


def cams_passthrough(
    vintages: Iterable[tuple[datetime, datetime, float | None]],
    issue_time: datetime,
    as_of: datetime,
    horizons: Sequence[int] = HORIZONS_H,
) -> tuple[list[Forecast], list[int]]:
    """Pass the stored CAMS forecast through for ``issue_time + h``.

    `vintages` are ``(vintage_issue_time, valid_time, pm25)``. For each target valid_time the row
    from the latest vintage with ``vintage_issue_time <= as_of`` is used (never a later one, so a
    backtest cannot peek). Returns the forecasts found and the horizons with no CAMS row.
    """
    as_of = require_utc(as_of)
    targets = valid_times(issue_time, horizons)
    best: dict[datetime, tuple[datetime, float]] = {}
    for vintage_issue, valid, value in vintages:
        vintage_issue, valid = require_utc(vintage_issue), require_utc(valid)
        if value is None or vintage_issue > as_of or not _usable(value):
            continue
        if valid not in best or vintage_issue > best[valid][0]:
            best[valid] = (vintage_issue, float(value))
    found: list[Forecast] = []
    missing: list[int] = []
    for h, vt in targets.items():
        if vt in best:
            found.append(Forecast(h, vt, best[vt][1]))
        else:
            missing.append(h)
    return found, missing
