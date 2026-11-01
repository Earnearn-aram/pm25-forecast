"""Leakage-safe rolling PM2.5 features for experimental smoothing models."""

import logging
from datetime import datetime, timedelta
from math import isfinite

from pm25_forecast.backtest import Observation
from pm25_forecast.models.baselines import require_utc

logger = logging.getLogger(__name__)


def rolling_values(
    observations: tuple[Observation, ...], origin: datetime, hours: int
) -> list[tuple[datetime, float]]:
    """Return available, finite values in [origin-hours, origin-1h]."""
    origin = require_utc(origin)
    start, end = origin - timedelta(hours=hours), origin - timedelta(hours=1)
    missing = sum(1 for o in observations if start <= o.ts <= end and o.available_at is None)
    if missing:
        logger.warning("Dropped %d rolling observations with missing available_at", missing)
    return [
        (o.ts, o.value)
        for o in observations
        if start <= o.ts <= end
        and o.available_at is not None
        and o.available_at <= origin
        and isfinite(o.value)
        and o.value >= 0
    ]


def exponential_mean(
    observations: tuple[Observation, ...], origin: datetime, half_life: float
) -> float | None:
    """Time-weighted mean of all available history using an hourly half-life."""
    if half_life <= 0:
        raise ValueError("half-life must be positive")
    origin = require_utc(origin)
    numerator = denominator = 0.0
    missing = sum(
        1 for o in observations if o.ts <= origin - timedelta(hours=1) and o.available_at is None
    )
    if missing:
        logger.warning("Dropped %d EWMA observations with missing available_at", missing)
    for o in observations:
        age = (origin - o.ts).total_seconds() / 3600
        if (
            age < 1
            or o.available_at is None
            or o.available_at > origin
            or not isfinite(o.value)
            or o.value < 0
        ):
            continue
        weight = 2 ** (-(age - 1) / half_life)
        numerator += weight * o.value
        denominator += weight
    return numerator / denominator if denominator else None


def same_hour_mean(
    observations: tuple[Observation, ...], origin: datetime, target: datetime, days: int
) -> float | None:
    """Mean at target's UTC hour in the preceding days, with the serving lag."""
    if days <= 0:
        raise ValueError("days must be positive")
    origin, target = require_utc(origin), require_utc(target)
    hourly: dict[datetime, tuple[datetime, datetime, float]] = {}
    start, end = origin - timedelta(days=days), origin - timedelta(hours=1)
    missing = sum(1 for o in observations if start <= o.ts <= end and o.available_at is None)
    if missing:
        logger.warning("Dropped %d climatology observations with missing available_at", missing)
    for o in observations:
        if (
            not start <= o.ts <= end
            or o.ts.hour != target.hour
            or o.available_at is None
            or o.available_at > origin
            or not isfinite(o.value)
            or o.value < 0
        ):
            continue
        hour = o.ts.replace(minute=0, second=0, microsecond=0)
        candidate = (o.ts, o.available_at, o.value)
        if hour not in hourly or candidate > hourly[hour]:
            hourly[hour] = candidate
    return sum(row[2] for row in hourly.values()) / len(hourly) if hourly else None
