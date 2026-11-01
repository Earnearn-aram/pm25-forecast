"""Leakage-safe rolling PM2.5 features for experimental smoothing models."""

from datetime import datetime, timedelta
from math import isfinite

from pm25_forecast.backtest import Observation
from pm25_forecast.models.baselines import require_utc


def rolling_values(
    observations: tuple[Observation, ...], origin: datetime, hours: int
) -> list[tuple[datetime, float]]:
    """Return available, finite values in [origin-hours, origin-1h]."""
    origin = require_utc(origin)
    start, end = origin - timedelta(hours=hours), origin - timedelta(hours=1)
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
    target = require_utc(target)
    values = [
        v for ts, v in rolling_values(observations, origin, days * 24) if ts.hour == target.hour
    ]
    return sum(values) / len(values) if values else None
