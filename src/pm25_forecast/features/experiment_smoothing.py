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
