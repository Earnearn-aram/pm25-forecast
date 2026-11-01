"""KAN-86 (E09) feature library v2: lags, rolling stats, calendar, cross-station, missingness.

Shared by training and serving, like ``build_features``. The v1 contract (``FEATURE_NAMES``,
``SMOOTHER_FEATURE_NAMES``) is untouched; experiments pick a v2 set by name from ``FEATURE_SETS``.

As-of rule (docs/experiments/protocol.md section 2): for origin ``t`` only readings with
``ts <= t - 1h`` and ``available_at <= t`` are read, so nothing after ``t - 1h`` can enter a
feature. A lag of ``k`` h is the reading at exactly ``ts = t - k h`` (lag 1 is the freshest
allowed reading). A rolling window of ``w`` h covers ``t - w h <= ts <= t - 1h``. Missing inputs
give ``None``; nothing is imputed. Calendar features describe ``valid_time`` (the target hour,
known in advance) in Bangkok local time (UTC+7, fixed offset, no DST).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pm25_forecast.features import _rows, _usable
from pm25_forecast.models.baselines import MAX_OBSERVATION_AGE, require_utc

HOUR = timedelta(hours=1)
SERVING_LAG = HOUR
LAGS_H: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48)
WINDOWS_H: tuple[int, ...] = (3, 6, 12, 24, 72)
ROLLING_STATS: tuple[str, ...] = ("mean", "std", "min", "max", "median")
EWMA_ALPHA = 0.3
EWMA_WINDOW_H = 24
DIFFS_H: tuple[int, ...] = (1, 24)
BANGKOK_OFFSET = timedelta(hours=7)

# Thai public holidays (including substitution days and cabinet special days), Bangkok local
# dates; cross-checked 2026-11-01 against secondary lists of the Bank of Thailand calendar.
# Static table: years outside HOLIDAY_YEARS give None for the holiday flag instead of a silent wrong 0.
HOLIDAY_YEARS: frozenset[int] = frozenset({2026})
THAI_HOLIDAYS: frozenset[date] = frozenset(
    date(2026, m, d)
    for m, d in (
        (1, 1),
        (1, 2),  # cabinet special holiday
        (3, 3),
        (4, 6),
        (4, 13),
        (4, 14),
        (4, 15),
        (5, 1),
        (5, 4),
        (5, 31),
        (6, 1),
        (6, 3),
        (7, 28),
        (7, 29),
        (7, 30),
        (8, 12),
        (10, 13),
        (10, 23),
        (12, 5),
        (12, 7),
        (12, 10),
        (12, 31),
    )
)

LAG_NAMES = tuple(f"pm25_lag_{k}h" for k in LAGS_H)
ROLLING_NAMES = tuple(f"pm25_{s}_{w}h" for w in WINDOWS_H for s in ROLLING_STATS)
EWMA_NAME = f"pm25_ewma_{EWMA_WINDOW_H}h"
DIFF_NAMES = tuple(f"pm25_diff_{d}h" for d in DIFFS_H)
CALENDAR_NAMES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_thai_holiday",
)
CROSS_STATION_NAMES = ("bkk_mean", "dev_from_bkk_mean", "nearest_station_pm25")
MISSING_NAMES = tuple(f"missing_lag_{k}h" for k in LAGS_H) + tuple(
    f"n_valid_{w}h" for w in WINDOWS_H
)

ALL_FEATURE_NAMES: tuple[str, ...] = (
    *LAG_NAMES,
    *ROLLING_NAMES,
    EWMA_NAME,
    *DIFF_NAMES,
    *CALENDAR_NAMES,
    *CROSS_STATION_NAMES,
    *MISSING_NAMES,
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "fs_v2_core": (
        *LAG_NAMES,
        *(f"pm25_mean_{w}h" for w in WINDOWS_H),
        *DIFF_NAMES,
        *CALENDAR_NAMES,
    ),
    "fs_v2_full": ALL_FEATURE_NAMES,
}


def feature_set_names(name: str) -> tuple[str, ...]:
    """Ordered feature names of a registered set (model manifests copy this tuple)."""
    try:
        return FEATURE_SETS[name]
    except KeyError:
        raise KeyError(f"unknown feature set {name!r}; known: {sorted(FEATURE_SETS)}") from None


def _series(records: Iterable[Any], origin: datetime) -> dict[datetime, float]:
    """Usable readings visible at ``origin``: ``ts <= origin - 1h``, ``available_at <= origin``."""
    cutoff = origin - SERVING_LAG
    out: dict[datetime, float] = {}
    for row in _rows(records, (("ts",), ("pm25", "value"), ("available_at", "ingested_at"))):
        ts, value = row["ts"], _usable(row["pm25"])
        if ts is None or value is None:
            continue
        ts = require_utc(ts)
        if ts > cutoff or ts < origin - timedelta(hours=max(WINDOWS_H) + 1):
            continue
        available_at = row["available_at"]
        if available_at is not None and require_utc(available_at) > origin:
            continue
        out[ts] = value
    return out


def _latest(series: Mapping[datetime, float], origin: datetime) -> float | None:
    fresh = [(ts, v) for ts, v in series.items() if origin - ts <= MAX_OBSERVATION_AGE]
    return max(fresh, key=lambda r: r[0])[1] if fresh else None


def _window(series: Mapping[datetime, float], origin: datetime, hours: int) -> list[float]:
    """Values for hours origin-1h back to origin-``hours``h, newest first."""
    return [v for k in range(1, hours + 1) if (v := series.get(origin - k * HOUR)) is not None]


def _calendar(valid_time: datetime) -> dict[str, float | None]:
    local = valid_time.astimezone(UTC) + BANGKOK_OFFSET
    hour_angle = 2 * math.pi * local.hour / 24
    dow_angle = 2 * math.pi * local.weekday() / 7
    day = local.date()
    return {
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "dow_sin": math.sin(dow_angle),
        "dow_cos": math.cos(dow_angle),
        "is_weekend": float(local.weekday() >= 5),
        "is_thai_holiday": float(day in THAI_HOLIDAYS) if day.year in HOLIDAY_YEARS else None,
    }


def _distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def build_features_v2(
    observations_by_station: Mapping[str, Iterable[Any]],
    station_id: str,
    prediction_time: datetime,
    valid_time: datetime,
    station_locations: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float | None]:
    """All v2 features for one station and origin, keyed by ``ALL_FEATURE_NAMES``.

    ``observations_by_station`` maps station id to observation records (same shapes as
    ``build_features``). ``station_locations`` maps station id to (lat, lon); without it the
    nearest-neighbour feature is None. Use ``build_feature_set`` to get a named subset.
    """
    origin, valid_time = require_utc(prediction_time), require_utc(valid_time)
    all_series = {s: _series(rows, origin) for s, rows in observations_by_station.items()}
    own = all_series.get(station_id, {})

    feats: dict[str, float | None] = {}
    for k, name in zip(LAGS_H, LAG_NAMES, strict=True):
        feats[name] = own.get(origin - k * HOUR)

    for w in WINDOWS_H:
        vals = _window(own, origin, w)
        feats[f"pm25_mean_{w}h"] = statistics.fmean(vals) if vals else None
        feats[f"pm25_std_{w}h"] = statistics.stdev(vals) if len(vals) >= 2 else None
        feats[f"pm25_min_{w}h"] = min(vals) if vals else None
        feats[f"pm25_max_{w}h"] = max(vals) if vals else None
        feats[f"pm25_median_{w}h"] = statistics.median(vals) if vals else None

    # EWMA over present values, newest first, weight (1 - alpha)^age_in_hours, normalised.
    weighted = [
        (own[ts], (1 - EWMA_ALPHA) ** (k - 1))
        for k in range(1, EWMA_WINDOW_H + 1)
        if (ts := origin - k * HOUR) in own
    ]
    total = sum(w for _, w in weighted)
    feats[EWMA_NAME] = sum(v * w for v, w in weighted) / total if weighted else None

    newest = own.get(origin - SERVING_LAG)
    for d in DIFFS_H:
        older = own.get(origin - SERVING_LAG - d * HOUR)
        feats[f"pm25_diff_{d}h"] = None if newest is None or older is None else newest - older

    feats.update(_calendar(valid_time))

    latest_by_station = {
        s: v for s, ser in all_series.items() if (v := _latest(ser, origin)) is not None
    }
    feats["bkk_mean"] = statistics.fmean(latest_by_station.values()) if latest_by_station else None
    own_latest = latest_by_station.get(station_id)
    feats["dev_from_bkk_mean"] = (
        None if own_latest is None or feats["bkk_mean"] is None else own_latest - feats["bkk_mean"]
    )
    feats["nearest_station_pm25"] = None
    here = (station_locations or {}).get(station_id)
    if here is not None:
        others = [
            (_distance_km(here, station_locations[s]), s)
            for s in sorted(latest_by_station)
            if s != station_id and s in station_locations
        ]
        if others:
            feats["nearest_station_pm25"] = latest_by_station[min(others)[1]]

    for k in LAGS_H:
        feats[f"missing_lag_{k}h"] = float(feats[f"pm25_lag_{k}h"] is None)
    for w in WINDOWS_H:
        feats[f"n_valid_{w}h"] = float(len(_window(own, origin, w)))

    return {name: feats[name] for name in ALL_FEATURE_NAMES}


def build_feature_set(
    name: str,
    observations_by_station: Mapping[str, Iterable[Any]],
    station_id: str,
    prediction_time: datetime,
    valid_time: datetime,
    station_locations: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float | None]:
    """The features of registered set ``name``, in the set's order."""
    names = feature_set_names(name)
    feats = build_features_v2(
        observations_by_station, station_id, prediction_time, valid_time, station_locations
    )
    return {n: feats[n] for n in names}
