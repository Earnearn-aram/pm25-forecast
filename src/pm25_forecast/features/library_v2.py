"""KAN-86 (E09) feature library v2: lags, rolling stats, calendar, cross-station, missingness.

Shared by training and serving, like ``build_features``. The v1 contract (``FEATURE_NAMES``,
``SMOOTHER_FEATURE_NAMES``) is untouched; experiments pick a v2 set by name from ``FEATURE_SETS``.

As-of rule (docs/experiments/protocol.md section 2): for origin ``t`` only readings with
``ts <= t - 1h`` and ``available_at <= t`` are read, so nothing after ``t - 1h`` can enter a
feature. A lag of ``k`` h is the reading at exactly ``ts = t - k h`` (lag 1 is the freshest
allowed reading). A rolling window of ``w`` h covers ``t - w h <= ts <= t - 1h``. Missing inputs
give ``None``; nothing is imputed. Calendar features describe ``valid_time`` (the target hour,
known in advance) in Bangkok local time (UTC+7, fixed offset, no DST).

Missing values: ``None`` means missing. Callers NaN-fill for LightGBM and impute for linear
models; nothing is imputed here. ``n_valid_*h`` and ``missing_lag_*h`` are floats (counts / 0-1).

Timestamps: readings are floored to the hour (a 10:30 reading counts as hour 10:00); the origin
must be on the hour (ValueError otherwise). If several readings fall in one hour, the one with
the latest ``available_at <= origin`` wins (a missing ``available_at`` ranks lowest), so the result
does not depend on row order.

Cross-station features use each station's latest reading within ``MAX_OBSERVATION_AGE`` (3 h, same
as ``pm25_latest``), while the own-station lags use exact hour slots. ``bkk_mean`` is the mean over
all stations that have such a reading, INCLUDING the own station, and shifts when stations drop out.

Performance: call ``prepare_series`` once per snapshot/fold and pass the result to
``build_features_v2`` / ``build_feature_set``; raw records are re-indexed on every call (fine for
live serving, quadratic in a walk-forward). Only the groups of the requested names are computed.

Holidays: the table covers 2026 only; other years give ``None`` for ``is_thai_holiday``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
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
# TODO(KAN-86): extend the table BEFORE 2027-01-01 (serving needs it); training is 2026 only.
# Static table: years outside HOLIDAY_YEARS give None for the holiday flag, not a silent 0.
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


_NO_TIME = datetime.min.replace(tzinfo=UTC)


class PreparedSeries:
    """Observations indexed once: station -> hour -> [(available_at, value)] (hour-floored UTC)."""

    def __init__(self, index: dict[str, dict[datetime, list[tuple[datetime, float]]]]):
        self._index = index

    @property
    def stations(self) -> tuple[str, ...]:
        return tuple(self._index)

    def value(self, station: str, hour: datetime, origin: datetime) -> float | None:
        """Reading of ``hour`` visible at ``origin`` (latest available_at <= origin), else None."""
        rows = self._index.get(station, {}).get(hour)
        if not rows:
            return None
        best: tuple[datetime, float] | None = None
        for row in rows:
            if row[0] <= origin and (best is None or row[0] >= best[0]):
                best = row
        return None if best is None else best[1]


def prepare_series(observations_by_station: Mapping[str, Iterable[Any]]) -> PreparedSeries:
    """Normalise and index records once (same record shapes as ``build_features``)."""
    index: dict[str, dict[datetime, list[tuple[datetime, float]]]] = {}
    for station, records in observations_by_station.items():
        hours = index.setdefault(station, {})
        fields = (("ts",), ("pm25", "value"), ("available_at", "ingested_at"))
        for row in _rows(records, fields):
            ts, value = row["ts"], _usable(row["pm25"])
            if ts is None or value is None:
                continue
            hour = require_utc(ts).replace(minute=0, second=0, microsecond=0)
            available_at = row["available_at"]
            avail = _NO_TIME if available_at is None else require_utc(available_at)
            hours.setdefault(hour, []).append((avail, value))
    return PreparedSeries(index)


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
    observations: Mapping[str, Iterable[Any]] | PreparedSeries,
    station_id: str,
    prediction_time: datetime,
    valid_time: datetime,
    station_locations: Mapping[str, tuple[float, float]] | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, float | None]:
    """Features for one station and origin, keyed by ``names`` (default ``ALL_FEATURE_NAMES``).

    ``observations`` is a ``PreparedSeries`` (preferred) or a mapping of station id to raw
    records. ``station_locations`` maps station id to (lat, lon); without it the nearest-neighbour
    feature is None. Only the groups needed for ``names`` are computed.
    """
    origin, valid_time = require_utc(prediction_time), require_utc(valid_time)
    if (origin.minute, origin.second, origin.microsecond) != (0, 0, 0):
        raise ValueError(f"prediction_time must be on the hour, got {origin.isoformat()}")
    want = tuple(ALL_FEATURE_NAMES if names is None else names)
    unknown = set(want) - set(ALL_FEATURE_NAMES)
    if unknown:
        raise KeyError(f"unknown features: {sorted(unknown)}")
    wanted = set(want)
    prep = (
        observations if isinstance(observations, PreparedSeries) else prepare_series(observations)
    )

    # Own readings for t-1h .. t-72h (None = missing), looked up once and shared by all groups.
    slots: list[float | None] = [
        prep.value(station_id, origin - k * HOUR, origin) for k in range(1, max(WINDOWS_H) + 1)
    ]

    def window(hours: int) -> list[float]:
        return [v for v in slots[:hours] if v is not None]

    def at(k: int) -> float | None:  # reading at ts = t - k h
        return slots[k - 1]

    feats: dict[str, float | None] = {}
    for k in LAGS_H:
        feats[f"pm25_lag_{k}h"] = at(k)
        feats[f"missing_lag_{k}h"] = float(at(k) is None)

    for w in WINDOWS_H:
        vals = window(w)
        feats[f"n_valid_{w}h"] = float(len(vals))
        feats[f"pm25_mean_{w}h"] = statistics.fmean(vals) if vals else None
        feats[f"pm25_std_{w}h"] = statistics.stdev(vals) if len(vals) >= 2 else None
        feats[f"pm25_min_{w}h"] = min(vals) if vals else None
        feats[f"pm25_max_{w}h"] = max(vals) if vals else None
        feats[f"pm25_median_{w}h"] = statistics.median(vals) if vals else None

    # EWMA over present values, newest first, weight (1 - alpha)^(age-1), normalised over them.
    weighted = [
        (v, (1 - EWMA_ALPHA) ** i) for i, v in enumerate(slots[:EWMA_WINDOW_H]) if v is not None
    ]
    total = sum(w for _, w in weighted)
    feats[EWMA_NAME] = sum(v * w for v, w in weighted) / total if weighted else None

    newest = at(1)
    for d in DIFFS_H:
        older = at(1 + d)
        feats[f"pm25_diff_{d}h"] = None if newest is None or older is None else newest - older

    if wanted & set(CALENDAR_NAMES):
        feats.update(_calendar(valid_time))

    if wanted & set(CROSS_STATION_NAMES):
        feats.update(_cross_station(prep, station_id, origin, station_locations))

    return {name: feats[name] for name in want}


def _latest_reading(prep: PreparedSeries, station: str, origin: datetime) -> float | None:
    for k in range(1, int(MAX_OBSERVATION_AGE / HOUR) + 1):  # ts = t-1h .. t-3h, newest first
        v = prep.value(station, origin - k * HOUR, origin)
        if v is not None:
            return v
    return None


def _cross_station(
    prep: PreparedSeries,
    station_id: str,
    origin: datetime,
    locations: Mapping[str, tuple[float, float]] | None,
) -> dict[str, float | None]:
    latest = {s: v for s in prep.stations if (v := _latest_reading(prep, s, origin)) is not None}
    mean = statistics.fmean(latest.values()) if latest else None
    own = latest.get(station_id)
    out: dict[str, float | None] = {
        "bkk_mean": mean,
        "dev_from_bkk_mean": None if own is None or mean is None else own - mean,
        "nearest_station_pm25": None,
    }
    here = (locations or {}).get(station_id)
    if here is not None and locations is not None:
        others = [
            (_distance_km(here, locations[s]), s)
            for s in sorted(latest)
            if s != station_id and s in locations
        ]
        if others:
            out["nearest_station_pm25"] = latest[min(others)[1]]
    return out


def build_feature_set(
    name: str,
    observations: Mapping[str, Iterable[Any]] | PreparedSeries,
    station_id: str,
    prediction_time: datetime,
    valid_time: datetime,
    station_locations: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float | None]:
    """The features of registered set ``name``, in the set's order (only those are computed)."""
    return build_features_v2(
        observations,
        station_id,
        prediction_time,
        valid_time,
        station_locations,
        names=feature_set_names(name),
    )
