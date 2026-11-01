"""KAN-87 (E10) CAMS forecast features, shared by training and serving.

For origin ``t`` (on the hour) and target hour ``valid_time`` a CAMS vintage is visible only if
``issue_time <= t`` and ``available_at <= t`` (docs/experiments/protocol.md section 2). Missing
inputs give ``None``; nothing is imputed.

Features (``CAMS_FEATURE_NAMES``):

* ``cams_pm25_target``: value for ``valid_time`` of the newest visible vintage (latest issue_time).
* ``cams_minus_latest``: that value minus the newest own-station reading (``ts`` in
  ``[t-3h, t-1h]``, same 3 h limit as ``pm25_latest``).
* ``cams_err_24h`` / ``cams_err_72h``: mean signed error ``observed - CAMS`` over hours ``v`` in
  ``[t-W, t-1h]``. Only matured targets count: the observation at ``v`` must have
  ``available_at <= t``. CAMS for ``v`` is the newest vintage with ``issue_time <= v`` (a real
  forecast, not a later re-analysis) that is visible at ``t``. Needs ``MIN_ERR_HOURS`` matched
  hours, else ``None``. Positive means CAMS under-predicts recently.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from pm25_forecast.features import _rows, _usable
from pm25_forecast.features.library_v2 import PreparedSeries
from pm25_forecast.models.baselines import MAX_OBSERVATION_AGE, require_utc

HOUR = timedelta(hours=1)
ERR_WINDOWS_H: tuple[int, ...] = (24, 72)
MIN_ERR_HOURS = 6
CAMS_MODELS = ("cams", "cams_global")
CAMS_FEATURE_NAMES: tuple[str, ...] = (
    "cams_pm25_target",
    "cams_minus_latest",
    "cams_err_24h",
    "cams_err_72h",
)


class PreparedCams:
    """Vintages indexed once: station -> valid hour -> [(issue_time, available_at, value)]."""

    def __init__(self, index: dict[str, dict[datetime, list[tuple[datetime, datetime, float]]]]):
        self._index = index

    def visible(
        self,
        station: str,
        valid_time: datetime,
        origin: datetime,
        issued_by: datetime | None = None,
    ) -> float | None:
        """Newest vintage for ``valid_time`` visible at ``origin`` (and issued by ``issued_by``)."""
        best: tuple[datetime, float] | None = None
        for issue, avail, value in self._index.get(station, {}).get(valid_time, ()):
            if issue > origin or avail > origin:
                continue
            if issued_by is not None and issue > issued_by:
                continue
            if best is None or issue >= best[0]:
                best = (issue, value)
        return None if best is None else best[1]

    def __bool__(self) -> bool:
        return any(self._index.values())


def prepare_cams(vintages_by_station: Mapping[str, Iterable[Any]]) -> PreparedCams:
    """Normalise and index vintage records (same shapes as ``build_features``)."""
    fields = (
        ("issue_time",),
        ("valid_time",),
        ("pm25", "pm25_pred", "value"),
        ("model", "model_version"),
        ("available_at", "created_at"),
    )
    index: dict[str, dict[datetime, list[tuple[datetime, datetime, float]]]] = {}
    for station, records in vintages_by_station.items():
        hours = index.setdefault(station, {})
        for row in _rows(records, fields):
            issue, valid, value = row["issue_time"], row["valid_time"], _usable(row["pm25"])
            avail = row["available_at"]
            if issue is None or valid is None or value is None or avail is None:
                continue  # no valid available_at: excluded, never filled (protocol section 2)
            if row["model"] is not None and row["model"] not in CAMS_MODELS:
                continue
            valid = require_utc(valid).replace(minute=0, second=0, microsecond=0)
            hours.setdefault(valid, []).append((require_utc(issue), require_utc(avail), value))
    return PreparedCams(index)


def _own_latest(obs: PreparedSeries, station: str, origin: datetime) -> float | None:
    for k in range(1, int(MAX_OBSERVATION_AGE / HOUR) + 1):
        if (v := obs.value(station, origin - k * HOUR, origin)) is not None:
            return v
    return None


def _recent_error(
    obs: PreparedSeries, cams: PreparedCams, station: str, origin: datetime, window_h: int
) -> float | None:
    errs = []
    for k in range(1, window_h + 1):
        hour = origin - k * HOUR  # ts <= t - 1h; obs.value also enforces available_at <= t
        actual = obs.value(station, hour, origin)
        if actual is None:
            continue
        forecast = cams.visible(station, hour, origin, issued_by=hour)
        if forecast is not None:
            errs.append(actual - forecast)
    return statistics.fmean(errs) if len(errs) >= MIN_ERR_HOURS else None


def build_cams_features(
    obs: PreparedSeries,
    cams: PreparedCams,
    station_id: str,
    prediction_time: datetime,
    valid_time: datetime,
) -> dict[str, float | None]:
    """The four CAMS features for one station and origin, keyed by ``CAMS_FEATURE_NAMES``."""
    origin, valid_time = require_utc(prediction_time), require_utc(valid_time)
    if (origin.minute, origin.second, origin.microsecond) != (0, 0, 0):
        raise ValueError(f"prediction_time must be on the hour, got {origin.isoformat()}")
    target = cams.visible(station_id, valid_time, origin)
    latest = _own_latest(obs, station_id, origin)
    out: dict[str, float | None] = {
        "cams_pm25_target": target,
        "cams_minus_latest": None if target is None or latest is None else target - latest,
    }
    for w in ERR_WINDOWS_H:
        out[f"cams_err_{w}h"] = _recent_error(obs, cams, station_id, origin, w)
    return out
