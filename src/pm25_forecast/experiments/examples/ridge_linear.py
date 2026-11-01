"""Minimal complete model factory for ``experiments.evaluate``: ridge on [1, last value, mean24].

    python -m pm25_forecast.experiments.evaluate --snapshot <dir> \
        --model pm25_forecast.experiments.examples.ridge_linear:fit

One ridge regression per horizon, fitted with numpy only. Features come from the shared module
``pm25_forecast.features`` (``build_features`` for the last value, ``mean_24h`` for the 24 h mean);
there is no private feature code. Copy this file as the starting point for a real model.

Data you get (``backtest.Observation`` / ``backtest.Vintage``, all UTC, timezone-aware):
    Observation(station_id, ts, value, available_at)
    Vintage(station_id, issue_time, valid_time, value, available_at, model)   # CAMS forecasts
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from pm25_forecast.backtest import Observation, Predictor, Snapshot, Vintage
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features import build_features, mean_24h

HOUR = timedelta(hours=1)
DEFAULT_ALPHA = 10.0


def _row(
    window: list[Observation], vintages: list[Vintage], origin: datetime, valid_time: datetime
) -> list[float] | None:
    """[1, last value, mean24] for one (origin, valid_time), or None when an input is missing.

    ``window`` must hold only observations with ``ts <= origin - 1 h`` and
    ``available_at <= origin`` (the serving lag); ``build_features`` and ``mean_24h`` apply
    the same rule again and ignore anything later.
    """
    latest = build_features(window, vintages, origin, valid_time)["pm25_latest"]
    mean24 = mean_24h(window, origin)
    if latest is None or mean24 is None:
        return None
    return [1.0, latest, mean24]


def _last_24h(by_ts: Mapping[datetime, Observation], origin: datetime) -> list[Observation]:
    return [o for k in range(1, 25) if (o := by_ts.get(origin - k * HOUR)) is not None]


def fit(train: TrainSnapshot, params: Mapping[str, Any] | None = None) -> Predictor:
    """Fit one ridge per horizon on every (origin, target) pair that had matured at refit time."""
    alpha = float((params or {}).get("alpha", DEFAULT_ALPHA))
    by_station: dict[str, dict[datetime, Observation]] = defaultdict(dict)
    for o in train.observations:  # already limited to ts/available_at <= refit_time
        by_station[o.station_id][o.ts] = o
    vintages: dict[str, list[Vintage]] = defaultdict(list)
    for v in train.vintages:
        vintages[v.station_id].append(v)

    coef: dict[int, np.ndarray] = {}
    for horizon in train.horizons:
        xs: list[list[float]] = []
        ys: list[float] = []
        for station, by_ts in by_station.items():
            for target, label in by_ts.items():  # the label is the observation at the target
                origin = target - horizon * HOUR
                # Usable pair: origin + h <= refit_time holds because target = ts <= refit_time,
                # and the label's available_at <= refit_time because it is in the train set.
                row = _row(_last_24h(by_ts, origin), vintages[station], origin, target)
                if row is not None:
                    xs.append(row)
                    ys.append(label.value)
        if len(xs) < 10:
            continue  # too little data: no forecast for this horizon (never invent one)
        x, y = np.array(xs), np.array(ys)
        penalty = alpha * np.eye(x.shape[1])
        penalty[0, 0] = 0.0  # do not shrink the intercept
        coef[horizon] = np.linalg.solve(x.T @ x + penalty, x.T @ y)

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        if horizon not in coef:
            return None
        origin = snapshot.prediction_time
        by_ts = {o.ts: o for o in snapshot.observations if o.ts >= origin - 24 * HOUR}
        row = _row(
            _last_24h(by_ts, origin), list(snapshot.vintages), origin, origin + horizon * HOUR
        )
        return None if row is None else float(np.dot(coef[horizon], row))

    return predict
