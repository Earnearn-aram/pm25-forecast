"""Classical time-series candidates for KAN-85 (E07): ARIMA, SARIMA and Holt-Winters per station.

Dev-only experiment code. Every factory has the runner signature ``fit(train, params)`` and is
called once per weekly fold; the returned predictor is called for every (origin, horizon).

Rules the code follows (protocol sections 2 and 3):

* A forecast from origin ``t`` uses only readings with ``ts <= t - 1 h`` and ``available_at <= t``.
  The series ends at the hourly grid point ``t - 1 h`` and the forecast ``h`` hours ahead is step
  ``h + 1`` from that point.
* Gaps: the hourly grid is reindexed and missing hours are NaN. The target is never filled. The
  state-space models (ARIMA/SARIMA) skip a missing hour in the Kalman update (the filter carries the
  state forward; no value is invented). Holt-Winters cannot take NaN, so it uses the newest gap-free
  trailing run and returns None when that run is shorter than three seasons.
* No forecast (None) when the newest reading is older than 3 h, like the persistence baseline.
* Parameters and the model order (AIC, on the training part of the fold only) are re-estimated at
  every weekly refit. Between refits the state is updated hourly with the parameters frozen
  (``results.extend``); there is no refit inside a fold.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from pm25_forecast.backtest import Observation, Predictor, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features import build_features

log = logging.getLogger(__name__)
HOUR = timedelta(hours=1)
MAX_HORIZON = 24
FALLBACK_WINDOW = 336  # hours used when a state chain cannot be extended (fresh instance)
TIMING_ENV = "PM25_TS_TIMING_LOG"  # optional JSONL sink for per-fold fit time and orders


def _series_by_ts(rows: tuple[Observation, ...] | list[Observation], origin: datetime):
    """Usable readings of one station keyed by ts, serving lag enforced a second time."""
    out: dict[datetime, Observation] = {}
    for o in rows:
        if o.ts > origin - HOUR or o.available_at > origin:
            continue
        prev = out.get(o.ts)
        if prev is None or o.available_at >= prev.available_at:
            out[o.ts] = o
    return out


def _grid(by_ts: Mapping[datetime, Any], start: datetime, end: datetime, value) -> np.ndarray:
    n = int((end - start) / HOUR) + 1
    out = np.full(n, np.nan)
    for k in range(n):
        o = by_ts.get(start + k * HOUR)
        if o is not None:
            out[k] = value(o)
    return out


def _trailing_run(y: np.ndarray) -> np.ndarray:
    """Newest gap-free run of ``y`` (empty when the last value is missing)."""
    bad = np.flatnonzero(np.isnan(y))
    return y[bad[-1] + 1 :] if bad.size else y


def _latest_ok(snapshot: Snapshot, horizon: int) -> bool:
    origin = snapshot.prediction_time
    tail = snapshot.observations[-6:]
    latest = build_features(tail, (), origin, origin + horizon * HOUR)["pm25_latest"]
    return latest is not None


def _log_fold(name: str, train: TrainSnapshot, seconds: float, detail: dict[str, Any]) -> None:
    rec = {"model": name, "refit_time": train.refit_time.isoformat(), "seconds": seconds, **detail}
    log.info("fit %s", json.dumps(rec))
    sink = os.environ.get(TIMING_ENV)
    if sink:
        path = Path(sink)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


def _train_series(train: TrainSnapshot) -> dict[str, dict[datetime, Observation]]:
    out: dict[str, dict[datetime, Observation]] = {s: {} for s in train.stations}
    for o in train.observations:
        if o.station_id in out and o.ts <= train.refit_time:  # matured rows only
            out[o.station_id][o.ts] = o
    return out


# ----------------------------------------------------------------------------- ARIMA / SARIMA


@dataclass
class _Chain:
    """Fitted results whose filter has been run up to grid hour ``end`` (inclusive)."""

    results: Any
    end: datetime
    last: float  # value at ``end`` (NaN if missing): consistency check for extension


def _orders(params: Mapping[str, Any]):
    d = int(params.get("d", 0))
    ps = [int(p) for p in params["p_grid"]]
    qs = [int(q) for q in params["q_grid"]]
    seasonal = params.get("seasonal_grid")  # list of [P, Q] or None
    period = int(params.get("seasonal_period", 0))
    out = []
    for p, q in product(ps, qs):
        if seasonal:
            out += [((p, d, q), (int(sp), 0, int(sq), period)) for sp, sq in seasonal]
        else:
            out.append(((p, d, q), (0, 0, 0, 0)))
    return out


def _fit_best(y: np.ndarray, params: Mapping[str, Any]):
    """Fit every order of the preregistered grid on ``y``; return (results, order, n_tried)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    trend = "c" if int(params.get("d", 0)) == 0 else None
    best = None
    tried = 0
    for order, sorder in _orders(params):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=sorder,
                    trend=trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit(disp=False, maxiter=int(params.get("maxiter", 50)))
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue
        tried += 1
        if not np.isfinite(res.aic):
            continue
        if best is None or res.aic < best[0].aic:
            best = (res, (order, sorder))
    return (best[0], best[1], tried) if best else (None, None, tried)


def arima_family(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    """ARIMA (AIC grid) or SARIMA (24 h season), one model per station, state updated hourly."""
    name = str(params.get("name", "arima"))
    fit_hours = int(params["fit_days"]) * 24 if params.get("fit_days") else None
    t0 = time.perf_counter()
    series = _train_series(train)
    chains: dict[str, _Chain] = {}
    chosen: dict[str, Any] = {}
    for station, by_ts in series.items():
        if not by_ts:
            continue
        end = max(by_ts)
        start = min(by_ts)
        if fit_hours is not None:
            start = max(start, end - (fit_hours - 1) * HOUR)
        y = _grid(by_ts, start, end, _value)
        if np.isfinite(y).sum() < 24 * 14:
            continue  # under two weeks of readings: no model (never invent one)
        res, order, tried = _fit_best(y, params)
        if res is None:
            continue
        chains[station] = _Chain(res, end, float(y[-1]))
        chosen[station] = {"order": order, "aic": float(res.aic), "tried": tried}
    _log_fold(name, train, time.perf_counter() - t0, {"stations": chosen})
    base = dict(chains)  # the refit results; ``chains`` advances hourly
    cache: dict[str, Any] = {"snap": None, "fc": None}

    def advance(station: str, by_ts: Mapping[datetime, Observation], end: datetime) -> Any:
        chain = chains[station]
        cur = by_ts.get(chain.end)
        same = (cur is None and np.isnan(chain.last)) or (
            cur is not None and cur.value == chain.last
        )
        if end == chain.end and same:
            return chain.results
        if end > chain.end and same:
            new = _grid(by_ts, chain.end + HOUR, end, _value)
            res = chain.results.extend(new)
        else:  # fresh instance / earlier origin / changed history: filter a recent window
            start = end - (FALLBACK_WINDOW - 1) * HOUR
            win = _grid(by_ts, start, end, _value)
            res = base[station].apply(win)
        chains[station] = _Chain(res, end, float(by_ts[end].value) if end in by_ts else np.nan)
        return res

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        station = snapshot.station_id
        if station not in chains or horizon > MAX_HORIZON or not _latest_ok(snapshot, horizon):
            return None
        if cache["snap"] is snapshot:
            fc = cache["fc"]
        else:
            origin = snapshot.prediction_time
            by_ts = _series_by_ts(snapshot.observations, origin)
            res = advance(station, by_ts, origin - HOUR)
            fc = np.asarray(res.forecast(MAX_HORIZON + 1), dtype=float)
            cache["snap"], cache["fc"] = snapshot, fc
        value = fc[horizon]  # step h + 1 from the last grid hour (origin - 1 h)
        return float(value) if np.isfinite(value) else None

    return predict


def arima(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    return arima_family(train, params)


def sarima(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    return arima_family(train, params)


# ----------------------------------------------------------------------------- Holt-Winters


def ets(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    """Additive Holt-Winters with a 24 h season; smoothing parameters re-estimated weekly.

    The weekly fit uses the newest gap-free run (at most ``fit_days``). At every origin the
    recursion is re-run with the frozen smoothing parameters over the newest gap-free run of
    ``state_hours`` readings (heuristic initial states), which is the hourly state update.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    name = str(params.get("name", "ets"))
    season = int(params.get("seasonal_periods", 24))
    trend = params.get("trend", "add")
    damped = bool(params.get("damped_trend", True)) and trend is not None
    fit_hours = int(params.get("fit_days", 28)) * 24
    state_hours = int(params.get("state_hours", 168))
    min_run = 3 * season

    def spec(y: np.ndarray, init: str):
        return ExponentialSmoothing(
            y,
            trend=trend,
            damped_trend=damped,
            seasonal="add",
            seasonal_periods=season,
            initialization_method=init,
        )

    t0 = time.perf_counter()
    series = _train_series(train)
    smoothing: dict[str, dict[str, float]] = {}
    for station, by_ts in series.items():
        if not by_ts:
            continue
        end, start = max(by_ts), min(by_ts)
        y = _trailing_run(_grid(by_ts, max(start, end - (fit_hours - 1) * HOUR), end, _value))
        if y.size < min_run:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = spec(y, "estimated").fit()
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue
        p = res.params
        fixed = {
            "smoothing_level": float(p["smoothing_level"]),
            "smoothing_seasonal": float(p["smoothing_seasonal"]),
        }
        if trend is not None:
            fixed["smoothing_trend"] = float(p["smoothing_trend"])
        if damped:
            fixed["damping_trend"] = float(p["damping_trend"])
        smoothing[station] = fixed
    _log_fold(name, train, time.perf_counter() - t0, {"stations": smoothing})
    cache: dict[str, Any] = {"snap": None, "fc": None}

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        station = snapshot.station_id
        if station not in smoothing or horizon > MAX_HORIZON or not _latest_ok(snapshot, horizon):
            return None
        if cache["snap"] is not snapshot:
            origin = snapshot.prediction_time
            by_ts = _series_by_ts(snapshot.observations, origin)
            end = origin - HOUR
            y = _grid(by_ts, end - (state_hours - 1) * HOUR, end, _value)
            y = _trailing_run(y)
            fc = None
            if y.size >= min_run:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = spec(y, "heuristic").fit(optimized=False, **smoothing[station])
                fc = np.asarray(res.forecast(MAX_HORIZON + 1), dtype=float)
            cache["snap"], cache["fc"] = snapshot, fc
        fc = cache["fc"]
        if fc is None:
            return None
        value = fc[horizon]
        return float(value) if np.isfinite(value) else None

    return predict


def _value(o: Observation) -> float:
    return o.value
