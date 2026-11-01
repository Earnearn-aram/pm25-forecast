"""KAN-84 (E06) linear models: OLS, ridge and lasso behind one factory for ``experiments.evaluate``.

    python -m pm25_forecast.experiments.evaluate --prereg docs/experiments/prereg/<id>.yaml \
        --snapshot ~/pm25-experiments/snapshots/e02-v1 \
        --model pm25_forecast.experiments.linear_models:fit

Variant axes (all in ``prereg.params``; defaults in ``DEFAULTS``):

- ``model``: ``ols`` | ``ridge`` | ``lasso``. Regularisation strength is chosen by an inner
  expanding-window time-series CV on the training rows of each weekly fold only (``select``).
- ``scope``: ``pooled`` (one model per horizon, station one-hot) | ``per_station``.
- ``target``: ``raw`` (ug/m3) | ``log1p`` (fit on log1p, back-transformed with expm1).
- ``output``: ``level`` (predict the value) | ``change`` (predict target minus the latest reading,
  the persistence value; in log1p space the difference of log1p values).
- ``feature_set``: a name from ``features.library_v2.FEATURE_SETS`` (default ``fs_v2_core``) or
  ``fs_v2_full_nocross``: ``fs_v2_full`` without the cross-station group, because the runner hands
  a predictor only the own-station readings.

Feature logic is only in ``pm25_forecast.features.library_v2`` (no private feature code). The
latest reading is the newest of ``pm25_lag_1h/2h/3h`` (the persistence rule of protocol section 1).
A row is skipped (``None``, never imputed) when there is no reading in ``[t-3h, t-1h]``; other
missing features are filled with the training mean (after the train/validation split inside the
inner CV, so nothing leaks) and then standardised. Negative predictions are clipped by the runner.

Diagnostics (chosen strength, standardised coefficients per fold) are appended to
``$PM25_KAN84_LOG_DIR/<variant>.jsonl`` (default ``~/pm25-experiments/cache/kan84``), outside git.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from pm25_forecast.backtest import Observation, Predictor, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features.library_v2 import (
    CROSS_STATION_NAMES,
    build_features_v2,
    feature_set_names,
    prepare_series,
)

HOUR = timedelta(hours=1)
LOOKBACK_H = 80  # fs_v2_* read at most 72 h before the origin
LATEST_LAGS = ("pm25_lag_1h", "pm25_lag_2h", "pm25_lag_3h")
MIN_TRAIN_ROWS = 200  # per model; fewer: no forecast for that (station, horizon)
INNER_BLOCKS = 5  # time blocks by origin; 4 expanding splits (block k+1 validates on blocks <= k)
MIN_INNER_TRAIN = 60
RIDGE_LAMBDAS = tuple(10.0**e for e in np.arange(-4.0, 0.01, 0.5))  # alpha_sk = n * lambda
LASSO_RATIOS = tuple(10.0**e for e in np.arange(-4.0, 0.01, 0.5))  # alpha = ratio * alpha_max
DEFAULTS: dict[str, Any] = {
    "model": "ridge",
    "scope": "pooled",
    "target": "raw",
    "output": "level",
    "feature_set": "fs_v2_core",
    "variant": "ad-hoc",
}
SETS_NO_CROSS = {"fs_v2_full_nocross": "fs_v2_full"}

_cache: dict[tuple, tuple[np.ndarray, float] | None] = {}
_dumps: dict[str, IO[str]] = {}


def clear_cache() -> None:
    """Forget cached training feature rows (tests use different data under the same keys)."""
    _cache.clear()


def names_for(feature_set: str) -> tuple[str, ...]:
    if feature_set == "fs_v2_full":
        raise ValueError("fs_v2_full needs other stations' readings; use fs_v2_full_nocross")
    if feature_set in SETS_NO_CROSS:
        return tuple(
            n for n in feature_set_names(SETS_NO_CROSS[feature_set]) if n not in CROSS_STATION_NAMES
        )
    return feature_set_names(feature_set)


# ---- target transform ----------------------------------------------------------------------


def to_target(y: np.ndarray, latest: np.ndarray, target: str, output: str) -> np.ndarray:
    z = np.log1p(y) if target == "log1p" else y
    if output == "change":
        z = z - (np.log1p(latest) if target == "log1p" else latest)
    return z


def from_target(p: np.ndarray, latest: np.ndarray, target: str, output: str) -> np.ndarray:
    if target == "log1p":
        z = p + (np.log1p(latest) if output == "change" else 0.0)
        return np.expm1(np.minimum(z, 20.0))
    return p + (latest if output == "change" else 0.0)


# ---- numerics ------------------------------------------------------------------------------


class Standardiser:
    """Training-mean imputation, then z-scores (constant columns get scale 1)."""

    def __init__(self, x: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column
            mean = np.nanmean(x, axis=0)
        self.mean = np.where(np.isnan(mean), 0.0, mean)
        filled = np.where(np.isnan(x), self.mean, x)
        scale = filled.std(axis=0)
        self.scale = np.where(scale < 1e-9, 1.0, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.where(np.isnan(x), self.mean, x) - self.mean) / self.scale


def fit_linear(
    z: np.ndarray, y: np.ndarray, model: str, strength: float
) -> tuple[float, np.ndarray]:
    """(intercept, coefficients) on standardised ``z``. ``strength``: ridge lambda, lasso ratio."""
    n = len(y)
    ybar = float(y.mean())
    yc = y - ybar
    if model == "ols":
        w = np.linalg.lstsq(z, yc, rcond=None)[0]
    elif model == "ridge":
        w = np.linalg.solve(z.T @ z + n * strength * np.eye(z.shape[1]), z.T @ yc)
    elif model == "lasso":
        alpha_max = float(np.max(np.abs(z.T @ yc)) / n)
        if alpha_max <= 0:
            return ybar, np.zeros(z.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            w = (
                Lasso(
                    alpha=strength * alpha_max,
                    fit_intercept=False,
                    max_iter=20000,
                    tol=1e-5,
                    selection="cyclic",
                )
                .fit(z, yc)
                .coef_
            )
    else:
        raise ValueError(f"unknown model {model!r}")
    return ybar, w


def grid_for(model: str) -> tuple[float, ...]:
    return {"ols": (0.0,), "ridge": RIDGE_LAMBDAS, "lasso": LASSO_RATIOS}[model]


def inner_splits(origins: np.ndarray, horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits on origin time: validate on block k, train on rows whose label
    had matured at the block start (``origin + h <= start``), never on later rows."""
    edges = np.quantile(origins, np.linspace(0, 1, INNER_BLOCKS + 1))
    out = []
    for k in range(1, INNER_BLOCKS):
        lo, hi = edges[k], edges[k + 1]
        val = (origins >= lo) & ((origins <= hi) if k == INNER_BLOCKS - 1 else (origins < hi))
        train = origins + horizon * 3600 <= lo
        if train.sum() >= MIN_INNER_TRAIN and val.any():
            out.append((np.flatnonzero(train), np.flatnonzero(val)))
    return out


def select(
    x: np.ndarray,
    z_target: np.ndarray,
    y: np.ndarray,
    latest: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    spec: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    """Strength with the lowest pooled inner-CV MAE (ug/m3). Returns (strength, mae by grid)."""
    grid = grid_for(spec["model"])
    if len(grid) == 1:
        return grid[0], {}
    splits = inner_splits(origins, horizon)
    if not splits:
        return grid[len(grid) // 2], {}
    errs = dict.fromkeys(grid, 0.0)
    count = 0
    for tr, va in splits:
        std = Standardiser(x[tr])
        ztr, zva = std.transform(x[tr]), std.transform(x[va])
        for g in grid:
            b, w = fit_linear(ztr, z_target[tr], spec["model"], g)
            pred = from_target(b + zva @ w, latest[va], spec["target"], spec["output"])
            errs[g] += float(np.abs(np.maximum(pred, 0.0) - y[va]).sum())
        count += len(va)
    mae = {g: e / count for g, e in errs.items()}
    return min(grid, key=lambda g: (mae[g], g)), mae


# ---- rows ----------------------------------------------------------------------------------


def row_vector(
    prep: Any, station: str, origin: datetime, target: datetime, feature_set: str
) -> tuple[np.ndarray, float] | None:
    """(feature vector with NaN for missing, latest reading) or None without a usable latest."""
    names = names_for(feature_set)
    feats = build_feature_set_names(prep, station, origin, target, names)
    latest = next((feats[n] for n in LATEST_LAGS if feats[n] is not None), None)
    if latest is None:
        return None
    return np.array([np.nan if feats[n] is None else feats[n] for n in names]), float(latest)


def build_feature_set_names(
    prep: Any, station: str, origin: datetime, target: datetime, names: Sequence[str]
) -> dict[str, float | None]:
    need = tuple(dict.fromkeys((*LATEST_LAGS, *names)))
    return build_features_v2(prep, station, origin, target, None, names=need)


def _train_rows(
    prep: Any, station: str, obs: Sequence[Observation], horizon: int, feature_set: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ls, os_ = [], [], [], []
    for label in obs:  # label at ts = target; origin = target - h; both <= refit_time
        origin = label.ts - horizon * HOUR
        key = (feature_set, station, origin, horizon)
        if key not in _cache:
            _cache[key] = row_vector(prep, station, origin, label.ts, feature_set)
        got = _cache[key]
        if got is None:
            continue
        xs.append(got[0])
        ys.append(label.value)
        ls.append(got[1])
        os_.append(origin.timestamp())
    p = len(names_for(feature_set))
    return (
        np.array(xs).reshape(-1, p),
        np.array(ys, dtype=float),
        np.array(ls, dtype=float),
        np.array(os_, dtype=float),
    )


# ---- model ---------------------------------------------------------------------------------


class Fitted:
    def __init__(self, std: Standardiser, b: float, w: np.ndarray, strength: float) -> None:
        self.std, self.b, self.w, self.strength = std, b, w, strength

    def predict_z(self, x: np.ndarray) -> float:
        return float(self.b + self.std.transform(x.reshape(1, -1))[0] @ self.w)


def _dump(variant: str) -> IO[str]:
    if variant not in _dumps:
        folder = Path(
            os.environ.get("PM25_KAN84_LOG_DIR", "~/pm25-experiments/cache/kan84")
        ).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        _dumps[variant] = (folder / f"{variant}.jsonl").open("w", buffering=1)
    return _dumps[variant]


def fit(train: TrainSnapshot, params: Mapping[str, Any] | None = None) -> Predictor:
    spec = {**DEFAULTS, **(params or {})}
    feature_set, scope = spec["feature_set"], spec["scope"]
    names = names_for(feature_set)
    stations = tuple(sorted(train.stations))
    by_station: dict[str, list[Observation]] = defaultdict(list)
    for o in train.observations:  # all have ts/available_at <= refit_time
        by_station[o.station_id].append(o)
    prep = prepare_series(by_station)
    onehot = {s: np.eye(len(stations))[i] for i, s in enumerate(stations)}

    models: dict[tuple[str, int], Fitted] = {}
    log: list[dict[str, Any]] = []
    for horizon in train.horizons:
        parts = {
            s: _train_rows(prep, s, by_station[s], horizon, feature_set)
            for s in stations
            if s in by_station
        }
        groups = {"*": list(parts)} if scope == "pooled" else {s: [s] for s in parts}
        for key, members in groups.items():
            x = np.vstack([parts[s][0] for s in members])
            if scope == "pooled":
                x = np.hstack(
                    [x, np.vstack([np.tile(onehot[s], (len(parts[s][1]), 1)) for s in members])]
                )
            y = np.concatenate([parts[s][1] for s in members])
            latest = np.concatenate([parts[s][2] for s in members])
            origins = np.concatenate([parts[s][3] for s in members])
            if len(y) < MIN_TRAIN_ROWS:
                continue
            zt = to_target(y, latest, spec["target"], spec["output"])
            strength, inner = select(x, zt, y, latest, origins, horizon, spec)
            std = Standardiser(x)
            b, w = fit_linear(std.transform(x), zt, spec["model"], strength)
            fitted = Fitted(std, b, w, strength)
            for s in members if scope == "per_station" else stations:
                models[s, horizon] = fitted
            cols = [*names, *(f"station_{s}" for s in stations)] if scope == "pooled" else names
            log.append(
                {
                    "refit_time": train.refit_time.isoformat(),
                    "h": horizon,
                    "key": key,
                    "n": len(y),
                    "strength": strength,
                    "inner_mae": {f"{g:.6g}": round(v, 4) for g, v in inner.items()},
                    "intercept": b,
                    "nonzero": int((np.abs(w) > 1e-12).sum()),
                    "coef_std": dict(zip(cols, map(float, w), strict=True)),
                    "scale": dict(zip(cols, map(float, std.scale), strict=True)),
                }
            )
    handle = _dump(spec["variant"])
    for rec in log:
        handle.write(json.dumps(rec) + "\n")

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        station = snapshot.station_id
        model = models.get((station, horizon))
        if model is None or station not in stations:
            return None
        origin = snapshot.prediction_time
        recent = [
            o for o in snapshot.observations[-LOOKBACK_H:] if o.ts >= origin - LOOKBACK_H * HOUR
        ]
        got = row_vector(
            prepare_series({station: recent}),
            station,
            origin,
            origin + horizon * HOUR,
            feature_set,
        )
        if got is None:
            return None
        x, latest = got
        if scope == "pooled":
            x = np.concatenate([x, onehot[station]])
        z = model.predict_z(x)
        return float(
            from_target(np.array([z]), np.array([latest]), spec["target"], spec["output"])[0]
        )

    return predict
