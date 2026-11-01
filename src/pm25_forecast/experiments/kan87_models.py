"""KAN-87 (E10) model factories: ridge and LightGBM on ``fs_v2_core`` with and without CAMS.

Factories for ``experiments.evaluate --model pm25_forecast.experiments.kan87_models:<name>``:
``ridge_base``, ``ridge_cams``, ``lgbm_base``, ``lgbm_cams``.

Identical rows: every variant returns None unless the CAMS features are all present, so the
with/without pairs are scored on the same rows. Training uses only rows with CAMS too, so with
no stored vintages in the training window the model has no rows and forecasts nothing (never an
invented value). One model per station and horizon. Features come from ``features.library_v2``
and ``features.cams``. Ridge fills missing features with the training mean; LightGBM takes NaN.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np

from pm25_forecast.backtest import Observation, Predictor, Snapshot, Vintage
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features.cams import (
    CAMS_FEATURE_NAMES,
    PreparedCams,
    build_cams_features,
    prepare_cams,
)
from pm25_forecast.features.library_v2 import (
    PreparedSeries,
    build_feature_set,
    feature_set_names,
    prepare_series,
)

HOUR = timedelta(hours=1)
BASE_SET = "fs_v2_core"
LOOKBACK_H = 80  # fs_v2_core reads <= 72 h; the CAMS error window is 72 h
MIN_TRAIN_ROWS = 200
DEFAULT_RIDGE = {"alpha": 30.0}
DEFAULT_LGBM = {
    "objective": "l1",
    "num_leaves": 7,
    "learning_rate": 0.05,
    "n_estimators": 150,
    "min_child_samples": 50,
    "feature_fraction": 0.9,
    "seed": 20261101,
    "n_seeds": 5,
}


def row_features(
    prep: PreparedSeries,
    cams: PreparedCams,
    station: str,
    origin: datetime,
    target: datetime,
    *,
    with_cams: bool,
) -> dict[str, float | None] | None:
    """Base (+ CAMS) features, or None when the matched-rows rule excludes the row."""
    feats = build_feature_set(BASE_SET, prep, station, origin, target)
    if feats["pm25_mean_3h"] is None:  # no reading in the last 3 h: never impute
        return None
    cf = build_cams_features(prep, cams, station, origin, target)
    if any(cf[n] is None for n in CAMS_FEATURE_NAMES):  # same rows with and without CAMS
        return None
    if with_cams:
        feats.update(cf)
    return feats


def _vector(feats: Mapping[str, float | None], names: tuple[str, ...]) -> np.ndarray:
    return np.array([np.nan if feats[n] is None else feats[n] for n in names], dtype=np.float64)


class _Ridge:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> _Ridge:
        self.mean = np.nanmean(x, axis=0)
        self.mean = np.where(np.isnan(self.mean), 0.0, self.mean)
        filled = np.where(np.isnan(x), self.mean, x)
        self.scale = filled.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        z1 = np.hstack([np.ones((len(x), 1)), (filled - self.mean) / self.scale])
        penalty = self.alpha * np.eye(z1.shape[1])
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(z1.T @ z1 + penalty, z1.T @ y)
        return self

    def predict(self, x: np.ndarray) -> float:
        filled = np.where(np.isnan(x), self.mean, x)
        return float(self.coef[0] + ((filled - self.mean) / self.scale) @ self.coef[1:])


class _Lgbm:
    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = {**DEFAULT_LGBM, **params}

    def fit(self, x: np.ndarray, y: np.ndarray) -> _Lgbm:
        p = self.params
        self.models = [
            lgb.LGBMRegressor(
                objective=p["objective"],
                num_leaves=int(p["num_leaves"]),
                learning_rate=float(p["learning_rate"]),
                n_estimators=int(p["n_estimators"]),
                min_child_samples=int(p["min_child_samples"]),
                colsample_bytree=float(p["feature_fraction"]),
                random_state=int(p["seed"]) + i,
                deterministic=True,
                force_row_wise=True,
                n_jobs=1,
                verbose=-1,
            ).fit(x, y)
            for i in range(int(p["n_seeds"]))
        ]
        return self

    def predict(self, x: np.ndarray) -> float:
        return float(np.mean([m.predict(x.reshape(1, -1))[0] for m in self.models]))


def _fit(
    train: TrainSnapshot, params: Mapping[str, Any] | None, *, family: str, with_cams: bool
) -> Predictor:
    names = feature_set_names(BASE_SET) + (CAMS_FEATURE_NAMES if with_cams else ())
    by_station: dict[str, list[Observation]] = defaultdict(list)
    for o in train.observations:  # all have ts/available_at <= refit_time
        by_station[o.station_id].append(o)
    vin_by_station: dict[str, list[Vintage]] = defaultdict(list)
    for v in train.vintages:  # all have issue_time/available_at <= refit_time
        vin_by_station[v.station_id].append(v)

    models: dict[tuple[str, int], _Ridge | _Lgbm] = {}
    for station, obs in by_station.items():
        prep = prepare_series({station: obs})
        cams = prepare_cams({station: vin_by_station.get(station, [])})
        if not cams:
            continue  # no vintages: no matched rows, no forecast
        for horizon in train.horizons:
            xs, ys = [], []
            for label in obs:  # label at ts = target; origin = target - h (both <= refit_time)
                feats = row_features(
                    prep, cams, station, label.ts - horizon * HOUR, label.ts, with_cams=with_cams
                )
                if feats is not None:
                    xs.append(_vector(feats, names))
                    ys.append(label.value)
            if len(xs) < MIN_TRAIN_ROWS:
                continue
            x, y = np.vstack(xs), np.array(ys)
            if family == "ridge":
                alpha = float({**DEFAULT_RIDGE, **(params or {})}["alpha"])
                models[station, horizon] = _Ridge(alpha).fit(x, y)
            else:
                models[station, horizon] = _Lgbm(params or {}).fit(x, y)

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        model = models.get((snapshot.station_id, horizon))
        if model is None:
            return None
        origin = snapshot.prediction_time
        recent = [o for o in snapshot.observations if o.ts >= origin - LOOKBACK_H * HOUR]
        prep = prepare_series({snapshot.station_id: recent})
        cams = prepare_cams({snapshot.station_id: list(snapshot.vintages)})
        feats = row_features(
            prep, cams, snapshot.station_id, origin, origin + horizon * HOUR, with_cams=with_cams
        )
        return None if feats is None else model.predict(_vector(feats, names))

    return predict


def _factory(family: str, with_cams: bool, variant: str):
    def fit(train: TrainSnapshot, params: Mapping[str, Any] | None = None) -> Predictor:
        return _fit(train, params, family=family, with_cams=with_cams)

    fit.__name__ = variant
    return fit


ridge_base = _factory("ridge", False, "ridge_base")
ridge_cams = _factory("ridge", True, "ridge_cams")
lgbm_base = _factory("lgbm", False, "lgbm_base")
lgbm_cams = _factory("lgbm", True, "lgbm_cams")
