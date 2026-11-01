"""KAN-88 (E11) model factories: ridge and LightGBM on ``fs_v2_core`` with and without weather.

Factories for ``experiments.evaluate --model pm25_forecast.experiments.kan88_models:<name>``:
``ridge_base``, ``ridge_wx``, ``lgbm_base``, ``lgbm_wx`` (primary, weather = ``previous_day1``) and
``ridge_wx_d2``, ``lgbm_wx_d2`` (sensitivity, ``previous_day2`` = strictly available at the origin).

Identical rows: every variant, with or without weather, returns None when the weather inputs for
the target hour are missing, so the scored rows are the same. One model per station and horizon.
Base features come from ``features.library_v2`` and weather from ``features.weather`` (no private
feature code). Ridge fills missing features with the training mean; LightGBM takes NaN.

Weather cache: ``$PM25_WEATHER_CACHE`` or ``~/pm25-experiments/cache/weather`` (filled by
``kan88_fetch_weather``). Predictions are also appended to ``$PM25_KAN88_PRED_DIR/<variant>.csv``
(default ``~/pm25-experiments/cache/kan88/preds``) for the paired analysis in ``kan88_analyze``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Any

import lightgbm as lgb
import numpy as np

from pm25_forecast.backtest import Observation, Predictor, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features.library_v2 import build_feature_set, feature_set_names, prepare_series
from pm25_forecast.features.weather import (
    WEATHER_FEATURE_NAMES,
    WeatherSeries,
    load_weather_cache,
    weather_features,
)

HOUR = timedelta(hours=1)
BASE_SET = "fs_v2_core"
LOOKBACK_H = 80  # fs_v2_core reads at most 72 h before the origin
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
MIN_TRAIN_ROWS = 200

_weather: WeatherSeries | None = None
_dumps: dict[str, IO[str]] = {}


def get_weather() -> WeatherSeries:
    global _weather
    if _weather is None:
        _weather = load_weather_cache(
            os.environ.get("PM25_WEATHER_CACHE", "~/pm25-experiments/cache/weather")
        )
    return _weather


def _dump(variant: str) -> IO[str]:
    if variant not in _dumps:
        folder = Path(
            os.environ.get("PM25_KAN88_PRED_DIR", "~/pm25-experiments/cache/kan88/preds")
        ).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        handle = (folder / f"{variant}.csv").open("w", buffering=1)
        handle.write("station,origin,h,pred\n")
        _dumps[variant] = handle
    return _dumps[variant]


def row_features(
    prep: Any,
    wx: WeatherSeries,
    station: str,
    origin: datetime,
    target: datetime,
    *,
    weather: bool,
    lead: str,
) -> dict[str, float | None] | None:
    """Base (+ weather) features, or None when the matched-rows rule excludes the row."""
    feats = build_feature_set(BASE_SET, prep, station, origin, target)
    if feats["pm25_mean_3h"] is None:  # no reading in the last 3 h: never impute
        return None
    wf = weather_features(wx, station, origin, target, lead)
    if any(wf[n] is None for n in REQUIRED_WX_NAMES):  # same rows with and without weather
        return None
    if weather:
        feats.update(wf)
    return feats


REQUIRED_WX_NAMES = (
    "wx_wind_speed",
    "wx_wind_dir_sin",
    "wx_rh",
    "wx_temp",
    "wx_precip",
    "wx_pressure",
    "wx_cloud",
    "wx_shortwave",
)


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
        z = (filled - self.mean) / self.scale
        z1 = np.hstack([np.ones((len(z), 1)), z])
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
    train: TrainSnapshot,
    params: Mapping[str, Any] | None,
    *,
    family: str,
    weather: bool,
    lead: str,
    variant: str,
) -> Predictor:
    wx = get_weather()
    names = feature_set_names(BASE_SET) + (WEATHER_FEATURE_NAMES if weather else ())
    by_station: dict[str, list[Observation]] = defaultdict(list)
    for o in train.observations:  # all have ts/available_at <= refit_time
        by_station[o.station_id].append(o)

    models: dict[tuple[str, int], _Ridge | _Lgbm] = {}
    for station, obs in by_station.items():
        prep = prepare_series({station: obs})
        for horizon in train.horizons:
            xs, ys = [], []
            for label in obs:  # label at ts = target; origin = target - h (both <= refit_time)
                origin = label.ts - horizon * HOUR
                feats = row_features(
                    prep, wx, station, origin, label.ts, weather=weather, lead=lead
                )
                if feats is not None:
                    xs.append(_vector(feats, names))
                    ys.append(label.value)
            if len(xs) < MIN_TRAIN_ROWS:
                continue
            x, y = np.vstack(xs), np.array(ys)
            if family == "ridge":
                models[station, horizon] = _Ridge(
                    float({**DEFAULT_RIDGE, **(params or {})}["alpha"])
                ).fit(x, y)
            else:
                models[station, horizon] = _Lgbm(params or {}).fit(x, y)

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        model = models.get((snapshot.station_id, horizon))
        if model is None:
            return None
        origin = snapshot.prediction_time
        recent = [o for o in snapshot.observations if o.ts >= origin - LOOKBACK_H * HOUR]
        prep = prepare_series({snapshot.station_id: recent})
        feats = row_features(
            prep,
            wx,
            snapshot.station_id,
            origin,
            origin + horizon * HOUR,
            weather=weather,
            lead=lead,
        )
        if feats is None:
            return None
        pred = model.predict(_vector(feats, names))
        _dump(variant).write(f"{snapshot.station_id},{origin.isoformat()},{horizon},{pred!r}\n")
        return pred

    return predict


def _factory(family: str, weather: bool, lead: str, variant: str):
    def fit(train: TrainSnapshot, params: Mapping[str, Any] | None = None) -> Predictor:
        return _fit(train, params, family=family, weather=weather, lead=lead, variant=variant)

    fit.__name__ = variant
    return fit


ridge_base = _factory("ridge", False, "previous_day1", "ridge_base")
ridge_wx = _factory("ridge", True, "previous_day1", "ridge_wx")
lgbm_base = _factory("lgbm", False, "previous_day1", "lgbm_base")
lgbm_wx = _factory("lgbm", True, "previous_day1", "lgbm_wx")
ridge_wx_d2 = _factory("ridge", True, "previous_day2", "ridge_wx_d2")
lgbm_wx_d2 = _factory("lgbm", True, "previous_day2", "lgbm_wx_d2")
