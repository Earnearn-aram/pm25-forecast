"""Parameterized dev-only smoothing candidates for KAN-83."""

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from pm25_forecast.backtest import Predictor, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features import build_features
from pm25_forecast.features.experiment_smoothing import rolling_values


def moving_average(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    """Predict each station from its recent readings without imputing stale input."""
    hours = int(params["window_hours"])
    stations = set(train.stations)

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        if snapshot.station_id not in stations:
            return None
        recent = snapshot.observations[-max(hours, 3) :]
        latest = build_features(
            recent,
            (),
            snapshot.prediction_time,
            snapshot.prediction_time + timedelta(hours=horizon),
        )["pm25_latest"]
        if latest is None:
            return None
        values = rolling_values(recent, snapshot.prediction_time, hours)
        return sum(v for _, v in values) / len(values) if values else None

    return predict


def ewma(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    """Exponentially weight available station history by elapsed hours."""
    from pm25_forecast.features.experiment_smoothing import exponential_mean

    half_life = float(params["half_life_hours"])
    stations = set(train.stations)
    last_snapshot = None
    last_result = None

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        nonlocal last_snapshot, last_result
        if snapshot is last_snapshot:
            return last_result
        last_snapshot = snapshot
        latest = build_features(
            snapshot.observations[-3:],
            (),
            snapshot.prediction_time,
            snapshot.prediction_time + timedelta(hours=horizon),
        )["pm25_latest"]
        last_result = (
            exponential_mean(snapshot.observations, snapshot.prediction_time, half_life)
            if snapshot.station_id in stations and latest is not None
            else None
        )
        return last_result

    return predict


def beta_smoother(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    """Blend latest and serving mean24 with a fixed preregistered coefficient."""
    from pm25_forecast.features import mean_24h

    beta = float(params["beta"])
    stations = set(train.stations)

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        if snapshot.station_id not in stations:
            return None
        recent = snapshot.observations[-24:]
        latest = build_features(
            recent,
            (),
            snapshot.prediction_time,
            snapshot.prediction_time + timedelta(hours=horizon),
        )["pm25_latest"]
        if latest is None:
            return None
        mean = mean_24h(recent, snapshot.prediction_time)
        return latest if mean is None else latest + beta * (mean - latest)

    return predict


def climatology(train: TrainSnapshot, params: Mapping[str, Any]) -> Predictor:
    """Station-specific same-hour mean, optionally blended with persistence."""
    from pm25_forecast.features.experiment_smoothing import same_hour_mean

    days = int(params["days"])
    weight = float(params["persistence_weight"])
    stations = set(train.stations)

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        if snapshot.station_id not in stations:
            return None
        origin = snapshot.prediction_time
        target = origin + timedelta(hours=horizon)
        recent = snapshot.observations[-days * 24 :]
        latest = build_features(recent[-3:], (), origin, target)["pm25_latest"]
        if latest is None:
            return None
        mean = same_hour_mean(recent, origin, target, days)
        return None if mean is None else weight * latest + (1 - weight) * mean

    return predict
