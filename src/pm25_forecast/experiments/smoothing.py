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
