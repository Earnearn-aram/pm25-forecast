from datetime import UTC, datetime, timedelta

import pytest

from pm25_forecast.backtest import Observation, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.experiments.smoothing import beta_smoother, climatology, ewma, moving_average
from pm25_forecast.features.experiment_smoothing import (
    exponential_mean,
    rolling_values,
    same_hour_mean,
)

T = datetime(2026, 9, 10, 12, tzinfo=UTC)


def ob(hours, value, available=None):
    ts = T - timedelta(hours=hours)
    return Observation("a", ts, value, available or ts + timedelta(hours=1))


def test_features_respect_lag_and_availability():
    rows = (ob(25, 10), ob(24, 20), ob(3, 30), ob(1, 40, T + timedelta(hours=1)), ob(0, 100))
    assert [v for _, v in rolling_values(rows, T, 24)] == [20, 30]
    assert same_hour_mean(rows, T, T + timedelta(hours=24), 2) == 20
    assert exponential_mean(rows, T, 2) < 30


def test_factories_refuse_stale_latest_and_other_stations():
    train = TrainSnapshot(T, ("a",), (24,), (), ())
    stale = Snapshot(T, "a", (ob(4, 10),), ())
    other = Snapshot(T, "b", (ob(1, 10),), ())
    factories = [
        (moving_average, {"window_hours": 24}),
        (ewma, {"half_life_hours": 2}),
        (beta_smoother, {"beta": 0.5}),
        (climatology, {"days": 7, "persistence_weight": 0.5}),
    ]
    for factory, params in factories:
        predictor = factory(train, params)
        assert predictor(stale, 24) is None
        assert predictor(other, 24) is None


def test_beta_half_matches_serving_smoother():
    train = TrainSnapshot(T, ("a",), (24,), (), ())
    snap = Snapshot(T, "a", (ob(24, 10), ob(2, 20), ob(1, 30)), ())
    assert beta_smoother(train, {"beta": 0.5})(snap, 24) == pytest.approx(25)
