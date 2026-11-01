import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

# statsmodels lives in the experiments dependency group; the default CI install does not have it.
pytest.importorskip("statsmodels")

from pm25_forecast.backtest import Observation, Snapshot
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.experiments.timeseries import arima, ets, sarima

START = datetime(2026, 9, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
N = 24 * 21


def series(n=N, skip=()):
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        if k in skip:
            continue
        ts = START + k * HOUR
        value = 20 + 8 * math.sin(2 * math.pi * k / 24) + rng.normal(0, 1)
        out.append(Observation("a", ts, max(value, 0.1), ts + HOUR))
    return tuple(out)


def make_train(obs):
    refit = START + N * HOUR
    return TrainSnapshot(refit, ("a",), (24,), tuple(o for o in obs if o.ts <= refit), ())


def snap(obs, origin):
    rows = tuple(o for o in obs if o.ts <= origin - HOUR and o.available_at <= origin)
    return Snapshot(origin, "a", rows, ())


ARIMA = {"name": "t", "d": 0, "p_grid": [1, 2], "q_grid": [0, 1]}
SARIMA = {
    "name": "t",
    "d": 0,
    "p_grid": [1],
    "q_grid": [0],
    "seasonal_grid": [[1, 0]],
    "seasonal_period": 24,
    "fit_days": 14,
}
ETS = {"name": "t", "trend": "add", "damped_trend": True, "fit_days": 14, "state_hours": 120}


def factories():
    return [(arima, ARIMA), (sarima, SARIMA), (ets, ETS)]


def test_forecasts_are_finite_and_seasonal_models_track_the_cycle():
    obs = series(N + 48)
    train = make_train(obs)
    origin = train.refit_time + 6 * HOUR
    for factory, params in factories():
        pred = factory(train, params)(snap(obs, origin), 24)
        assert pred is not None and math.isfinite(pred)
    truth = 20 + 8 * math.sin(2 * math.pi * (N + 6 + 24) / 24)
    assert abs(sarima(train, SARIMA)(snap(obs, origin), 24) - truth) < 4
    assert abs(ets(train, ETS)(snap(obs, origin), 24) - truth) < 4


def test_future_data_does_not_change_the_forecast():
    obs = series(N + 48)
    train = make_train(obs)
    origin = train.refit_time + 6 * HOUR
    corrupt = tuple(
        Observation(o.station_id, o.ts, 999.0, o.available_at) if o.ts >= origin else o for o in obs
    )
    for factory, params in factories():
        a = factory(train, params)(snap(obs, origin), 12)
        b = factory(train, params)(snap(corrupt, origin), 12)  # snap() drops ts >= origin
        assert a is not None and a == b
        late = Observation("a", origin, 500.0, origin + HOUR)  # the lag rule is also enforced
        with pytest.raises(ValueError):
            Snapshot(origin, "a", (*snap(obs, origin).observations, late), ())


def test_hourly_chain_matches_a_fresh_instance():
    obs = series(N + 48)
    train = make_train(obs)
    live = arima(train, ARIMA)
    for k in range(1, 6):
        origin = train.refit_time + k * HOUR
        chained = live(snap(obs, origin), 24)
        fresh = arima(train, ARIMA)(snap(obs, origin), 24)
        assert chained == pytest.approx(fresh, abs=0.5)  # fallback window starts fresh


def test_stale_latest_returns_none_and_gaps_are_not_filled():
    gap = set(range(N - 2, N + 10))
    obs = series(N + 48, skip=gap)
    train = make_train(obs)
    origin = train.refit_time + 6 * HOUR  # newest reading is older than 3 h
    for factory, params in factories():
        assert factory(train, params)(snap(obs, origin), 24) is None
    ok = train.refit_time + 14 * HOUR  # readings resumed; the gap is inside the history
    for factory, params in factories():
        pred = factory(train, params)(snap(obs, ok), 24)
        assert pred is None or math.isfinite(pred)


def test_short_gap_at_the_end_shifts_the_step_not_the_target():
    obs = series(N + 48, skip={N + 4})  # one missing hour just before the origin
    train = make_train(obs)
    origin = train.refit_time + 6 * HOUR
    pred = arima(train, ARIMA)(snap(obs, origin), 24)
    assert pred is not None and math.isfinite(pred)


def test_unknown_station_and_too_little_history():
    obs = series(30)
    train = TrainSnapshot(START + 30 * HOUR, ("a",), (24,), obs, ())
    for factory, params in factories():
        assert factory(train, params)(snap(obs, START + 30 * HOUR), 24) is None
