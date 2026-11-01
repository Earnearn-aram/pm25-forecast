"""KAN-88: weather feature group (as-of rule, sin/cos, 24 h windows, tolerant parsing)."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pm25_forecast.experiments.kan88_models import _Ridge
from pm25_forecast.features.weather import (
    WEATHER_FEATURE_NAMES,
    WeatherSeries,
    weather_features,
)

FIXTURE = Path(__file__).parent / "fixtures" / "open_meteo_previous_runs" / "recorded_59t.json"
T0 = datetime(2026, 10, 20, tzinfo=UTC)  # first fixture hour
HOUR = timedelta(hours=1)


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def series(payload):
    s = WeatherSeries()
    s.add("59t", payload)
    return s


def test_all_keys_and_direction_is_sin_cos(series, payload):
    origin, valid = T0 + 24 * HOUR, T0 + 48 * HOUR  # +24 h, nominal issue time == origin
    feats = weather_features(series, "59t", origin, valid)
    assert tuple(feats) == WEATHER_FEATURE_NAMES
    i = payload["hourly"]["time"].index(int(valid.timestamp()))
    direction = payload["hourly"]["wind_direction_10m_previous_day1"][i]
    assert feats["wx_wind_dir_sin"] == pytest.approx(math.sin(math.radians(direction)))
    assert feats["wx_wind_dir_cos"] == pytest.approx(math.cos(math.radians(direction)))
    assert feats["wx_temp"] == payload["hourly"]["temperature_2m_previous_day1"][i]


def test_nominal_issue_time_after_origin_is_not_used(series):
    valid = T0 + 48 * HOUR
    early = weather_features(series, "59t", valid - 25 * HOUR, valid)  # issued after the origin
    assert all(v is None for v in early.values())
    # previous_day2 for the same hour was issued 24 h before the origin: available
    late = weather_features(series, "59t", valid - 24 * HOUR, valid, lead="previous_day2")
    assert late["wx_temp"] is not None
    # ... but not one hour earlier than that
    assert (
        weather_features(series, "59t", valid - 49 * HOUR, valid, "previous_day2")["wx_temp"]
        is None
    )


def test_window_features_need_all_24_hours(series, payload):
    valid = T0 + 48 * HOUR
    feats = weather_features(series, "59t", valid - 24 * HOUR, valid)
    ts = payload["hourly"]["time"]
    i = ts.index(int(valid.timestamp()))
    precip = payload["hourly"]["precipitation_previous_day1"][i - 23 : i + 1]
    assert feats["wx_precip_sum_24h"] == pytest.approx(sum(precip))
    # the first 23 fixture hours have no full window
    assert (
        weather_features(series, "59t", T0 + 30 * HOUR, T0 + 10 * HOUR)["wx_precip_sum_24h"] is None
    )


def test_unknown_station_and_bad_payload(series):
    assert all(v is None for v in weather_features(series, "nope", T0, T0 + 24 * HOUR).values())
    with pytest.raises(ValueError):
        WeatherSeries().add("x", {"hourly": {"temperature_2m_previous_day1": [1.0]}})
    with pytest.raises(ValueError):
        WeatherSeries().add("x", {"nonsense": 1})


def test_missing_column_gives_none_not_error(payload):
    del payload["hourly"]["cloud_cover_previous_day1"]
    s = WeatherSeries()
    s.add("59t", payload)
    feats = weather_features(s, "59t", T0 + 24 * HOUR, T0 + 48 * HOUR)
    assert feats["wx_cloud"] is None and feats["wx_temp"] is not None


def test_boundary_layer_height_is_null_upstream_and_unused(payload):
    assert all(v is None for v in payload["hourly"]["boundary_layer_height_previous_day1"])
    assert not any("boundary" in n for n in WEATHER_FEATURE_NAMES)


def test_ridge_recovers_linear_signal_and_fills_nan():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = 2 * x[:, 0] + 5
    model = _Ridge(alpha=0.1).fit(x, y)
    assert model.predict(np.array([1.0, 0.0, 0.0])) == pytest.approx(7.0, abs=0.2)
    assert math.isfinite(model.predict(np.array([np.nan, 0.0, 0.0])))
