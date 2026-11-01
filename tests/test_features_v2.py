import math
import time
from datetime import UTC, datetime, timedelta

import pytest

from pm25_forecast.backtest import Observation
from pm25_forecast.features import FEATURE_NAMES, SMOOTHER_FEATURE_NAMES, build_features
from pm25_forecast.features import library_v2 as v2

T = datetime(2026, 10, 28, 12, tzinfo=UTC)  # Wednesday 19:00 Bangkok
TARGET = T + timedelta(hours=24)
H = timedelta(hours=1)
LOCS = {"a": (13.75, 100.50), "b": (13.80, 100.55), "c": (14.50, 101.00)}


def value_at(k: int) -> float:
    return 10.0 + k  # reading k hours before T


def series(station="a", ks=range(1, 80), offset=0.0):
    return [Observation(station, T - k * H, value_at(k) + offset, T - k * H + H) for k in ks]


def data(**kw):
    return {"a": series("a"), "b": series("b", offset=10.0), "c": series("c", offset=100.0)} | kw


def build(obs=None, **kw):
    return v2.build_features_v2(data() if obs is None else obs, "a", T, TARGET, LOCS, **kw)


def test_v1_contract_unchanged():
    assert FEATURE_NAMES == ("pm25_latest", "pm25_same_hour_24h", "cams_pm25")
    assert SMOOTHER_FEATURE_NAMES == ("pm25_latest", "pm25_mean_24h")
    target = T + 22 * H
    obs = [(T - H, 20.0), (target - 24 * H, 18.0)]
    assert build_features(obs, [], T, target) == {
        "pm25_latest": 20.0,
        "pm25_same_hour_24h": 18.0,
        "cams_pm25": None,
    }


def test_lags_are_exact_hour_slots_and_gap_gives_none():
    obs = data(a=series("a", ks=[k for k in range(1, 80) if k != 6]))
    f = build(obs)
    for k in v2.LAGS_H:
        assert f[f"pm25_lag_{k}h"] == (None if k == 6 else value_at(k))
    assert f["missing_lag_6h"] == 1.0
    assert f["missing_lag_1h"] == 0.0


def test_rolling_stats_over_window():
    f = build()
    vals = [value_at(k) for k in range(1, 4)]  # 11, 12, 13
    assert f["pm25_mean_3h"] == pytest.approx(12.0)
    assert f["pm25_std_3h"] == pytest.approx(1.0)
    assert (f["pm25_min_3h"], f["pm25_max_3h"], f["pm25_median_3h"]) == (11.0, 13.0, 12.0)
    assert f["pm25_mean_72h"] == pytest.approx(sum(value_at(k) for k in range(1, 73)) / 72)
    assert vals  # window excludes t itself and older than t-3h
    assert f["n_valid_3h"] == 3.0
    assert f["n_valid_72h"] == 72.0


def test_rolling_with_gaps_and_sparse_windows():
    obs = data(a=series("a", ks=[2, 30]))
    f = build(obs)
    assert f["n_valid_3h"] == 1.0 and f["n_valid_6h"] == 1.0 and f["n_valid_24h"] == 1.0
    assert f["n_valid_72h"] == 2.0
    assert f["pm25_mean_3h"] == value_at(2)
    assert f["pm25_std_3h"] is None  # one value: no sample std
    assert f["pm25_std_72h"] is not None
    empty = build(data(a=[]))
    assert empty["pm25_mean_3h"] is None and empty["n_valid_3h"] == 0.0
    assert empty["pm25_ewma_24h"] is None and empty["pm25_diff_1h"] is None


def test_ewma_diffs():
    f = build()
    w = [(1 - v2.EWMA_ALPHA) ** (k - 1) for k in range(1, 25)]
    expected = sum(value_at(k) * w[k - 1] for k in range(1, 25)) / sum(w)
    assert f["pm25_ewma_24h"] == pytest.approx(expected)
    assert f["pm25_diff_1h"] == value_at(1) - value_at(2) == -1.0
    assert f["pm25_diff_24h"] == value_at(1) - value_at(25)
    gap = build(data(a=series("a", ks=[k for k in range(1, 80) if k != 2])))
    assert gap["pm25_diff_1h"] is None and gap["pm25_diff_24h"] is not None


def test_calendar_bangkok_time_cyclic_weekend_holiday():
    f = v2._calendar(datetime(2026, 10, 28, 12, tzinfo=UTC))  # Wed 19:00 local
    assert f["hour_sin"] == pytest.approx(math.sin(2 * math.pi * 19 / 24))
    assert f["dow_cos"] == pytest.approx(math.cos(2 * math.pi * 2 / 7))
    assert (f["is_weekend"], f["is_thai_holiday"]) == (0.0, 0.0)
    # 2026-10-23 17:30 UTC is Sat 2026-10-24 00:30 local: weekend, not the Oct 23 holiday
    sat = v2._calendar(datetime(2026, 10, 23, 17, 30, tzinfo=UTC))
    assert (sat["is_weekend"], sat["is_thai_holiday"]) == (1.0, 0.0)
    # 2026-10-22 18:00 UTC is Fri 2026-10-23 01:00 local: Chulalongkorn Day
    assert v2._calendar(datetime(2026, 10, 22, 18, tzinfo=UTC))["is_thai_holiday"] == 1.0
    # Outside the static table: unknown, not a silent 0
    assert v2._calendar(datetime(2027, 3, 3, 5, tzinfo=UTC))["is_thai_holiday"] is None


def test_cross_station_mean_deviation_and_nearest():
    f = build()
    assert f["bkk_mean"] == pytest.approx((11 + 21 + 111) / 3)
    assert f["dev_from_bkk_mean"] == pytest.approx(11 - (11 + 21 + 111) / 3)
    assert f["nearest_station_pm25"] == 21.0  # b is closest to a
    # b has no reading: nearest falls to the next station that does
    f = build(data(b=[]))
    assert f["nearest_station_pm25"] == 111.0
    assert f["bkk_mean"] == pytest.approx((11 + 111) / 2)
    # No locations: no neighbour; own station silent: no deviation
    assert v2.build_features_v2(data(), "a", T, TARGET)["nearest_station_pm25"] is None
    assert build(data(a=[]))["dev_from_bkk_mean"] is None


def test_feature_set_registry():
    assert set(v2.FEATURE_SETS) >= {"fs_v2_core", "fs_v2_full"}
    assert set(v2.FEATURE_SETS["fs_v2_core"]) < set(v2.FEATURE_SETS["fs_v2_full"])
    assert v2.feature_set_names("fs_v2_full") == v2.ALL_FEATURE_NAMES
    assert len(set(v2.ALL_FEATURE_NAMES)) == len(v2.ALL_FEATURE_NAMES)
    for name in v2.FEATURE_SETS:
        got = v2.build_feature_set(name, data(), "a", T, TARGET, LOCS)
        assert tuple(got) == v2.feature_set_names(name)
    with pytest.raises(KeyError, match="unknown feature set"):
        v2.feature_set_names("nope")
    assert tuple(build()) == v2.ALL_FEATURE_NAMES


def test_training_and_serving_shapes_agree_with_available_at():
    d = data()
    as_tuples = {s: [(o.ts, o.value, o.available_at) for o in rows] for s, rows in d.items()}
    as_dicts = {
        s: [{"ts": o.ts, "pm25": o.value, "available_at": o.available_at} for o in rows]
        for s, rows in d.items()
    }
    assert build(as_tuples) == build(as_dicts) == build(d)
    # Prepared once (training) and raw records (serving) give identical features.
    assert build(v2.prepare_series(d)) == build(d)
    # available_at must matter: delaying every row past t removes them all.
    late = {s: [(o.ts, o.value, T + H) for o in rows] for s, rows in d.items()}
    assert build(late)["n_valid_72h"] == 0.0


def test_ewma_with_gap_renormalises_over_present_values():
    gap = build(data(a=series("a", ks=[k for k in range(1, 80) if k not in (2, 3)])))
    ks = [k for k in range(1, 25) if k not in (2, 3)]
    w = {k: (1 - v2.EWMA_ALPHA) ** (k - 1) for k in ks}
    expected = sum(value_at(k) * w[k] for k in ks) / sum(w.values())
    assert gap["pm25_ewma_24h"] == pytest.approx(expected)


def test_duplicate_hour_takes_latest_available_at_not_row_order():
    rows = [
        Observation("a", T - H, 1.0, T - H + 2 * H // 2),  # available at T: visible
        Observation("a", T - H, 2.0, T - H + 30 * H // 60),  # earlier availability
        Observation("a", T - H, 3.0, T + H),  # not yet available at T
    ]
    for order in (rows, rows[::-1]):
        assert build(data(a=order))["pm25_lag_1h"] == 1.0


def test_minute_timestamps_floor_to_the_hour_and_origin_must_be_on_the_hour():
    obs = data(a=[Observation("a", T - H + timedelta(minutes=30), 42.0, T - H)])
    assert build(obs)["pm25_lag_1h"] == 42.0
    with pytest.raises(ValueError, match="on the hour"):
        v2.build_features_v2(data(), "a", T + timedelta(minutes=5), TARGET)


def test_only_requested_features_are_computed():
    core = v2.build_feature_set("fs_v2_core", data(), "a", T, TARGET)
    assert "bkk_mean" not in core and "pm25_std_3h" not in core
    assert v2.build_features_v2(data(), "a", T, TARGET, names=["bkk_mean"]) == {
        "bkk_mean": pytest.approx((11 + 21 + 111) / 3)
    }
    with pytest.raises(KeyError):
        v2.build_features_v2(data(), "a", T, TARGET, names=["nope"])


def test_walk_forward_scale_is_fast_with_prepared_series():
    stations = [f"s{i}" for i in range(5)]
    start = T - timedelta(hours=3000)
    big = {
        s: [Observation(s, start + i * H, 20.0 + i % 17, start + (i + 1) * H) for i in range(3000)]
        for s in stations
    }
    locs = {s: (13.7 + 0.01 * i, 100.5) for i, s in enumerate(stations)}
    began = time.perf_counter()
    prep = v2.prepare_series(big)
    for n in range(500):
        origin = T - n * H
        for s in stations:
            v2.build_feature_set("fs_v2_full", prep, s, origin, origin + 24 * H, locs)
    assert time.perf_counter() - began < 5.0


def test_as_of_guard_nothing_after_t_minus_1h_is_read():
    base = build()
    # Poison: readings at t, t+1h.. and with ts in the past but available only after t.
    poison = [Observation(s, T + k * H, 9999.0, T + k * H) for s in "abc" for k in range(0, 80)]
    late = [Observation(s, T - k * H, 8888.0, T + H) for s in "abc" for k in range(1, 80)]
    poisoned = {s: [*data()[s], *[p for p in poison if p.station_id == s]] for s in "abc"}
    assert build(poisoned) == base
    # available_at > t: the same ts rows, if they would replace, must be ignored
    replaced = {s: [*data()[s], *[p for p in late if p.station_id == s]] for s in "abc"}
    assert build(replaced) == base
    # Truncating all data at t - 1h changes nothing, for every feature
    cut = {s: [o for o in rows if o.ts <= T - H] for s, rows in data().items()}
    assert build(cut) == base


def test_as_of_guard_reading_at_t_is_not_a_lag():
    obs = data(a=[Observation("a", T, 500.0, T), *series("a")])
    f = build(obs)
    assert f["pm25_lag_1h"] == value_at(1)
    assert f["pm25_max_3h"] == value_at(1) + 2
    assert f["dev_from_bkk_mean"] == pytest.approx(11 - (11 + 21 + 111) / 3)


def test_zero_reading_is_a_valid_value_in_cross_station_features():
    f = build(data(b=[Observation("b", T - H, 0.0, T)]))
    assert f["nearest_station_pm25"] == 0.0
    assert f["bkk_mean"] == pytest.approx((11 + 0 + 111) / 3)
