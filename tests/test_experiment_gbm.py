"""KAN-89: pooled LightGBM factory on synthetic data. No network, no database."""

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pm25_forecast.backtest import Observation, Snapshot
from pm25_forecast.experiments import gbm
from pm25_forecast.experiments.evaluate import TrainSnapshot

HOUR = timedelta(hours=1)
T0 = datetime(2026, 9, 1, tzinfo=UTC)
STATIONS = ("s1", "s2", "s3")


def make_obs(n_hours=24 * 12):
    rng = np.random.default_rng(1)
    out = []
    for i, s in enumerate(STATIONS):
        noise = 0.0
        for k in range(n_hours):
            noise = 0.9 * noise + rng.normal(0, 2)
            ts = T0 + k * HOUR
            v = max(0.0, 25 + 6 * math.sin(2 * math.pi * ts.hour / 24) + 2 * i + noise)
            out.append(Observation(s, ts, round(v, 3), ts + HOUR))
    return out


def test_selected_names_groups():
    full = gbm.selected_names("fs_v2_full")
    assert "nearest_station_pm25" not in full
    assert "bkk_mean" in full and "hour_sin" in full
    no_cross = gbm.selected_names("fs_v2_full", ("cross",))
    assert set(full) - set(no_cross) == {"bkk_mean", "dev_from_bkk_mean"}
    core_no_roll = gbm.selected_names("fs_v2_core", ("rolling",))
    assert not any(n.startswith("pm25_mean_") for n in core_no_roll)
    groups = [gbm.GROUPS[g] for g in gbm.GROUPS]
    assert sum(len(g) for g in groups) == len(set().union(*groups))  # groups are disjoint
    with pytest.raises(KeyError):
        gbm.selected_names("fs_v2_full", ("nope",))


def test_config_grid_is_fixed_budget_and_seeded():
    a, b = gbm.config_grid(6, 7), gbm.config_grid(6, 7)
    assert a == b and len(a) == 6 and a[0] == gbm.DEFAULT_CONFIG
    assert gbm.config_grid(6, 8) != a


@pytest.mark.parametrize("target", ["level", "delta"])
def test_fit_predict_no_future(target, tmp_path):
    obs = make_obs()
    refit = T0 + 10 * 24 * HOUR
    train_obs = tuple(o for o in obs if o.ts <= refit and o.available_at <= refit)
    train = TrainSnapshot(refit, STATIONS, (24,), train_obs, ())
    params = {
        "feature_set": "fs_v2_full",
        "objective": "huber",
        "target": target,
        "budget": 2,
        "inner_folds": 2,
        "num_threads": 1,
        "log_dir": str(tmp_path),
        "log_name": "t",
    }
    predict = gbm.fit(train, params)
    assert (tmp_path / "t.tuning.jsonl").exists()

    def snap(station, origin):
        rows = tuple(
            o for o in obs if o.station_id == station and o.ts < origin and o.available_at <= origin
        )
        return Snapshot(origin, station, rows, ())

    origin = refit + 3 * HOUR
    p1 = predict(snap("s1", origin), 24)
    assert p1 is not None and math.isfinite(p1) and 0 < p1 < 100
    # Stale input: no reading within 3 h -> no forecast (never impute).
    stale = Snapshot(
        origin,
        "s1",
        tuple(o for o in snap("s1", origin).observations if o.ts <= origin - 6 * HOUR),
        (),
    )
    assert predict(stale, 24) is None
    assert predict(snap("s1", origin), 6) is None  # horizon not trained
