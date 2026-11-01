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


def _train(obs=None, refit=None):
    obs = obs if obs is not None else make_obs()
    refit = refit or T0 + 10 * 24 * HOUR
    rows = tuple(o for o in obs if o.ts <= refit and o.available_at <= refit)
    return obs, refit, TrainSnapshot(refit, STATIONS, (24,), rows, ())


def _params(tmp_path=None, **kw):
    p = {
        "feature_set": "fs_v2_full",
        "objective": "l2",
        "target": "delta",
        "budget": 2,
        "inner_folds": 2,
        "num_threads": 1,
    }
    if tmp_path is not None:
        p.update(log_dir=str(tmp_path), log_name="t")
    p.update(kw)
    return p


def _snap(obs, station, origin):
    rows = tuple(
        o for o in obs if o.station_id == station and o.ts < origin and o.available_at <= origin
    )
    return Snapshot(origin, station, rows, ())


def test_inner_cv_never_uses_labels_maturing_after_validation_origin():
    rng = np.random.default_rng(0)
    for h in (6, 12, 24):
        tgt = np.sort(rng.integers(0, 200 * 24, 3000)).astype(float) * 3600
        folds = list(gbm.inner_cv_masks(tgt, h, 3))
        assert len(folds) == 3
        for tr, val in folds:
            assert tr.any() and val.any()
            first_val_origin = tgt[val].min() - h * 3600
            # label of the newest training pair is available_at = ts + 1 h
            assert tgt[tr].max() + 3600 <= first_val_origin
            assert not (tr & val).any()


def test_same_seed_gives_identical_predictions():
    obs, refit, train = _train()
    origin = refit + 3 * HOUR
    out = []
    for _ in range(2):
        predict = gbm.fit(train, _params(seed=5))
        out.append([predict(_snap(obs, s, origin), 24) for s in STATIONS])
    assert out[0] == out[1] and all(v is not None for v in out[0])


def test_tuning_log_is_idempotent_and_summary_keeps_latest(tmp_path):
    from pm25_forecast.experiments import gbm_summary

    _, _, train = _train()
    for _ in range(2):
        gbm.fit(train, _params(tmp_path))
    path = tmp_path / "t.tuning.jsonl"
    assert len(path.read_text().splitlines()) == 1  # one horizon, one fold
    # Legacy file with a duplicated entry: the summary counts it once, latest line wins.
    line = path.read_text().splitlines()[0]
    path.write_text(line + "\n" + line.replace('"chosen": 0', '"chosen": 1') + "\n")
    entries = gbm_summary.load_log(path)
    assert len(entries) == 1


def test_train_cache_does_not_leak_across_snapshots():
    obs, refit, train_a = _train()
    shifted = [Observation(o.station_id, o.ts, o.value + 7.0, o.available_at) for o in obs]
    _, _, train_b = _train(shifted, refit)
    gbm.fit(train_a, _params())
    fp_a = gbm._TRAIN_CACHE_FP[0]
    p_b = gbm.fit(train_b, _params())
    assert gbm._TRAIN_CACHE_FP[0] != fp_a
    origin = refit + 3 * HOUR
    stale_free = [p_b(_snap(shifted, s, origin), 24) for s in STATIONS]
    gbm._TRAIN_CACHE.clear()
    gbm._TRAIN_CACHE_FP[0] = None
    fresh = gbm.fit(train_b, _params())
    assert stale_free == [fresh(_snap(shifted, s, origin), 24) for s in STATIONS]


def test_station_order_within_origin_does_not_change_predictions():
    obs, refit, train = _train()
    o1, o2 = refit + 2 * HOUR, refit + 3 * HOUR
    results = []
    for order1, order2 in (
        (STATIONS, STATIONS),
        (STATIONS[::-1], (STATIONS[1], STATIONS[2], STATIONS[0])),
    ):
        predict = gbm.fit(train, _params())
        for s in order1:
            predict(_snap(obs, s, o1), 24)
        results.append({s: predict(_snap(obs, s, o2), 24) for s in order2})
    assert results[0] == results[1] and all(v is not None for v in results[0].values())
