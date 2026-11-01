"""KAN-84 linear models: target transforms, inner CV leakage, serving lag, end-to-end fit."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pm25_forecast.backtest import Observation, Snapshot
from pm25_forecast.experiments import linear_models as lm
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features import build_features
from pm25_forecast.features.library_v2 import prepare_series

T0 = datetime(2026, 8, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)


@pytest.fixture(autouse=True)
def _fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PM25_KAN84_LOG_DIR", str(tmp_path))
    lm.clear_cache()
    lm._dumps.clear()


def series(station: str, n: int, seed: int) -> list[Observation]:
    rng = np.random.default_rng(seed)
    out, level = [], 30.0
    for i in range(n):
        level = 0.9 * level + 3.0 + rng.normal(0, 2.0)
        ts = T0 + i * HOUR
        out.append(Observation(station, ts, max(level, 1.0), ts + HOUR))
    return out


def train_snapshot(n: int = 700, stations=("a", "b")) -> TrainSnapshot:
    obs = tuple(o for k, s in enumerate(stations) for o in series(s, n, k))
    return TrainSnapshot(T0 + n * HOUR, tuple(stations), (24,), obs, ())


def test_target_roundtrip():
    y, latest = np.array([5.0, 40.0]), np.array([7.0, 30.0])
    for target in ("raw", "log1p"):
        for output in ("level", "change"):
            z = lm.to_target(y, latest, target, output)
            assert np.allclose(lm.from_target(z, latest, target, output), y)


def test_change_target_zero_means_persistence():
    latest = np.array([12.0])
    for target in ("raw", "log1p"):
        assert np.allclose(lm.from_target(np.zeros(1), latest, target, "change"), latest)


def test_inner_splits_never_train_on_unmatured_labels():
    origins = np.arange(0, 600) * 3600.0
    for h in (6, 24):
        splits = lm.inner_splits(origins, h)
        assert len(splits) >= 3
        for tr, va in splits:
            assert origins[tr].max() + h * 3600 <= origins[va].min()
            assert not set(tr) & set(va)


def test_latest_matches_shared_feature_module():
    obs = series("a", 100, 0)
    origin = T0 + 90 * HOUR
    got = lm.row_vector(prepare_series({"a": obs}), "a", origin, origin + 24 * HOUR, "fs_v2_core")
    shared = build_features([o for o in obs if o.ts < origin], (), origin, origin + 24 * HOUR)
    assert got is not None
    assert got[1] == shared["pm25_latest"]


def test_features_ignore_origin_hour_and_later():
    """Serving lag: changing readings at ts >= t must not change the row."""
    obs = series("a", 100, 0)
    origin = T0 + 90 * HOUR
    a = lm.row_vector(prepare_series({"a": obs}), "a", origin, origin + 24 * HOUR, "fs_v2_core")
    changed = [
        Observation(o.station_id, o.ts, o.value + 99, o.available_at) if o.ts >= origin else o
        for o in obs
    ]
    b = lm.row_vector(prepare_series({"a": changed}), "a", origin, origin + 24 * HOUR, "fs_v2_core")
    assert a is not None and b is not None
    assert np.array_equal(a[0], b[0], equal_nan=True) and a[1] == b[1]


def test_no_reading_in_last_3h_gives_no_row():
    obs = [o for o in series("a", 100, 0) if not (T0 + 86 * HOUR <= o.ts <= T0 + 89 * HOUR)]
    origin = T0 + 90 * HOUR
    assert (
        lm.row_vector(prepare_series({"a": obs}), "a", origin, origin + 24 * HOUR, "fs_v2_core")
        is None
    )


def test_full_set_with_cross_station_is_refused():
    with pytest.raises(ValueError):
        lm.names_for("fs_v2_full")
    assert not set(lm.names_for("fs_v2_full_nocross")) & set(lm.CROSS_STATION_NAMES)


@pytest.mark.parametrize("model", ["ols", "ridge", "lasso"])
@pytest.mark.parametrize("scope", ["pooled", "per_station"])
@pytest.mark.parametrize(("target", "output"), [("raw", "level"), ("log1p", "change")])
def test_fit_predicts_finite_and_sensible(model, scope, target, output):
    train = train_snapshot()
    predict = lm.fit(
        train, {"model": model, "scope": scope, "target": target, "output": output, "variant": "t"}
    )
    origin = train.refit_time
    obs = tuple(o for o in train.observations if o.station_id == "a" and o.ts < origin)
    snap = Snapshot(origin, "a", obs, ())
    p = predict(snap, 24)
    assert p is not None and math.isfinite(p) and 0 < p < 100
    assert predict(Snapshot(origin, "zzz", obs, ()), 24) is None
    stale = tuple(o for o in obs if o.ts < origin - 5 * HOUR)
    assert predict(Snapshot(origin, "a", stale, ()), 24) is None


def test_lasso_zeroes_some_coefficients_and_logs_diagnostics(tmp_path):
    import json

    train = train_snapshot(900, ("a",))
    lm.fit(train, {"model": "lasso", "scope": "per_station", "variant": "lasso-t"})
    lm._dumps["lasso-t"].flush()
    rec = json.loads((tmp_path / "lasso-t.jsonl").read_text().splitlines()[0])
    assert rec["nonzero"] < len(rec["coef_std"])
    assert "pm25" in max(rec["coef_std"], key=lambda k: abs(rec["coef_std"][k]))


def test_prereg_grid_is_24_unique_valid_preregs():
    from pm25_forecast.experiments import kan84_prereg as kp
    from pm25_forecast.experiments.prereg import Prereg

    specs = [kp.prereg_dict(*row) for row in kp.grid()]
    assert len(specs) == 24
    assert len({s["id"] for s in specs}) == 24
    for s in specs:
        p = Prereg.model_validate(s)
        assert p.split == "dev" and not p.final and p.params["primary_ci_block_hours"] == 72
    assert kp.prereg_dict("ridge", "pooled", "raw", "change", "fs_v2_full_nocross")["id"].endswith(
        "-full"
    )
