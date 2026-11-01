"""KAN-92 (E15): ensemble factory on synthetic data. No network, no database, no weather cache."""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pm25_forecast.backtest import Observation, Snapshot
from pm25_forecast.experiments import ensembles
from pm25_forecast.experiments.evaluate import TrainSnapshot, _ProbeSnapshot

HOUR = timedelta(hours=1)
T0 = datetime(2026, 9, 1, tzinfo=UTC)
STATIONS = ("s1", "s2", "s3")
MEMBERS = ("ridge", "clim", "smoother")  # the members that need no weather cache or statsmodels


def make_obs(n_hours=24 * 14):
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


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PM25_KAN92_LOG_DIR", str(tmp_path))
    ensembles.clear_cache()
    for v in ("equal", "invmae", "stack"):
        monkeypatch.setitem(ensembles.VARIANT_MEMBERS, v, MEMBERS)
    yield
    ensembles.clear_cache()


def _train(obs, refit):
    rows = tuple(o for o in obs if o.ts <= refit and o.available_at <= refit)
    return TrainSnapshot(refit, STATIONS, (24,), rows, ())


def _snap(obs, station, origin):
    rows = tuple(
        o
        for o in obs
        if o.station_id == station and o.ts <= origin - HOUR and o.available_at <= origin
    )
    return Snapshot(origin, station, rows, ())


def test_inverse_mae_weights_use_recent_rows_only():
    y = np.zeros(10)
    t = np.arange(10) * 86400.0
    p = np.column_stack(
        [np.r_[np.full(5, 9.0), np.full(5, 1.0)], np.r_[np.full(5, 1.0), np.full(5, 2.0)]]
    )
    w, err, used = ensembles.inverse_mae_weights(y, p, t, window_days=3)  # last 4 rows
    assert used == 4 and err.tolist() == [1.0, 2.0]
    assert w.tolist() == pytest.approx([2 / 3, 1 / 3]) and w.sum() == pytest.approx(1)
    nan = p.copy()
    nan[9, 0] = np.nan  # a row without a prediction from every member is dropped
    assert (
        ensembles.inverse_mae_weights(y, nan, t, window_days=3)[2] == 4
    )  # the window moves to the newest complete row


def test_stacker_recovers_nonnegative_weights_and_combine():
    rng = np.random.default_rng(0)
    p = rng.normal(30, 5, (500, 3))
    y = 2.0 + 0.7 * p[:, 0] + 0.3 * p[:, 2]
    b, w = ensembles.fit_stacker(y, p)
    assert b == pytest.approx(2.0, abs=1e-6)
    assert w == pytest.approx([0.7, 0.0, 0.3], abs=1e-6)
    assert ensembles.combine([10, 20, 30], w, b) == pytest.approx(2 + 7 + 9)
    y2 = -p[:, 1]  # a negative relation cannot get a negative weight
    assert (ensembles.fit_stacker(y2, p)[1] >= 0).all()


def test_clean_snapshot_filters_by_time_not_position():
    obs = [o for o in make_obs(100) if o.station_id == "s1"]
    origin = T0 + 60 * HOUR
    wide = _ProbeSnapshot(origin, "s1", tuple(obs), ())  # holds rows after the origin
    clean = ensembles.clean_snapshot(wide)
    assert max(o.ts for o in clean.observations) == origin - HOUR
    assert all(o.available_at <= origin for o in clean.observations)


def test_component_params_equal_the_preregistered_ones():
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1] / "docs" / "experiments" / "prereg"
    for name, prereg_id in ensembles.COMPONENT_PREREG.items():
        got = yaml.safe_load((root / f"{prereg_id}.yaml").read_text())["params"]
        mine = ensembles.COMPONENT_PARAMS[name]
        if name == "ridge":  # the YAML also holds descriptive text keys
            assert {k: got[k] for k in mine if k != "variant"} == {
                k: v for k, v in mine.items() if k != "variant"
            }
        elif name == "gbm":  # log routing keys are dropped on purpose
            assert {k: v for k, v in got.items() if k not in ("log_dir", "log_name")} == mine
        elif name == "arima":  # `stateless` is a runtime flag of this ticket
            assert {k: got[k] for k in mine if k != "stateless"} == {
                k: v for k, v in mine.items() if k != "stateless"
            }
        else:
            assert {k: got[k] for k in mine} == mine


@pytest.mark.parametrize("variant", ["equal", "invmae", "stack"])
def test_fit_and_predict_match_the_weights(variant):
    obs = make_obs()
    refit = T0 + 12 * 24 * HOUR
    predict = ensembles.fit(_train(obs, refit), {"variant": variant, "min_oof_rows": 100})
    origin = refit + 3 * HOUR
    snap = _snap(obs, "s1", origin)
    got = predict(snap, 24)
    assert got is not None and math.isfinite(got) and 0 < got < 100
    diag = ensembles.log_dir() / f"{variant}.json"
    info = __import__("json").loads(diag.read_text())[refit.isoformat()]["horizons"]["24"]
    assert not info["fallback_equal"] or variant == "equal"
    w = np.array([info["weights"][m] for m in MEMBERS])
    if variant == "equal":
        assert w.tolist() == pytest.approx([1 / 3] * 3)
    if variant == "invmae":
        assert w.sum() == pytest.approx(1) and (w > 0).all()
    parts = [
        ensembles.member_factory(m)(_train(obs, refit))(snap, 24) for m in MEMBERS
    ]  # members refit on the same train snapshot give the same member forecasts
    assert got == pytest.approx(ensembles.combine(parts, w, info["intercept"]), rel=1e-9)
    stale = Snapshot(
        origin, "s1", tuple(o for o in snap.observations if o.ts <= origin - 6 * HOUR), ()
    )
    assert predict(stale, 24) is None  # a member without a forecast: no forecast, no imputation
    assert predict(snap, 6) is None  # horizon not fitted


def test_widened_snapshot_gives_the_same_prediction():
    """The serving-lag probe: rows after the origin (shifted) must not change a forecast."""
    obs = make_obs()
    refit = T0 + 12 * 24 * HOUR
    train = _train(obs, refit)
    origin = refit + 5 * HOUR
    base = _snap(obs, "s2", origin)
    extra = [
        o for o in obs if o.station_id == "s2" and (o.ts > origin - HOUR or o.available_at > origin)
    ]
    for shift in (1000.0, -999.0):
        wide = _ProbeSnapshot(
            origin,
            "s2",
            tuple(
                sorted(
                    [
                        *base.observations,
                        *(Observation("s2", o.ts, o.value + shift, o.available_at) for o in extra),
                    ],
                    key=lambda o: o.ts,
                )
            ),
            (),
        )
        a = ensembles.fit(train, {"variant": "invmae", "min_oof_rows": 100})(base, 24)
        b = ensembles.fit(train, {"variant": "invmae", "min_oof_rows": 100})(wide, 24)
        assert a is not None and a == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_out_of_fold_predictions_respect_the_purge(monkeypatch):
    """Every inner fit is trained on labels available before its first validation origin."""
    obs = make_obs()
    refit = T0 + 12 * 24 * HOUR
    train = _train(obs, refit)
    seen: list[dict] = []
    real = ensembles.member_factory

    def spy(name):
        inner_fit = real(name)

        def factory(tr):
            rec = {
                "refit": tr.refit_time,
                "max_ts": max(o.ts for o in tr.observations),
                "origins": [],
            }
            assert all(o.available_at <= tr.refit_time for o in tr.observations)
            seen.append(rec)
            predictor = inner_fit(tr)

            def predict(snap, h):
                rec["origins"].append(snap.prediction_time)
                return predictor(snap, h)

            return predict

        return factory

    monkeypatch.setattr(ensembles, "member_factory", spy)
    oof = ensembles.oof_predictions(train, MEMBERS, 24, 3)
    assert len(oof["y"]) > 300 and oof["preds"].shape[1] == 3
    assert (oof["target_s"] <= refit.timestamp()).all()
    inner = [r for r in seen if r["origins"]]
    assert len(inner) == 3 * len(MEMBERS)
    for r in inner:
        assert (
            r["max_ts"] + HOUR <= r["refit"] <= min(r["origins"])
        )  # label matured before 1st origin
        assert r["refit"] < refit


def test_arima_stateless_flag_is_reproducible_on_fresh_instances():
    pytest.importorskip("statsmodels")
    from pm25_forecast.experiments import timeseries

    obs = make_obs(24 * 20)
    refit = T0 + 16 * 24 * HOUR
    train = _train(obs, refit)
    params = {**ensembles.COMPONENT_PARAMS["arima"], "p_grid": [1], "q_grid": [0]}
    live = timeseries.arima(train, params)
    values = []
    for k in range(4):  # the live instance moves forward hour by hour
        values.append(live(_snap(obs, "s1", refit + (3 + k) * HOUR), 24))
    fresh = timeseries.arima(train, params)
    again = fresh(_snap(obs, "s1", refit + 6 * HOUR), 24)  # a fresh instance asked out of order
    assert again is not None and again == pytest.approx(values[-1], rel=1e-12, abs=1e-12)
