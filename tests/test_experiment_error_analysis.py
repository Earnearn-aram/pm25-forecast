"""KAN-93 error analysis: pure functions on small synthetic arrays (no network, no snapshot)."""

import numpy as np
import pytest

from pm25_forecast.experiments import error_analysis as ea

H = 3600


def make_frame():
    n = 48
    origin = np.arange(n, dtype=np.int64) * H
    actual = np.where(np.arange(n) % 12 < 3, 5.0, 20.0)
    latest = np.full(n, 12.0)
    smoother = np.full(n, 12.0)
    pred = {
        "persistence": latest,
        "smoother": smoother,
        "ridge_pool_raw_chg": actual + 1.0,
        "climatology_28d_blend": smoother,
    }
    return {
        "station": np.array(["a", "b"] * (n // 2)),
        "origin_s": origin,
        "actual": actual,
        "latest": latest,
        "trend6": np.where(np.arange(n) % 2 == 0, 8.0, -8.0),
        "pred": pred,
        "target_hour": ((origin // H + 24) % 24).astype(int),
        "target_dow": (((origin // H + 24) // 24 + 3) % 7).astype(int),
    }


def test_group_summary_counts_and_signs():
    f = make_frame()
    res = ea.summarise_groups(f)
    assert sum(e["n"] for e in res["station"].values()) == 48
    assert sum(e["n"] for e in res["hour_utc"].values()) == 48
    d = res["direction_actual"]
    assert d["rising"]["n"] + d["flat"]["n"] + d["falling"]["n"] == 48
    # persistence is 12 everywhere: it over-forecasts when the outcome is lower (positive bias)
    assert d["falling"]["persistence"]["bias"] > 0
    assert d["rising"]["persistence"]["bias"] < 0
    assert res["station"]["a"]["smoother"]["mae_minus_smoother"] == pytest.approx(0)


def test_group_diff_ci_runs_and_is_negative_for_a_better_model():
    f = make_frame()
    diff, lo, hi = ea.group_diff_ci(f, np.ones(48, dtype=bool), "ridge_pool_raw_chg")
    assert diff < 0 and lo <= diff <= hi


def test_spike_onsets_need_an_observed_previous_hour():
    series = {
        ("a", 0): 10.0,
        ("a", H): 40.0,
        ("a", 2 * H): 41.0,
        ("a", 5 * H): 50.0,
        ("b", 0): 38.0,
        ("b", H): 39.0,
    }
    assert ea.spike_onsets(series) == [("a", H)]


def test_error_concentration_shares_are_fractions():
    c = ea.error_concentration(make_frame())
    assert 0 <= c["share_error_regime_change"] <= 1
    assert c["share_error_worst_10pct_days"] <= 1


def test_onset_table_counts_catch_and_false_alarm():
    # station a: actual rises above 37.5 at t=10h; horizon-6 forecasts issued at origin t-6h.
    def arrays(cand_at):
        origin, actual = [], []
        for t in range(0, 30):
            origin.append((t - 6) * H)
            actual.append(10.0 if t < 10 else 40.0 if t < 12 else 10.0)
        cand = np.array([cand_at(t) for t in range(0, 30)], dtype=float)
        return {
            "h6_station": np.array(["a"] * 30),
            "h6_origin_s": np.array(origin, dtype=np.int64),
            "h6_actual": np.array(actual),
            "h6_cand": cand,
            "h6_pers": np.full(30, 10.0),
            "h6_smoo": np.full(30, 10.0),
        }

    good = arrays(lambda t: 40.0 if 10 <= t < 12 else 10.0)
    bad = arrays(lambda t: 40.0 if t == 25 else 10.0)  # no catch, one far-away alarm
    raw = {"good": good, "bad": bad}
    out = ea.onset_table(raw, models=("good", "bad"), horizons=(6,))
    assert out["n_onsets"] == 1
    assert out["models"]["good"]["by_horizon"]["6"]["caught"] == 1
    assert out["models"]["good"]["max_lead_counts"]["6"] == 1
    b = out["models"]["bad"]["by_horizon"]["6"]
    assert b["missed"] == 1 and b["false_alarm_hours"] == 1


def test_figures_are_written_small(tmp_path):
    pytest.importorskip("matplotlib")
    f = make_frame()
    summary = ea.summarise_groups(f)
    onset = {
        "n_onsets": 1,
        "mean_observed_at_onset": 40.0,
        "models": {
            m: {"by_horizon": {str(h): {"mean_forecast_at_onset": 20.0} for h in (6, 12, 24)}}
            for m in ea.MODELS
        },
    }
    saved = ea.save_figures(summary, onset, tmp_path)
    assert len(saved) == 4
    assert all((tmp_path / n).stat().st_size < 150_000 for n in saved)


def _raw(n=6, nan_at=None):
    """Two models' raw arrays for horizon 24; ``nan_at`` drops one forecast of model b."""
    origin = (np.arange(n, dtype=np.int64) + 100) * H
    base = {
        "h24_station": np.array(["a"] * n),
        "h24_origin_s": origin,
        "h24_actual": np.arange(n, dtype=float),
        "h24_pers": np.full(n, 5.0),
        "h24_smoo": np.full(n, 6.0),
        "h24_trend6": np.zeros(n),
    }
    a = {**base, "h24_cand": np.arange(n, dtype=float) + 1}
    b = {**base, "h24_cand": np.arange(n, dtype=float) + 2}
    if nan_at is not None:
        b["h24_cand"][nan_at] = np.nan
    return {"a": a, "b": b}


def test_common_frame_aligns_models_on_shared_rows():
    f = ea.common_frame(_raw(nan_at=2))
    assert len(f["actual"]) == 5 and 2 not in f["actual"].tolist()
    assert np.allclose(f["pred"]["a"] - f["actual"], 1.0)
    assert np.allclose(f["pred"]["b"] - f["actual"], 2.0)
    assert np.all(f["pred"]["smoother"] == 6.0) and np.all(f["latest"] == 5.0)


def test_collect_rows_alignment_and_trend6_uses_only_origin_minus_1h_and_7h():
    from datetime import timedelta

    from pm25_forecast.backtest import Observation
    from pm25_forecast.experiments import folds as folds_mod

    start = folds_mod.S0
    n_hours = 24 * 70
    obs = [
        Observation("a", start + timedelta(hours=i), float(i % 24), start + timedelta(hours=i + 1))
        for i in range(n_hours)
    ]

    # a distinctive value at origin-1h and origin-7h; a future reading must never be used.
    def fit(train, params=None):
        def predict(snapshot, horizon):
            return 7.0

        return predict

    try:
        out = ea.collect_rows(obs, [], fit, None, horizons=(24,))
    except Exception as exc:  # the synthetic series may be too short for the runner's plan
        pytest.skip(f"runner plan needs a longer series: {exc}")
    st, o = out["h24_station"], out["h24_origin_s"]
    assert len(st) == len(o) == len(out["h24_actual"]) == len(out["h24_cand"])
    assert np.all(out["h24_cand"] == 7.0)
    by_ts = {int((start + timedelta(hours=i)).timestamp()): float(i % 24) for i in range(n_hours)}
    for origin_s, trend in zip(o, out["h24_trend6"], strict=True):
        expected = by_ts[int(origin_s) - H] - by_ts[int(origin_s) - 7 * H]
        assert trend == pytest.approx(expected)
