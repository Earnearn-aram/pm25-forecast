"""Unit tests for the KAN-65 offline experiment helpers (no real data, no DB)."""

from datetime import UTC, datetime

import numpy as np

from pm25_forecast.experiments.common import STATIONS, Grid, day_block_ci
from pm25_forecast.experiments.dataset import build_table
from pm25_forecast.experiments.fitters import forward, inverse, ridge_fit
from pm25_forecast.experiments.run_all import boot_draws, holm


def _grid(n=24 * 40, seed=0):
    g = object.__new__(Grid)
    g.sha = "x"
    g.t0 = datetime(2026, 1, 1, tzinfo=UTC)
    g.n = n
    g.t_last = g.t0 + (n - 1) * (datetime(2026, 1, 1, 1, tzinfo=UTC) - g.t0)
    rng = np.random.default_rng(seed)
    g.values = {
        s: 20 + 5 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 1, n) for s in STATIONS
    }
    g.source = {s: np.ones(n, dtype=np.int8) for s in STATIONS}
    return g


def test_transforms_roundtrip():
    y, lat = np.array([5.0, 30.0]), np.array([7.0, 20.0])
    for kind in ("direct", "resid", "logratio"):
        assert np.allclose(inverse(kind, forward(kind, y, lat), lat), y)


def test_features_are_as_of():
    g = _grid()
    t1 = build_table(g, 24)
    # corrupt every reading at or after the origin hour: features must not change
    row = len(t1.y) // 2
    o, s = int(t1.origin_idx[row]), str(t1.station[row])
    g2 = _grid()
    for st in STATIONS:
        g2.values[st][o:] = 1e6
    t2 = build_table(g2, 24)
    sel = (t2.station == s) & (t2.origin_idx == o)
    for name in (
        "pm25_latest",
        "pm25_t48",
        "pm25_t168",
        "mean24",
        "max6",
        "trend3",
        "nbr_latest_mean",
        "nbr_mean24",
        "pm25_t24",
    ):
        a, b = t1.feats[name][row], t2.feats[name][sel]
        assert (np.isnan(a) and np.isnan(b[0])) or a == b[0], name
    assert np.isnan(t1.feats["pm25_t24"]).all()  # reading at the origin hour is never servable


def test_gap_not_sampled():
    g = _grid()
    for st in ("bkp100t", "bkp101t"):
        g.values[st][200:260] = np.nan
    t = build_table(g, 24)
    bad = np.isin(t.station, ["bkp100t", "bkp101t"]) & (t.target_idx >= 200) & (t.target_idx < 260)
    assert not bad.any()


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    y = 3 * x[:, 0] - x[:, 1] + 2
    r = ridge_fit(x, y, ["a", "b"], alpha=0.01)
    assert np.abs(r.predict(x) - y).mean() < 0.1


def test_bootstrap_and_holm():
    diff = np.r_[np.full(48, -1.0), np.full(48, -1.0)]
    days = np.repeat(np.arange(4), 24)
    dr = boot_draws(diff, days, 500)
    assert np.allclose(dr, -1.0)
    m, lo, hi = day_block_ci(diff, days, n_boot=500)
    assert lo == hi == m == -1.0
    adj = holm(np.array([0.01, 0.04, 0.5]))
    assert adj[0] == 0.03 and adj[2] == 0.5 and np.all(adj <= 1)


def test_review_block_bootstrap_and_events():
    from pm25_forecast.experiments.run_review import boot_draws as bd
    from pm25_forecast.experiments.run_review import event_block, hp_block

    days = np.repeat(np.arange(14), 4)
    diff = np.full(len(days), -2.0)
    dr, nb = bd(diff, days, 7, n_boot=200)
    assert nb == 2 and np.allclose(dr, -2.0)
    lat = np.full(len(days), 10.0)
    y = lat.copy()
    y[:3] = 40.0  # three onsets
    p = lat.copy()
    p[0] = 20.0  # one caught
    e = event_block(y, p, lat, days)
    assert e["onsets"] == 3 and abs(e["caught_share"] - 1 / 3) < 1e-9 and not e["testable"]
    assert hp_block(y, p, lat, days)["status"] == "untestable"  # no latest > 37.5
