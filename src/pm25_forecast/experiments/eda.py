"""KAN-65 EDA. Design decisions use only data with target before the KAN-15 test start; the
holdout is shown separately as description only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import HIGH, STATIONS, Grid
from .dataset import BKK, build_table


def _acf(a: np.ndarray, lag: int, lo: int, hi: int) -> float:
    x, y = a[lo : hi - lag], a[lo + lag : hi]
    ok = ~np.isnan(x) & ~np.isnan(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else float("nan")


def run(grid: Grid) -> dict:
    out: dict = {}
    t24 = build_table(grid, 24)
    ts_idx = t24.test_start_idx
    out["test_start_utc"] = grid.time(ts_idx).isoformat()
    # 1 station x source stats (all data)
    rows = {}
    for s in STATIONS:
        v, src = grid.values[s], grid.source[s]
        for code, nm in ((1, "openaq"), (2, "air4thai")):
            x = v[src == code]
            if len(x):
                rows[f"{s}/{nm}"] = {
                    "n": len(x),
                    "mean": float(x.mean()),
                    "median": float(np.median(x)),
                    "sd": float(x.std()),
                    "p90": float(np.quantile(x, 0.9)),
                    "max": float(x.max()),
                    "share_above_37.5": float((x > HIGH).mean()),
                }
    out["station_source_stats"] = rows
    first_air = {s: grid.time(int(np.argmax(grid.source[s] == 2))).isoformat() for s in STATIONS}
    out["air4thai_first_ts"] = first_air
    # 2 missingness: coverage per station of the hourly grid, gaps > 6 h
    cov, gaps = {}, {}
    for s in STATIONS:
        v = grid.values[s]
        present = ~np.isnan(v)
        first = int(np.argmax(present))
        span = len(v) - first
        cov[s] = float(present[first:].mean())
        idx = np.where(present)[0]
        d = np.diff(idx)
        big = np.where(d > 6)[0]
        gaps[s] = [
            {
                "from": grid.time(int(idx[i])).isoformat(),
                "to": grid.time(int(idx[i + 1])).isoformat(),
                "hours": int(d[i] - 1),
            }
            for i in big
            if d[i] - 1 >= 48
        ]
        del span
    out["grid_coverage"] = cov
    out["gaps_ge_48h"] = gaps
    # 3 diurnal and weekly profile (local time), pooled stations, training span only
    prof_h = np.zeros(24)
    cnt_h = np.zeros(24)
    prof_d = np.zeros(7)
    cnt_d = np.zeros(7)
    for s in STATIONS:
        v = grid.values[s][:ts_idx]
        for i in np.where(~np.isnan(v))[0]:
            t = grid.time(int(i)) + BKK
            prof_h[t.hour] += v[i]
            cnt_h[t.hour] += 1
            prof_d[t.weekday()] += v[i]
            cnt_d[t.weekday()] += 1
    out["diurnal_mean_by_local_hour_train"] = (prof_h / cnt_h).round(2).tolist()
    out["weekday_mean_train_mon_first"] = (prof_d / cnt_d).round(2).tolist()
    # 4 autocorrelation at lags (pooled mean of per-station correlation), train span
    acf = {}
    for lag in (1, 6, 12, 24, 48, 168):
        acf[str(lag)] = float(np.nanmean([_acf(grid.values[s], lag, 0, ts_idx) for s in STATIONS]))
    out["acf_train"] = acf
    acf_h = {}
    for lag in (1, 6, 12, 24, 48, 168):
        acf_h[str(lag)] = float(
            np.nanmean([_acf(grid.values[s], lag, ts_idx, grid.n) for s in STATIONS])
        )
    out["acf_holdout_span_description"] = acf_h
    # 5 persistence error at +24 by group (train targets vs holdout)
    err = np.abs(t24.y - t24.latest)
    train, hold = t24.is_train, t24.is_test
    loc = [grid.time(int(i)) + BKK for i in t24.target_idx]
    lh = np.array([x.hour for x in loc])
    mo = np.array([x.strftime("%Y-%m") for x in loc])

    def grp(mask):
        d = {
            "n": int(mask.sum()),
            "mae": float(err[mask].mean()),
            "bias_pred_minus_y": float((t24.latest - t24.y)[mask].mean()),
            "mean_y": float(t24.y[mask].mean()),
        }
        return d

    out["persistence_mae_overall"] = {"train": grp(train), "holdout": grp(hold)}
    out["persistence_mae_by_hour_group"] = {
        f"{a:02d}-{b:02d}": {
            "train": grp(train & (lh >= a) & (lh <= b)),
            "holdout": grp(hold & (lh >= a) & (lh <= b)),
        }
        for a, b in ((0, 5), (6, 11), (12, 17), (18, 23))
    }
    out["persistence_mae_by_station"] = {
        s: {"train": grp(train & (t24.station == s)), "holdout": grp(hold & (t24.station == s))}
        for s in STATIONS
    }
    out["persistence_mae_by_month"] = {
        m: grp(np.isin(mo, [m]) & (train | hold)) for m in sorted(set(mo))
    }
    bins = [(0, 15), (15, 25), (25, 37.5), (37.5, 1e9)]
    out["persistence_mae_by_latest_level"] = {
        f"{lo}-{hi}": {
            "train": grp(train & (t24.latest > lo) & (t24.latest <= hi)),
            "holdout": grp(hold & (t24.latest > lo) & (t24.latest <= hi)),
        }
        for lo, hi in bins
    }
    out["persistence_mae_by_actual_level"] = {
        f"{lo}-{hi}": {
            "train": grp(train & (t24.y > lo) & (t24.y <= hi)),
            "holdout": grp(hold & (t24.y > lo) & (t24.y <= hi)),
        }
        for lo, hi in bins
    }
    # 6 cross-station correlation and lead-lag (train span), concurrent and lagged
    mat = {}
    for a in STATIONS:
        mat[a] = {}
        for b in STATIONS:
            x, y = grid.values[a][:ts_idx], grid.values[b][:ts_idx]
            ok = ~np.isnan(x) & ~np.isnan(y)
            mat[a][b] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    out["cross_corr_concurrent_train"] = mat
    ll = {}
    for k in (-3, -2, -1, 0, 1, 2, 3):
        vals = []
        for a in STATIONS:
            for b in STATIONS:
                if a == b:
                    continue
                x = grid.values[a][:ts_idx]
                y = grid.values[b][:ts_idx]
                xs, ys = (x[: len(x) - k], y[k:]) if k >= 0 else (x[-k:], y[: len(y) + k])
                ok = ~np.isnan(xs) & ~np.isnan(ys)
                vals.append(np.corrcoef(xs[ok], ys[ok])[0, 1])
        ll[str(k)] = float(np.mean(vals))
    out["leadlag_mean_corr_a_t_vs_b_t_plus_k"] = ll
    # 7 design probes on TRAIN rows only (+24 h): simple predictors, in-sample descriptive
    tr = train & ~np.isnan(t24.feats["mean24"]) & ~np.isnan(t24.feats["nbr_latest_mean"])
    y = t24.y[tr]
    cand = {
        "latest": t24.latest[tr],
        "mean24": t24.feats["mean24"][tr],
        "0.5latest+0.5mean24": 0.5 * t24.latest[tr] + 0.5 * t24.feats["mean24"][tr],
        "nbr_latest_mean": t24.feats["nbr_latest_mean"][tr],
        "0.5latest+0.5nbr": 0.5 * t24.latest[tr] + 0.5 * t24.feats["nbr_latest_mean"][tr],
        "pm25_t48": t24.feats["pm25_t48"][tr],
        "pm25_t168": t24.feats["pm25_t168"][tr],
    }
    out["train_probe_mae_plus24"] = {k: float(np.nanmean(np.abs(y - v))) for k, v in cand.items()}
    out["train_probe_n"] = int(tr.sum())
    # OLS slope of y on latest (shrinkage toward the mean?) on train
    x = t24.latest[train]
    yy = t24.y[train]
    out["train_slope_y_on_latest"] = float(np.polyfit(x, yy, 1)[0])
    out["train_rel_change_cv"] = float(np.std(yy - x) / np.mean(yy))
    # 8 episode proxy: persistence error when PM is rising vs falling over the last 6 h
    tr6 = train & ~np.isnan(t24.feats["trend6"])
    for nm, m in (
        ("rising>3", t24.feats["trend6"] > 3),
        ("falling<-3", t24.feats["trend6"] < -3),
        ("flat", np.abs(t24.feats["trend6"]) <= 3),
    ):
        mm = tr6 & m
        out.setdefault("train_persistence_by_6h_trend", {})[nm] = {
            "n": int(mm.sum()),
            "mae": float(err[mm].mean()),
            "bias_pred_minus_y": float((t24.latest - t24.y)[mm].mean()),
        }
    out["data_span"] = {"first": grid.t0.isoformat(), "last": grid.t_last.isoformat()}
    return out


if __name__ == "__main__":
    import sys

    g = Grid(Path(sys.argv[1]))
    r = run(g)
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[2]).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
