"""KAN-65 v2 (post-review redesign, docs/kan-65-ds-plan.md Amendments 5 and 5b).

PYTHONPATH=src python -m pm25_forecast.experiments.run_review \
    --database data/history.duckdb --out out/kan65_v2
Fixed configurations, no tuning except the smoother beta; non-overlapping regime-B folds;
1-day and 7-day block bootstrap; forecast-time high-pollution rule; onset-event metric; H-MA test.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .common import BOOT_SEED, EXPECTED_SHA, HIGH, STATIONS, Grid
from .dataset import BKK, SET_FEATURES, build_table
from .fitters import FIT_COUNT, forward, inverse, lgb_params, lgb_refit, ridge_fit
from .run_all import holm, metrics

HORIZONS = (6, 12, 24)
BETAS = tuple(round(0.1 * i, 1) for i in range(10))
FAMILY = ("smoother", "lgbm_direct", "lgbm_logratio", "ridge", "current")
GROUPS = {"HOLDOUT": ["HOLDOUT"], "REGIME_B": ["G1", "G2", "G3"], "REGIME_A": ["F1", "F2", "F3"]}
HEADLINE_GROUPS = ("HOLDOUT", "REGIME_B")
N_BOOT = 40000
ONSET_RISE = 15.0
CATCH_RISE = 7.5
ML_ROUNDS = 150
RIDGE_ALPHA = 10.0


def d(y, m, dd):
    return datetime(y, m, dd, tzinfo=UTC)


def make_runs(grid: Grid, test_start_idx: int):
    ix = grid.idx
    end = grid.n + 10**6
    return [
        {
            "name": "F1",
            "B": ix(d(2025, 12, 31)),
            "t0": ix(d(2026, 1, 1)),
            "t1": ix(d(2026, 2, 12)),
            "vd": 45,
            "native": False,
            "select": True,
        },
        {
            "name": "F2",
            "B": ix(d(2026, 3, 31)),
            "t0": ix(d(2026, 4, 1)),
            "t1": ix(d(2026, 5, 13)),
            "vd": 45,
            "native": False,
            "select": True,
        },
        {
            "name": "F3",
            "B": ix(d(2026, 5, 31)),
            "t0": ix(d(2026, 6, 1)),
            "t1": ix(d(2026, 7, 13)),
            "vd": 45,
            "native": False,
            "select": True,
        },
        {
            "name": "G1",
            "B": ix(d(2026, 8, 22)),
            "t0": ix(d(2026, 8, 22)),
            "t1": ix(d(2026, 9, 12)),
            "vd": 10,
            "native": True,
            "select": True,
        },
        {
            "name": "G2",
            "B": ix(d(2026, 9, 12)),
            "t0": ix(d(2026, 9, 12)),
            "t1": ix(d(2026, 10, 3)),
            "vd": 10,
            "native": True,
            "select": True,
        },
        {
            "name": "G3",
            "B": ix(d(2026, 10, 3)),
            "t0": ix(d(2026, 10, 3)),
            "t1": end,
            "vd": 10,
            "native": True,
            "select": True,
        },
        {
            "name": "HOLDOUT",
            "B": test_start_idx,
            "t0": test_start_idx,
            "t1": end,
            "vd": 45,
            "native": False,
            "select": False,
        },
    ]


def split(tab, run):
    o, t = tab.origin_idx, tab.target_idx
    vs = run["B"] - run["vd"] * 24
    m = {
        "train": t < run["B"],
        "val": (o >= vs) & (t < run["B"]),
        "test": (o >= run["t0"]) & (o < run["t1"]),
    }
    if run["native"]:
        m["val"] &= tab.source_target == 2
    return m


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def smoother(f, beta):
    return f["pm25_latest"] + beta * (f["mean24"] - f["pm25_latest"])


def ml_block(tab, m, native):
    """Fixed-config ML candidates. Native (regime-B) folds: A+B rows with source flag (main) and
    B-only rows (descriptive)."""
    te = m["test"]
    lat_t = tab.latest[te]
    out = {}
    variants = [("", "ab"), ("_bonly", "bonly")] if native else [("", "all")]
    for suffix, var in variants:
        names = list(SET_FEATURES["D"]) + (["src_air4thai"] if var == "ab" else [])
        rows = m["train"] & ((tab.source_target == 2) if var == "bonly" else True)
        if rows.sum() < 300:
            continue
        X = tab.X(names)
        for kind, nm in (("direct", "lgbm_direct"), ("logratio", "lgbm_logratio")):
            yt = forward(kind, tab.y, tab.latest)
            b = lgb_refit(lgb_params("l1", 7), X[rows], yt[rows], names, ["station_id"], ML_ROUNDS)
            out[nm + suffix] = inverse(kind, b.predict(X[te]), lat_t)
        yr = forward("resid", tab.y, tab.latest)
        r = ridge_fit(X[rows], yr[rows], names, RIDGE_ALPHA)
        out["ridge" + suffix] = inverse("resid", r.predict(X[te]), lat_t)
    # current production-style model (KAN-15 params, untuned)
    names = ["pm25_latest", "pm25_t24"]
    Xc = tab.X(names)
    tr = m["train"]
    cp = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "seed": 20261029,
        "feature_fraction_seed": 20261029,
        "bagging_seed": 20261029,
        "num_threads": 1,
    }
    bst = lgb.train(cp, lgb.Dataset(Xc[tr], label=tab.y[tr], feature_name=names), 200)
    out["current"] = np.maximum(0.0, bst.predict(Xc[te]))
    return out


def global_block(tabs, run):
    native = run["native"]
    base = [n for n in SET_FEATURES["D"] if n != "station_id"]
    masks = {h: split(tabs[h], run) for h in HORIZONS}
    res = {}
    for gv in ("global", "global_id"):
        names = base + (["station_id"] if gv == "global_id" else [])
        names = names + (["src_air4thai"] if native else []) + ["horizon"]
        xs, ys = [], []
        for h in HORIZONS:
            tab, mk = tabs[h], masks[h]
            x = tab.X(names[:-1])[mk["train"]]
            xs.append(np.column_stack([x, np.full(len(x), float(h))]))
            ys.append(tab.y[mk["train"]])
        b = lgb_refit(
            lgb_params("l1", 7),
            np.vstack(xs),
            np.concatenate(ys),
            names,
            ["station_id"] if gv == "global_id" else "auto",
            ML_ROUNDS,
        )
        res[gv] = {}
        for h in HORIZONS:
            tab, te = tabs[h], masks[h]["test"]
            x = np.column_stack([tab.X(names[:-1])[te], np.full(te.sum(), float(h))])
            res[gv][h] = np.maximum(0.0, b.predict(x))
    return res


# ------------------------------------------------------------------ statistics
def boot_draws(diff, days, block, n_boot=N_BOOT, seed=BOOT_SEED):
    blk = (days - days.min()) // block
    uniq, pos = np.unique(blk, return_inverse=True)
    nb = len(uniq)
    sums = np.bincount(pos, weights=diff, minlength=nb)
    cnts = np.bincount(pos, minlength=nb).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(0, n_boot, 5000):
        k = min(5000, n_boot - i)
        dr = rng.integers(0, nb, size=(k, nb))
        out[i : i + k] = sums[dr].sum(1) / cnts[dr].sum(1)
    return out, nb


def ci(diff, days, block, q_bonf=None):
    dr, nb = boot_draws(diff, days, block)
    res = {
        "n": len(diff),
        "days": len(np.unique(days)),
        "blocks": int(nb),
        "diff": float(diff.mean()),
        "lo": float(np.quantile(dr, 0.025)),
        "hi": float(np.quantile(dr, 0.975)),
        "p": float((min((dr <= 0).sum(), (dr >= 0).sum()) * 2 + 1) / (len(dr) + 1)),
    }
    if q_bonf:
        res["lo_bonf"] = float(np.quantile(dr, q_bonf / 2))
        res["hi_bonf"] = float(np.quantile(dr, 1 - q_bonf / 2))
    return res


def ratio_ci(num, den, days, block, n_boot=20000):
    blk = (days - days.min()) // block
    uniq, pos = np.unique(blk, return_inverse=True)
    nb = len(uniq)
    sn = np.bincount(pos, weights=num.astype(float), minlength=nb)
    sd = np.bincount(pos, weights=den.astype(float), minlength=nb)
    rng = np.random.default_rng(BOOT_SEED)
    dr = rng.integers(0, nb, size=(n_boot, nb))
    dsum = sd[dr].sum(1)
    ok = dsum > 0
    r = sn[dr].sum(1)[ok] / dsum[ok]
    return (
        float(num.sum() / max(den.sum(), 1)),
        float(np.quantile(r, 0.025)),
        float(np.quantile(r, 0.975)),
    )


def testable_hp(n, days):
    return n >= 200 and days >= 10


def hp_block(y, p, lat, days):
    mk = lat > HIGH
    n, nd = int(mk.sum()), len(np.unique(days[mk])) if mk.any() else 0
    out = {"n": n, "days": nd, "testable": bool(testable_hp(n, nd))}
    if n >= 10:
        diff = np.abs(p[mk] - y[mk]) - np.abs(lat[mk] - y[mk])
        out["ci_1d"] = ci(diff, days[mk], 1)
        out["ci_7d"] = ci(diff, days[mk], 7)
        if out["testable"]:
            out["status"] = "FAIL" if out["ci_7d"]["lo"] > 0 else "PASS"
        else:
            out["status"] = "untestable"
    else:
        out["status"] = "untestable"
    return out


def event_block(y, p, lat, days):
    onset = (y - lat) >= ONSET_RISE
    alarm = (p - lat) >= CATCH_RISE
    n_on = int(onset.sum())
    out = {
        "onsets": n_on,
        "onset_days": len(np.unique(days[onset])) if n_on else 0,
        "testable": n_on >= 30,
    }
    if n_on == 0:
        return out
    hit = onset & alarm
    out["caught_share"], out["caught_lo7"], out["caught_hi7"] = ratio_ci(hit, onset, days, 7)
    non = ~onset
    out["false_alarm_share"] = float((alarm & non).sum() / max(non.sum(), 1))
    out["mean_error_at_onset"] = float((p - y)[onset].mean())
    out["mae_at_onset"] = float(np.abs(p - y)[onset].mean())
    out["persistence_mae_at_onset"] = float(np.abs(lat - y)[onset].mean())
    if n_on >= 30:
        diff = np.abs(p[onset] - y[onset]) - np.abs(lat[onset] - y[onset])
        out["onset_mae_diff_7d"] = ci(diff, days[onset], 7)
    return out


# ------------------------------------------------------------------ H-MA test
def _stats(series: dict, hour_of: np.ndarray) -> dict:
    d1, same, _n1, a1, a24, flat, lvl, cnt = [], [], [], [], [], [], [], 0
    for _s, v in series.items():
        ok = ~np.isnan(v)
        cnt += int(ok.sum())
        if ok.sum() < 100:
            continue
        lvl.append(np.nanmean(v))
        pair = ok[1:] & ok[:-1]
        dd = np.abs(v[1:] - v[:-1])[pair]
        d1.append(dd.mean())
        same.append(np.mean(np.round(v[1:][pair], 1) == np.round(v[:-1][pair], 1)))
        for lag, store in ((1, a1), (24, a24)):
            pr = ok[lag:] & ok[:-lag]
            if pr.sum() > 50:
                store.append(np.corrcoef(v[lag:][pr], v[:-lag][pr])[0, 1])
        prof = np.array([np.nanmean(v[(hour_of == h) & ok]) for h in range(24)])
        flat.append((prof.max() - prof.min()) / np.nanmean(v[ok]))
    return {
        "n_values": cnt,
        "mean": float(np.mean(lvl)),
        "mean_abs_1h_change": float(np.mean(d1)),
        "identical_consecutive_share": float(np.mean(same)),
        "acf_lag1": float(np.mean(a1)),
        "acf_lag24": float(np.mean(a24)),
        "diurnal_flatness": float(np.mean(flat)),
    }


def hma_test(grid: Grid) -> dict:
    hour_of = np.array([(grid.time(i) + BKK).hour for i in range(grid.n)])
    months = np.array([grid.time(i).month for i in range(grid.n)])
    years = np.array([grid.time(i).year for i in range(grid.n)])
    raw, ma, old25, old_all = {}, {}, {}, {}
    for s in STATIONS:
        v, src = grid.values[s], grid.source[s]
        a = np.where(src == 2, v, np.nan)
        raw[s] = a
        m = np.full(grid.n, np.nan)
        for i in range(23, grid.n):
            w = a[i - 23 : i + 1]
            if np.sum(~np.isnan(w)) >= 18:
                m[i] = np.nanmean(w)
        ma[s] = m
        o = np.where(src == 1, v, np.nan)
        old_all[s] = o
        old25[s] = np.where((years == 2025) & (months >= 8) & (months <= 10), o, np.nan)
    res = {
        "air4thai_raw_hourly": _stats(raw, hour_of),
        "air4thai_trailing_24h_mean": _stats(ma, hour_of),
        "openaq_AugOct2025": _stats(old25, hour_of),
        "openaq_all": _stats(old_all, hour_of),
    }
    o, r, m = (
        res["openaq_AugOct2025"],
        res["air4thai_raw_hourly"],
        res["air4thai_trailing_24h_mean"],
    )

    def ratio(a, b):
        return a / b if b > 0 else float("inf")

    r_ma_d = ratio(m["mean_abs_1h_change"], o["mean_abs_1h_change"])
    r_ma_f = ratio(m["diurnal_flatness"], o["diurnal_flatness"])
    r_raw_d = ratio(r["mean_abs_1h_change"], o["mean_abs_1h_change"])
    r_old_f_raw = ratio(o["diurnal_flatness"], r["diurnal_flatness"])
    supported = (
        1 / 1.5 <= r_ma_d <= 1.5
        and 1 / 1.5 <= r_ma_f <= 1.5
        and m["acf_lag1"] > 0.99
        and o["acf_lag1"] > 0.99
        and (r_raw_d > 2 or r_raw_d < 0.5)
    )
    rejected = r_ma_d > 2 or r_ma_d < 0.5 or (1 / 1.5 <= r_old_f_raw <= 1.5)
    res["ratios"] = {
        "ma_vs_old_mean_abs_1h_change": r_ma_d,
        "ma_vs_old_diurnal_flatness": r_ma_f,
        "raw_vs_old_mean_abs_1h_change": r_raw_d,
        "old_vs_raw_diurnal_flatness": r_old_f_raw,
    }
    res["verdict"] = "supported" if supported else ("rejected" if rejected else "undecided")
    return res


# ------------------------------------------------------------------ main
def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", type=Path, default=Path("data/history.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("out/kan65_v2"))
    args = ap.parse_args(argv)
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    def log(s):
        print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    grid = Grid(args.database)
    info = {
        "snapshot_sha256": grid.sha,
        "matches_kan15_snapshot": grid.sha == EXPECTED_SHA,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "model_seed": 20261029,
    }
    log(f"snapshot matches={info['matches_kan15_snapshot']}")
    tabs = {h: build_table(grid, h) for h in HORIZONS}
    runs = make_runs(grid, tabs[24].test_start_idx)
    hma = hma_test(grid)
    log(f"H-MA verdict: {hma['verdict']} {hma['ratios']}")

    store, beta_info = {}, {}
    for run in runs:
        gl = global_block(tabs, run)
        for h in HORIZONS:
            tab = tabs[h]
            m = split(tab, run)
            te = m["test"]
            f_te = {k: v[te] for k, v in tab.feats.items()}
            f_val = {k: v[m["val"]] for k, v in tab.feats.items()}
            y_val = tab.y[m["val"]]
            preds = ml_block(tab, m, run["native"])
            if run["select"] and m["val"].sum() >= 50:
                vm = {b: mae(smoother(f_val, b), y_val) for b in BETAS}
                beta = min(BETAS, key=lambda b: (vm[b], b))
                how = f"validation ({int(m['val'].sum())} rows)"
            else:
                beta, vm, how = 0.5, {}, "fixed 0.5 (no same-regime validation)"
            preds["smoother"] = smoother(f_te, beta)
            preds["smoother_b05"] = smoother(f_te, 0.5)
            for b in BETAS[1:]:
                preds[f"sm_{b}"] = smoother(f_te, b)
            for gv in gl:
                preds[gv] = gl[gv][h]
            y, lat = tab.y[te], tab.latest[te]
            oracle = min(BETAS, key=lambda b: (mae(smoother(f_te, b), y), b))
            beta_info[f"{run['name']}/+{h}"] = {"chosen": beta, "how": how, "test_oracle": oracle}
            store[(run["name"], h)] = {
                "y": y,
                "lat": lat,
                "station": tab.station[te],
                "day": tab.target_day[te],
                "origin": tab.origin_idx[te],
                "src": tab.source_target[te],
                "preds": preds,
            }
            log(
                f"{run['name']} +{h}: n={len(y)} pers {mae(lat, y):.3f} smoother(b={beta}) "
                f"{mae(preds['smoother'], y):.3f} direct {mae(preds['lgbm_direct'], y):.3f} "
                f"logr {mae(preds['lgbm_logratio'], y):.3f} ridge {mae(preds['ridge'], y):.3f}"
            )

    def pooled(members, h):
        recs = [store[(r, h)] for r in members]
        P = {
            k: np.concatenate([r[k] for r in recs])
            for k in ("y", "lat", "station", "day", "origin", "src")
        }
        keys = set.intersection(*[set(r["preds"]) for r in recs])
        P["preds"] = {c: np.concatenate([r["preds"][c] for r in recs]) for c in keys}
        P["run"] = np.concatenate(
            [np.full(len(r["y"]), n) for r, n in zip(recs, members, strict=True)]
        )
        return P

    results = {
        "info": info,
        "hma": hma,
        "beta": beta_info,
        "runs": {
            r["name"]: {
                k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in r.items()
            }
            for r in runs
        },
    }
    unique = {}
    for g, members in GROUPS.items():
        P = pooled(members, 24)
        keys = set(zip(P["station"].tolist(), P["origin"].tolist(), strict=True))
        assert len(keys) == len(P["y"]), f"duplicate station-hours in {g}"
        unique[g] = {
            "unique_station_hours_plus24": len(keys),
            "unique_target_days": len(np.unique(P["day"])),
        }
    results["unique_counts"] = unique
    log(f"unique counts {unique}")

    n_claims = len(FAMILY) * len(HORIZONS) * len(HEADLINE_GROUPS)
    qb = 0.05 / n_claims
    groups: dict = {}
    claims = []
    for g, members in GROUPS.items():
        for h in HORIZONS:
            P = pooled(members, h)
            G = {"persistence": metrics(P["y"], P["lat"], P["lat"])}
            for c, p in P["preds"].items():
                diff = np.abs(p - P["y"]) - np.abs(P["lat"] - P["y"])
                r = metrics(P["y"], p, P["lat"])
                fam = c in FAMILY and g in HEADLINE_GROUPS
                r["vs_persistence_1d"] = ci(diff, P["day"], 1, qb if fam else None)
                r["vs_persistence_7d"] = ci(diff, P["day"], 7, qb if fam else None)
                r["high_latest_gt_37.5"] = hp_block(P["y"], p, P["lat"], P["day"])
                r["events"] = event_block(P["y"], p, P["lat"], P["day"])
                if fam:
                    claims.append((g, h, c))
                G[c] = r
            G["persistence_events"] = event_block(P["y"], P["lat"], P["lat"], P["day"])
            G["runs_beating_persistence"] = {
                c: f"{sum(mae(store[(rn, h)]['preds'][c], store[(rn, h)]['y']) < mae(store[(rn, h)]['lat'], store[(rn, h)]['y']) for rn in members)}/{len(members)}"
                for c in P["preds"]
            }
            groups[f"{g}/+{h}"] = G
        log(f"group {g} done")
    for kind in ("1d", "7d"):
        ps = np.array([groups[f"{g}/+{h}"][c][f"vs_persistence_{kind}"]["p"] for g, h, c in claims])
        adj = holm(ps)
        for (g, h, c), a in zip(claims, adj, strict=True):
            groups[f"{g}/+{h}"][c][f"vs_persistence_{kind}"]["p_holm"] = float(a)
    results["groups"] = groups
    results["n_claims"] = n_claims

    per_run = {}
    for (rn, h), rec in store.items():
        e = {"persistence_mae": mae(rec["lat"], rec["y"]), "n": len(rec["y"])}
        for c, p in rec["preds"].items():
            diff = np.abs(p - rec["y"]) - np.abs(rec["lat"] - rec["y"])
            e[c] = {"mae": mae(p, rec["y"]), "diff": float(diff.mean())}
            if c in FAMILY or c == "smoother_b05":
                e[c]["ci_1d"] = ci(diff, rec["day"], 1)
        per_run[f"{rn}/+{h}"] = e
    results["per_run"] = per_run
    results["fit_counts"] = dict(FIT_COUNT)
    results["total_runtime_s"] = time.time() - t0
    (args.out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
