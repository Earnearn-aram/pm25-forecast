"""KAN-65 single entry point: EDA, leakage audit, walk-forward backtest, report JSON.

PYTHONPATH=src python -m pm25_forecast.experiments.run_all --database data/history.duckdb \
    --out out/kan65
Offline and read-only. See docs/kan-65-ds-plan.md (pre-registration and amendments).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import scipy

from . import eda
from .common import BOOT_SEED, EXPECTED_SHA, HIGH, STATIONS, Grid
from .dataset import ALL_FEATURES, BKK, SET_FEATURES, build_table
from .fitters import (
    FIT_COUNT,
    forward,
    inverse,
    lgb_fit_es,
    lgb_params,
    lgb_refit,
    ridge_fit,
)

HORIZONS = (6, 12, 24)
SETS = ("A", "B", "D", "E")
KINDS = ("direct", "resid", "logratio")
LGB_GRID = [(o, nl) for o in ("l2", "l1") for nl in (7, 31)]
ALPHAS = (0.1, 1.0, 10.0, 100.0)
CANDS = (
    "seasonal",
    "current",
    "smooth",
    "ridge",
    "lgbm_direct",
    "lgbm_resid",
    "ens",
    "shrink",
    "global",
    "global_id",
)
GROUPS = {
    "HOLDOUT": ["HOLDOUT"],
    "REGIME_B": ["F4", "R1", "R2", "R3"],
    "REGIME_A": ["F1", "F2", "F3"],
}


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
        },
        {
            "name": "F2",
            "B": ix(d(2026, 3, 31)),
            "t0": ix(d(2026, 4, 1)),
            "t1": ix(d(2026, 5, 13)),
            "vd": 45,
        },
        {
            "name": "F3",
            "B": ix(d(2026, 5, 31)),
            "t0": ix(d(2026, 6, 1)),
            "t1": ix(d(2026, 7, 13)),
            "vd": 45,
        },
        {
            "name": "F4",
            "B": ix(d(2026, 7, 31)),
            "t0": ix(d(2026, 8, 1)),
            "t1": ix(d(2026, 9, 12)),
            "vd": 45,
        },
        {
            "name": "R1",
            "B": ix(d(2026, 9, 1)),
            "t0": ix(d(2026, 9, 1)),
            "t1": ix(d(2026, 9, 22)),
            "vd": 10,
            "native": True,
        },
        {
            "name": "R2",
            "B": ix(d(2026, 9, 22)),
            "t0": ix(d(2026, 9, 22)),
            "t1": ix(d(2026, 10, 13)),
            "vd": 10,
            "native": True,
        },
        {
            "name": "R3",
            "B": ix(d(2026, 10, 13)),
            "t0": ix(d(2026, 10, 13)),
            "t1": end,
            "vd": 10,
            "native": True,
        },
        {"name": "HOLDOUT", "B": test_start_idx, "t0": test_start_idx, "t1": end, "vd": 45},
    ]


def split(tab, run):
    o, t = tab.origin_idx, tab.target_idx
    vs = run["B"] - run["vd"] * 24
    m = {
        "train": t < run["B"],
        "fit": t < vs,
        "val": (o >= vs) & (t < run["B"]),
        "test": (o >= run["t0"]) & (o < run["t1"]),
    }
    if run.get("native"):
        m["val"] &= tab.source_target == 2
    return m


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def search_horizon(tab, run, h, log):
    m = split(tab, run)
    native = run.get("native", False)
    variants = ("bonly", "ab") if native else ("all",)
    best: dict = {}

    def upd(fam, val_mae, vpred, meta):
        if fam not in best or val_mae < best[fam]["val_mae"]:
            best[fam] = {"val_mae": val_mae, "vpred": vpred, "meta": meta}

    lat_v, y_v = tab.latest[m["val"]], tab.y[m["val"]]
    for var in variants:
        rf = (tab.source_target == 2) if var == "bonly" else np.ones(len(tab.y), bool)
        extra = ["src_air4thai"] if var == "ab" else []
        fit_m, _tr_m = m["fit"] & rf, m["train"] & rf
        if fit_m.sum() < 300 or m["val"].sum() < 50:
            log(f"  {run['name']} +{h}: skip {var} (fit={fit_m.sum()}, val={m['val'].sum()})")
            continue
        for sname in SETS:
            names = SET_FEATURES[sname] + extra
            X = tab.X(names)
            cat = ["station_id"] if "station_id" in names else "auto"
            for kind in KINDS:
                yt = forward(kind, tab.y, tab.latest)
                fam = "lgbm_direct" if kind == "direct" else "lgbm_resid"
                for obj, nl in LGB_GRID:
                    b, it = lgb_fit_es(
                        lgb_params(obj, nl),
                        X[fit_m],
                        yt[fit_m],
                        X[m["val"]],
                        yt[m["val"]],
                        names,
                        cat,
                    )
                    vp = inverse(kind, b.predict(X[m["val"]], num_iteration=it), lat_v)
                    upd(
                        fam,
                        mae(vp, y_v),
                        vp,
                        {
                            "model": "lgbm",
                            "var": var,
                            "set": sname,
                            "kind": kind,
                            "obj": obj,
                            "leaves": nl,
                            "iters": it,
                            "names": names,
                            "cat": cat,
                        },
                    )
                for al in ALPHAS:
                    r = ridge_fit(X[fit_m], yt[fit_m], names, al)
                    vp = inverse(kind, r.predict(X[m["val"]]), lat_v)
                    upd(
                        "ridge",
                        mae(vp, y_v),
                        vp,
                        {
                            "model": "ridge",
                            "var": var,
                            "set": sname,
                            "kind": kind,
                            "alpha": al,
                            "names": names,
                        },
                    )
    # simple smoothers on validation (no fitting beyond choosing the variant)
    f = tab.feats
    sm_opts = {
        "latest": lambda s: s["pm25_latest"],
        "mean3": lambda s: s["mean3"],
        "mean6": lambda s: s["mean6"],
        "mean24": lambda s: s["mean24"],
    }
    for beta in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        sm_opts[f"shrink_mean24_{beta}"] = lambda s, b=beta: (
            s["pm25_latest"] + b * (s["mean24"] - s["pm25_latest"])
        )
    vfeats = {k: v[m["val"]] for k, v in f.items()}
    for nm, fn in sm_opts.items():
        vp = fn(vfeats)
        upd("smooth", mae(vp, y_v), vp, {"model": "smooth", "name": nm})
    return best, m


def predict_test(best, tab, m, run, h):
    """Refit each selected config on all train rows (variant rows only), predict test rows."""
    te = m["test"]
    lat_t = tab.latest[te]
    out = {}
    for fam in ("lgbm_direct", "lgbm_resid", "ridge", "smooth"):
        if fam not in best:
            continue
        meta = best[fam]["meta"]
        if meta["model"] == "smooth":
            fs = {k: v[te] for k, v in tab.feats.items()}
            nm = meta["name"]
            if nm.startswith("shrink_mean24_"):
                b = float(nm.rsplit("_", 1)[1])
                out[fam] = fs["pm25_latest"] + b * (fs["mean24"] - fs["pm25_latest"])
            else:
                out[fam] = fs["pm25_latest" if nm == "latest" else nm]
            continue
        rf = (tab.source_target == 2) if meta["var"] == "bonly" else np.ones(len(tab.y), bool)
        tr = m["train"] & rf
        X = tab.X(meta["names"])
        yt = forward(meta["kind"], tab.y, tab.latest)
        if meta["model"] == "lgbm":
            bst = lgb_refit(
                lgb_params(meta["obj"], meta["leaves"]),
                X[tr],
                yt[tr],
                meta["names"],
                meta["cat"],
                meta["iters"],
            )
            out[fam] = inverse(meta["kind"], bst.predict(X[te]), lat_t)
        else:
            r = ridge_fit(X[tr], yt[tr], meta["names"], meta["alpha"])
            out[fam] = inverse(meta["kind"], r.predict(X[te]), lat_t)
    # ensemble and shrink toward persistence
    if "ridge" in out and "lgbm_resid" in out:
        out["ens"] = 0.5 * (out["ridge"] + out["lgbm_resid"])
    if "lgbm_resid" in out:
        vp, lat_v, y_v = best["lgbm_resid"]["vpred"], tab.latest[m["val"]], tab.y[m["val"]]
        ws = (0.25, 0.5, 0.75)
        w = min(ws, key=lambda w_: mae(lat_v + w_ * (vp - lat_v), y_v))
        out["shrink"] = lat_t + w * (out["lgbm_resid"] - lat_t)
        best["lgbm_resid"]["meta"]["shrink_w"] = w
    # current production-style per-horizon model: latest + same-hour-24h reading, direct y,
    # KAN-15 parameters (regression, 200 rounds), untuned, trained on all rows before B
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
    FIT_COUNT["lgbm_refit"] += 1
    out["current"] = np.maximum(0.0, bst.predict(Xc[te]))
    out["seasonal"] = tab.feats["pm25_t24" if h < 24 else "pm25_t48"][te]
    return out


def global_search(tabs, run, log):
    """One LightGBM over stacked horizons 6/12/24 with horizon as a feature (G, G+id)."""
    native = run.get("native", False)
    variants = ("bonly", "ab") if native else ("all",)
    base = [n for n in SET_FEATURES["D"] if n != "station_id"]
    res = {}
    masks = {h: split(tabs[h], run) for h in HORIZONS}

    def stack(var, part, names, kind):
        xs, ys, ls, hs = [], [], [], []
        for h in HORIZONS:
            tab, mk = tabs[h], masks[h]
            rf = (tab.source_target == 2) if var == "bonly" and part != "val" else True
            sel = mk[part] & rf
            x = tab.X(names)[sel]
            xs.append(np.column_stack([x, np.full(sel.sum(), float(h))]))
            ys.append(forward(kind, tab.y, tab.latest)[sel])
            ls.append(tab.latest[sel])
            hs.append(tab.y[sel])
        return np.vstack(xs), np.concatenate(ys), np.concatenate(ls), np.concatenate(hs)

    for gv in ("global", "global_id"):
        best = None
        for var in variants:
            extra = ["src_air4thai"] if var == "ab" else []
            names = base + (["station_id"] if gv == "global_id" else []) + extra + ["horizon"]
            cat = ["station_id"] if gv == "global_id" else "auto"
            for kind in ("direct", "resid"):
                xf, yf, _, _ = stack(var, "fit", names[:-1], kind)
                xv, yv, lv, ytrue = stack(var, "val", names[:-1], kind)
                if len(xf) < 600 or len(xv) < 100:
                    continue
                for obj, nl in LGB_GRID:
                    b, it = lgb_fit_es(lgb_params(obj, nl), xf, yf, xv, yv, names, cat)
                    vp = inverse(kind, b.predict(xv, num_iteration=it), lv)
                    v = mae(vp, ytrue)
                    if best is None or v < best["val_mae"]:
                        best = {
                            "val_mae": v,
                            "var": var,
                            "kind": kind,
                            "obj": obj,
                            "leaves": nl,
                            "iters": it,
                            "names": names,
                            "cat": cat,
                        }
        if best is None:
            continue
        xt, yt_, _, _ = stack(best["var"], "train", best["names"][:-1], best["kind"])
        bst = lgb_refit(
            lgb_params(best["obj"], best["leaves"]),
            xt,
            yt_,
            best["names"],
            best["cat"],
            best["iters"],
        )
        preds = {}
        for h in HORIZONS:
            tab, te = tabs[h], masks[h]["test"]
            x = np.column_stack([tab.X(best["names"][:-1])[te], np.full(te.sum(), float(h))])
            preds[h] = inverse(best["kind"], bst.predict(x), tab.latest[te])
        res[gv] = {"preds": preds, "meta": {k: v for k, v in best.items() if k != "names"}}
        log(f"  {run['name']} {gv}: {res[gv]['meta']}")
    return res


# ---------------------------------------------------------------- statistics
def boot_draws(diff, days, n_boot, seed=BOOT_SEED):
    uniq, pos = np.unique(days, return_inverse=True)
    nb = len(uniq)
    sums = np.bincount(pos, weights=diff, minlength=nb)
    cnts = np.bincount(pos, minlength=nb).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    step = 5000
    for i in range(0, n_boot, step):
        dr = rng.integers(0, nb, size=(min(step, n_boot - i), nb))
        out[i : i + step] = sums[dr].sum(1) / cnts[dr].sum(1)
    return out


def ci_pack(diff, days, n_boot, q_bonf=None):
    dr = boot_draws(diff, days, n_boot)
    res = {
        "n": len(diff),
        "days": len(np.unique(days)),
        "diff": float(diff.mean()),
        "lo": float(np.quantile(dr, 0.025)),
        "hi": float(np.quantile(dr, 0.975)),
        "p": float((min((dr <= 0).sum(), (dr >= 0).sum()) * 2 + 1) / (n_boot + 1)),
    }
    if q_bonf:
        res["lo_bonf"] = float(np.quantile(dr, q_bonf / 2))
        res["hi_bonf"] = float(np.quantile(dr, 1 - q_bonf / 2))
    return res


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run_max, m = 0.0, len(ps)
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * ps[i]))
        adj[i] = run_max
    return adj


def metrics(y, p, lat):
    e = p - y
    out = {
        "n": len(y),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "bias": float(e.mean()),
    }
    hi = y > HIGH
    out["high_n"] = int(hi.sum())
    out["high_mae"] = float(np.abs(e[hi]).mean()) if hi.any() else None
    for nm, lo_, hi_ in (("le15", -1, 15), ("15_37.5", 15, HIGH), ("gt37.5", HIGH, 1e9)):
        mk = (y > lo_) & (y <= hi_)
        out[f"mae_band_{nm}"] = float(np.abs(e[mk]).mean()) if mk.any() else None
        out[f"bias_band_{nm}"] = float(e[mk].mean()) if mk.any() else None
        out[f"n_band_{nm}"] = int(mk.sum())
    return out


# ---------------------------------------------------------------- leakage audit
def audit(grid: Grid, tabs, seed=7, n=300):
    rng = np.random.default_rng(seed)
    out = {}
    for h, tab in tabs.items():
        rows = rng.choice(len(tab.y), size=n, replace=False)
        worst: dict = dict.fromkeys(ALL_FEATURES, 0.0)
        mism = 0
        for r in rows:
            s, o = str(tab.station[r]), int(tab.origin_idx[r])
            t = int(tab.target_idx[r])

            def get(st, i, o=o):
                if i < 0 or i > o - 1 or i >= grid.n:  # as-of: nothing after r = o - 1
                    return np.nan
                return grid.values[st][i]

            def latest(st, o=o):
                for k in (1, 2, 3):
                    v = get(st, o - k)
                    if not np.isnan(v):
                        return v
                return np.nan

            def win(st, w, o=o):
                v = np.array([get(st, o - k) for k in range(1, w + 1)])
                v = v[~np.isnan(v)]
                if len(v) == 0:
                    return np.nan, np.nan, np.nan
                return v.mean(), v.max(), (v.std(ddof=1) if len(v) > 1 else np.nan)

            lt = grid.time(t) + BKK
            exp = {"pm25_latest": latest(s)}
            exp["hour_sin"] = np.sin(2 * np.pi * lt.hour / 24)
            exp["hour_cos"] = np.cos(2 * np.pi * lt.hour / 24)
            exp["dow_sin"] = np.sin(2 * np.pi * lt.weekday() / 7)
            exp["dow_cos"] = np.cos(2 * np.pi * lt.weekday() / 7)
            exp["month_sin"] = np.sin(2 * np.pi * (lt.month - 1) / 12)
            exp["month_cos"] = np.cos(2 * np.pi * (lt.month - 1) / 12)
            exp["station_id"] = float(STATIONS.index(s))
            for nm, back in (
                ("pm25_t24", 24),
                ("pm25_t48", 48),
                ("pm25_t72", 72),
                ("pm25_t168", 168),
            ):
                exp[nm] = get(s, t - back)
            for w in (3, 6, 24):
                exp[f"mean{w}"], exp[f"max{w}"], exp[f"std{w}"] = win(s, w)
            exp["trend3"] = exp["pm25_latest"] - get(s, o - 4)
            exp["trend6"] = exp["pm25_latest"] - get(s, o - 7)
            oth = [x for x in STATIONS if x != s]
            nl = [latest(x) for x in oth]
            nm24 = [win(x, 24)[0] for x in oth]
            exp["nbr_latest_mean"] = np.nan if np.all(np.isnan(nl)) else np.nanmean(nl)
            exp["nbr_mean24"] = np.nan if np.all(np.isnan(nm24)) else np.nanmean(nm24)
            for k in ALL_FEATURES:
                a, b = tab.feats[k][r], exp[k]
                if np.isnan(a) and np.isnan(b):
                    continue
                dd = abs(a - b) if not (np.isnan(a) or np.isnan(b)) else np.inf
                worst[k] = max(worst[k], dd)
                mism += dd > 1e-9
        out[f"+{h}"] = {
            "rows": n,
            "mismatching_cells": int(mism),
            "max_abs_diff": {k: float(v) for k, v in worst.items()},
            "t24_all_nan_at_h24": bool(h != 24 or np.isnan(tab.feats["pm25_t24"]).all()),
        }
    return out


def leaky_control(tabs, runs, grid):
    tab = tabs[24]
    run = next(r for r in runs if r["name"] == "HOLDOUT")
    m = split(tab, run)
    names = SET_FEATURES["D"] + ["leak_nbr_target"]
    X = tab.X(names)
    b = lgb_refit(
        lgb_params("l2", 31), X[m["train"]], tab.y[m["train"]], names, ["station_id"], 200
    )
    p = np.maximum(0, b.predict(X[m["test"]]))
    y, lat = tab.y[m["test"]], tab.latest[m["test"]]
    diff = np.abs(p - y) - np.abs(lat - y)
    return {
        "description": "set D + neighbours' mean reading AT the target hour (future info)",
        "mae": mae(p, y),
        "persistence_mae": mae(lat, y),
        "paired": ci_pack(diff, tab.target_day[m["test"]], 5000),
    }


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", type=Path, default=Path("data/history.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("out/kan65"))
    ap.add_argument("--only", default="", help="comma list of run names (debug)")
    args = ap.parse_args(argv)
    t_start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    def log(s):
        print(f"[{time.time() - t_start:7.0f}s] {s}", flush=True)

    grid = Grid(args.database)
    info = {
        "snapshot_sha256": grid.sha,
        "matches_kan15_snapshot": grid.sha == EXPECTED_SHA,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lgb.__version__,
        "duckdb": duckdb.__version__,
        "boot_seed": BOOT_SEED,
    }
    log(f"snapshot {grid.sha} matches={info['matches_kan15_snapshot']}")
    tabs = {h: build_table(grid, h) for h in HORIZONS}
    runs = make_runs(grid, tabs[24].test_start_idx)
    all_runs = list(runs)
    if args.only:
        runs = [r for r in runs if r["name"] in args.only.split(",")]
    log(f"pairs per horizon: { {h: len(t.y) for h, t in tabs.items()} }")

    (args.out / "eda.json").write_text(json.dumps(eda.run(grid), indent=1))
    log("eda done")
    aud = audit(grid, tabs)
    log(f"audit: { {k: v['mismatching_cells'] for k, v in aud.items()} }")

    store: dict = {}  # (run, h) -> dict(y, lat, ... preds{})
    selections: dict = {}
    for run in runs:
        gl = global_search(tabs, run, log)
        for h in HORIZONS:
            tab = tabs[h]
            best, m = search_horizon(tab, run, h, log)
            if "lgbm_resid" not in best:
                log(f"{run['name']} +{h}: no data for tuning, skipped")
                continue
            preds = predict_test(best, tab, m, run, h)
            for gv, g in gl.items():
                preds[gv] = g["preds"][h]
            te = m["test"]
            rec = {
                "y": tab.y[te],
                "lat": tab.latest[te],
                "station": tab.station[te],
                "day": tab.target_day[te],
                "src": tab.source_target[te],
                "local": [grid.time(int(i)) + BKK for i in tab.target_idx[te]],
                "preds": preds,
            }
            store[(run["name"], h)] = rec
            selections[f"{run['name']}/+{h}"] = {
                fam: {k: v for k, v in b["meta"].items() if k not in ("names", "cat")}
                | {"val_mae": b["val_mae"], "names": b["meta"].get("names")}
                for fam, b in best.items()
            } | {"global": {k: v["meta"] for k, v in gl.items()}}
            log(
                f"{run['name']} +{h}: n_test={int(te.sum())} persistence MAE "
                f"{mae(rec['lat'], rec['y']):.3f}  "
                + "  ".join(
                    f"{c}={mae(p[np.isfinite(p)], rec['y'][np.isfinite(p)]):.3f}"
                    for c, p in preds.items()
                    if c in CANDS
                )
            )

    # ---------------- statistics
    results: dict = {
        "info": info,
        "audit": aud,
        "selections": selections,
        "runs": {
            r["name"]: {
                k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in r.items()
            }
            for r in runs
        },
    }
    claims = []
    per_run: dict = {}
    nclaim = len(CANDS) * len(HORIZONS) * len(GROUPS)
    qb = 0.05 / nclaim

    def pooled(runs_in, h):
        recs = [store[(r, h)] for r in runs_in if (r, h) in store]
        if not recs:
            return None
        cat = {
            k: np.concatenate([r[k] for r in recs]) for k in ("y", "lat", "station", "day", "src")
        }
        cat["local"] = [x for r in recs for x in r["local"]]
        cat["preds"] = {
            c: np.concatenate([r["preds"][c] for r in recs if c in r["preds"]])
            for c in CANDS
            if all(c in r["preds"] for r in recs)
        }
        return cat

    for h in HORIZONS:
        for (rn, hh), rec in store.items():
            if hh != h:
                continue
            e = {"persistence": metrics(rec["y"], rec["lat"], rec["lat"])}
            for c, p in rec["preds"].items():
                ok = np.isfinite(p)
                e[c] = metrics(rec["y"][ok], p[ok], rec["lat"][ok])
                diff = np.abs(p[ok] - rec["y"][ok]) - np.abs(rec["lat"][ok] - rec["y"][ok])
                e[c]["vs_persistence"] = ci_pack(diff, rec["day"][ok], 10000)
            per_run[f"{rn}/+{h}"] = e
        log(f"per-run stats +{h} done")
    results["per_run"] = per_run

    group_res: dict = {}
    for gname, members in GROUPS.items():
        for h in HORIZONS:
            P = pooled(members, h)
            if P is None:
                continue
            gr = {"persistence": metrics(P["y"], P["lat"], P["lat"])}
            cur = P["preds"].get("current")
            for c, p in P["preds"].items():
                ok = np.isfinite(p)
                y, lat, days = P["y"][ok], P["lat"][ok], P["day"][ok]
                diff = np.abs(p[ok] - y) - np.abs(lat - y)
                r = metrics(y, p[ok], lat)
                r["vs_persistence"] = ci_pack(diff, days, 50000, q_bonf=qb)
                # point-estimate difference in the two high-pollution subsets (target / latest)
                for nm, mk in (("high_by_target", y > HIGH), ("high_by_latest", lat > HIGH)):
                    r[f"vs_persistence_{nm}"] = (
                        ci_pack(diff[mk], days[mk], 10000) if mk.sum() >= 10 else None
                    )
                if c != "current" and cur is not None:
                    dc = np.abs(p - P["y"]) - np.abs(cur - P["y"])
                    okc = np.isfinite(dc)
                    r["vs_current"] = ci_pack(dc[okc], P["day"][okc], 20000)
                gr[c] = r
                if c in CANDS:
                    claims.append((gname, h, c, r["vs_persistence"]["p"]))
            # stability across member runs (point MAE beats persistence)
            stab = {}
            for c in CANDS:
                wins, tot = 0, 0
                for rn in members:
                    rec = store.get((rn, h))
                    if rec is None or c not in rec["preds"]:
                        continue
                    ok = np.isfinite(rec["preds"][c])
                    tot += 1
                    wins += mae(rec["preds"][c][ok], rec["y"][ok]) < mae(
                        rec["lat"][ok], rec["y"][ok]
                    )
                stab[c] = f"{wins}/{tot}"
            gr["stability_runs_beating_persistence"] = stab
            group_res[f"{gname}/+{h}"] = gr
        log(f"group stats {gname} done")
    adj = holm(np.array([c[3] for c in claims]))
    for (gname, h, c, _), a in zip(claims, adj, strict=True):
        group_res[f"{gname}/+{h}"][c]["vs_persistence"]["p_holm"] = float(a)
    results["groups"] = group_res
    results["n_claims"] = len(claims)

    # breakdowns at +24 (diagnostic, descriptive)
    brk: dict = {}
    for gname, members in GROUPS.items():
        P = pooled(members, 24)
        if P is None:
            continue
        loc = P["local"]
        hg = np.array([x.hour // 6 for x in loc])
        mo = np.array([x.strftime("%Y-%m") for x in loc])
        views = {"station": P["station"], "hour_group_local": hg, "month": mo, "source": P["src"]}
        b = {}
        for vname, arr in views.items():
            b[vname] = {}
            for key in sorted(set(arr.tolist())):
                mk = arr == key
                row = {"n": int(mk.sum()), "persistence": mae(P["lat"][mk], P["y"][mk])}
                for c, p in P["preds"].items():
                    ok = mk & np.isfinite(p)
                    row[c] = mae(p[ok], P["y"][ok]) if ok.any() else None
                b[vname][str(key)] = row
        brk[gname] = b
    results["breakdown_plus24"] = brk
    results["leaky_control"] = leaky_control(tabs, all_runs, grid)
    log(f"leaky control: {results['leaky_control']}")
    results["fit_counts"] = dict(FIT_COUNT)
    results["total_runtime_s"] = time.time() - t_start
    (args.out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    log(f"done: fits {FIT_COUNT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
