"""KAN-75 PoC: do lead-matched +24 h weather forecasts improve the +24 h PM2.5 forecast?

Follows docs/kan-75-poc-plan.md (pre-registered). All numbers go to out/kan75/results.json.
Run from the worktree root:  PYTHONPATH=src python -m pm25_forecast.experiments.kan75_run
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .common import STATIONS, Grid
from .dataset import Table, build_table
from .kan75_fetch import RAW_DIR, VARS

DB = Path("/Users/earn/AI Project/pm25-forecast/data/history.duckdb")
OUT = Path("out/kan75")
START = datetime(2026, 7, 31, 17, tzinfo=UTC)
END = datetime(2026, 10, 26, tzinfo=UTC)  # KAN-74 cut: exclusive
FOLDS = {  # test windows by origin time [t0, t1)
    "G1": (datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 22, tzinfo=UTC)),
    "G2": (datetime(2026, 9, 22, tzinfo=UTC), datetime(2026, 10, 13, tzinfo=UTC)),
    "G3": (datetime(2026, 10, 13, tzinfo=UTC), END),
}
VAL_DAYS = 10
BETAS = [round(0.1 * i, 1) for i in range(10)]
SEEDS = [20261029 + i for i in range(5)]
BOOT_SEED = 20261030
N_BOOT = 10000
ONSET_RISE, CATCH_RISE = 15.0, 7.5
PM_FEATS = [
    "pm25_latest", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "station_id", "pm25_t24", "pm25_t48", "pm25_t72", "pm25_t168", "mean3", "max3", "std3",
    "mean6", "max6", "std6", "mean24", "max24", "std24", "trend3", "trend6",
]  # fmt: skip
WX_FEATS = [
    "wx_wind_speed", "wx_wind_dir_sin", "wx_wind_dir_cos", "wx_rh", "wx_temp", "wx_precip",
    "wx_pressure", "wx_cloud", "wx_d_temp", "wx_d_pressure", "wx_d_wind",
]  # fmt: skip
VAR_KEY = {
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir",
    "relative_humidity_2m": "rh", "temperature_2m": "temp", "precipitation": "precip",
    "surface_pressure": "pressure", "cloud_cover": "cloud",
}  # fmt: skip


# ------------------------------------------------------------------ weather
def load_weather(grid: Grid, lead: str) -> dict[str, dict[str, np.ndarray]]:
    """{station: {var_key: array on the grid}} from the cached Previous Runs responses."""
    out: dict[str, dict[str, np.ndarray]] = {}
    t0 = int(grid.t0.timestamp())
    for st in STATIONS:
        h = json.loads((RAW_DIR / f"{st}.json").read_text())["hourly"]
        idx = (np.asarray(h["time"]) - t0) // 3600
        d = {}
        for v in VARS:
            a = np.full(grid.n, np.nan)
            vals = np.array([np.nan if x is None else x for x in h[f"{v}_{lead}"]], float)
            ok = (idx >= 0) & (idx < grid.n)
            a[idx[ok]] = vals[ok]
            d[VAR_KEY[v]] = a
        out[st] = d
    return out


def add_weather(tab: Table, wx: dict[str, dict[str, np.ndarray]]) -> None:
    """Features at the target hour t (valid time) and the change t minus origin hour o."""
    cols = {k: np.full(len(tab.y), np.nan) for k in WX_FEATS}
    for st in STATIONS:
        m = tab.station == st
        o, t = tab.origin_idx[m], tab.target_idx[m]
        w = wx[st]
        cols["wx_wind_speed"][m] = w["wind_speed"][t]
        rad = np.deg2rad(w["wind_dir"][t])
        cols["wx_wind_dir_sin"][m], cols["wx_wind_dir_cos"][m] = np.sin(rad), np.cos(rad)
        cols["wx_rh"][m] = w["rh"][t]
        cols["wx_temp"][m] = w["temp"][t]
        cols["wx_precip"][m] = w["precip"][t]
        cols["wx_pressure"][m] = w["pressure"][t]
        cols["wx_cloud"][m] = w["cloud"][t]
        cols["wx_d_temp"][m] = w["temp"][t] - w["temp"][o]
        cols["wx_d_pressure"][m] = w["pressure"][t] - w["pressure"][o]
        cols["wx_d_wind"][m] = w["wind_speed"][t] - w["wind_speed"][o]
    tab.feats.update(cols)


# ------------------------------------------------------------------ models
def lgb_params(seed: int) -> dict:
    return {
        "objective": "l1", "metric": "l1", "learning_rate": 0.05, "num_leaves": 7,
        "min_child_samples": 50, "feature_fraction": 0.9, "seed": seed,
        "feature_fraction_seed": seed, "bagging_seed": seed, "deterministic": True,
        "force_row_wise": True, "verbosity": -1, "num_threads": 1,
    }  # fmt: skip


def fit_lgb(tab: Table, m: np.ndarray, names: list[str], seed: int) -> lgb.Booster:
    ds = lgb.Dataset(
        tab.X(names)[m], label=tab.y[m], feature_name=names, categorical_feature=["station_id"]
    )
    return lgb.train(lgb_params(seed), ds, num_boost_round=150)


def pick_beta(tab: Table, train: np.ndarray, val: np.ndarray) -> float:
    ok = val & np.isfinite(tab.feats["mean24"])
    lat, m24, y = tab.latest[ok], tab.feats["mean24"][ok], tab.y[ok]
    maes = [np.abs(lat + b * (m24 - lat) - y).mean() for b in BETAS]
    return BETAS[int(np.argmin(maes))]  # ties go to the smaller beta


# ------------------------------------------------------------------ stats
class Blocks:
    """Calendar 7-day blocks of target date; bootstrap resamples whole blocks."""

    def __init__(self, days: np.ndarray):
        self.blk = (days - days.min()) // 7
        self.nb = int(self.blk.max()) + 1
        self.draws = np.random.default_rng(BOOT_SEED).integers(0, self.nb, (N_BOOT, self.nb))

    def sums(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        w = x if mask is None else np.where(mask, x, 0.0)
        return np.bincount(self.blk, weights=w, minlength=self.nb)

    def ci(self, num: np.ndarray, den: np.ndarray, level: float = 0.95):
        """Ratio-of-sums statistic sum(num)/sum(den) over blocks: (point, lo, hi)."""
        s, c = self.sums(num), self.sums(den)
        pt = s.sum() / c.sum()
        bs = s[self.draws].sum(1) / c[self.draws].sum(1)
        a = (1 - level) / 2
        return float(pt), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))

    def gain_ci(self, e_new: np.ndarray, e_ref: np.ndarray):
        s_n, s_r = self.sums(e_new), self.sums(e_ref)
        bs = 1 - s_n[self.draws].sum(1) / s_r[self.draws].sum(1)
        pt = 1 - s_n.sum() / s_r.sum()
        return float(pt), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def pack_diff(bl: Blocks, e_a: np.ndarray, e_b: np.ndarray, n_comp: int = 3) -> dict:
    """Paired MAE difference a-b (negative: a better), 95% and Bonferroni(n_comp) CIs."""
    d = e_a - e_b
    one = np.ones_like(d)
    r = {"diff": bl.ci(d, one)}
    r["diff_bonf"] = bl.ci(d, one, level=1 - 0.05 / n_comp)[1:]
    r["rel_gain_a_vs_b"] = bl.gain_ci(e_a, e_b)
    r["share_rows_a_better"] = float((d < 0).mean())
    return r


def onset_stats(bl: Blocks, y, lat, preds: dict[str, np.ndarray]) -> dict:
    onset = (y - lat) >= ONSET_RISE
    out = {"n_onsets": int(onset.sum()), "n_onset_days": len(np.unique(bl.blk[onset]))}
    hit, far = {}, {}
    for k, p in preds.items():
        alarm = (p - lat) >= CATCH_RISE
        hit[k], far[k] = (alarm & onset).astype(float), (alarm & ~onset).astype(float)
        s_on, s_non = bl.sums(hit[k]), bl.sums(far[k])
        out[k] = {
            "caught_share": float(hit[k].sum() / max(onset.sum(), 1)),
            "false_alarm_share": float(far[k].sum() / max((~onset).sum(), 1)),
            "n_alarms": int((alarm).sum()),
            "_s": (s_on.tolist(), s_non.tolist()),
        }
    for k in out.copy():
        if isinstance(out[k], dict):
            out[k].pop("_s")
    on_w, on_n = bl.sums(onset.astype(float)), bl.sums((~onset).astype(float))

    def paired(a: str, b: str):
        d_hit = bl.sums(hit[a] - hit[b])
        d_far = bl.sums(far[a] - far[b])
        dr = bl.draws
        h = d_hit[dr].sum(1) / np.maximum(on_w[dr].sum(1), 1)
        f = d_far[dr].sum(1) / on_n[dr].sum(1)
        q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
        return {
            "caught_diff": float(d_hit.sum() / on_w.sum()), "caught_diff_ci": q(h),
            "false_alarm_diff": float(d_far.sum() / on_n.sum()), "false_alarm_diff_ci": q(f),
        }  # fmt: skip

    if "lgb_wx" in preds and "lgb_pm" in preds:
        out["wx_vs_pm_paired"] = paired("lgb_wx", "lgb_pm")
    return out


# ------------------------------------------------------------------ main
def run_variant(tab: Table, lead_tag: str, grid: Grid, log, drop: tuple[str, ...] = ()) -> dict:
    ix = grid.idx
    o, t = tab.origin_idx, tab.target_idx
    n = len(tab.y)
    names_pm = [f for f in PM_FEATS if f not in drop]
    names_wx = names_pm + WX_FEATS
    seed_preds = {
        "lgb_pm": np.full((len(SEEDS), n), np.nan),
        "lgb_wx": np.full((len(SEEDS), n), np.nan),
    }
    smooth = np.full(n, np.nan)
    betas, ev = {}, np.zeros(n, bool)
    fold_of = np.full(n, "", dtype=object)
    for name, (a, b) in FOLDS.items():
        B = ix(a)
        train = t < B
        test = (o >= B) & (o < ix(b))
        val = (o >= B - VAL_DAYS * 24) & train
        beta = pick_beta(tab, train, val)
        betas[name] = beta
        lat, m24 = tab.latest, tab.feats["mean24"]
        smooth[test] = lat[test] + beta * (m24[test] - lat[test])
        for si, sd in enumerate(SEEDS):
            for key, names in (("lgb_pm", names_pm), ("lgb_wx", names_wx)):
                bst = fit_lgb(tab, train, names, sd)
                seed_preds[key][si, test] = np.maximum(0.0, bst.predict(tab.X(names)[test]))
        ev |= test
        fold_of[test] = name
        log(f"[{lead_tag}] {name}: train={int(train.sum())} test={int(test.sum())} beta={beta}")
    ev &= np.isfinite(tab.y) & np.isfinite(tab.latest) & np.isfinite(tab.feats["mean24"])
    ev &= np.isfinite(tab.X(WX_FEATS)).all(1) & np.isfinite(smooth)
    y, lat = tab.y[ev], tab.latest[ev]
    preds = {
        "persistence": lat,
        "smoother": smooth[ev],
        "lgb_pm": seed_preds["lgb_pm"][:, ev].mean(0),
        "lgb_wx": seed_preds["lgb_wx"][:, ev].mean(0),
    }
    err = {k: np.abs(p - y) for k, p in preds.items()}
    days = tab.target_day[ev]
    bl = Blocks(days)
    res = {
        "lead": lead_tag,
        "n_rows": int(ev.sum()),
        "n_target_days": len(np.unique(days)),
        "n_blocks": bl.nb,
        "betas": betas,
        "mae": {k: float(e.mean()) for k, e in err.items()},
        "mae_ci7": {k: bl.ci(e, np.ones_like(e))[1:] for k, e in err.items()},
        "seed_pooled_mae": {
            k: [
                float(np.abs(np.maximum(0, seed_preds[k][i, ev]) - y).mean())
                for i in range(len(SEEDS))
            ]
            for k in ("lgb_pm", "lgb_wx")
        },
        "by_fold_mae": {
            f: {k: float(e[fold_of[ev] == f].mean()) for k, e in err.items()}
            | {"n": int((fold_of[ev] == f).sum())}
            for f in FOLDS
        },
        "wx_vs_smoother": pack_diff(bl, err["lgb_wx"], err["smoother"]),
        "wx_vs_persistence": pack_diff(bl, err["lgb_wx"], err["persistence"]),
        "wx_vs_lgb_pm": pack_diff(bl, err["lgb_wx"], err["lgb_pm"]),
        "lgb_pm_vs_smoother": pack_diff(bl, err["lgb_pm"], err["smoother"]),
        "smoother_vs_persistence": pack_diff(bl, err["smoother"], err["persistence"]),
        "onset": onset_stats(bl, y, lat, {k: v for k, v in preds.items() if k != "persistence"}),
        "bias": {k: float((p - y).mean()) for k, p in preds.items()},
    }
    # per-fold wx-vs-pm win count (consistency)
    res["wx_beats_pm_by_fold"] = {
        f: float(err["lgb_wx"][fold_of[ev] == f].mean() - err["lgb_pm"][fold_of[ev] == f].mean())
        for f in FOLDS
    }
    return res


def importance(tab: Table, grid: Grid) -> dict:
    """Descriptive: gain importance and mean |SHAP| of a weather LightGBM fit on all rows."""
    names = PM_FEATS + WX_FEATS
    m = np.isfinite(tab.y) & (tab.target_idx < grid.idx(END))
    bst = fit_lgb(tab, m, names, SEEDS[0])
    shap = bst.predict(tab.X(names)[m], pred_contrib=True)[:, :-1]
    gain = bst.feature_importance("gain")
    ms = np.abs(shap).mean(0)
    rows = sorted(zip(names, ms, gain / gain.sum(), strict=True), key=lambda r: -r[1])
    return {
        "n_fit_rows": int(m.sum()),
        "ranked": [
            {"feature": n, "mean_abs_shap": float(s), "gain_share": float(g)} for n, s, g in rows
        ],
        "weather_share_of_total_shap": float(
            sum(r[1] for r in rows if r[0].startswith("wx_")) / ms.sum()
        ),
    }  # in-sample on the same rows used to fit (descriptive only)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(s: str) -> None:
        print(s)
        lines.append(s)

    grid = Grid(DB, source="air4thai", start=START, end=END)
    log(f"db sha256 {grid.sha}")
    log(f"grid {grid.time(0)} .. {grid.t_last} ({grid.n} h)")
    results: dict = {"db_sha256": grid.sha, "grid_start": grid.t0.isoformat(),
                     "grid_end": grid.t_last.isoformat()}  # fmt: skip
    results["raw_rows_per_station"] = {s: int(np.isfinite(grid.values[s]).sum()) for s in STATIONS}
    results["variants"] = {}
    for lead in ("previous_day1", "previous_day2"):
        tab = build_table(grid)
        add_weather(tab, load_weather(grid, lead))
        results["variants"][lead] = run_variant(tab, lead, grid, log)
        if lead == "previous_day1":
            # POST-HOC (not in the pre-registered plan): drop month_sin/month_cos, which cannot
            # generalise with 3 calendar months of data. Descriptive sensitivity only.
            results["variants"]["previous_day1_posthoc_no_month"] = run_variant(
                tab, "previous_day1_posthoc_no_month", grid, log, drop=("month_sin", "month_cos")
            )
            results["importance_previous_day1"] = importance(tab, grid)
            # weather coverage check on the eval rows
            results["n_rows_table"] = len(tab.y)
            results["pm_nan_share"] = {
                k: float(np.isnan(tab.feats[k]).mean()) for k in ("pm25_t168", "mean24", "trend6")
            }
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    (OUT / "run.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
