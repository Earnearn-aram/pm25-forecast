"""KAN-93 (E16) error analysis: where do persistence, the smoother and the best models fail?

Exploratory and dev-only (the holdout is never read). Predictions are recomputed with the
evaluation runner's own pair generation (``evaluate.make_plan``, ``select_cohort`` and
``_walk_forward``); there is no private scoring loop. Nothing is logged to MLflow.

    python -m pm25_forecast.experiments.error_analysis collect --snapshot <dir> --out <dir>
    python -m pm25_forecast.experiments.error_analysis report --cache <dir> --figures <dir>

``collect`` writes one ``.npz`` of per-pair arrays per model OUTSIDE git (it holds raw
readings). ``report`` prints aggregates as JSON and saves aggregate-only PNG figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pm25_forecast.experiments import evaluate as ev
from pm25_forecast.experiments import folds as folds_mod
from pm25_forecast.experiments import stats
from pm25_forecast.experiments.dataset import load_snapshot
from pm25_forecast.experiments.prereg import load_prereg

HOUR = ev.HOUR
THRESHOLD = ev.SPIKE_THRESHOLD
HORIZONS = (6, 12, 24)
MAIN_H = 24
BANDS = (
    ("<15", 0.0, 15.0),
    ("15-25", 15.0, 25.0),
    ("25-37.5", 25.0, THRESHOLD),
    (">37.5", THRESHOLD, 1e9),
)
FLAT_BAND = 5.0  # |actual - latest| <= 5 ug/m3 counts as "flat"
MODELS = ("persistence", "smoother", "climatology_28d_blend", "ridge_pool_raw_chg")
SPECS = {  # name -> (prereg id, factory); params come from the frozen prereg
    "climatology_28d_blend": (
        "e05-clim-28d-blend",
        "pm25_forecast.experiments.smoothing:climatology",
    ),
    "ridge_pool_raw_chg": ("e06-ridge-pool-raw-chg", "pm25_forecast.experiments.linear_models:fit"),
}
PREREG_DIR = Path(__file__).resolve().parents[3] / "docs" / "experiments" / "prereg"


# ---- collection (reuses the runner) --------------------------------------------------------


def collect_rows(
    observations: Sequence[Any],
    vintages: Sequence[Any],
    fit: ev.Factory,
    params: Mapping[str, Any] | None,
    horizons: Sequence[int] = HORIZONS,
) -> dict[str, np.ndarray]:
    """Dev-split per-pair arrays for ``fit`` plus persistence and smoother on the same rows.

    Keys are ``h{H}_{field}`` with fields station, origin_s (epoch s), actual, pers, smoo, cand,
    plus ``trend6`` (ex-ante: reading at origin-1h minus origin-7h, NaN if missing).
    """
    usable = [o for o in observations if folds_mod.S0 <= o.ts < folds_mod.TARGET_CUT]
    plan = ev.make_plan(max(o.ts for o in usable), "dev")
    data = [o for o in usable if o.ts < plan.limit]
    cohort = ev.select_cohort(data, folds_mod.S0, plan.dev_limit)
    kept = [v for v in vintages if v.valid_time < plan.limit]
    rows, _ = ev._walk_forward(data, kept, cohort, plan, list(horizons), fit, params)
    value = {(o.station_id, o.ts): o.value for o in data}
    out: dict[str, np.ndarray] = {}
    for h, r in rows.items():
        out[f"h{h}_station"] = np.array(r.station)
        out[f"h{h}_origin_s"] = np.array([int(o.timestamp()) for o in r.origin], dtype=np.int64)
        for name in ("actual", "pers", "smoo", "cand"):
            out[f"h{h}_{name}"] = np.array(getattr(r, name), dtype=float)
        trend = []
        for s, o in zip(r.station, r.origin, strict=True):
            a, b = value.get((s, o - HOUR)), value.get((s, o - 7 * HOUR))
            trend.append(np.nan if a is None or b is None else a - b)
        out[f"h{h}_trend6"] = np.array(trend)
    return out


def collect_all(snapshot: Path, out_dir: Path) -> dict[str, str]:
    observations, vintages, _ = load_snapshot(snapshot)
    written = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (prereg_id, spec) in SPECS.items():
        prereg = load_prereg(PREREG_DIR / f"{prereg_id}.yaml")
        fit, _ = ev.resolve_factory(spec)
        arrays = collect_rows(observations, vintages, fit, prereg.params)
        path = out_dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        written[name] = path.name
    return written


def load_cache(cache: Path) -> dict[str, dict[str, np.ndarray]]:
    """``{model: {field: array}}``: the raw per-pair arrays of every horizon, one dict per model.

    No filtering or alignment happens here; ``common_frame`` aligns the models on the (station,
    origin) rows where all have a forecast. Persistence and smoother columns are the same in
    every file (they come from the runner's baselines on the same rows).
    """
    raw = {m: dict(np.load(cache / f"{m}.npz")) for m in SPECS}
    return raw


# ---- pure analysis -------------------------------------------------------------------------


def common_frame(raw: Mapping[str, Mapping[str, np.ndarray]], h: int = MAIN_H) -> dict[str, Any]:
    """Rows of horizon ``h`` where all models forecast; columns aligned across models."""
    keys = {}
    for m, d in raw.items():
        keys[m] = {
            (s, o): i
            for i, (s, o) in enumerate(zip(d[f"h{h}_station"], d[f"h{h}_origin_s"], strict=True))
            if np.isfinite(d[f"h{h}_cand"][i])
        }
    shared = sorted(set.intersection(*(set(k) for k in keys.values())))
    first = next(iter(raw))
    idx0 = np.array([keys[first][k] for k in shared])
    d0 = raw[first]
    frame: dict[str, Any] = {
        "station": np.array([k[0] for k in shared]),
        "origin_s": np.array([k[1] for k in shared], dtype=np.int64),
        "actual": d0[f"h{h}_actual"][idx0],
        "latest": d0[f"h{h}_pers"][idx0],
        "trend6": d0[f"h{h}_trend6"][idx0],
        "pred": {"persistence": d0[f"h{h}_pers"][idx0], "smoother": d0[f"h{h}_smoo"][idx0]},
    }
    for m, d in raw.items():
        idx = np.array([keys[m][k] for k in shared])
        frame["pred"][m] = d[f"h{h}_cand"][idx]
    frame["target_hour"] = ((frame["origin_s"] // 3600 + h) % 24).astype(int)
    frame["target_dow"] = (((frame["origin_s"] // 3600 + h) // 24 + 3) % 7).astype(int)  # Mon=0
    return frame


def group_masks(frame: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    actual, latest, trend = frame["actual"], frame["latest"], frame["trend6"]
    change = actual - latest
    out: dict[str, dict[str, np.ndarray]] = {
        "station": {s: frame["station"] == s for s in sorted(set(frame["station"].tolist()))},
        "hour_utc": {f"{h:02d}": frame["target_hour"] == h for h in range(24)},
        "day_of_week": {
            d: frame["target_dow"] == i
            for i, d in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
        },
        "level_band_actual": {n: (actual >= lo) & (actual < hi) for n, lo, hi in BANDS},
        "level_band_latest": {n: (latest >= lo) & (latest < hi) for n, lo, hi in BANDS},
        "direction_actual": {
            "rising": change > FLAT_BAND,
            "flat": np.abs(change) <= FLAT_BAND,
            "falling": change < -FLAT_BAND,
        },
        "trend6_known_at_origin": {
            "rising": trend > FLAT_BAND,
            "flat": np.abs(trend) <= FLAT_BAND,
            "falling": trend < -FLAT_BAND,
        },
    }
    return out


def summarise_groups(
    frame: Mapping[str, Any], models: Sequence[str] = MODELS
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per group and model: n, MAE, bias (pred - actual, signed) and MAE minus smoother MAE."""
    res: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for dim, groups in group_masks(frame).items():
        for g, mask in groups.items():
            n = int(mask.sum())
            entry: dict[str, Any] = {"n": n}
            if n:
                smo_mae = float(
                    np.abs(frame["pred"]["smoother"][mask] - frame["actual"][mask]).mean()
                )
                for m in models:
                    e = frame["pred"][m][mask] - frame["actual"][mask]
                    entry[m] = {
                        "mae": float(np.abs(e).mean()),
                        "bias": float(e.mean()),
                        "mae_minus_smoother": float(np.abs(e).mean() - smo_mae),
                    }
            res[dim][g] = entry
    return dict(res)


def group_diff_ci(
    frame: Mapping[str, Any], mask: np.ndarray, model: str, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean |model err| - |smoother err| in a subset with the runner's 72 h block bootstrap.

    Blocks run over the origin-hour grid, so hours outside the subset simply carry zero weight.
    """
    d = np.abs(frame["pred"][model] - frame["actual"]) - np.abs(
        frame["pred"]["smoother"] - frame["actual"]
    )
    grid_h = frame["origin_s"] // 3600
    idx = (grid_h - grid_h.min()).astype(int)
    size = int(idx.max()) + 1
    s = np.bincount(idx[mask], weights=d[mask], minlength=size)
    n = np.bincount(idx[mask], minlength=size).astype(float)
    lo, hi, _ = stats.moving_block_bootstrap_ci(s, n, block=stats.PRIMARY_BLOCK_H, level=level)
    return float(s.sum() / n.sum()), float(lo), float(hi)


def error_concentration(frame: Mapping[str, Any], model: str = "smoother") -> dict[str, Any]:
    """How concentrated the absolute error is: shares by regime and by target day."""
    err = np.abs(frame["pred"][model] - frame["actual"])
    change = frame["actual"] - frame["latest"]
    total = float(err.sum())
    regime = np.abs(change) > FLAT_BAND
    days = (frame["origin_s"] // 3600 + MAIN_H) // 24
    per_day = np.array([err[days == d].sum() for d in np.unique(days)])
    per_day = np.sort(per_day)[::-1]
    top = max(1, round(0.1 * len(per_day)))
    return {
        "model": model,
        "share_pairs_regime_change": float(regime.mean()),
        "share_error_regime_change": float(err[regime].sum() / total),
        "share_pairs_above_25": float((frame["actual"] >= 25).mean()),
        "share_error_above_25": float(err[frame["actual"] >= 25].sum() / total),
        "n_days": len(per_day),
        "share_error_worst_10pct_days": float(per_day[:top].sum() / total),
    }


def station_levels(frame: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out = {}
    for s in sorted(set(frame["station"].tolist())):
        m = frame["station"] == s
        a = frame["actual"][m]
        out[s] = {
            "mean": float(a.mean()),
            "sd": float(a.std()),
            "share_regime_change": float((np.abs(a - frame["latest"][m]) > FLAT_BAND).mean()),
        }
    return out


def spike_onsets(
    raw_series: Mapping[tuple[str, int], float], threshold: float = THRESHOLD
) -> list[tuple[str, int]]:
    """Onsets: hours (epoch s) with value > threshold whose previous hour is <= threshold.

    The previous hour must be observed; an onset right after a gap is not counted.
    """
    onsets = []
    for (s, t), v in raw_series.items():
        prev = raw_series.get((s, t - 3600))
        if v > threshold and prev is not None and prev <= threshold:
            onsets.append((s, t))
    return sorted(onsets)


def onset_table(
    raw: Mapping[str, Mapping[str, np.ndarray]],
    models: Sequence[str] = MODELS,
    horizons: Sequence[int] = HORIZONS,
    window_h: int = 12,
) -> dict[str, Any]:
    """Descriptive spike-onset scoring from the forecasts made h hours ahead (h in horizons).

    The actual hourly series is rebuilt from the pair rows (actual at target time). For every
    onset, a model "catches" it at lead h if the forecast issued h hours before the onset hour
    exceeds the threshold; ``max_lead`` is the longest horizon that catches it. A false alarm is
    a forecast above the threshold at a target hour that is not within ``window_h`` hours
    after or before an observed exceedance of the same station (i.e. no spike nearby).
    """
    base = raw[next(iter(raw))]
    actual: dict[tuple[str, int], float] = {}
    for h in horizons:
        for s, o, a in zip(
            base[f"h{h}_station"], base[f"h{h}_origin_s"], base[f"h{h}_actual"], strict=True
        ):
            actual[(str(s), int(o) + h * 3600)] = float(a)
    onsets = spike_onsets(actual)
    exceed_by_station: dict[str, list[int]] = defaultdict(list)
    for (s, t), v in actual.items():
        if v > THRESHOLD:
            exceed_by_station[s].append(t)
    exceed_arr = {s: np.array(sorted(ts)) for s, ts in exceed_by_station.items()}

    def near_exceedance(s: str, t: int) -> bool:
        arr = exceed_arr.get(s)
        return arr is not None and bool((np.abs(arr - t) <= window_h * 3600).any())

    at = [actual[k] for k in onsets]
    out: dict[str, Any] = {
        "n_onsets": len(onsets),
        "mean_observed_at_onset": float(np.mean(at)) if at else None,
        "window_h": window_h,
        "models": {},
    }
    for m in models:
        fc: dict[int, dict[tuple[str, int], float]] = {}
        for h in horizons:
            d = raw.get(m, base)
            cand = (
                d[f"h{h}_cand"]
                if m in raw
                else base[f"h{h}_{'pers' if m == 'persistence' else 'smoo'}"]
            )
            fc[h] = {
                (str(s), int(o) + h * 3600): float(c)
                for s, o, c in zip(d[f"h{h}_station"], d[f"h{h}_origin_s"], cand, strict=True)
                if np.isfinite(c)
            }
        per_h = {}
        for h in horizons:
            caught = missed = nofc = 0
            for k in onsets:
                v = fc[h].get(k)
                if v is None:
                    nofc += 1
                elif v > THRESHOLD:
                    caught += 1
                else:
                    missed += 1
            alarms = [k for k, v in fc[h].items() if v > THRESHOLD]
            false_alarms = sum(not near_exceedance(s, t) for s, t in alarms)
            at_onset = [fc[h][k] for k in onsets if k in fc[h]]
            per_h[str(h)] = {
                "mean_forecast_at_onset": float(np.mean(at_onset)) if at_onset else None,
                "max_forecast_any_hour": float(max(fc[h].values())) if fc[h] else None,
                "caught": caught,
                "missed": missed,
                "no_forecast": nofc,
                "alarm_hours": len(alarms),
                "false_alarm_hours": false_alarms,
            }
        leads = []
        for k in onsets:
            caught_h = [h for h in horizons if fc[h].get(k, -1.0) > THRESHOLD]
            leads.append(max(caught_h) if caught_h else 0)
        out["models"][m] = {
            "by_horizon": per_h,
            "max_lead_counts": {str(h): int(sum(x == h for x in leads)) for h in (0, *horizons)},
        }
    return out


# ---- figures (aggregates only) -------------------------------------------------------------

COLORS = {
    "persistence": "#888888",
    "smoother": "#1f77b4",
    "climatology_28d_blend": "#2ca02c",
    "ridge_pool_raw_chg": "#d62728",
}
LABELS = {
    "persistence": "persistence",
    "smoother": "smoother",
    "climatology_28d_blend": "climatology 28 d blend",
    "ridge_pool_raw_chg": "ridge (pooled, change)",
}


def _grouped_bars(
    ax: Any, groups: Mapping[str, Mapping[str, Any]], key: str, models: Sequence[str]
) -> None:
    names = [g for g, e in groups.items() if e["n"]]
    width = 0.8 / len(models)
    x = np.arange(len(names))
    for i, m in enumerate(models):
        ax.bar(
            x + i * width - 0.4 + width / 2,
            [groups[g][m][key] for g in names],
            width,
            label=LABELS[m],
            color=COLORS[m],
        )
    ax.set_xticks(x, [f"{g}\n(n={groups[g]['n']})" for g in names], fontsize=7)


def save_figures(summary: Mapping[str, Any], onset: Mapping[str, Any], figures: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    saved = []

    def finish(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(figures / name, dpi=80)
        plt.close(fig)
        saved.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _grouped_bars(axes[0], summary["station"], "mae", MODELS)
    axes[0].set_title("MAE by station, +24 h, dev")
    axes[0].set_ylabel("MAE (ug/m3)")
    axes[0].legend(fontsize=7)
    _grouped_bars(axes[1], summary["level_band_actual"], "bias", MODELS)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("Bias (forecast - observed) by observed level band")
    axes[1].set_ylabel("mean signed error (ug/m3)")
    axes[1].set_xlabel("observed PM2.5 band (ug/m3)")
    finish(fig, "e16-station-and-level-band.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in MODELS:
        hrs = sorted(summary["hour_utc"])
        axes[0].plot(
            [int(h) for h in hrs],
            [summary["hour_utc"][h][m]["mae"] for h in hrs],
            label=LABELS[m],
            color=COLORS[m],
        )
    axes[0].set_xlabel("target hour (UTC)")
    axes[0].set_ylabel("MAE (ug/m3)")
    axes[0].set_title("MAE by target hour of day, +24 h, dev")
    axes[0].legend(fontsize=7)
    _grouped_bars(axes[1], summary["day_of_week"], "mae", MODELS)
    axes[1].set_title("MAE by target day of week")
    axes[1].set_ylabel("MAE (ug/m3)")
    finish(fig, "e16-hour-and-weekday.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _grouped_bars(axes[0], summary["direction_actual"], "bias", MODELS)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("Bias when observed change vs latest is +/-5 ug/m3")
    axes[0].set_ylabel("mean signed error (ug/m3)")
    _grouped_bars(axes[1], summary["trend6_known_at_origin"], "mae", MODELS)
    axes[1].set_title("MAE by 6 h trend known at forecast time")
    axes[1].set_ylabel("MAE (ug/m3)")
    axes[1].legend(fontsize=7)
    finish(fig, "e16-rising-falling.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    hs = sorted(int(h) for h in onset["models"]["smoother"]["by_horizon"])
    for m in MODELS:
        by = onset["models"][m]["by_horizon"]
        ax.plot(
            hs,
            [by[str(h)]["mean_forecast_at_onset"] for h in hs],
            "o-",
            label=LABELS[m],
            color=COLORS[m],
        )
    ax.axhline(THRESHOLD, color="k", ls="--", lw=0.8, label="spike threshold 37.5")
    ax.axhline(
        onset["mean_observed_at_onset"], color="orange", ls=":", label="mean observed at onset"
    )
    ax.set_xticks(hs, [f"+{h} h" for h in hs])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("mean forecast at onset hour (ug/m3)")
    ax.set_title(f"Forecast at the {onset['n_onsets']} spike-onset hours (descriptive), dev")
    ax.legend(fontsize=7)
    finish(fig, "e16-spike-onsets.png")
    return saved


# ---- CLI -----------------------------------------------------------------------------------


def report(cache: Path, figures: Path | None) -> dict[str, Any]:
    raw = load_cache(cache)
    frame = common_frame(raw)
    summary = summarise_groups(frame)
    onset = onset_table({m: {**raw[m]} for m in SPECS}, models=MODELS)
    overall = {m: float(np.abs(frame["pred"][m] - frame["actual"]).mean()) for m in MODELS}
    masks = group_masks(frame)
    ci = {}
    for dim in ("station", "level_band_actual", "direction_actual"):
        for name, mask in masks[dim].items():
            if mask.sum() >= 30:
                ci[f"{dim}:{name}"] = {
                    m: group_diff_ci(frame, mask, m) for m in ("ridge_pool_raw_chg", "persistence")
                }
    out = {
        "n_pairs": len(frame["actual"]),
        "overall_mae": overall,
        "groups": summary,
        "diff_vs_smoother_ci": ci,
        "concentration": {
            m: error_concentration(frame, m) for m in ("smoother", "ridge_pool_raw_chg")
        },
        "station_levels": station_levels(frame),
        "onsets": onset,
    }
    if figures is not None:
        out["figures"] = save_figures(summary, onset, figures)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--snapshot", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--figures", type=Path, default=None)
    r.add_argument("--json-out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.cmd == "collect":
        print(json.dumps(collect_all(a.snapshot, a.out)))
    else:
        res = report(a.cache, a.figures)
        text = json.dumps(res, indent=1, sort_keys=True)
        if a.json_out:
            a.json_out.write_text(text + "\n")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
