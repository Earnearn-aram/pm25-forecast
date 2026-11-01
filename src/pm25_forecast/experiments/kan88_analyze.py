"""KAN-88 (E11) paired analysis of the runs made through ``experiments.evaluate``.

    python -m pm25_forecast.experiments.kan88_analyze --snapshot <dir> [--preds <dir>] [--json out]

Reads the +24 h predictions the six KAN-88 factories dumped, joins them to the frozen snapshot and
the baselines (same code as the runner), and computes on the IDENTICAL rows (intersection of all
variants and the baselines): MAE, paired differences with the 72 h moving-block bootstrap of
``stats`` (95 %, and 97.5 % = Bonferroni over the 2 model families for the weather increment),
the KAN-75 pre-declared success rule (A: CI of weather minus smoother below 0 and MAE gain versus
persistence >= 15 %; B: spike-onset catch share), per-fold and per-station diffs, and the
go / pivot / stop mapping of ``docs/experiments/prereg/e11-*.yaml``. Dev split only.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pm25_forecast.experiments import folds as folds_mod
from pm25_forecast.experiments import stats
from pm25_forecast.experiments.dataset import load_snapshot
from pm25_forecast.experiments.evaluate import (
    make_plan,
    persistence_value,
    select_cohort,
    smoother_value,
)

HOUR = timedelta(hours=1)
H = 24
VARIANTS = ("ridge_base", "ridge_wx", "lgbm_base", "lgbm_wx", "ridge_wx_d2", "lgbm_wx_d2")
FAMILIES = ("ridge", "lgbm")
ONSET_RISE, CATCH_RISE = 15.0, 7.5
MIN_ONSETS, MIN_ONSET_DAYS = 30, 8
GAIN_NEEDED = 0.15
LEVEL, LEVEL_BONF = 0.95, 1 - 0.05 / len(FAMILIES)


def _read_preds(folder: Path) -> dict[str, dict[tuple[str, datetime], float]]:
    out: dict[str, dict[tuple[str, datetime], float]] = {}
    for v in VARIANTS:
        rows: dict[tuple[str, datetime], float] = {}
        with (folder / f"{v}.csv").open() as fh:
            for r in csv.DictReader(fh):
                if int(r["h"]) == H:
                    rows[r["station"], datetime.fromisoformat(r["origin"])] = float(r["pred"])
        out[v] = rows
    return out


def _ci(
    diff: np.ndarray, hour_idx: np.ndarray, grid: int, level: float, block: int = 72
) -> dict[str, Any]:
    s = np.bincount(hour_idx, weights=diff, minlength=grid)
    n = np.bincount(hour_idx, minlength=grid).astype(float)
    lo, hi, est = stats.moving_block_bootstrap_ci(s, n, block=block, level=level)
    return {"diff": est, "lo": lo, "hi": hi}


def analyze(snapshot: Path, preds_dir: Path) -> dict[str, Any]:
    observations, _, _ = load_snapshot(snapshot)
    usable = [o for o in observations if folds_mod.S0 <= o.ts < folds_mod.TARGET_CUT]
    plan = make_plan(max(o.ts for o in usable), "dev")
    data = [o for o in usable if o.ts < plan.limit]
    cohort = select_cohort(data, folds_mod.S0, plan.dev_limit)
    by_station: dict[str, dict[datetime, Any]] = {s: {} for s in cohort}
    for o in data:
        if o.station_id in by_station:
            by_station[o.station_id][o.ts] = o

    preds = _read_preds(preds_dir)
    keys = set.intersection(*(set(p) for p in preds.values()))
    rows = []
    for station, origin in sorted(keys, key=lambda k: (k[1], k[0])):
        lookup = by_station[station].get
        pers, smoo = persistence_value(lookup, origin), smoother_value(lookup, origin)
        actual = lookup(origin + H * HOUR)
        if pers is None or smoo is None or actual is None:
            continue
        fold = next((i for i, (a, b) in enumerate(plan.folds) if a <= origin + H * HOUR < b), None)
        rows.append((station, origin, fold, actual.value, pers, smoo))
    n = len(rows)
    station = np.array([r[0] for r in rows])
    origin = [r[1] for r in rows]
    fold = np.array([r[2] for r in rows])
    y = np.array([r[3] for r in rows])
    pers = np.array([r[4] for r in rows])
    smoo = np.array([r[5] for r in rows])
    t0 = min(origin)
    hidx = np.array([int((o - t0) / HOUR) for o in origin])
    grid = int(hidx.max()) + 1
    day = np.array([(o + H * HOUR).date().isoformat() for o in origin])
    pred = {v: np.array([preds[v][r[0], r[1]] for r in rows]) for v in VARIANTS}
    pred = {v: np.maximum(p, 0.0) for v, p in pred.items()}  # runner clips negatives at 0
    err = {v: np.abs(p - y) for v, p in pred.items()}
    err["persistence"], err["smoother"] = np.abs(pers - y), np.abs(smoo - y)
    mae = {k: float(e.mean()) for k, e in err.items()}

    def paired(a: str, b: str, level: float = LEVEL) -> dict[str, Any]:
        d = _ci(err[a] - err[b], hidx, grid, level)
        d["fragile"] = bool(
            (d["hi"] < 0 or d["lo"] > 0)
            and any(
                (lambda c: c["lo"] <= 0 <= c["hi"])(_ci(err[a] - err[b], hidx, grid, level, blk))
                for blk in stats.SENSITIVITY_BLOCKS_H
            )
        )
        d["by_fold"] = {
            int(f): float((err[a] - err[b])[fold == f].mean()) for f in sorted(set(fold.tolist()))
        }
        d["by_station"] = {
            s: float((err[a] - err[b])[station == s].mean()) for s in sorted(set(station.tolist()))
        }
        return d

    onset = (y - pers) >= ONSET_RISE
    onset_days = len(set(day[onset].tolist()))
    testable = int(onset.sum()) >= MIN_ONSETS and onset_days >= MIN_ONSET_DAYS

    def spike(v: str) -> dict[str, Any]:
        caught = pred[v] >= pers + CATCH_RISE
        return {
            "caught_share": float(caught[onset].mean()) if onset.any() else None,
            "false_alarm_share": float(caught[~onset].mean()),
        }

    def onset_increment(a: str, b: str) -> dict[str, Any] | None:
        if not testable:
            return None
        ca, cb = pred[a] >= pers + CATCH_RISE, pred[b] >= pers + CATCH_RISE
        s_hits = np.bincount(
            hidx[onset], weights=(ca[onset].astype(float) - cb[onset]), minlength=grid
        )
        n_hits = np.bincount(hidx[onset], minlength=grid).astype(float)
        lo, hi, est = stats.moving_block_bootstrap_ci(s_hits, n_hits, block=72, level=LEVEL)
        fa = float(ca[~onset].mean() - cb[~onset].mean())
        return {"catch_share_diff": est, "lo": lo, "hi": hi, "false_alarm_diff": fa}

    result: dict[str, Any] = {
        "n_pairs": n,
        "n_hourly_origins": len(set(hidx.tolist())),
        "coverage_per_variant_before_intersection": {v: len(p) for v, p in preds.items()},
        "stations": sorted(set(station.tolist())),
        "mae": mae,
        "mean_target": float(y.mean()),
        "vs_persistence": {v: paired(v, "persistence") for v in VARIANTS},
        "vs_smoother": {v: paired(v, "smoother") for v in VARIANTS},
        "smoother_vs_persistence": paired("smoother", "persistence"),
        "increment_95": {},
        "increment_bonferroni": {},
        "increment_d2_vs_base_95": {},
        "spike": {
            "n_onsets": int(onset.sum()),
            "onset_days": onset_days,
            "testable": testable,
            **{v: spike(v) for v in VARIANTS},
            "increment": {f: onset_increment(f"{f}_wx", f"{f}_base") for f in FAMILIES},
        },
        "rule": {},
    }
    for f in FAMILIES:
        result["increment_95"][f] = paired(f"{f}_wx", f"{f}_base", LEVEL)
        result["increment_bonferroni"][f] = paired(f"{f}_wx", f"{f}_base", LEVEL_BONF)
        result["increment_d2_vs_base_95"][f] = paired(f"{f}_wx_d2", f"{f}_base", LEVEL)

    verdict = "stop"
    for f in FAMILIES:
        w = f"{f}_wx"
        vs_s, vs_p = result["vs_smoother"][w], result["vs_persistence"][w]
        gain = 1 - mae[w] / mae["persistence"]
        gain_ci = [-vs_p["hi"] / mae["persistence"], -vs_p["lo"] / mae["persistence"]]
        a = bool(vs_s["hi"] < 0 and gain >= GAIN_NEEDED)
        inc = result["spike"]["increment"][f]
        b = bool(
            inc is not None
            and inc["catch_share_diff"] >= 0.10
            and inc["lo"] > 0
            and inc["false_alarm_diff"] <= 0.02
        )
        inc_ok = bool(result["increment_bonferroni"][f]["hi"] < 0)
        result["rule"][f] = {
            "A_ci_vs_smoother_below_0": bool(vs_s["hi"] < 0),
            "A_gain_vs_persistence": gain,
            "A_gain_ci": gain_ci,
            "A": a,
            "B": b,
            "increment_ci_bonferroni_below_0": inc_ok,
            "go": bool((a or b) and inc_ok),
            "increment_point_negative": bool(result["increment_95"][f]["diff"] < 0),
        }
    if any(r["go"] for r in result["rule"].values()):
        verdict = "go"
    elif any(r["increment_point_negative"] for r in result["rule"].values()):
        verdict = "pivot"
    result["verdict"] = verdict
    return result


def _fmt(d: dict[str, Any]) -> str:
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def to_markdown(r: dict[str, Any]) -> str:
    lines = [
        f"Identical rows: {r['n_pairs']} station-hours at +24 h over "
        f"{r['n_hourly_origins']} hourly origins, stations {', '.join(r['stations'])}. "
        "MAE in ug/m3; diff = MAE(row) - MAE(reference), "
        "negative is better; brackets are the 72 h block-bootstrap CI.",
        "",
        "| model | MAE | vs persistence (95%) | vs smoother (95%) | skill vs persistence |",
        "|---|---|---|---|---|",
    ]
    for k in ("persistence", "smoother", *VARIANTS):
        vp = r["vs_persistence"].get(k)
        vs = r["vs_smoother"].get(k)
        skill = 1 - r["mae"][k] / r["mae"]["persistence"]
        lines.append(
            f"| {k} | {r['mae'][k]:.3f} | {_fmt(vp) if vp else ''} | "
            f"{_fmt(vs) if vs else ''} | {skill:+.3f} |"
        )
    lines += [
        "",
        "Weather increment, same family and same rows (wx minus base):",
        "",
        "| family | variant | diff, 95% CI | diff, 97.5% CI (Bonferroni, 2 families) "
        "| by fold | fragile |",
        "|---|---|---|---|---|---|",
    ]
    for f in FAMILIES:
        i95, ib = r["increment_95"][f], r["increment_bonferroni"][f]
        folds = ", ".join(f"{v:+.2f}" for v in i95["by_fold"].values())
        lines.append(
            f"| {f} | day1 (primary) | {_fmt(i95)} | {_fmt(ib)} | {folds} | {i95['fragile']} |"
        )
        d2 = r["increment_d2_vs_base_95"][f]
        folds2 = ", ".join(f"{v:+.2f}" for v in d2["by_fold"].values())
        lines.append(
            f"| {f} | day2 (strict availability) | {_fmt(d2)} |  | {folds2} | {d2['fragile']} |"
        )
    sp = r["spike"]
    lines += [
        "",
        f"Spike onsets (y - latest >= 15): {sp['n_onsets']} on {sp['onset_days']} days, "
        f"testable={sp['testable']}.",
    ]
    for v in VARIANTS[:4]:
        lines.append(
            f"- {v}: caught {sp[v]['caught_share']}, "
            f"false-alarm share {sp[v]['false_alarm_share']:.4f}"
            if sp[v]["caught_share"] is not None
            else f"- {v}: no onsets"
        )
    lines += ["", "Pre-declared KAN-75 rule and mapping:", ""]
    for f, rule in r["rule"].items():
        lines.append(
            f"- {f}_wx: A(CI vs smoother < 0) = {rule['A_ci_vs_smoother_below_0']}, "
            f"gain vs persistence {rule['A_gain_vs_persistence']:+.3f} "
            f"[{rule['A_gain_ci'][0]:+.3f}, {rule['A_gain_ci'][1]:+.3f}] "
            f"(needs >= 0.15) -> A = {rule['A']}; B = {rule['B']}; increment CI(97.5%) < 0 = "
            f"{rule['increment_ci_bonferroni_below_0']}; go = {rule['go']}"
        )
    lines += ["", f"Verdict by the pre-declared mapping: **{r['verdict']}**"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--preds", type=Path, default=Path("~/pm25-experiments/cache/kan88/preds"))
    p.add_argument("--json", type=Path)
    p.add_argument("--markdown", type=Path)
    a = p.parse_args(argv)
    result = analyze(a.snapshot.expanduser(), a.preds.expanduser())
    text = to_markdown(result)
    print(text)
    if a.json:
        a.json.expanduser().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if a.markdown:
        a.markdown.expanduser().write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
