"""KAN-89 (E12): tables for docs/experiments/e12-gbm.md from the MLflow store and tuning logs.

    python -m pm25_forecast.experiments.gbm_summary [--log-dir out/kan89]

Reads the dev runs of ticket KAN-89 (experiments ``kan78/e12-*``), prints markdown tables:
primary variants, ablation, hyper-parameter choices and gain importance of the best model.
Nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pm25_forecast.experiments import tracking


def _runs() -> list[dict[str, Any]]:
    client = tracking._client()
    out = []
    for exp in client.search_experiments(filter_string="name LIKE 'kan78/e12-%'"):
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string="tags.ticket = 'KAN-89' and tags.split = 'dev'",
            order_by=["attributes.start_time DESC"],
            max_results=5,
        )
        for run in runs:
            if run.info.status != "FINISHED" or run.data.tags.get("invalid") == "true":
                continue
            m, t = run.data.metrics, run.data.tags
            path = client.download_artifacts(run.info.run_id, "evaluation.json")
            ev = json.loads(Path(path).read_text())
            s24 = ev["scores"]["24"]
            folds = s24["by_fold"]
            out.append(
                {
                    "id": t["prereg"],
                    "run_id": run.info.run_id,
                    "dirty": t.get("git_dirty"),
                    "mae": m["mae_24h"],
                    "skill": m.get("skill_vs_persistence_24h"),
                    "d_s": s24["vs_smoother"]["diff"],
                    "d_s_ci": tuple(s24["vs_smoother"]["ci"]),
                    "d_p": s24["vs_persistence"]["diff"],
                    "d_p_ci": tuple(s24["vs_persistence"]["ci"]),
                    "dm_p": m.get("dm_pvalue_vs_smoother_24h"),
                    "mae6": m.get("mae_6h"),
                    "mae12": m.get("mae_12h"),
                    "fragile": s24["vs_smoother"].get("fragile"),
                    "folds_neg_smoother": sum(f["diff_vs_smoother"] < 0 for f in folds.values()),
                    "n_folds": len(folds),
                    "smoother_mae": s24["baselines"]["smoother"]["mae"],
                    "persistence_mae": s24["baselines"]["persistence"]["mae"],
                    "ci_level": ev["ci_level"],
                }
            )
            break
    return sorted(out, key=lambda r: r["id"])


def _f(x: float | None, d: int = 3, sign: bool = False) -> str:
    return "" if x is None else format(x, f"+.{d}f" if sign else f".{d}f")


def _ci(c: tuple[float | None, float | None]) -> str:
    return f"[{_f(c[0], 2, True)}, {_f(c[1], 2, True)}]"


def table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| run | mae_24h | mae_6h | mae_12h | skill vs pers. | diff vs pers. [CI] | "
        "diff vs smoother [CI] | folds < 0 vs smoother | MLflow run |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['id']} | {_f(r['mae'])} | {_f(r['mae6'])} | {_f(r['mae12'])} | {_f(r['skill'])} | "
        f"{_f(r['d_p'], 3, True)} {_ci(r['d_p_ci'])} | "
        f"{_f(r['d_s'], 3, True)} {_ci(r['d_s_ci'])} | "
        f"{r['folds_neg_smoother']}/{r['n_folds']} | {r['run_id'][:8]} |\n"
        for r in rows
    )
    return head + body


def load_log(path: Path) -> list[dict[str, Any]]:
    """Tuning-log entries, one per (refit_time, horizon); a later line replaces an earlier one."""
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            latest[(e["refit_time"], e["horizon"])] = e
    return list(latest.values())


def tuning_tables(log_dir: Path, best_id: str) -> str:
    out = []
    chosen: Counter = Counter()
    n = 0
    for p in sorted(log_dir.glob("e12-*.tuning.jsonl")):
        if "abl" in p.name:
            continue
        for e in load_log(p):
            if e["horizon"] == 24:
                chosen[json.dumps(e["config"], sort_keys=True)] += 1
                n += 1
    out.append(f"Chosen configurations at +24 h over {n} fits (5 folds x 12 primary variants):\n")
    for cfg, c in chosen.most_common(5):
        out.append(f"- {c}x {cfg}")
    gains: dict[str, float] = defaultdict(float)
    count = 0
    f = log_dir / f"{best_id}.tuning.jsonl"
    for e in load_log(f):
        if e["horizon"] != 24:
            continue
        tot = sum(e["gain"].values())
        for k, v in e["gain"].items():
            gains[k] += v / tot
        count += 1
    out.append(f"\nGain importance of {best_id} at +24 h (mean share over {count} fold fits):\n")
    out.append("| feature | share |\n|---|---|")
    for k, v in sorted(gains.items(), key=lambda kv: -kv[1])[:15]:
        out.append(f"| {k} | {v / count:.3f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="out/kan89")
    a = ap.parse_args()
    rows = _runs()
    prim = [r for r in rows if r["id"].startswith("e12-core") or r["id"].startswith("e12-full")]
    abl = [r for r in rows if r["id"].startswith("e12-abl")]
    best = min(prim, key=lambda r: r["mae"])
    print(f"runs: {len(rows)}; dirty: {[r['id'] for r in rows if r['dirty'] != 'false']}")
    print(
        f"smoother mae_24h {rows[0]['smoother_mae']:.3f}, "
        f"persistence {rows[0]['persistence_mae']:.3f}, "
        f"ci_level {rows[0]['ci_level']}"
    )
    print("\n## primary\n" + table(prim))
    print(f"best: {best['id']}")
    print("\n## ablation (reference = best)\n" + table([best, *abl]))
    print("\n" + tuning_tables(Path(a.log_dir), best["id"]))


if __name__ == "__main__":
    main()
