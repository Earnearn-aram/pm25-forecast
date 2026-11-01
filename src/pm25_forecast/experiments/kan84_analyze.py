"""KAN-84 (E06) analysis: results table of the KAN-84 dev runs and coefficients of one variant.

    python -m pm25_forecast.experiments.kan84_analyze table [--experiment-prefix kan78/e06-]
    python -m pm25_forecast.experiments.kan84_analyze coef <variant-id>

Reads the shared MLflow store (runs tagged ticket=KAN-84, split=dev, valid) and the per-fold
diagnostics written by ``linear_models`` (``$PM25_KAN84_LOG_DIR``). Nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pm25_forecast.experiments import tracking


def load_runs(prefix: str) -> list[dict[str, Any]]:
    client = tracking._client()
    out = []
    for exp in client.search_experiments(filter_string=f"name LIKE '{prefix}%'"):
        for run in client.search_runs(
            [exp.experiment_id],
            filter_string="tags.ticket = 'KAN-84' and tags.split = 'dev'",
            order_by=["start_time DESC"],
        ):
            if run.data.tags.get("invalid") == "true" or run.info.status != "FINISHED":
                continue
            with tempfile.TemporaryDirectory() as tmp:
                path = client.download_artifacts(run.info.run_id, "evaluation.json", tmp)
                ev = json.loads(Path(path).read_text())
            out.append(
                {
                    "experiment": exp.name,
                    "run_id": run.info.run_id,
                    "prereg": run.data.tags.get("prereg", ""),
                    "dirty": run.data.tags.get("git_dirty"),
                    "git_sha": run.data.tags.get("git_sha", "")[:10],
                    "ci_level": ev["ci_level"],
                    "ev": ev,
                }
            )
            break  # newest valid run per experiment
    return out


def summarise(run: dict[str, Any]) -> dict[str, Any]:
    s = run["ev"]["scores"]["24"]
    pers, smo = s["vs_persistence"], s["vs_smoother"]
    rel = {st: v["model"] / v["persistence"] - 1 for st, v in s["mae_by_station"].items()}
    worst = max(rel, key=rel.get)
    return {
        "prereg": run["prereg"],
        "run": run["run_id"][:8],
        "mae24": s["mae"],
        "mae12": run["ev"]["scores"]["12"]["mae"],
        "mae6": run["ev"]["scores"]["6"]["mae"],
        "skill": s["skill_vs_persistence"],
        "d_pers": pers["diff"],
        "ci_pers": pers.get("ci"),
        "d_smo": smo["diff"],
        "ci_smo": smo.get("ci"),
        "ci_smo_24": smo.get("ci_sensitivity", {}).get("24"),
        "ci_smo_168": smo.get("ci_sensitivity", {}).get("168"),
        "fragile": smo.get("fragile"),
        "folds_neg_smo": sum(f["diff_vs_smoother"] < 0 for f in s["by_fold"].values()),
        "folds_neg_pers": sum(f["diff_vs_persistence"] < 0 for f in s["by_fold"].values()),
        "n_folds": len(s["by_fold"]),
        "worst_station": worst,
        "worst_rel": rel[worst],
        "dm_p": smo.get("dm", {}).get("167", {}).get("pvalue"),
        "n_pairs": s["n_pairs"],
        "coverage": s["coverage"],
        "invalid": run["ev"]["invalid_reasons"],
        "clipped": run["ev"]["counters"].get("clipped"),
        "dirty": run["dirty"],
    }


def _ci(ci: list[float] | None) -> str:
    return "n/a" if not ci else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def cmd_table(prefix: str) -> None:
    rows = sorted((summarise(r) for r in load_runs(prefix)), key=lambda r: r["mae24"])
    print(
        "| prereg | run | MAE24 | MAE12 | MAE6 | skill | vs persistence (72h CI) | vs smoother"
        " (72h CI) | 24h/168h CI vs smoother | folds<0 smo/pers | worst stn rel | clipped |"
    )
    print("|" + "---|" * 12)
    for r in rows:
        print(
            f"| {r['prereg']} | {r['run']} | {r['mae24']:.3f} | {r['mae12']:.3f} | "
            f"{r['mae6']:.3f} | {r['skill']:+.3f} | {r['d_pers']:+.3f} {_ci(r['ci_pers'])} | "
            f"{r['d_smo']:+.3f} {_ci(r['ci_smo'])} | {_ci(r['ci_smo_24'])} {_ci(r['ci_smo_168'])}"
            f"{' FRAGILE' if r['fragile'] else ''} | {r['folds_neg_smo']}/{r['folds_neg_pers']}"
            f" of {r['n_folds']} | {r['worst_station']} {r['worst_rel']:+.1%} | {r['clipped']} |"
        )
    print(
        f"\nruns: {len(rows)}; invalid/dirty: "
        f"{[r['prereg'] for r in rows if r['invalid'] or r['dirty'] == 'true']}"
    )


def cmd_coef(variant: str, horizon: int) -> None:
    folder = Path(
        os.environ.get("PM25_KAN84_LOG_DIR", "~/pm25-experiments/cache/kan84")
    ).expanduser()
    recs = [json.loads(x) for x in (folder / f"{variant}.jsonl").read_text().splitlines()]
    print(f"strength chosen per refit (h={horizon}):")
    for r in recs:
        if r["h"] == horizon:
            print(
                f"  {r['refit_time'][:10]} key={r['key']} n={r['n']} strength={r['strength']:.4g}"
                f" nonzero={r['nonzero']}/{len(r['coef_std'])}"
            )
    last = max(r["refit_time"] for r in recs)
    for r in recs:
        if r["h"] == horizon and r["refit_time"] == last:
            print(
                f"\nlast refit {last}, key={r['key']}, intercept={r['intercept']:.3f}"
                " (coefficients per 1 SD of the feature, in target units)"
            )
            for k, v in sorted(r["coef_std"].items(), key=lambda kv: -abs(kv[1])):
                print(f"  {k:22s} {v:+.4f}  (SD {r['scale'][k]:.3g})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--experiment-prefix", default="kan78/e06-")
    c = sub.add_parser("coef")
    c.add_argument("variant")
    c.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    if args.cmd == "table":
        cmd_table(args.experiment_prefix)
    else:
        cmd_coef(args.variant, args.horizon)


if __name__ == "__main__":
    main()
