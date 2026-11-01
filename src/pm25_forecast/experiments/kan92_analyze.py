"""KAN-92 (E15): tables for ``docs/experiments/e15-ensembles.md`` from the tracked dev runs.

    python -m pm25_forecast.experiments.kan92_analyze

Reads the finished, valid dev runs of the E15 preregistrations (and the family-best members as
references, by the run ids listed in ``MEMBER_RUNS``) from the shared MLflow store, with their
``evaluation.json`` artifacts, and the per-fold weights written by ``ensembles.fit``. Nothing is
typed by hand: every number in the write-up is printed here. Dev runs only; holdout runs are never
read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pm25_forecast.experiments import ensembles, tracking

HORIZON = "24"
E15_IDS = ("e15-equal5", "e15-invmae5", "e15-stack5", "e15-smoother-gbm", "e15-ridge-clim")
BONF_SUFFIX = "-bonf5"
# Family-best member runs from the earlier tickets (dev, +24 h, 4,200 pairs), for reference.
MEMBER_RUNS = {
    "ridge e06-ridge-pool-raw-chg": "9215183065474135bcbfa8de637f44e5",  # pragma: allowlist secret
    "clim e05-clim-28d-blend": "444eecf8cc5d4dc492e157ee0fa51104",  # pragma: allowlist secret
    "gbm e12-full-l2-delta": "76298486bbca4bc2b88987b3c742bd64",  # pragma: allowlist secret
    "arima e07-arima-d1": "ba06af54de50496796d873c595e8eb17",  # pragma: allowlist secret
    "wx e11-lgbm-wx-d2": "62adf1482ff2458e99130b356cae87da",  # pragma: allowlist secret
    "wx e11-lgbm-wx": "e41b2f96731e4d698e176c017aa926cd",  # pragma: allowlist secret
}


def _ci(pair: Any) -> str:
    return "n/a" if not pair else f"[{pair[0]:+.2f}, {pair[1]:+.2f}]"


def load(client: Any, run_id: str) -> dict[str, Any]:
    run = client.get_run(run_id)
    path = client.download_artifacts(run_id, "evaluation.json")
    data = json.loads(Path(path).read_text())
    s = data["scores"][HORIZON]
    folds = [s["by_fold"][k].get("diff_vs_smoother") for k in sorted(s["by_fold"], key=int)]
    sm, ps = s["vs_smoother"], s["vs_persistence"]
    return {
        "run": run_id[:8],
        "prereg": run.data.tags.get("prereg", ""),
        "dirty": run.data.tags.get("git_dirty"),
        "invalid": run.data.tags.get("invalid"),
        "mae": s["mae"],
        "mae_pers": s["baselines"]["persistence"]["mae"],
        "mae_smoo": s["baselines"]["smoother"]["mae"],
        "skill": s["skill_vs_persistence"],
        "n": s["n_pairs"],
        "coverage": s["coverage"],
        "d_pers": ps.get("diff"),
        "ci_pers": ps.get("ci"),
        "d_smoo": sm.get("diff"),
        "ci_smoo": sm.get("ci"),
        "sens_smoo": sm.get("ci_sensitivity") or {},
        "fragile": sm.get("fragile"),
        "dm_p": ((sm.get("dm") or {}).get("167") or {}).get("pvalue"),
        "dm_all": {k: v.get("pvalue") for k, v in (sm.get("dm") or {}).items()},
        "folds": folds,
        "level": data["ci_level"],
        "gate": s.get("station_gate") or {"status": "n/a"},
        "lag": data.get("serving_lag")
        or {"status": "n/a", "checked": 0, "mismatches": 0, "errors": 0},
        "reasons": data["invalid_reasons"],
        "by_station": s["mae_by_station"],
    }


def _find_runs(client: Any) -> dict[str, list[Any]]:
    exps = client.search_experiments(filter_string="name LIKE 'kan78/e15-%'")
    out: dict[str, list[Any]] = {}
    for run in client.search_runs(
        [e.experiment_id for e in exps],
        filter_string="tags.ticket = 'KAN-92' and tags.split = 'dev'",
        order_by=["attributes.start_time ASC"],
        max_results=1000,
    ):
        if run.info.status == "FINISHED" and run.info.lifecycle_stage == "active":
            out.setdefault(run.data.tags.get("prereg", ""), []).append(run)
    return out


def _row(name: str, r: dict[str, Any]) -> str:
    folds = [f for f in r["folds"] if f is not None]
    dm = "n/a" if r["dm_p"] is None else f"{r['dm_p']:.3f}"
    if r["d_smoo"] is None:
        return f"| {name} | {r['mae']:.3f} |" + " n/a |" * 9 + f" {r['run']} |"
    pers = f"{r['d_pers']:+.3f} {_ci(r['ci_pers'])}"
    smoo = f"{r['d_smoo']:+.3f} {_ci(r['ci_smoo'])}"
    lag = f"{r['lag']['status']} ({r['lag']['checked']})"
    return (
        f"| {name} | {r['mae']:.3f} | {r['skill']:+.3f} | {pers} | {smoo} | {r['level']:.4f} "
        f"| {sum(f < 0 for f in folds)}/{len(folds)} | {dm} | {r['fragile']} "
        f"| {r['gate']['status']} | {lag} | {r['run']} |"
    )


def table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    head = (
        "| run | mae_24h | skill vs pers. | diff vs pers. [CI] | diff vs smoother [CI] | "
        "level | folds < 0 | DM p | fragile | station gate | serving lag | MLflow |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(_row(n, r) for n, r in rows)


def weights_summary() -> str:
    out = []
    for variant in ("invmae", "stack"):
        path = ensembles.log_dir() / f"{variant}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        ws = [h["24"] for h in data.values() for h in [h["horizons"]]]
        members = ensembles.VARIANT_MEMBERS[variant]
        mat = np.array([[w["weights"][m] for m in members] for w in ws])
        icpt = np.array([w["intercept"] for w in ws])
        out.append(
            f"{variant}: folds={len(ws)} mean weights "
            + ", ".join(f"{m}={x:.2f}" for m, x in zip(members, mat.mean(axis=0), strict=True))
            + f"; sum of weights per fold {np.round(mat.sum(axis=1), 2).tolist()}"
            + f"; intercepts {np.round(icpt, 2).tolist()}"
            + f"; fallback_equal folds={sum(w['fallback_equal'] for w in ws)}"
            + f"; oof rows {[w.get('oof_rows') for w in ws]}"
        )
        for key in ("window_mae", "oof_mae"):
            if key in ws[-1]:
                means = np.array([[w[key][m] for m in members] for w in ws if key in w])
                out.append(
                    f"  {key} mean over folds: "
                    + ", ".join(
                        f"{m}={x:.2f}" for m, x in zip(members, means.mean(axis=0), strict=True)
                    )
                )
        ens = [w["oof_mae_ensemble"] for w in ws if "oof_mae_ensemble" in w]
        out.append(f"  in-sample/OOF ensemble MAE by fold: {np.round(ens, 3).tolist()}")
    return "\n".join(out)


def main() -> int:
    client = tracking._client()
    runs = _find_runs(client)
    rows, bonf, members = [], [], []
    for pid in E15_IDS:
        for run in runs.get(pid, []):
            r = load(client, run.info.run_id)
            (bonf if abs(r["level"] - 0.95) > 1e-6 else rows).append((pid, r))
    for name, rid in MEMBER_RUNS.items():
        members.append((name, load(client, rid)))
    print("## Ensembles (dev, +24 h, 95% CI, 72 h blocks)\n")
    print(table(rows))
    print("\n## Bonferroni sensitivity runs (K = 5, level 0.99)\n")
    print(table(bonf) if bonf else "none")
    print("\n## Members (earlier tickets)\n")
    print(table(members))
    print("\n## Gate details\n")
    for name, r in rows + bonf:
        g, lag = r["gate"], r["lag"]
        worst = g.get("worst_station")
        print(
            f"{name} [{r['run']}]: station gate {g['status']}, worst {worst} "
            f"{g.get('worst_rel_diff', float('nan')):+.3f} "
            f"(CI high {g.get('worst_rel_diff_ci_high', float('nan')):+.3f}), "
            f"failing {g.get('failing')}; lag {lag['status']} checked={lag['checked']} "
            f"mismatches={lag['mismatches']} errors={lag['errors']}; coverage {r['coverage']:.3f}; "
            f"invalid {r['reasons']}; sens CI vs smoother {{"
            + ", ".join(f"{k}: {_ci(v)}" for k, v in r["sens_smoo"].items())
            + f"}}; folds {[None if f is None else round(f, 3) for f in r['folds']]}; "
            f"DM p by bandwidth {r['dm_all']}; dirty={r['dirty']} invalid={r['invalid']}"
        )
    print("\n## Fitted weights\n")
    print(weights_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
