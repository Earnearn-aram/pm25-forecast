"""Print the E07 (KAN-85) results table and fit-time summary from MLflow and the timing logs.

    python -m pm25_forecast.experiments.e07_summary [--timing-dir ~/pm25-experiments/e07-timing]

Every number in docs/experiments/e07-timeseries.md comes from this output.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from pm25_forecast.experiments import tracking

IDS = ["e07-arima-d0", "e07-arima-d1", "e07-sarima", "e07-ets-damped", "e07-ets-notrend"]


def _ci(block: dict | None) -> str:
    if not block or not block.get("ci"):
        return "n/a"
    low, high = block["ci"]
    return f"{block['diff']:+.3f} [{low:+.3f}, {high:+.3f}]"


def _find(client, prereg: str):
    runs = []
    for exp in client.search_experiments():
        runs += client.search_runs(
            [exp.experiment_id], f"tags.prereg = '{prereg}' and tags.split = 'dev'"
        )
    runs = [r for r in runs if "invalid" not in r.data.tags]
    return max(runs, key=lambda r: r.info.start_time) if runs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-dir", default=str(Path.home() / "pm25-experiments/e07-timing"))
    args = ap.parse_args()
    client = tracking._client()
    print(
        "| variant | run | git_sha | dirty | mae_24h | mae_6h | mae_12h "
        "| vs persistence 24h | vs smoother 24h | fragile |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for pid in IDS:
        run = _find(client, pid)
        if run is None:
            print(f"| {pid} | missing |")
            continue
        path = client.download_artifacts(run.info.run_id, "evaluation.json")
        ev = json.loads(Path(path).read_text())
        s = ev["scores"]
        vp, vs = s["24"]["vs_persistence"], s["24"]["vs_smoother"]
        p168, p24 = vs["ci_sensitivity"]["168"], vs["ci_sensitivity"]["24"]
        print(
            f"| {pid} | {run.info.run_id[:8]} | {run.data.tags.get('git_sha', '')[:9]} "
            f"| {run.data.tags.get('git_dirty', '')} | {s['24']['mae']:.3f} | {s['6']['mae']:.3f} "
            f"| {s['12']['mae']:.3f} | {_ci(vp)} | {_ci(vs)} | {vs.get('fragile')} |"
        )
        print(f"  sensitivity vs smoother: 24h block {p24}, 168h block {p168}")
        t = Path(args.timing_dir) / f"{pid}.jsonl"
        if t.exists():
            recs = [json.loads(line) for line in t.read_text().splitlines()]
            secs = [r["seconds"] for r in recs]
            orders = Counter(
                json.dumps(v.get("order")) for r in recs for v in r["stations"].values()
            )
            print(
                f"  fit seconds per fold ({len(secs)} folds): "
                + ", ".join(f"{x:.1f}" for x in secs)
                + f"; median {statistics.median(secs):.1f}, total {sum(secs):.1f}"
            )
            if any(orders) and "null" not in orders:
                print(f"  orders chosen (station-folds): {dict(orders)}")


if __name__ == "__main__":
    main()
