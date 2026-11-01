"""KAN-87 (E10) coverage of CAMS vintages over the origins of a split.

    .venv/bin/python -m pm25_forecast.experiments.kan87_coverage \
        --snapshot ~/pm25-experiments/snapshots/e02-v1 [--split dev]

For every cohort station and every +24 h origin whose target lies in a fold of the split, counts
the origins with (a) a visible vintage for the target hour and (b) all four CAMS features. It
applies the same data limit as the runner: vintages with ``valid_time >= limit`` (H0 for dev)
are never read, so this is the coverage a dev run would really see. Dev split by default;
``--split holdout`` is refused (the holdout is locked, KAN-94).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from pm25_forecast.backtest import Observation, Vintage
from pm25_forecast.experiments import folds as folds_mod
from pm25_forecast.experiments.dataset import load_snapshot
from pm25_forecast.experiments.evaluate import make_plan, select_cohort
from pm25_forecast.features.cams import CAMS_FEATURE_NAMES, build_cams_features, prepare_cams
from pm25_forecast.features.library_v2 import prepare_series

HOUR = timedelta(hours=1)
HORIZON = 24


def coverage(
    observations: Sequence[Observation], vintages: Sequence[Vintage], split: str = "dev"
) -> dict:
    usable = [o for o in observations if folds_mod.S0 <= o.ts < folds_mod.TARGET_CUT]
    plan = make_plan(max(o.ts for o in usable), split)
    data = [o for o in usable if o.ts < plan.limit]
    cohort = select_cohort(data, folds_mod.S0, plan.dev_limit)
    kept = [v for v in vintages if v.valid_time < plan.limit]
    out: dict = {
        "split": split,
        "H0": plan.h0.isoformat() if plan.h0 else None,
        "limit": plan.limit.isoformat(),
        "stations": cohort,
        "vintages_in_snapshot": len(vintages),
        "vintages_read": len(kept),
        "first_vintage_valid_time": min((v.valid_time for v in vintages), default=None),
        "origins": 0,
        "with_target_vintage": 0,
        "with_all_features": 0,
    }
    if out["first_vintage_valid_time"] is not None:
        out["first_vintage_valid_time"] = out["first_vintage_valid_time"].isoformat()
    for station in cohort:
        prep = prepare_series({station: [o for o in data if o.station_id == station]})
        cams = prepare_cams({station: [v for v in kept if v.station_id == station]})
        for start, end in plan.folds:
            target = start
            while target < end:
                origin = target - HORIZON * HOUR
                f = build_cams_features(prep, cams, station, origin, target)
                out["origins"] += 1
                out["with_target_vintage"] += f["cams_pm25_target"] is not None
                out["with_all_features"] += all(f[n] is not None for n in CAMS_FEATURE_NAMES)
                target += HOUR
    n = out["origins"]
    out["share_target_vintage"] = out["with_target_vintage"] / n if n else None
    out["share_all_features"] = out["with_all_features"] / n if n else None
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--split", default="dev", choices=["dev"])
    args = p.parse_args(argv)
    obs, vin, _ = load_snapshot(Path(args.snapshot).expanduser())
    print(json.dumps(coverage(obs, vin, args.split), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
