"""KAN-84 (E06): generate the 24 pre-registrations of the linear-model grid (and the sensitivity).

    python -m pm25_forecast.experiments.kan84_prereg [--sensitivity <variant-id>]

Grid: model {ols, ridge, lasso} x scope {pooled, per_station} x target {raw, log1p} x output
{level, change} = 24 variants on ``fs_v2_core``. One further, pre-declared sensitivity (the best
variant on ``fs_v2_full_nocross``) brings the planned count to K = 25 (protocol section 8).

Note: the preregs are unchanged by the purge fix in ``linear_models.inner_splits`` (documented
deviation: the first runs used a 1 h short purge and were superseded by a full re-run).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

K_PLANNED = 25
SHORT = {
    "scope": {"pooled": "pool", "per_station": "stn"},
    "target": {"raw": "raw", "log1p": "log"},
    "output": {"level": "lvl", "change": "chg"},
}


def variant_id(model: str, scope: str, target: str, output: str, feature_set: str) -> str:
    base = (
        f"e06-{model}-{SHORT['scope'][scope]}-{SHORT['target'][target]}-{SHORT['output'][output]}"
    )
    return base if feature_set == "fs_v2_core" else f"{base}-full"


def prereg_dict(
    model: str, scope: str, target: str, output: str, feature_set: str = "fs_v2_core"
) -> dict:
    vid = variant_id(model, scope, target, output, feature_set)
    reg = {
        "ols": "none (ordinary least squares)",
        "ridge": "ridge lambda from a 9-point grid 1e-4..1 (half decades)",
        "lasso": "lasso alpha = ratio x alpha_max, ratio from a 9-point grid 1e-4..1",
    }[model]
    return {
        "id": vid,
        "ticket": "KAN-84",
        "hypothesis": (
            f"A {model} model ({scope.replace('_', '-')}, {target} target, predicting the "
            f"{output}) on {feature_set} lowers +24 h dev MAE versus the smoother (E06 grid of "
            f"24 variants plus 1 planned sensitivity, K = {K_PLANNED})."
        ),
        "model_family": "linear",
        "model": vid.removeprefix("e06-"),
        "feature_set": feature_set,
        "params": {
            "variant": vid,
            "model": model,
            "scope": scope,
            "target": target,
            "output": output,
            "feature_set": feature_set,
            "regularisation": reg,
            "inner_cv": "expanding window, 5 time blocks by origin, 4 splits, MAE in ug/m3, "
            "training part of each fold only",
            "primary_ci_block_hours": 72,
            "planned_comparisons_K": K_PLANNED,
        },
        "horizons": [6, 12, 24],
        "expected_effect": {
            "metric": "diff_vs_smoother_24h",
            "direction": "negative",
            "magnitude": -0.15,
        },
        "decision_rule": (
            "Dev exploration. Primary CI: 72 h moving-block bootstrap (24 h and 168 h as "
            "sensitivities). The best variant is the valid one with the lowest dev mae_24h. It "
            f"counts as better than the smoother only if its 72 h CI at the Bonferroni level "
            f"1 - 0.05/{K_PLANNED} (K = {K_PLANNED} planned variants) is entirely below zero; "
            "the unadjusted 95% CI is descriptive. No holdout scoring."
        ),
        "split": "dev",
        "final": False,
        "finalists": 1,
        "author_agent": "claude-data-scientist",
    }


def grid() -> list[tuple[str, str, str, str]]:
    return list(
        itertools.product(
            ("ols", "ridge", "lasso"),
            ("pooled", "per_station"),
            ("raw", "log1p"),
            ("level", "change"),
        )
    )


def write(out: Path, spec: dict) -> Path:
    path = out / f"{spec['id']}.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False, width=100), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/prereg"))
    parser.add_argument("--sensitivity", default=None, help="variant id to repeat on the full set")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.sensitivity:
        rows = [r for r in grid() if variant_id(*r, "fs_v2_core") == args.sensitivity]
        if len(rows) != 1:
            raise SystemExit(f"unknown variant {args.sensitivity!r}")
        print(write(args.out, prereg_dict(*rows[0], "fs_v2_full_nocross")))
        return
    for row in grid():
        print(write(args.out, prereg_dict(*row)))


if __name__ == "__main__":
    main()
