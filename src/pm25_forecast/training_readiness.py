"""Report retrain data readiness from a local export, without training or scoring.

The cutoff must be chosen before running this command. Only rows both dated and available before
it enter the temporary training export; the report never evaluates the data after the cutoff.
"""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from pm25_forecast.models.baselines import HORIZONS_H, require_utc
from pm25_forecast.retrain import make_training_export
from pm25_forecast.train_model import MIN_TRAINING_EXAMPLES, _validation_indices, build_samples


def report_readiness(export_path: Path, cutoff: datetime, safe_before: datetime) -> dict[str, Any]:
    """Use the retrain filter and sample builder to report counts, not model performance."""
    cutoff = require_utc(cutoff)
    safe_before = require_utc(safe_before)
    if cutoff > safe_before:
        raise ValueError("cutoff exceeds the operator-verified non-holdout boundary")
    if not export_path.is_file():
        raise FileNotFoundError(f"export not found: {export_path}")
    with tempfile.TemporaryDirectory() as tmp:
        training_export = Path(tmp) / "training.duckdb"
        make_training_export(export_path, training_export, cutoff)
        samples, labels, _ = build_samples(training_export, as_of=cutoff)

    horizons: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS_H:
        rows = sorted(samples[horizon], key=lambda row: row[2])
        usable = len(rows)
        train_count = validation_count = None
        if usable >= MIN_TRAINING_EXAMPLES:
            train, validation = _validation_indices(
                [row[2] for row in rows], [row[3] for row in rows], horizon
            )
            train_count, validation_count = len(train), len(validation)
        horizons[str(horizon)] = {
            "usable_examples": usable,
            "training_after_embargo": train_count,
            "validation_examples": validation_count,
            "ready": train_count is not None and train_count >= MIN_TRAINING_EXAMPLES,
        }
    return {
        "cutoff": cutoff.isoformat(),
        "status": "count_ready" if all(h["ready"] for h in horizons.values()) else "not_ready",
        "operator_verified_safe_before": safe_before.isoformat(),
        "availability_history_verified": False,
        "minimum_training_examples_per_horizon": MIN_TRAINING_EXAMPLES,
        "usable_mature_observations_before_cutoff": len(labels),
        "horizons": horizons,
        "limitations": [
            "Current observations are mutable: overwritten values and earlier availability "
            "cannot be reconstructed.",
            "Availability timestamps are conservative recorded times, not proof of transaction "
            "commit visibility or provider availability.",
            "Readiness is a sample-count check, not model quality or promotion approval.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True, help="Existing local DuckDB export")
    parser.add_argument("--cutoff", required=True, help="Exclusive timezone-aware training cutoff")
    parser.add_argument(
        "--safe-before",
        required=True,
        help="Operator-verified exclusive boundary before any locked holdout in the export",
    )
    args = parser.parse_args(argv)
    try:
        cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
        safe_before = datetime.fromisoformat(args.safe_before.replace("Z", "+00:00"))
        report = report_readiness(args.export, cutoff, safe_before)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
