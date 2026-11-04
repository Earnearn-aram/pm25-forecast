"""Report export-level training readiness without asserting operational availability."""

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pm25_forecast.models.baselines import HORIZONS_H, require_utc
from pm25_forecast.train_model import MIN_TRAINING_EXAMPLES, build_samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_readiness(cutoff: datetime, export: Path | None = None) -> dict[str, Any]:
    """Summarize claimed export counts while always failing closed on missing provenance.

    ``available_at`` in legacy exports may derive from mutable ``ingested_at``/``created_at``
    fields. These counts are diagnostic only and cannot establish historical visibility.
    """
    cutoff = require_utc(cutoff)
    report: dict[str, Any] = {
        "cutoff": cutoff.isoformat(),
        "cutoff_semantics": "exclusive",
        "status": "blocked_unverified_export_boundary",
        "ready": False,
        "availability_history_verified": False,
        "immutable_correction_history_verified": False,
        "reason": (
            "No trusted, versioned provenance source binds an operational export to a verified "
            "non-holdout boundary. Export-level counts below are diagnostic only; no historical "
            "availability or correction history is established."
        ),
        "export_opened": False,
        "export_read_attempted": False,
        "export_sha256": None,
        "outcome_maturity_export_level": None,
        "horizons": None,
        "retention_and_rollout": (
            "Preserve immutable timestamped captures and issue snapshots for the full training "
            "lookback, outcome-maturity window, and audit margin. Do not enable training until "
            "the visibility contract, retention budget, and controlled rollout are approved."
        ),
    }
    if export is None:
        report["reason"] = (
            "No trusted, versioned provenance source binds an operational export to a verified "
            "non-holdout boundary. No export data was read."
        )
        return report

    path = export.expanduser().resolve()
    if not path.is_file():
        report["status"] = "blocked_export_unavailable"
        report["reason"] = "The requested local export is not a readable file."
        return report

    report["export_sha256"] = _sha256(path)
    report["export_read_attempted"] = True
    # build_samples opens the DuckDB file read-only and applies the same availability cutoff
    # used by training. The cutoff is exclusive, so make its as-of instant just before it.
    samples, labels, _vintages = build_samples(path, cutoff - timedelta(microseconds=1))
    mature = [
        (target, available)
        for (_station, target), (_value, available) in labels.items()
        if target < cutoff and available < cutoff
    ]
    late = [
        (target, available)
        for (_station, target), (_value, available) in labels.items()
        if target < cutoff and available >= cutoff
    ]
    report["export_opened"] = True
    report["outcome_maturity_export_level"] = {
        "mature_outcomes_before_cutoff": len(mature),
        "late_outcomes_at_or_after_cutoff": len(late),
    }
    report["horizons"] = {
        f"+{horizon}h": {
            "usable_examples_export_level": len(samples[horizon]),
            "minimum_examples": MIN_TRAINING_EXAMPLES,
            "count_threshold_met_export_level": len(samples[horizon]) >= MIN_TRAINING_EXAMPLES,
        }
        for horizon in HORIZONS_H
    }
    report["maturity_cutoff"] = cutoff.isoformat()
    report["correction_history"] = (
        "unavailable: a current-state export cannot reconstruct overwritten values or their "
        "earlier arrival times"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Fail closed until trusted provenance binds the export to a non-holdout boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="Exclusive timezone-aware UTC cutoff")
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional local DuckDB export for diagnostic-only counts; never used to train",
    )
    args = parser.parse_args(argv)
    try:
        cutoff = require_utc(datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")))
    except ValueError as exc:
        parser.error(str(exc))

    try:
        report = report_readiness(cutoff, args.export)
    except Exception as exc:
        report = report_readiness(cutoff)
        report.update(
            status="blocked_invalid_export",
            export_read_attempted=args.export is not None,
            reason=f"Export diagnostic failed ({type(exc).__name__}); no readiness was granted.",
        )
    print(json.dumps(report, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
