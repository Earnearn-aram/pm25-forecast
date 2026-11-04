"""Report why operational training readiness cannot currently be verified."""

import argparse
import json
from datetime import datetime

from pm25_forecast.models.baselines import require_utc


def main(argv: list[str] | None = None) -> int:
    """Fail closed until trusted provenance binds an export to a non-holdout boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="Exclusive timezone-aware UTC cutoff")
    args = parser.parse_args(argv)
    try:
        cutoff = require_utc(datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")))
    except ValueError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "cutoff": cutoff.isoformat(),
                "status": "blocked_unverified_export_boundary",
                "ready": False,
                "export_opened": False,
                "availability_history_verified": False,
                "reason": (
                    "No trusted, versioned provenance source binds an operational export to a "
                    "verified non-holdout boundary. No export data was read."
                ),
            },
            indent=2,
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
