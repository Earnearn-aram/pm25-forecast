import json
from datetime import UTC, datetime

import pytest

from pm25_forecast import training_readiness


def test_cli_fails_closed_without_export_access(capsys):
    cutoff = "2026-01-03T00:00:00Z"
    assert training_readiness.main(["--cutoff", cutoff]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "cutoff": datetime(2026, 1, 3, tzinfo=UTC).isoformat(),
        "status": "blocked_unverified_export_boundary",
        "ready": False,
        "export_opened": False,
        "availability_history_verified": False,
        "reason": (
            "No trusted, versioned provenance source binds an operational export to a verified "
            "non-holdout boundary. No export data was read."
        ),
    }


def test_runtime_module_exposes_no_export_reading_api():
    assert not hasattr(training_readiness, "report_readiness")
    assert not hasattr(training_readiness, "make_training_export")


@pytest.mark.parametrize("cutoff", ["2026-01-03T00:00:00", "not-a-timestamp"])
def test_cli_rejects_naive_or_invalid_cutoff(cutoff):
    with pytest.raises(SystemExit) as exc:
        training_readiness.main(["--cutoff", cutoff])
    assert exc.value.code == 2
