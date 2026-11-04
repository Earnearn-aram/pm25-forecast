import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from pm25_forecast import training_readiness


def test_cli_fails_closed_without_export_access(capsys):
    cutoff = "2026-01-03T00:00:00Z"
    assert training_readiness.main(["--cutoff", cutoff]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "cutoff": datetime(2026, 1, 3, tzinfo=UTC).isoformat(),
        "cutoff_semantics": "exclusive",
        "status": "blocked_unverified_export_boundary",
        "ready": False,
        "availability_history_verified": False,
        "immutable_correction_history_verified": False,
        "reason": (
            "No trusted, versioned provenance source binds an operational export to a verified "
            "non-holdout boundary. No export data was read."
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


def _write_export(path: Path, *, corrected_at: datetime | None = None) -> datetime:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    cutoff = start + timedelta(hours=72)
    rows = []
    for hour in range(72):
        observed_at = start + timedelta(hours=hour)
        value, available_at = 10.0, observed_at
        if corrected_at is not None and observed_at == start + timedelta(hours=60):
            # Current state contains the corrected value and later arrival only; old state is gone.
            value, available_at = 25.0, corrected_at
        rows.append(("59t", observed_at, value, "fixture", available_at))
    with duckdb.connect(str(path)) as con:
        con.execute(
            "CREATE TABLE observations (station_id VARCHAR, ts TIMESTAMPTZ, pm25 DOUBLE, "
            "source VARCHAR, available_at TIMESTAMPTZ)"
        )
        con.execute(
            "CREATE TABLE forecast_vintages (station_id VARCHAR, issue_time TIMESTAMPTZ, "
            "valid_time TIMESTAMPTZ, model_version VARCHAR, run_id VARCHAR, pm25_pred DOUBLE, "
            "available_at TIMESTAMPTZ)"
        )
        con.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?)", rows)
    return cutoff


def test_export_diagnostic_reports_maturity_and_stays_not_ready(tmp_path, capsys):
    export = tmp_path / "history.duckdb"
    cutoff = _write_export(export)

    assert training_readiness.main(["--cutoff", cutoff.isoformat(), "--export", str(export)]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["status"] == "blocked_unverified_export_boundary"
    assert report["ready"] is False
    assert report["availability_history_verified"] is False
    assert report["immutable_correction_history_verified"] is False
    assert report["export_opened"] is True
    assert len(report["export_sha256"]) == 64
    assert report["correction_history"].startswith("unavailable:")
    assert report["outcome_maturity_export_level"] == {
        "mature_outcomes_before_cutoff": 72,
        "late_outcomes_at_or_after_cutoff": 0,
    }
    assert set(report["horizons"]) == {"+6h", "+12h", "+24h"}
    for horizon in report["horizons"].values():
        assert horizon["usable_examples_export_level"] > 0
        assert horizon["minimum_examples"] == 20


def test_late_corrected_current_value_is_excluded_and_history_remains_unverified(tmp_path, capsys):
    clean_export = tmp_path / "clean-history.duckdb"
    export = tmp_path / "corrected-history.duckdb"
    cutoff = _write_export(clean_export)
    _write_export(export, corrected_at=datetime(2026, 1, 5, 3, tzinfo=UTC))
    clean_report = training_readiness.report_readiness(cutoff, clean_export)

    assert training_readiness.main(["--cutoff", cutoff.isoformat(), "--export", str(export)]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["ready"] is False
    assert report["immutable_correction_history_verified"] is False
    assert report["outcome_maturity_export_level"] == {
        "mature_outcomes_before_cutoff": 71,
        "late_outcomes_at_or_after_cutoff": 1,
    }
    for horizon in report["horizons"].values():
        assert "mature_outcomes_export_level" not in horizon
    for key, horizon in report["horizons"].items():
        assert horizon["usable_examples_export_level"] == (
            clean_report["horizons"][key]["usable_examples_export_level"] - 1
        )


def test_invalid_export_is_blocked_without_claiming_it_opened(tmp_path, capsys):
    export = tmp_path / "not-a-duckdb-file.duckdb"
    export.write_text("not a database")
    cutoff = datetime(2026, 1, 3, tzinfo=UTC)

    assert training_readiness.main(["--cutoff", cutoff.isoformat(), "--export", str(export)]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["status"] == "blocked_invalid_export"
    assert report["ready"] is False
    assert report["export_read_attempted"] is True
    assert report["export_opened"] is False
    assert report["horizons"] is None


@pytest.mark.parametrize("cutoff", ["2026-01-03T00:00:00", "not-a-timestamp"])
def test_cli_rejects_naive_or_invalid_cutoff(cutoff):
    with pytest.raises(SystemExit) as exc:
        training_readiness.main(["--cutoff", cutoff])
    assert exc.value.code == 2
