from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from pm25_forecast.retrain import make_training_export
from pm25_forecast.train_model import _validation_indices, build_samples
from pm25_forecast.training_readiness import report_readiness


def make_export(path, observations):
    with duckdb.connect(str(path)) as con:
        con.execute(
            "CREATE TABLE observations (station_id VARCHAR, ts TIMESTAMPTZ, pm25 DOUBLE, "
            "source VARCHAR, available_at TIMESTAMPTZ)"
        )
        con.execute(
            "CREATE TABLE forecast_vintages (station_id VARCHAR, issue_time TIMESTAMPTZ, "
            "valid_time TIMESTAMPTZ, model_version VARCHAR, run_id VARCHAR, "
            "pm25_pred DOUBLE, available_at TIMESTAMPTZ)"
        )
        if observations:
            con.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?)", observations)


def test_report_excludes_late_arrivals_and_future_rows_without_scoring(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    cutoff = start + timedelta(hours=100)
    data = [
        ("s", start + timedelta(hours=i), float(i), "air4thai", start + timedelta(hours=i))
        for i in range(100)
    ]
    # A corrected current value is only available after the cutoff; do not invent the old value.
    data[10] = ("s", start + timedelta(hours=10), 999.0, "air4thai", cutoff + timedelta(hours=1))
    data.append(("s", cutoff, 5.0, "air4thai", cutoff))
    path = tmp_path / "export.duckdb"
    make_export(path, data)
    result = report_readiness(path, cutoff, cutoff)
    assert result["usable_mature_observations_before_cutoff"] == 99
    assert result["cutoff"] == cutoff.isoformat()
    assert result["horizons"]["24"]["usable_examples"] > 0
    assert "overwritten values" in result["limitations"][0]
    with duckdb.connect(str(path), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM observations").fetchone()[0] == 101


def test_empty_and_all_late_data_are_not_ready(tmp_path):
    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    path = tmp_path / "export.duckdb"
    make_export(path, [("s", cutoff - timedelta(days=1), 20.0, "air4thai", cutoff)])
    result = report_readiness(path, cutoff, cutoff)
    assert result["status"] == "not_ready"
    assert result["usable_mature_observations_before_cutoff"] == 0
    assert all(h["usable_examples"] == 0 for h in result["horizons"].values())
    assert all(h["training_after_embargo"] is None for h in result["horizons"].values())


def test_cutoff_must_be_timezone_aware(tmp_path):
    path = tmp_path / "export.duckdb"
    make_export(path, [])
    with pytest.raises(ValueError, match="timezone-aware"):
        report_readiness(path, datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=UTC))  # noqa: DTZ001


def test_refuses_cutoff_past_declared_safe_boundary_before_opening_file(tmp_path):
    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-holdout boundary"):
        report_readiness(tmp_path / "not-opened.duckdb", cutoff, cutoff - timedelta(days=1))


def test_count_ready_matches_trainer_sample_and_embargo_counts(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    cutoff = start + timedelta(hours=180)
    data = [
        ("s", start + timedelta(hours=i), float(i), "air4thai", start + timedelta(hours=i))
        for i in reversed(range(180))
    ]
    path = tmp_path / "export.duckdb"
    filtered = tmp_path / "training.duckdb"
    make_export(path, data)
    result = report_readiness(path, cutoff, cutoff)
    make_training_export(path, filtered, cutoff)
    samples, _, _ = build_samples(filtered, as_of=cutoff)
    assert result["status"] == "count_ready"
    assert result["availability_history_verified"] is False
    for horizon, rows in samples.items():
        rows.sort(key=lambda row: row[2])
        train, validation = _validation_indices([r[2] for r in rows], [r[3] for r in rows], horizon)
        actual = result["horizons"][str(horizon)]
        assert actual["usable_examples"] == len(rows)
        assert actual["training_after_embargo"] == len(train)
        assert actual["validation_examples"] == len(validation)
