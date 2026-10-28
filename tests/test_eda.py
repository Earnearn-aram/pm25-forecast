import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import pm25_forecast.eda as eda
from pm25_forecast.eda import (
    EdaDatabaseError,
    Observation,
    analyze,
    load_observations,
    render_bar_svg,
    render_markdown,
    write_report,
)


def row(station: str, ts: datetime, pm25: float | None, source: str = "openaq") -> Observation:
    return Observation(station, ts, pm25, source)


def test_analyze_reports_missingness_seasonality_and_sources() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    metrics = analyze(
        [
            row("02t", start, 100.0),
            row("02t", start + timedelta(hours=1), 80.0, "air4thai"),
            row("02t", start + timedelta(hours=3), None),
            row("02t", datetime(2025, 5, 1, tzinfo=UTC), 20.0),
            row("59t", start, -1.0),
            row("59t", start + timedelta(hours=1), 40.0),
        ]
    )
    first, second = metrics["stations"]
    assert first["usable_rows"] == 3 and first["missing_hours"] > 0 and first["max_gap_hours"] > 0
    assert second["invalid_rows"] == 1
    assert metrics["burning_season"]["mean_pm25"] == 73.33
    assert metrics["burning_season"]["non_burning_mean_pm25"] == 20.0
    assert metrics["source_counts"] == {"air4thai": 1, "openaq": 5}


def test_report_and_charts_contain_aggregates() -> None:
    metrics = analyze(
        [
            row("02t", datetime(2025, 1, 1, tzinfo=UTC), 50.0),
            row("02t", datetime(2025, 5, 1, tzinfo=UTC), 25.0),
        ]
    )
    assert "January-April mean PM2.5" in render_markdown(metrics)
    assert "DATABASE_URL=..." in render_markdown(metrics)
    assert "<svg" in render_bar_svg(metrics["monthly_mean_pm25"], "Monthly", ["1", "5"])


def test_analyze_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        analyze([row("02t", datetime(2025, 1, 1), 50.0)])  # noqa: DTZ001


def test_leading_trailing_nulls_and_all_null_station_are_full_gaps() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    metrics = analyze(
        [
            row("02t", start, None),
            row("02t", start + timedelta(hours=1), 10.0),
            row("02t", start + timedelta(hours=4), None),
            row("59t", start, None),
            row("59t", start + timedelta(hours=3), None),
        ]
    )

    first, second = metrics["stations"]
    assert first["max_gap_hours"] == 3
    assert second["expected_hours"] == 4
    assert second["max_gap_hours"] == 4


def test_zero_values_render_without_division_by_zero() -> None:
    svg = render_bar_svg({"0": 0.0, "1": 0.0}, "Zeros", ["0", "1"])
    assert "<svg" in svg
    assert 'height="0.0"' in svg


def test_missing_seasons_and_all_invalid_station_use_strict_json(tmp_path: Path) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    metrics = analyze([row("02t", start, -1.0)], station_ids=["02t"])

    assert metrics["stations"][0]["mean_pm25"] is None
    assert metrics["burning_season"]["mean_pm25"] is None
    assert metrics["burning_season"]["non_burning_mean_pm25"] is None
    write_report(metrics, tmp_path)
    encoded = (tmp_path / "kan-13-metrics.json").read_text()
    assert "NaN" not in encoded
    assert json.loads(encoded)["stations"][0]["p95_pm25"] is None


def test_configured_station_with_no_rows_remains_visible() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    metrics = analyze([row("02t", start, 10.0)], station_ids=["02t", "59t"])

    empty = metrics["stations"][1]
    assert empty["station_id"] == "59t"
    assert empty["rows"] == 0
    assert empty["coverage_pct"] is None
    assert "| 59t | 0 | 0 | 0 | n/a |" in render_markdown(metrics)


def test_database_errors_are_sanitised(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "postgresql://user:password@secret-host/db"  # pragma: allowlist secret

    def fail(_url: str):
        raise RuntimeError(f"could not connect to {secret}")

    monkeypatch.setattr(eda, "connect", fail)
    with pytest.raises(EdaDatabaseError) as caught:
        load_observations(secret, ["02t"])

    message = str(caught.value)
    assert message == "database query failed (RuntimeError)"
    assert "password" not in message and "secret-host" not in message
    assert caught.value.__cause__ is None


def test_coverage_counts_distinct_utc_hour_buckets_and_never_exceeds_100() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    metrics = analyze(
        [
            row("02t", start, 10.0),
            row("02t", start + timedelta(minutes=20), 12.0),
            row("02t", start + timedelta(hours=1, minutes=5), 14.0),
        ],
        station_ids=["02t"],
    )

    station = metrics["stations"][0]
    assert station["expected_hours"] == 2
    assert station["usable_rows"] == 3
    assert station["coverage_pct"] == 100.0
    assert station["missing_hours"] == 0
    assert metrics["duplicate_station_hours"] == 1
