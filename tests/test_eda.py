from datetime import UTC, datetime, timedelta

import pytest

from pm25_forecast.eda import Observation, analyze, render_bar_svg, render_markdown


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
