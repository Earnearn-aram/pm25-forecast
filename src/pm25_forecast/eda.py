"""Aggregate-only exploratory analysis for stored PM2.5 observations."""

# Report prose is intentionally readable Markdown rather than line-wrapped source.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from pm25_forecast.storage.db import connect

TARGET_STATIONS = ("02t", "59t", "61t", "bkp100t", "bkp101t")


@dataclass(frozen=True)
class Observation:
    station_id: str
    ts: datetime
    pm25: float | None
    source: str


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observation timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _max_gap_hours(timestamps: Iterable[datetime]) -> int:
    ordered = sorted(set(timestamps))
    if len(ordered) < 2:
        return 0
    return max(
        max(int((right - left).total_seconds() // 3600) - 1, 0) for left, right in pairwise(ordered)
    )


def analyze(observations: Iterable[Observation]) -> dict[str, Any]:
    """Return JSON-serialisable aggregate metrics for observations."""
    rows = [Observation(row.station_id, _utc(row.ts), row.pm25, row.source) for row in observations]
    by_station: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        by_station[row.station_id].append(row)
    station_metrics: list[dict[str, Any]] = []
    hourly: dict[int, list[float]] = defaultdict(list)
    weekday: dict[int, list[float]] = defaultdict(list)
    monthly: dict[int, list[float]] = defaultdict(list)
    burning: list[float] = []
    other: list[float] = []
    for station_id in sorted(by_station):
        station_rows = by_station[station_id]
        usable = [row for row in station_rows if row.pm25 is not None]
        valid = [row for row in usable if 0 <= float(row.pm25) <= 1000]
        values = [float(row.pm25) for row in valid if row.pm25 is not None]
        timestamps = [row.ts for row in valid]
        first, last = min(row.ts for row in station_rows), max(row.ts for row in station_rows)
        expected = int((last - first).total_seconds() // 3600) + 1
        distinct_usable = len(set(timestamps))
        station_metrics.append(
            {
                "station_id": station_id,
                "rows": len(station_rows),
                "usable_rows": len(usable),
                "openaq_rows": sum(row.source == "openaq" for row in station_rows),
                "air4thai_rows": sum(row.source == "air4thai" for row in station_rows),
                "first_ts": first.isoformat(),
                "last_ts": last.isoformat(),
                "expected_hours": expected,
                "missing_hours": max(expected - distinct_usable, 0),
                "coverage_pct": round(100 * distinct_usable / expected, 2),
                "invalid_rows": len(usable) - len(valid),
                "mean_pm25": round(statistics.fmean(values), 2) if values else math.nan,
                "median_pm25": round(statistics.median(values), 2) if values else math.nan,
                "p95_pm25": round(_percentile(values, 0.95), 2),
                "max_gap_hours": _max_gap_hours(timestamps),
            }
        )
        for row in valid:
            value = float(row.pm25)  # type: ignore[arg-type]
            hourly[row.ts.hour].append(value)
            weekday[row.ts.weekday()].append(value)
            monthly[row.ts.month].append(value)
            (burning if row.ts.month <= 4 else other).append(value)

    def means(groups: dict[int, list[float]], indexes: Iterable[int]) -> dict[str, float | None]:
        return {
            str(index): round(statistics.fmean(groups[index]), 2) if groups[index] else None
            for index in indexes
        }

    burning_mean = statistics.fmean(burning) if burning else math.nan
    other_mean = statistics.fmean(other) if other else math.nan
    shift = burning_mean - other_mean
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "stations": station_metrics,
        "duplicate_station_hours": len(rows) - len({(row.station_id, row.ts) for row in rows}),
        "hourly_mean_pm25": means(hourly, range(24)),
        "weekday_mean_pm25": means(weekday, range(7)),
        "monthly_mean_pm25": means(monthly, range(1, 13)),
        "burning_season": {
            "definition": "January-April (UTC month)",
            "rows": len(burning),
            "mean_pm25": round(burning_mean, 2),
            "median_pm25": round(statistics.median(burning), 2) if burning else None,
            "non_burning_rows": len(other),
            "non_burning_mean_pm25": round(other_mean, 2),
            "non_burning_median_pm25": round(statistics.median(other), 2) if other else None,
            "mean_shift_pm25": round(shift, 2),
            "mean_shift_pct": round(100 * shift / other_mean, 2)
            if other and other_mean
            else math.nan,
        },
        "source_counts": dict(sorted(Counter(row.source for row in rows).items())),
    }


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:,}" if isinstance(value, int) else str(value)


def render_markdown(metrics: dict[str, Any]) -> str:
    station_rows = "\n".join(
        "| {station_id} | {usable_rows:,} | {openaq_rows:,} | {air4thai_rows:,} | "
        "{coverage_pct:.2f}% | {missing_hours:,} | {max_gap_hours:,} | {median_pm25} | "
        "{p95_pm25} | {invalid_rows:,} |".format(**row)
        for row in metrics["stations"]
    )
    burning = metrics["burning_season"]
    monthly = metrics["monthly_mean_pm25"]
    monthly_rows = "\n".join(f"| {month} | {_fmt(monthly[str(month)])} |" for month in range(1, 13))
    return f"""# KAN-13 PM2.5 exploratory data analysis

Generated from the production `observations` table at `{metrics["generated_at"]}`. The generator exports aggregate statistics only; it does not export production rows or credentials.

## Executive findings

- January-April mean PM2.5 is **{_fmt(burning["mean_pm25"])} µg/m³**, versus **{_fmt(burning["non_burning_mean_pm25"])} µg/m³** in May-December: a shift of **{_fmt(burning["mean_shift_pm25"])} µg/m³ ({_fmt(burning["mean_shift_pct"])}%)**.
- The extract contains **{metrics["duplicate_station_hours"]} duplicate station-hours** (the production primary key should keep this at zero).
- Invalid usable readings outside 0-1000 µg/m³ are reported per station and should be excluded from modeling.

## Coverage, missingness, and target distribution

| Station | Usable rows | OpenAQ rows | Air4Thai rows | Coverage | Missing hours | Longest gap (h) | Median | P95 | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{station_rows}

Coverage spans each station's first through last timestamp, inclusive. A missing hour has no valid PM2.5 reading.

## Monthly seasonality

| UTC month | Mean PM2.5 (µg/m³) |
| ---: | ---: |
{monthly_rows}

![Monthly mean PM2.5](kan-13-monthly.svg)

## Intraday pattern

![Hourly mean PM2.5](kan-13-hourly.svg)

## Modeling implications

1. Use `(station_id, ts)` as the unique target key and retain source provenance.
2. Exclude null and out-of-range PM2.5; do not interpolate gaps until a backtest establishes an imputation policy.
3. Include station identity, cyclical hour/month features, and a burning-season indicator.
4. Split validation chronologically; random row splits leak seasonal and adjacent-hour information.
5. Score by station and burning/non-burning season so aggregate accuracy cannot hide high-pollution failures.

## Reproduce

`DATABASE_URL=... uv run python -m pm25_forecast.eda --output-dir docs/generated`

The database URL is read from the environment and never written to the report.
"""


def render_bar_svg(values: dict[str, float | None], title: str, labels: Sequence[str]) -> str:
    width, height, left, bottom, top = 960, 420, 64, 54, 54
    plot_width, plot_height = width - left - 20, height - top - bottom
    ceiling = (
        max((float(value) for value in values.values() if value is not None), default=1.0) * 1.1
    )
    bar_width = plot_width / max(len(labels), 1)
    bars: list[str] = []
    for index, label in enumerate(labels):
        value = values.get(label)
        if value is None:
            continue
        bar_height = plot_height * float(value) / ceiling
        x, y = left + index * bar_width + bar_width * 0.12, top + plot_height - bar_height
        bars.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.76:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>',
                f'<text x="{x + bar_width * 0.38:.1f}" y="{height - 28}" text-anchor="middle" font-size="12">{label}</text>',
            ]
        )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{width - 20}" y2="{top + plot_height}" stroke="#374151"/>',
            *bars,
            "</svg>",
        ]
    )


def load_observations(database_url: str, stations: Sequence[str]) -> list[Observation]:
    """Load the bounded target dataset without logging the connection string."""
    with connect(database_url) as con, con.cursor() as cur:
        cur.execute(
            "SELECT station_id, ts, pm25, source FROM observations "
            "WHERE station_id = ANY(%s) ORDER BY station_id, ts",
            (list(stations),),
        )
        return [Observation(station, ts, pm25, source) for station, ts, pm25, source in cur]


def write_report(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kan-13-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "kan-13-eda.md").write_text(render_markdown(metrics))
    (output_dir / "kan-13-monthly.svg").write_text(
        render_bar_svg(
            metrics["monthly_mean_pm25"], "Monthly mean PM2.5", [str(v) for v in range(1, 13)]
        )
    )
    (output_dir / "kan-13-hourly.svg").write_text(
        render_bar_svg(
            metrics["hourly_mean_pm25"], "Hourly mean PM2.5 (UTC)", [str(v) for v in range(24)]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/generated"))
    parser.add_argument("--station", action="append", dest="stations")
    args = parser.parse_args()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL must be set")
    write_report(
        analyze(load_observations(database_url, args.stations or TARGET_STATIONS)), args.output_dir
    )
    print(f"wrote aggregate EDA report to {args.output_dir}")


if __name__ == "__main__":
    main()
