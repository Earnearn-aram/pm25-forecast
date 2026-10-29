"""Offline retrospective LightGBM evaluation for the direct +24h horizon.

The reporting delay is an explicit scenario assumption. Historical ``available_at`` values in the
bulk export are not used to infer when measurements originally arrived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np

from pm25_forecast.features import FEATURE_NAMES as SHARED_FEATURE_NAMES
from pm25_forecast.features import build_features

HORIZON_H = 24
FEATURE_NAMES = tuple(name for name in SHARED_FEATURE_NAMES if name != "cams_pm25")
CATEGORY_EDGES = (15.0, 25.0, 37.5, 75.0)
HOUR = timedelta(hours=1)
DAY = timedelta(hours=HORIZON_H)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int]:
    errors = [prediction - target for target, prediction in zip(actual, predicted, strict=True)]
    return {
        "n": len(actual),
        "mae": fmean(abs(error) for error in errors) if errors else 0.0,
        "rmse": math.sqrt(fmean(error * error for error in errors)) if errors else 0.0,
    }


def _category(value: float) -> int:
    return sum(value > boundary for boundary in CATEGORY_EDGES)


def _category_metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    confusion = [
        [0 for _ in range(len(CATEGORY_EDGES) + 1)] for _ in range(len(CATEGORY_EDGES) + 1)
    ]
    for target, estimate in zip(actual, predicted, strict=True):
        confusion[_category(target)][_category(estimate)] += 1
    n = len(actual)
    return {
        "accuracy": sum(confusion[i][i] for i in range(len(confusion))) / n if n else None,
        "boundaries_ug_m3": list(CATEGORY_EDGES),
        "bands": ["<=15", "15-25", "25-37.5", "37.5-75", ">75"],
        "actual_counts": [sum(row) for row in confusion],
        "predicted_counts": [
            sum(confusion[i][j] for i in range(len(confusion))) for j in range(len(confusion))
        ],
        "confusion_matrix_actual_rows_predicted_columns": confusion,
        "definition": "exploratory hourly concentration-band proxy; not official AQI",
    }


def _read_snapshot(path: Path) -> tuple[dict[str, dict[datetime, float]], str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with duckdb.connect(str(path), read_only=True) as con:
        con.execute("SET TimeZone = 'UTC'")
        rows = con.execute(
            "SELECT station_id, ts, pm25 FROM observations "
            "WHERE pm25 IS NOT NULL ORDER BY station_id, ts"
        ).fetchall()
    observations: dict[str, dict[datetime, float]] = defaultdict(dict)
    for station, ts, value in rows:
        if value is not None and math.isfinite(value) and value >= 0:
            observations[station][ts.astimezone(UTC)] = float(value)
    if not observations:
        raise ValueError("snapshot contains no usable observations")
    return dict(observations), digest


def _make_rows(
    observations: dict[str, dict[datetime, float]], lag_hours: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if lag_hours < 0:
        raise ValueError("lag_hours must be non-negative")
    station_times = {station: sorted(values) for station, values in observations.items()}
    all_times = [ts for times in station_times.values() for ts in times]
    first = min(all_times).replace(minute=0, second=0, microsecond=0)
    last_origin = max(all_times) - DAY
    if last_origin < first:
        raise ValueError("snapshot is shorter than the 24-hour forecast horizon")
    origin_count = int((last_origin - first) / HOUR) + 1
    test_start = first + timedelta(hours=int(origin_count * 0.85))
    origin_end = last_origin + HOUR
    rows: list[dict[str, Any]] = []
    for origin_i in range(origin_count):
        origin = first + origin_i * HOUR
        target_time = origin + DAY
        cutoff = origin - timedelta(hours=lag_hours)
        for station, history in observations.items():
            actual = history.get(target_time)
            if actual is None:
                continue
            times = station_times[station]
            end = bisect_right(times, cutoff)
            recent_start = bisect_left(times, origin - timedelta(hours=3))
            recent = (
                [(ts, history[ts]) for ts in times[max(0, min(recent_start, end - 1)) : end]]
                if end
                else []
            )
            # The shared feature builder applies timestamp and freshness rules. The input slice
            # contains only eligible recent observations, avoiding a full-history scan per row.
            features = build_features(recent, (), origin, target_time)
            vector = [features[name] for name in FEATURE_NAMES]
            if all(value is None for value in vector):
                continue
            latest = vector[0]
            seasonal = vector[1]
            rows.append(
                {
                    "station": station,
                    "origin": origin,
                    "target": target_time,
                    "actual": actual,
                    "features": vector,
                    "persistence": latest,
                    "seasonal_naive": seasonal,
                    "is_test": origin >= test_start,
                    "is_train": origin + DAY < test_start,
                }
            )
    metadata = {
        "origins_utc": {"start": _iso(first), "end_exclusive": _iso(origin_end)},
        "test_start_origin_utc": _iso(test_start),
        "test_end_origin_exclusive_utc": _iso(origin_end),
        "horizon_hours": HORIZON_H,
        "origin_step_hours": 1,
        "observation_availability_assumption": f"measurement timestamp <= origin - {lag_hours}h",
        "label_embargo": (
            "training target timestamp must be strictly earlier than first test origin"
        ),
    }
    return rows, metadata


def evaluate(
    path: Path, artifact_path: Path, manifest_path: Path, lag_hours: int = 0
) -> dict[str, Any]:
    observations, snapshot_hash = _read_snapshot(path)
    rows, split = _make_rows(observations, lag_hours)
    train_rows = [row for row in rows if row["is_train"]]
    # Exact same test intersection as the fixed baseline evaluator: valid actual, persistence,
    # and target-minus-24h measurement all exist on the same station/origin/target pair.
    test_rows = [
        row
        for row in rows
        if row["is_test"] and row["persistence"] is not None and row["seasonal_naive"] is not None
    ]
    if len(train_rows) < 20 or len(test_rows) < 1:
        raise ValueError("not enough leakage-safe training examples or common test pairs")
    x_train = np.asarray(
        [[math.nan if x is None else x for x in r["features"]] for r in train_rows]
    )
    y_train = np.asarray([r["actual"] for r in train_rows], dtype=float)
    x_test = np.asarray([[math.nan if x is None else x for x in r["features"]] for r in test_rows])
    y_test = [float(r["actual"]) for r in test_rows]
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "seed": 20261029,
        "feature_fraction_seed": 20261029,
        "bagging_seed": 20261029,
        "num_threads": 1,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train, feature_name=list(FEATURE_NAMES)),
        num_boost_round=200,
    )
    predictions = np.maximum(0.0, booster.predict(x_test)).astype(float).tolist()
    artifact = json.dumps(
        {"format": 1, "horizon_hours": HORIZON_H, "booster": booster.model_to_string()},
        separators=(",", ":"),
    ).encode()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact)
    train_origins = [row["origin"] for row in train_rows]
    train_targets = [row["target"] for row in train_rows]
    report: dict[str, Any] = {
        "status": "ok",
        "source": {
            "snapshot_sha256": snapshot_hash,
            "station_count": len(observations),
            "stations": sorted(observations),
            "observation_rows_used": sum(
                len(rows_by_station) for rows_by_station in observations.values()
            ),
            "observation_span_utc": {
                "start": _iso(min(t for h in observations.values() for t in h)),
                "end": _iso(max(t for h in observations.values() for t in h)),
            },
        },
        "design": {
            **split,
            "assumed_reporting_lag_hours": lag_hours,
            "retrospective_availability_notice": (
                "The export's available_at values are bulk-ingest timestamps. The assumed lag "
                "is a scenario only and does not establish historical real availability."
            ),
            "features": list(FEATURE_NAMES),
            "cams_features_used": False,
            "preprocessing": "none; LightGBM handles missing values natively",
            "training_rows": len(train_rows),
            "test_common_pairs": len(test_rows),
            "train_origin_range_utc": {
                "start": _iso(min(train_origins)),
                "end": _iso(max(train_origins)),
            },
            "train_target_range_utc": {
                "start": _iso(min(train_targets)),
                "end": _iso(max(train_targets)),
            },
            "test_pairs_are_common_to": [
                "LightGBM",
                "persistence_latest_3h",
                "seasonal_target_minus_24h",
            ],
            "category_proxy": _category_metrics(y_test, predictions)["definition"],
        },
        "test_metrics": {
            "lightgbm_direct_24h": {
                **_metrics(y_test, predictions),
                "category": _category_metrics(y_test, predictions),
            },
            "persistence_latest_3h": {
                **_metrics(y_test, [float(r["persistence"]) for r in test_rows]),
                "category": _category_metrics(y_test, [float(r["persistence"]) for r in test_rows]),
            },
            "seasonal_target_minus_24h": {
                **_metrics(y_test, [float(r["seasonal_naive"]) for r in test_rows]),
                "category": _category_metrics(
                    y_test, [float(r["seasonal_naive"]) for r in test_rows]
                ),
            },
        },
        "coverage": {
            "test_pairs": len(test_rows),
            "test_station_counts": {
                station: sum(row["station"] == station for row in test_rows)
                for station in sorted(observations)
            },
            "test_target_dates_utc": sorted(
                {row["target"].date().isoformat() for row in test_rows}
            ),
            "test_target_date_range_utc": {
                "start": min(row["target"].date().isoformat() for row in test_rows),
                "end": max(row["target"].date().isoformat() for row in test_rows),
            },
        },
    }
    manifest = {
        "version": "kan15-retrospective-24h",
        "model_type": "lightgbm-retrospective-direct-24h",
        "metrics": {"24": report["test_metrics"]["lightgbm_direct_24h"]},
        "training_window": {"start": _iso(min(train_origins)), "end": _iso(max(train_origins))},
        "feature_list": list(FEATURE_NAMES),
        "data_sha256": snapshot_hash,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "assumed_reporting_lag_hours": lag_hours,
        "test_start_origin_utc": split["test_start_origin_utc"],
        "trained_at": datetime.now(UTC).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["artifact"] = {
        "sha256": manifest["artifact_sha256"],
        "format": "JSON LightGBM model bundle",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/history.duckdb"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lag-hours", type=int, default=0)
    args = parser.parse_args(argv)
    report = evaluate(args.database, args.artifact, args.manifest, args.lag_hours)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.report),
                "test_metrics": report["test_metrics"],
                "test_pairs": report["coverage"]["test_pairs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
