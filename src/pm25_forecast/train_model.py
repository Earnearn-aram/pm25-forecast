"""Fit local direct LightGBM models from a read-only DuckDB history export."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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

from pm25_forecast.collector import ConfigError, open_db
from pm25_forecast.features import FEATURE_NAMES, build_features
from pm25_forecast.settings import get_settings
from pm25_forecast.storage import db
from pm25_forecast.storage.registry import publish_model, read_model

logger = logging.getLogger(__name__)
HORIZONS_H = (6, 12, 24)


class InsufficientTrainingDataError(ValueError):
    """The export does not contain enough causally eligible rows to fit every horizon."""


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("export timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _rows(con: Any, query: str) -> list[tuple]:
    return con.execute(query).fetchall()


def make_training_rows(
    export_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray, list[datetime]]]:
    """Build as-of features and pair them with labels mature in the local export."""
    return _make_training_rows_with_availability(export_path)[0]


def _make_training_rows_with_availability(
    export_path: Path,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray, list[datetime]]],
    dict[int, list[datetime]],
]:
    with duckdb.connect(str(export_path), read_only=True) as con:
        con.execute("SET TimeZone = 'UTC'")
        observations = _rows(
            con,
            "SELECT station_id, ts, pm25, available_at FROM observations ORDER BY station_id, ts",
        )
        vintages = _rows(
            con,
            "SELECT station_id, issue_time, valid_time, pm25_pred, model_version, available_at "
            "FROM forecast_vintages ORDER BY station_id, issue_time, valid_time",
        )

    by_station_obs: dict[str, list[tuple[datetime, float, datetime]]] = defaultdict(list)
    by_station_vintages: dict[str, list[tuple[datetime, datetime, float, datetime, str]]] = (
        defaultdict(list)
    )
    labels: dict[tuple[str, datetime], tuple[float, datetime]] = {}
    for station, ts, value, available in observations:
        if value is None or not math.isfinite(value) or value < 0:
            continue
        by_station_obs[station].append((ts, float(value), available))
        labels[(station, ts)] = (float(value), available)
    for station, issue, valid, value, model_version, available in vintages:
        if value is not None and math.isfinite(value) and value >= 0:
            by_station_vintages[station].append(
                (issue, valid, float(value), available, model_version)
            )
    observation_times = {
        station: [row[0] for row in rows] for station, rows in by_station_obs.items()
    }
    first_available_by_station = {
        station: min(
            [row[2] for row in by_station_obs.get(station, [])]
            + [row[3] for row in by_station_vintages.get(station, [])],
            default=None,
        )
        for station in {station for station, _ in labels}
    }

    samples: dict[int, list[tuple[list[float], float, datetime, datetime]]] = {
        h: [] for h in HORIZONS_H
    }
    training_as_of = datetime.now(UTC)
    for horizon in HORIZONS_H:
        for (station, target_time), (target, target_available) in labels.items():
            if target_time.minute or target_time.second or target_time.microsecond:
                continue
            if target_time > training_as_of or target_available > training_as_of:
                continue
            issue = target_time - timedelta(hours=horizon)
            first_available = first_available_by_station[station]
            if first_available is None or first_available > issue:
                continue
            history = by_station_obs[station]
            history_times = observation_times[station]
            history_start = bisect_left(history_times, issue - timedelta(hours=24))
            history_end = bisect_right(history_times, issue)
            feature_observations = [
                (ts, value)
                for ts, value, available in history[history_start:history_end]
                if available <= issue
            ]
            feature_vintages = [
                (v_issue, valid, value, model)
                for v_issue, valid, value, available, model in by_station_vintages[station]
                if v_issue <= issue and available <= issue
            ]
            target_time = issue + timedelta(hours=horizon)
            feature_values = build_features(
                feature_observations, feature_vintages, issue, target_time
            )
            if set(feature_values) != set(FEATURE_NAMES):
                raise ValueError("shared feature output does not match FEATURE_NAMES")
            vector = [
                float("nan") if feature_values[name] is None else float(feature_values[name])
                for name in FEATURE_NAMES
            ]
            if all(math.isnan(x) for x in vector):
                continue
            samples[horizon].append((vector, target, issue, target_available))

    result = {}
    label_availabilities = {}
    counts = {horizon: len(rows) for horizon, rows in samples.items()}
    insufficient = {horizon: count for horizon, count in counts.items() if count < 20}
    if insufficient:
        availability_origin_count = sum(
            available <= issue for (_, issue), (_, available) in labels.items()
        )
        availability_vintage_count = sum(
            available <= issue
            for rows in by_station_vintages.values()
            for issue, _, _, available, _ in rows
        )
        per_horizon = ", ".join(f"+{horizon}h={counts[horizon]}" for horizon in HORIZONS_H)
        raise InsufficientTrainingDataError(
            "Insufficient leakage-safe training data: "
            f"usable examples by horizon ({per_horizon}); need 20 per horizon. "
            f"{availability_origin_count}/{len(labels)} usable observations have "
            f"available_at <= their observation time; {availability_vintage_count}/"
            f"{sum(map(len, by_station_vintages.values()))} usable forecast vintages have "
            "available_at <= their issue time. These are export-level counts; the strict "
            "feature policy requires available_at <= simulated issue time. "
            "Do not treat a historical timestamp or a later ingested_at as proof of past "
            "availability; preserve immutable timestamped captures and collect enough mature "
            "+24h outcomes before training."
        )

    for horizon, rows in samples.items():
        rows.sort(key=lambda row: row[2])
        result[horizon] = (
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
            [row[2] for row in rows],
        )
        label_availabilities[horizon] = [row[3] for row in rows]
    return result, label_availabilities


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    return {
        "mae": fmean(abs(float(error)) for error in errors),
        "rmse": math.sqrt(fmean(float(error * error) for error in errors)),
    }


def _validation_indices(
    origins: list[datetime],
    label_availabilities: list[datetime],
    horizon: int,
) -> tuple[list[int], list[int]]:
    """Split chronologically; embargo target times and label arrivals at the fold boundary."""
    if len(origins) != len(label_availabilities):
        raise ValueError("training origins and label availability must have the same length")
    split = max(1, min(len(origins) - 1, int(len(origins) * 0.8)))
    validation_start = origins[split]
    train_indices = [
        index
        for index, origin in enumerate(origins[:split])
        if origin + timedelta(hours=horizon) < validation_start
        and label_availabilities[index] < validation_start
    ]
    return train_indices, list(range(split, len(origins)))


def load_model_bundle(artifact: bytes) -> dict[int, lgb.Booster]:
    """Deserialize the locally trained bundle and validate its shared feature contract."""
    payload = json.loads(artifact)
    if payload.get("format") != 1 or set(payload.get("horizons", {})) != {
        str(horizon) for horizon in HORIZONS_H
    }:
        raise ValueError("unsupported or incomplete model bundle")
    boosters = {
        int(horizon): lgb.Booster(model_str=model_text)
        for horizon, model_text in payload["horizons"].items()
    }
    if any(booster.feature_name() != list(FEATURE_NAMES) for booster in boosters.values()):
        raise ValueError("model feature names do not match the shared feature contract")
    return boosters


def train(export_path: Path, model_version: str) -> tuple[bytes, dict[str, Any]]:
    data, label_availabilities = _make_training_rows_with_availability(export_path)
    boosters: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    first_train: datetime | None = None
    last_train: datetime | None = None
    for horizon, (features, target, origins) in data.items():
        train_indices, valid_indices = _validation_indices(
            origins, label_availabilities[horizon], horizon
        )
        if len(train_indices) < 20:
            raise ValueError(
                f"horizon +{horizon}h has fewer than 20 training examples after the "
                "validation embargo"
            )
        train_x, valid_x = features[train_indices], features[valid_indices]
        train_y, valid_y = target[train_indices], target[valid_indices]
        params = {"objective": "regression", "metric": "mae", "verbosity": -1, "seed": 2026}
        booster = lgb.train(
            params,
            lgb.Dataset(train_x, label=train_y, feature_name=list(FEATURE_NAMES)),
            num_boost_round=100,
        )
        predicted = np.maximum(0.0, booster.predict(valid_x))
        boosters[str(horizon)] = booster.model_to_string()
        metrics[str(horizon)] = {
            **_metrics(valid_y, predicted),
            "validation_examples": len(valid_indices),
            "training_examples": len(train_y),
        }
        start_train = origins[train_indices[0]]
        end_train = origins[train_indices[-1]]
        first_train = min(first_train, start_train) if first_train else start_train
        last_train = max(last_train, end_train) if last_train else end_train

    artifact = json.dumps({"format": 1, "horizons": boosters}, separators=(",", ":")).encode()
    load_model_bundle(artifact)
    data_hash = hashlib.sha256(export_path.read_bytes()).hexdigest()
    manifest = {
        "version": model_version,
        "model_type": "lightgbm-direct-by-horizon",
        "metrics": metrics,
        "training_window": {"start": _iso(first_train), "end": _iso(last_train)},
        "feature_list": list(FEATURE_NAMES),
        "data_sha256": data_hash,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "trained_at": datetime.now(UTC).isoformat(),
    }
    return artifact, manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=Path("data/history.duckdb"))
    parser.add_argument("--version", default=datetime.now(UTC).strftime("lgbm-%Y%m%dT%H%M%SZ"))
    parser.add_argument(
        "--push",
        action="store_true",
        help="write and read back the artifact in the hosted registry",
    )
    parser.add_argument("--artifact", type=Path, default=Path("data/model.json"))
    try:
        args = parser.parse_args(argv)
        artifact, manifest = train(args.export, args.version)
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_bytes(artifact)
        manifest_path = args.artifact.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.push:
            settings = get_settings()
            if settings.database_url is None:
                raise ConfigError("DATABASE_URL is not set")
            database_url = settings.database_url.get_secret_value()
            publish_model(database_url, args.version, artifact, manifest)
            with open_db(db.connect, database_url) as con:
                stored_artifact, stored_manifest = read_model(con, args.version)
            load_model_bundle(stored_artifact)
            if stored_manifest != manifest:
                raise ValueError("stored model manifest does not match the published manifest")
            logger.info("model %s stored and read back successfully", args.version)
        return 0
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except InsufficientTrainingDataError as exc:
        logger.error("training data rejected: %s", exc)
        return 1
    except Exception as exc:
        # DB errors can echo DATABASE_URL; only the exception type is safe to log.
        logger.error("model training or registry operation failed (%s)", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
