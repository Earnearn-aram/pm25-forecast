"""Fit local direct LightGBM models from a read-only DuckDB history export."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("export timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _rows(con: Any, query: str) -> list[tuple]:
    return con.execute(query).fetchall()


def make_training_rows(
    export_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray, list[datetime]]]:
    """Build leakage-safe samples; all feature/label availability is bounded by issue time."""
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

    samples: dict[int, list[tuple[list[float], float, datetime]]] = {h: [] for h in HORIZONS_H}
    for (station, issue), (_current, issue_available) in labels.items():
        # The origin itself must have been present by prediction time.
        if issue_available > issue:
            continue
        history = by_station_obs[station]
        feature_observations = [
            (ts, value) for ts, value, available in history if ts <= issue and available <= issue
        ]
        feature_vintages = [
            (v_issue, valid, value, model)
            for v_issue, valid, value, available, model in by_station_vintages[station]
            if v_issue <= issue and available <= issue
        ]
        for horizon in HORIZONS_H:
            target_time = issue + timedelta(hours=horizon)
            label = labels.get((station, target_time))
            if label is None or label[1] > datetime.now(UTC):
                continue
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
            samples[horizon].append((vector, label[0], issue))

    result = {}
    for horizon, rows in samples.items():
        rows.sort(key=lambda row: row[2])
        if len(rows) < 20:
            raise ValueError(f"horizon +{horizon}h has only {len(rows)} usable examples; need 20")
        result[horizon] = (
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
            [row[2] for row in rows],
        )
    return result


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    return {
        "mae": fmean(abs(float(error)) for error in errors),
        "rmse": math.sqrt(fmean(float(error * error) for error in errors)),
    }


def load_model_bundle(artifact: bytes) -> dict[int, lgb.Booster]:
    """Deserialize the model format consumed by the hourly inference code."""
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
    data = make_training_rows(export_path)
    boosters: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    first_train: datetime | None = None
    last_train: datetime | None = None
    for horizon, (features, target, origins) in data.items():
        split = max(1, min(len(origins) - 1, int(len(origins) * 0.8)))
        validation_start = origins[split]
        train_indices = [
            index
            for index, origin in enumerate(origins[:split])
            if origin + timedelta(hours=horizon) < validation_start
        ]
        valid_indices = list(range(split, len(origins)))
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
    except Exception as exc:
        # DB errors can echo DATABASE_URL; only the exception type is safe to log.
        logger.error("model training or registry operation failed (%s)", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
