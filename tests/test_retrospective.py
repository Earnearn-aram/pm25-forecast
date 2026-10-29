import json
from datetime import UTC, datetime, timedelta

import duckdb
import lightgbm as lgb
import pytest

from pm25_forecast.models.serving import ModelLoadError, load_serving_model
from pm25_forecast.retrospective import FEATURE_NAMES, evaluate
from pm25_forecast.storage.registry import read_model, write_model


class LocalRegistry:
    """Small in-memory stand-in for verifying a local artifact/manifest round trip."""

    def __init__(self):
        self.row = None

    def execute(self, statement, params):
        if statement.lstrip().startswith("INSERT"):
            self.row = (params["sha256"], params["artifact"], params["manifest"])
        else:
            self.selected = self.row
        return self

    def fetchone(self):
        return self.selected


def _snapshot(tmp_path):
    database = tmp_path / "history.duckdb"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with duckdb.connect(str(database)) as con:
        con.execute(
            "CREATE TABLE observations "
            "(station_id VARCHAR, ts TIMESTAMPTZ, pm25 DOUBLE, available_at TIMESTAMPTZ)"
        )
        values = [
            (station, start + timedelta(hours=i), 8.0 + (i % 18) + station_index)
            for station_index, station in enumerate(("a", "b"))
            for i in range(300)
        ]
        con.executemany(
            "INSERT INTO observations VALUES (?, ?, ?, ?)",
            [(station, ts, value, ts + timedelta(days=200)) for station, ts, value in values],
        )
    return database


def test_retrospective_model_has_embargo_common_pairs_and_local_registry_readback(tmp_path):
    database = _snapshot(tmp_path)
    artifact_path = tmp_path / "kan15-model.json"
    manifest_path = tmp_path / "kan15-model.manifest.json"
    report = evaluate(database, artifact_path, manifest_path)

    assert report["design"]["cams_features_used"] is False
    assert (
        "does not establish historical real availability"
        in report["design"]["retrospective_availability_notice"]
    )
    assert report["design"]["training_rows"] > 20
    assert report["design"]["test_pairs_are_common_to"] == [
        "LightGBM",
        "persistence_latest_3h",
        "seasonal_target_minus_24h",
    ]
    assert report["coverage"]["test_pairs"] == 2 * 42
    assert report["test_metrics"]["lightgbm_direct_24h"]["n"] == report["coverage"]["test_pairs"]

    model = json.loads(artifact_path.read_bytes())
    booster = lgb.Booster(model_str=model["booster"])
    assert model["horizon_hours"] == 24
    assert booster.feature_name() == list(FEATURE_NAMES)
    manifest = json.loads(manifest_path.read_text())
    registry = LocalRegistry()
    write_model(registry, manifest["version"], artifact_path.read_bytes(), manifest)
    stored_artifact, stored_manifest = read_model(registry, manifest["version"])
    assert stored_artifact == artifact_path.read_bytes()
    assert stored_manifest == manifest


def test_one_hour_lag_compares_with_persistence_only_and_is_servable(tmp_path):
    database = _snapshot(tmp_path)
    artifact_path = tmp_path / "lag1.json"
    manifest_path = tmp_path / "lag1.manifest.json"
    report = evaluate(database, artifact_path, manifest_path, lag_hours=1)

    assert report["design"]["assumed_reporting_lag_hours"] == 1
    assert report["design"]["test_pairs_are_common_to"] == ["LightGBM", "persistence_latest_3h"]
    assert set(report["test_metrics"]) == {"lightgbm_direct_24h", "persistence_latest_3h"}
    assert "never available" in report["seasonal_naive_not_applicable"]
    assert report["coverage"]["test_pairs"] > 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == "kan15-retrospective-24h-lag1h"
    assert manifest["assumed_reporting_lag_hours"] == 1
    model = load_serving_model(manifest["version"], artifact_path.read_bytes(), manifest)
    assert model.horizons == (24,)


def test_zero_lag_model_is_still_refused_by_the_serving_loader(tmp_path):
    database = _snapshot(tmp_path)
    artifact_path = tmp_path / "lag0.json"
    manifest_path = tmp_path / "lag0.manifest.json"
    report = evaluate(database, artifact_path, manifest_path)
    assert "seasonal_naive_not_applicable" not in report
    manifest = json.loads(manifest_path.read_text())
    with pytest.raises(ModelLoadError, match="assumed_reporting_lag_hours=0"):
        load_serving_model(manifest["version"], artifact_path.read_bytes(), manifest)
