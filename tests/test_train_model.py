import json
from datetime import UTC, datetime, timedelta

import duckdb

from pm25_forecast.features import FEATURE_NAMES
from pm25_forecast.train_model import train


def test_train_reads_only_local_export_and_writes_feature_manifest(tmp_path):
    export_path = tmp_path / "history.duckdb"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with duckdb.connect(str(export_path)) as con:
        con.execute(
            "CREATE TABLE observations "
            "(station_id VARCHAR, ts TIMESTAMPTZ, pm25 DOUBLE, available_at TIMESTAMPTZ)"
        )
        con.execute(
            "CREATE TABLE forecast_vintages "
            "(station_id VARCHAR, issue_time TIMESTAMPTZ, valid_time TIMESTAMPTZ, "
            "pm25_pred DOUBLE, model_version VARCHAR, available_at TIMESTAMPTZ)"
        )
        observations = [
            ("59t", start + timedelta(hours=i), 10.0 + i % 7, start + timedelta(hours=i))
            for i in range(80)
        ]
        con.executemany("INSERT INTO observations VALUES (?, ?, ?, ?)", observations)

    artifact, manifest = train(export_path, "test-local-v1")
    bundle = json.loads(artifact)

    assert bundle["format"] == 1
    assert set(bundle["horizons"]) == {"6", "12", "24"}
    assert manifest["feature_list"] == list(FEATURE_NAMES)
    assert manifest["metrics"]["24"]["validation_examples"] > 0
    assert manifest["training_window"]["start"].endswith("+00:00")
