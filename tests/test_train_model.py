import json
from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from pm25_forecast.features import FEATURE_NAMES
from pm25_forecast.train_model import (
    InsufficientTrainingDataError,
    _validation_indices,
    make_training_rows,
    train,
)


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


def test_delayed_issue_observation_is_excluded_but_earlier_available_rows_are_used(tmp_path):
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
            (
                "59t",
                start + timedelta(hours=i),
                10.0 + i % 7,
                start + timedelta(hours=i, minutes=1),
            )
            for i in range(80)
        ]
        con.executemany("INSERT INTO observations VALUES (?, ?, ?, ?)", observations)

    data = make_training_rows(export_path)

    for features, _targets, origins in data.values():
        assert origins[0] == start + timedelta(hours=1)
        assert features[0, 0] == 10.0


def test_training_fails_closed_with_all_horizon_availability_counts(tmp_path):
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
            (
                "59t",
                start + timedelta(hours=i),
                10.0 + i % 7,
                start + timedelta(hours=i, minutes=1),
            )
            for i in range(30)
        ]
        observations.append(
            (
                "59t",
                start + timedelta(hours=30, minutes=30),
                15.0,
                start + timedelta(hours=30, minutes=31),
            )
        )
        con.executemany("INSERT INTO observations VALUES (?, ?, ?, ?)", observations)

    with pytest.raises(InsufficientTrainingDataError) as exc_info:
        make_training_rows(export_path)
    message = str(exc_info.value)

    assert "usable examples by horizon (+6h=23, +12h=17, +24h=5)" in message
    assert "0/31 usable observations have available_at <= their observation time" in message
    assert "0/0 usable forecast vintages have available_at <= their issue time" in message
    assert "strict feature policy requires available_at <= simulated issue time" in message


def test_validation_embargo_excludes_labels_that_arrived_after_fold_start():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    origins = [start + timedelta(hours=i) for i in range(10)]
    label_availabilities = [origin + timedelta(hours=1) for origin in origins]
    label_availabilities[2] = origins[8] + timedelta(minutes=1)

    training, validation = _validation_indices(origins, label_availabilities, horizon=1)

    assert training == [0, 1, 3, 4, 5, 6]
    assert validation == [8, 9]
