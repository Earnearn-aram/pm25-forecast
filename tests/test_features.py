from datetime import UTC, datetime, timedelta

from pm25_forecast.backtest import Observation, Snapshot, Vintage
from pm25_forecast.features import FEATURE_NAMES, build_features

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)
TARGET = NOW + timedelta(hours=6)


def test_training_and_live_paths_produce_identical_ordered_features():
    observations = (
        Observation("59t", NOW, 21.5, NOW),
        Observation("59t", TARGET - timedelta(hours=24), 19.0, TARGET - timedelta(hours=24)),
        Observation("59t", NOW + timedelta(hours=1), 999.0, NOW + timedelta(hours=1)),
    )
    vintages = (
        Vintage("59t", NOW - timedelta(hours=2), TARGET, 27.0, NOW - timedelta(hours=2)),
        Vintage("59t", NOW + timedelta(minutes=1), TARGET, 888.0, NOW + timedelta(minutes=1)),
    )
    training = Snapshot(NOW, "59t", observations[:2], vintages[:1]).features(TARGET)

    # Live rows use the same normalized raw inputs loaded from the database.
    live = build_features(
        [(row.ts, row.value) for row in observations],
        [(row.issue_time, row.valid_time, row.value, row.model) for row in vintages],
        NOW,
        TARGET,
    )

    assert tuple(training) == FEATURE_NAMES
    assert tuple(live) == FEATURE_NAMES
    assert (
        training
        == live
        == {
            "pm25_latest": 21.5,
            "pm25_same_hour_24h": 19.0,
            "cams_pm25": 27.0,
        }
    )


def test_feature_builder_ignores_future_or_not_yet_available_inputs():
    features = build_features(
        [
            {"ts": NOW - timedelta(hours=1), "pm25": 12.0, "available_at": NOW},
            {"ts": NOW, "pm25": 99.0, "available_at": NOW + timedelta(seconds=1)},
            {"ts": NOW + timedelta(hours=1), "pm25": 88.0, "available_at": NOW},
        ],
        [
            {
                "issue_time": NOW - timedelta(hours=1),
                "valid_time": TARGET,
                "pm25_pred": 24.0,
                "model_version": "cams_global",
                "available_at": NOW,
            },
            {
                "issue_time": NOW + timedelta(seconds=1),
                "valid_time": TARGET,
                "pm25_pred": 999.0,
                "model_version": "cams_global",
                "available_at": NOW,
            },
            {
                "issue_time": NOW - timedelta(minutes=1),
                "valid_time": TARGET,
                "pm25_pred": 777.0,
                "model_version": "cams_global",
                "available_at": NOW + timedelta(seconds=1),
            },
        ],
        NOW,
        TARGET,
    )

    assert features == {
        "pm25_latest": 12.0,
        "pm25_same_hour_24h": None,
        "cams_pm25": 24.0,
    }
