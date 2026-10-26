"""Upsert statements (named parameters; valid on PostgreSQL and SQLite >= 3.24)."""

from datetime import UTC, datetime


def require_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalise to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


UPSERT_OBSERVATION = """
INSERT INTO observations (station_id, ts, pm25, source)
VALUES (:station_id, :ts, :pm25, :source)
ON CONFLICT (station_id, ts) DO UPDATE SET pm25 = excluded.pm25, source = excluded.source
"""

# Append-only forecast vintage: re-running the same issue never overwrites the first write.
INSERT_PREDICTION = """
INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id, pm25_pred)
VALUES (:station_id, :issue_time, :valid_time, :model_version, :run_id, :pm25_pred)
ON CONFLICT (station_id, issue_time, valid_time, model_version) DO NOTHING
"""

UPSERT_SCORE = """
INSERT INTO scores (station_id, issue_time, valid_time, model_version, pm25_obs, abs_error)
VALUES (:station_id, :issue_time, :valid_time, :model_version, :pm25_obs, :abs_error)
ON CONFLICT (station_id, issue_time, valid_time, model_version)
DO UPDATE SET pm25_obs = excluded.pm25_obs, abs_error = excluded.abs_error
"""
