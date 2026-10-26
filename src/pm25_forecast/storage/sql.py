"""Upsert statements (named parameters; valid on PostgreSQL and SQLite >= 3.24)."""

import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def require_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalise to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


UPSERT_OBSERVATION = """
INSERT INTO observations (station_id, ts, pm25, source)
VALUES (:station_id, :ts, :pm25, :source)
ON CONFLICT (station_id, ts) DO UPDATE SET
    pm25 = COALESCE(excluded.pm25, observations.pm25),
    source = excluded.source,
    ingested_at = CURRENT_TIMESTAMP
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


def write_rows(
    con: Any,
    statement: str,
    rows: Iterable[Mapping[str, Any]],
    adapt: Callable[[datetime], Any] = lambda value: value,
) -> int:
    """Execute `statement` once per row after rejecting naive datetimes; return rows affected.

    `con` is any DB-API-style connection whose `execute(sql, params)` returns a cursor with
    `rowcount`. Datetimes are normalised to UTC via `require_utc` (naive ones raise ValueError
    before anything is executed), then passed through `adapt` (e.g. `isoformat` for SQLite).
    For append-only DO NOTHING statements the shortfall is the number of ignored duplicates.
    """
    prepared = [
        {k: adapt(require_utc(v)) if isinstance(v, datetime) else v for k, v in row.items()}
        for row in rows
    ]
    affected = 0
    for params in prepared:
        affected += max(con.execute(statement, params).rowcount, 0)
    logger.info(
        "write_rows: %d inserted/updated, %d ignored (of %d)",
        affected,
        len(prepared) - affected,
        len(prepared),
    )
    return affected
