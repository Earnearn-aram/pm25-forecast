"""Statements using psycopg 3 style `%(name)s` placeholders."""

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


# Overlap precedence (KAN-49): usable Air4Thai > OpenAQ, regardless of arrival order.
# OpenAQ fills gaps, never displaces another source. A null Air4Thai value cannot steal
# OpenAQ provenance; a later usable Air4Thai reading replaces it. Same-source updates
# remain supported. The WHERE is evaluated atomically by the database on conflict.
UPSERT_OBSERVATION = """
INSERT INTO observations (station_id, ts, pm25, source)
VALUES (%(station_id)s, %(ts)s, %(pm25)s, %(source)s)
ON CONFLICT (station_id, ts) DO UPDATE SET
    pm25 = COALESCE(excluded.pm25, observations.pm25),
    source = excluded.source,
    ingested_at = CURRENT_TIMESTAMP
WHERE observations.source = excluded.source
   OR (excluded.source = 'air4thai' AND excluded.pm25 IS NOT NULL)
   OR (excluded.source = 'openaq' AND observations.pm25 IS NULL
       AND excluded.pm25 IS NOT NULL)
   OR (excluded.source NOT IN ('air4thai', 'openaq')
       AND observations.source NOT IN ('air4thai', 'openaq'))
"""

# Append-only forecast vintage: re-running the same issue never overwrites the first write.
INSERT_PREDICTION = """
INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id, pm25_pred)
VALUES (
    %(station_id)s, %(issue_time)s, %(valid_time)s, %(model_version)s, %(run_id)s, %(pm25_pred)s
)
ON CONFLICT (station_id, issue_time, valid_time, model_version) DO NOTHING
"""

UPSERT_SCORE = """
INSERT INTO scores (station_id, issue_time, valid_time, model_version, pm25_obs, abs_error)
VALUES (
    %(station_id)s, %(issue_time)s, %(valid_time)s, %(model_version)s, %(pm25_obs)s, %(abs_error)s
)
ON CONFLICT (station_id, issue_time, valid_time, model_version)
DO UPDATE SET pm25_obs = excluded.pm25_obs, abs_error = excluded.abs_error
"""

# Reads used by the collector (KAN-18): catch-up point per station and the latest forecast vintage.
SELECT_LAST_OBSERVATION_TS = """
SELECT station_id, max(ts) FROM observations WHERE source = %(source)s GROUP BY station_id
"""

# Heartbeat (KAN-19): observations whose data timestamp is at or after `since`.
SELECT_OBSERVATION_COUNT_SINCE = """
SELECT count(*) FROM observations WHERE source = %(source)s AND ts >= %(since)s
"""

# Backfill (KAN-12): rows already stored for a station in a range, to report how many were new.
COUNT_OBSERVATIONS_IN_RANGE = """
SELECT count(*) FROM observations
WHERE station_id = %(station_id)s AND ts >= %(start)s AND ts <= %(end)s
"""

SELECT_LATEST_VINTAGE = """
SELECT issue_time, valid_time, pm25_pred FROM predictions
WHERE station_id = %(station_id)s AND model_version = %(model_version)s
  AND issue_time = (
      SELECT max(issue_time) FROM predictions
      WHERE station_id = %(station_id)s AND model_version = %(model_version)s
  )
"""

# Predict step (KAN-28): usable observations available by issue_time, newest first.
SELECT_OBSERVATIONS_BETWEEN = """
SELECT ts, pm25, ingested_at FROM observations
WHERE station_id = %(station_id)s AND ts >= %(since)s AND ts <= %(until)s
  AND ingested_at <= %(as_of)s AND pm25 IS NOT NULL
ORDER BY ts DESC
"""

# Predict step (KAN-28): stored forecast rows (the collector's CAMS vintages) for the target
# valid_times, from vintages issued at or before `as_of`.
SELECT_FORECAST_ROWS = """
SELECT issue_time, valid_time, pm25_pred, created_at FROM predictions
WHERE station_id = %(station_id)s AND model_version = %(model_version)s
  AND issue_time <= %(as_of)s AND valid_time >= %(valid_from)s AND valid_time <= %(valid_to)s
  AND created_at <= %(as_of)s
"""


# Scoring (KAN-20): one model's forecasts whose valid_time is in [since, now], with the actual
# for that station and hour (NULL = not arrived yet) and the stored score (NULL = not scored).
# The stored pm25_obs lets the caller write only new or changed scores.
SELECT_MATURED_FORECASTS = """
SELECT p.station_id, p.issue_time, p.valid_time, p.pm25_pred, o.pm25, s.pm25_obs
FROM predictions p
LEFT JOIN observations o
  ON o.station_id = p.station_id AND o.ts = p.valid_time AND o.pm25 IS NOT NULL
LEFT JOIN scores s
  ON s.station_id = p.station_id AND s.issue_time = p.issue_time
 AND s.valid_time = p.valid_time AND s.model_version = p.model_version
WHERE p.model_version = %(model_version)s
  AND p.valid_time >= %(since)s AND p.valid_time <= %(now)s
"""

# Scoring (KAN-20): one model's scores in a rolling window of valid_time, for metrics. Only
# scores whose actual is still usable count: an observation later corrected to NULL or negative
# drops its score from the metrics (the row is kept, never deleted).
SELECT_SCORES_BETWEEN = """
SELECT s.station_id, s.issue_time, s.valid_time, s.abs_error FROM scores s
JOIN observations o
  ON o.station_id = s.station_id AND o.ts = s.valid_time
 AND o.pm25 IS NOT NULL AND o.pm25 >= 0
WHERE s.model_version = %(model_version)s
  AND s.valid_time >= %(since)s AND s.valid_time <= %(now)s
"""


def write_rows(
    con: Any,
    statement: str,
    rows: Iterable[Mapping[str, Any]],
    adapt: Callable[[datetime], Any] = lambda value: value,
) -> int:
    """Execute `statement` once per row after rejecting naive datetimes; return rows affected.

    `con` is any DB-API-style connection whose `cursor()` is a context manager with
    `executemany(sql, rows)` and a `rowcount` that totals the rows affected (psycopg 3 does).
    Datetimes are normalised to UTC via `require_utc` (naive ones raise ValueError before
    anything is executed), then passed through `adapt` (e.g. `isoformat` for SQLite).
    For append-only DO NOTHING statements the shortfall is the number of ignored duplicates.
    """
    prepared = [
        {k: adapt(require_utc(v)) if isinstance(v, datetime) else v for k, v in row.items()}
        for row in rows
    ]
    affected = 0
    if prepared:
        with con.cursor() as cur:
            cur.executemany(statement, prepared)
            affected = max(cur.rowcount, 0)
    logger.info(
        "write_rows: %d inserted/updated, %d ignored (of %d)",
        affected,
        len(prepared) - affected,
        len(prepared),
    )
    return affected


# Retention (Decided): raw prediction vintages are kept 90 days. Run by a job added later
# (no scheduler in KAN-17). Scores are aggregated to daily before their raw rows are purged.
PURGE_OLD_PREDICTIONS = """
DELETE FROM predictions WHERE issue_time < now() - interval '90 days'
"""
