"""Read-only export of hosted observations and CAMS forecast vintages to DuckDB."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb

from pm25_forecast.collector import ConfigError
from pm25_forecast.settings import get_settings
from pm25_forecast.storage import db

logger = logging.getLogger(__name__)

OBSERVATIONS_QUERY = """
SELECT station_id, ts, pm25, source, ingested_at AS available_at
FROM observations
ORDER BY station_id, ts
"""
VINTAGES_QUERY = """
SELECT station_id, issue_time, valid_time, model_version, run_id, pm25_pred,
       created_at AS available_at
FROM predictions
WHERE model_version = 'cams_global'
ORDER BY station_id, issue_time, valid_time
"""


def export_history(
    destination: Path,
    database_url: str,
    *,
    connect=db.connect,
) -> dict[str, int]:
    """Export natural-key rows without writing to the source database."""
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with connect(database_url) as source:
        # PostgreSQL rejects every data-modifying statement within this transaction.
        source.execute("SET TRANSACTION READ ONLY")
        with duckdb.connect(str(destination)) as target:
            target.execute("SET TimeZone = 'UTC'")
            target.execute("BEGIN TRANSACTION")
            try:
                for table, query, schema in (
                    (
                        "observations",
                        OBSERVATIONS_QUERY,
                        "station_id VARCHAR, ts TIMESTAMPTZ, pm25 DOUBLE, source VARCHAR, "
                        "available_at TIMESTAMPTZ",
                    ),
                    (
                        "forecast_vintages",
                        VINTAGES_QUERY,
                        "station_id VARCHAR, issue_time TIMESTAMPTZ, valid_time TIMESTAMPTZ, "
                        "model_version VARCHAR, run_id VARCHAR, pm25_pred DOUBLE, "
                        "available_at TIMESTAMPTZ",
                    ),
                ):
                    cursor = source.execute(query)
                    rows = cursor.fetchall()
                    target.execute(f"CREATE OR REPLACE TABLE {table} ({schema})")
                    if rows:
                        placeholders = ", ".join("?" for _ in rows[0])
                        target.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                    counts[table] = len(rows)
                target.execute("COMMIT")
            except Exception:
                target.execute("ROLLBACK")
                raise
    logger.info("export complete: observations=%d forecast_vintages=%d", *counts.values())
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/history.duckdb"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        args = _parse_args(argv)
        settings = get_settings()
        if settings.database_url is None:
            raise ConfigError("DATABASE_URL is not set")
        export_history(args.output, settings.database_url.get_secret_value())
        return 0
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except Exception as exc:
        # Driver exceptions can include connection details; emit only the exception type.
        logger.error("history export failed (%s)", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
