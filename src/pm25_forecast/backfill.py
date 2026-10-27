"""Air4Thai history backfill (KAN-12 part 1, KAN-34). Run with ``python -m pm25_forecast.backfill``.

Air4Thai's history endpoint keeps only a rolling window (KAN-9: every station starts
2026-08-01 00:00 Bangkok time), so this backfills whatever the window still holds. A 2-year
history needs another source (KAN-12 part 2: OpenAQ/PCD).

Per station: fetch ``now - lookback .. now`` in chunks of at most 31 days, drop rows with no
PM2.5 or dated in the future, and upsert into ``observations`` in that station's own
transaction (natural key ``(station_id, ts)``, same statement as the collector). Re-runs and
overlaps with the collector therefore never duplicate rows, so this is safe to run daily and
doubles as gap recovery for the collector, which only ever sees the latest reading.

Exit codes: 0 if at least one station delivered usable rows (failures of other stations are
logged and alerted), 1 on total failure (no usable rows at all) or when the DB is down,
2 on configuration errors. Content from the API is data; nothing here builds SQL from it.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from pm25_forecast.collector import (
    MAX_FUTURE_SKEW,
    ConfigError,
    _safe_alert,
    as_utc,
    make_alert,
    open_db,
)
from pm25_forecast.ingest.air4thai import HISTORY_PAUSE_S, Air4ThaiClient, Observation
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 120  # longer than the ~3-month window, so nothing it holds is missed
EXIT_FAILED, EXIT_CONFIG = 1, 2


@dataclass
class StationReport:
    station_id: str
    rows_fetched: int = 0
    rows_dropped: int = 0
    rows_upserted: int = 0
    rows_new: int = 0
    earliest: datetime | None = None
    latest: datetime | None = None
    chunks: int = 0
    empty_chunks: int = 0
    failed_chunks: int = 0
    error: str | None = None

    def log_line(self) -> str:
        span = (
            f"{self.earliest.isoformat()} .. {self.latest.isoformat()}"
            if self.earliest and self.latest
            else "no rows"
        )
        return (
            f"station {self.station_id}: {self.rows_fetched} fetched, {self.rows_dropped} dropped, "
            f"{self.rows_upserted} upserted ({self.rows_new} new), {span}, "
            f"{self.chunks} chunks ({self.empty_chunks} empty, {self.failed_chunks} failed)"
            + (f", error: {self.error}" if self.error else "")
        )


@dataclass
class BackfillResult:
    stations: list[StationReport] = field(default_factory=list)

    @property
    def total_failure(self) -> bool:
        """No station delivered a single usable row (or there were no stations)."""
        return not any(r.rows_fetched - r.rows_dropped > 0 for r in self.stations)

    @property
    def failures(self) -> list[str]:
        return [
            r.log_line()
            for r in self.stations
            if r.error or r.failed_chunks or r.rows_fetched == r.rows_dropped
        ]


def usable(observations: Sequence[Observation], now: datetime) -> list[Observation]:
    """Rows worth storing: a PM2.5 value and a timestamp not in the future."""
    return [o for o in observations if o.pm25 is not None and o.ts <= now + MAX_FUTURE_SKEW]


def write_station(con: Any, station_id: str, rows: Sequence[Observation]) -> tuple[int, int]:
    """Idempotent upsert of one station's rows; returns (rows affected, rows that were new)."""
    if not rows:
        return 0, 0
    span = {"station_id": station_id, "start": rows[0].ts, "end": rows[-1].ts}
    (before,) = con.execute(sql.COUNT_OBSERVATIONS_IN_RANGE, span).fetchone()
    upserted = sql.write_rows(con, sql.UPSERT_OBSERVATION, [o.model_dump() for o in rows])
    (after,) = con.execute(sql.COUNT_OBSERVATIONS_IN_RANGE, span).fetchone()
    return upserted, int(after) - int(before)


def resolve_station_ids(settings: Settings, override: Sequence[str] = ()) -> list[str]:
    """Explicit ids win; otherwise the collector's ``STATION_LOCATIONS``."""
    ids = [s.strip() for s in override if s.strip()]
    if not ids:
        try:
            ids = [sid for sid, _lat, _lon in settings.station_location_list]
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
    if not ids:
        raise ConfigError("no stations: set STATION_LOCATIONS or pass --stations")
    return list(dict.fromkeys(ids))


def run(
    settings: Settings,
    *,
    station_ids: Sequence[str] = (),
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    http: httpx.Client | None = None,
    pause: float = HISTORY_PAUSE_S,
    sleep: Callable[[float], None] = time.sleep,
    client_kwargs: dict[str, Any] | None = None,
) -> BackfillResult:
    """One backfill run. Driver connection errors propagate for `run_entrypoint` to map."""
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    if lookback_days < 1:
        raise ConfigError("lookback must be at least 1 day")
    ids = resolve_station_ids(settings, station_ids)
    url = settings.database_url.get_secret_value()
    now = as_utc(clock())
    start = now - timedelta(days=lookback_days)
    result = BackfillResult()
    logger.info("backfill %d stations from %s to %s", len(ids), start.isoformat(), now.isoformat())

    client = Air4ThaiClient(
        client=http,
        ca_bundle=None if http is not None else settings.air4thai_ca_bundle,
        **(client_kwargs or {}),
    )
    try:
        for index, station_id in enumerate(ids):
            if index and pause > 0:
                sleep(pause)
            report = StationReport(station_id)
            result.stations.append(report)
            try:
                history = client.fetch_history(station_id, start, now, pause=pause, sleep=sleep)
            except Exception as exc:  # one station must never stop the others
                report.error = f"fetch failed: {type(exc).__name__}"
                logger.error("station %s: %s", station_id, report.error)
                continue
            report.chunks = len(history.chunks)
            report.empty_chunks = len(history.empty_chunks)
            report.failed_chunks = len(history.failed_chunks)
            report.rows_fetched = len(history.observations)
            rows = usable(history.observations, now)
            report.rows_dropped = report.rows_fetched - len(rows)
            if rows:
                report.earliest, report.latest = rows[0].ts, rows[-1].ts
                with open_db(connect, url) as con:  # one transaction per station
                    report.rows_upserted, report.rows_new = write_station(con, station_id, rows)
            if history.total_failure:
                report.error = "every chunk failed"
            log = logger.warning if report.error or report.failed_chunks else logger.info
            log("%s", report.log_line())
    finally:
        client.close()

    logger.info(
        "backfill done: %d stations, %d rows upserted (%d new), %d stations with problems",
        len(result.stations),
        sum(r.rows_upserted for r in result.stations),
        sum(r.rows_new for r in result.stations),
        len(result.failures),
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pm25_forecast.backfill",
        description="Backfill Air4Thai hourly PM2.5 history (whatever its window holds).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"days back from now to request (default {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--stations",
        default="",
        help="comma-separated station ids (default: the ids in STATION_LOCATIONS)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines log full request URLs
    args = parse_args(argv)
    alert = make_alert(None)
    try:
        settings = get_settings()
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook)
        result, code = run_entrypoint(
            lambda: run(
                settings,
                station_ids=args.stations.split(","),
                lookback_days=args.lookback_days,
            ),
            alert,
            connection_errors=db.CONNECTION_ERRORS,
        )
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        _safe_alert(alert, f"backfill configuration error: {exc}")
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.exception("backfill crashed")
        _safe_alert(alert, f"backfill crashed: {type(exc).__name__}")
        return EXIT_FAILED
    if isinstance(result, RunSkipped):
        return code
    if result.total_failure:
        _safe_alert(alert, "backfill failed: no usable rows from any station")
        return EXIT_FAILED
    if result.failures:
        _safe_alert(alert, "backfill completed with problems: " + "; ".join(result.failures))
    return code


if __name__ == "__main__":
    sys.exit(main())
