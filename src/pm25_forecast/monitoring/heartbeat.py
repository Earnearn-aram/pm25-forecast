"""Daily heartbeat (KAN-19). Run with ``python -m pm25_forecast.monitoring.heartbeat``.

Posts one line to ALERT_WEBHOOK_URL: the newest observation per station and how many
observation rows have a timestamp in the last 24 h, read through the storage layer. Silence on a
day means the heartbeat itself did not run; an external dead-man switch for that is KAN-30.

* DB unreachable: alert instead of the summary, exit 1 (the storage guard's skip policy).
* No observations in the last 24 h: the line is sent as an alert, exit 1.
* Stations older than the collector's stale threshold are flagged in the line (exit 0: the
  collector already alerts on them).
* Missing DATABASE_URL: config alert, exit 2. Missing ALERT_WEBHOOK_URL: warning, line logged only.
"""

import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pm25_forecast.collector import (
    AIR4THAI_SOURCE,
    STALE_AFTER,
    ConfigError,
    as_utc,
    open_db,
)
from pm25_forecast.monitoring.alerts import (
    make_alert,
    post_message,
    safe_alert,
    warn_missing_webhook,
)
from pm25_forecast.monitoring.logs import configure_logging, log_summary
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

logger = logging.getLogger(__name__)

PREFIX = "[pm25 heartbeat]"
WINDOW = timedelta(hours=24)
EXIT_OK, EXIT_UNHEALTHY, EXIT_CONFIG = 0, 1, 2


@dataclass(frozen=True)
class Status:
    checked_at: datetime
    last_ts: Mapping[str, datetime]  # station id -> newest observation ts (UTC)
    rows_24h: int
    expected: Sequence[str] = ()  # configured station ids (STATION_LOCATIONS), may be empty

    @property
    def stale(self) -> list[str]:
        ids = sorted(set(self.expected) | set(self.last_ts))
        return [
            s
            for s in ids
            if s not in self.last_ts or self.checked_at - self.last_ts[s] > STALE_AFTER
        ]

    @property
    def healthy(self) -> bool:
        return self.rows_24h > 0


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def format_message(status: Status) -> str:
    """One line, e.g. ``OK 2026-10-27T00:15Z: 48 observations in the last 24 h; last ts:
    TST01 2026-10-27T00:00Z (0.2 h ago), TST02 never (STALE)``."""
    stale = set(status.stale)
    entries = []
    for sid in sorted(set(status.expected) | set(status.last_ts)):
        ts = status.last_ts.get(sid)
        if ts is None:
            entry = f"{sid} never"
        else:
            age_h = (status.checked_at - ts) / timedelta(hours=1)
            entry = f"{sid} {_utc(ts)} ({age_h:.1f} h ago)"
        entries.append(entry + (" (STALE)" if sid in stale else ""))
    level = "OK" if status.healthy and not stale else ("WARN" if status.healthy else "FAIL")
    return (
        f"{level} {_utc(status.checked_at)}: {status.rows_24h} observations in the last 24 h; "
        f"last ts: {', '.join(entries) or 'no stations'}"
    )


def read_status(con: Any, now: datetime, expected: Sequence[str] = ()) -> Status:
    last_ts = {
        station_id: as_utc(ts)
        for station_id, ts in con.execute(
            sql.SELECT_LAST_OBSERVATION_TS, {"source": AIR4THAI_SOURCE}
        ).fetchall()
        if ts is not None
    }
    (count,) = con.execute(
        sql.SELECT_OBSERVATION_COUNT_SINCE,
        {"source": AIR4THAI_SOURCE, "since": sql.require_utc(now - WINDOW)},
    ).fetchone()
    return Status(checked_at=now, last_ts=last_ts, rows_24h=int(count), expected=tuple(expected))


def _expected_stations(settings: Settings) -> list[str]:
    try:
        return [sid for sid, _lat, _lon in settings.station_location_list]
    except ValueError:  # the collector reports malformed STATION_LOCATIONS; just skip it here
        logger.warning("STATION_LOCATIONS is malformed; reporting stored stations only")
        return []


def check(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Status:
    """Read the status. Driver connection errors propagate for `run_entrypoint` to map."""
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    now = clock()
    with open_db(connect, settings.database_url.get_secret_value()) as con:
        return read_status(con, now, _expected_stations(settings))


def main(
    settings: Settings | None = None,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    alert = make_alert(None, prefix=PREFIX)  # replaced once settings are read
    webhook: str | None = None
    try:
        settings = settings or get_settings()
        if settings.alert_webhook_url:
            webhook = settings.alert_webhook_url.get_secret_value()
        alert = make_alert(webhook, prefix=PREFIX)
        result, code = run_entrypoint(
            lambda: check(settings, connect=connect, clock=clock),
            alert,
            connection_errors=db.CONNECTION_ERRORS,
        )
    except ConfigError as exc:  # messages never contain secret values
        safe_alert(alert, f"configuration error: {exc}")
        log_summary(logger, "heartbeat", {"status": "config_error"})
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.exception("heartbeat crashed")
        safe_alert(alert, f"heartbeat crashed: {type(exc).__name__}")
        log_summary(logger, "heartbeat", {"status": "crashed"})
        return EXIT_UNHEALTHY
    if isinstance(result, RunSkipped):  # DB down: the guard has already alerted
        log_summary(logger, "heartbeat", {"status": "db_unreachable"})
        return code
    message = format_message(result)
    log_summary(
        logger,
        "heartbeat",
        {
            "status": "ok" if result.healthy else "no_recent_data",
            "checked_at": result.checked_at,
            "rows_24h": result.rows_24h,
            "stale_stations": result.stale,
            "last_ts": dict(sorted(result.last_ts.items())),
        },
    )
    if not result.healthy:
        safe_alert(alert, message)
        return EXIT_UNHEALTHY
    logger.info("HEARTBEAT: %s", message)
    if not webhook:
        warn_missing_webhook("the heartbeat")
        return EXIT_OK
    try:
        post_message(webhook, f"{PREFIX} {message}")
    except Exception as exc:  # WebhookError messages never contain the URL
        logger.error("could not deliver the heartbeat: %s", exc)
        return EXIT_UNHEALTHY
    return EXIT_OK


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
