"""Data quality checks (KAN-21). Run with ``python -m pm25_forecast.quality``.

Per run, per source and station, five checks over the database (see docs/data-quality.md):

* ``freshness``: age in hours of the newest usable observation (per observation source), of the
  newest ``cams_global`` vintage and of the newest forecast per served model.
* ``missing_hours``: whole hours in the last 24 without a usable observation (air4thai).
* ``out_of_range``: observations in the last 24 h that are negative, above 1000 or non-finite.
* ``expected_station``: an expected station (STATION_LOCATIONS) with no air4thai rows in 24 h.
* ``null_rate``: share of rows in the last 24 h with a NULL pm25.

Every result is upserted into ``data_quality_checks`` keyed on (checked_at hour, source, station,
check), so a re-run in the same hour overwrites instead of duplicating.

Alerting: one alert per run, listing only the ``fail`` items (freshness and out-of-range).
Missing hours are recorded as ``warn`` and never alerted: the Collector already alerts on gaps.
A run that found failing data still exits 0 (the alert is the signal, and a red step would make
the workflow's failure alert fire a second time); exit 1 means the run itself failed.

Exit codes: 0 ok, 1 failed / DB down, 2 config error.
"""

import logging
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import AwareDatetime, BaseModel, ValidationError

from pm25_forecast.collector import ConfigError, default_run_id, open_db
from pm25_forecast.ingest.air4thai import SOURCE as AIR4THAI_SOURCE
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
from pm25_forecast.models.baselines import PERSISTENCE_MODEL_VERSION
from pm25_forecast.monitoring.alerts import make_alert, safe_alert
from pm25_forecast.monitoring.logs import configure_logging, log_summary
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

logger = logging.getLogger(__name__)

EXIT_FAILED, EXIT_CONFIG = 1, 2
OPENAQ_SOURCE = "openaq"
OK, WARN, FAIL = "ok", "warn", "fail"

# --- Thresholds: the single place to tune them. Mirrored in docs/data-quality.md. ------------
WINDOW_H = 24  # look-back for missing hours, out-of-range, expected station and null rate
PM25_MIN, PM25_MAX = 0.0, 1000.0  # usable = finite and within [PM25_MIN, PM25_MAX]
MISSING_HOURS_WARN = 3  # never fails: the Collector alerts on gaps
NULL_RATE_WARN = 0.2  # share of rows with NULL pm25; never fails
EXPECTED_STATION_MIN_ROWS = 1  # fewer rows than this in the window = warn


@dataclass(frozen=True)
class Limit:
    """Age limits in hours; ``fail`` None means the check can only warn."""

    warn: float
    fail: float | None


OBSERVATION_FRESHNESS = {
    AIR4THAI_SOURCE: Limit(warn=3, fail=6),
    OPENAQ_SOURCE: Limit(warn=24, fail=None),  # gap filler / backfill: stale is normal
}
CAMS_FRESHNESS = Limit(warn=24, fail=48)  # a new vintage a day is enough; duplicates are skipped
PREDICTION_FRESHNESS = Limit(warn=3, fail=6)  # predict runs hourly


class ObsRow(BaseModel):
    station_id: str
    ts: AwareDatetime
    pm25: float | None
    source: str


class Newest(BaseModel):
    station_id: str
    newest: AwareDatetime


@dataclass(frozen=True)
class Check:
    source: str
    station_id: str  # '' = source-wide
    check: str
    status: str
    value: float | None
    threshold: float | None
    detail: str

    def row(self, checked_at: datetime) -> dict[str, Any]:
        return {
            "checked_at": checked_at,
            "source": self.source,
            "station_id": self.station_id,
            "check_name": self.check,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }

    def label(self) -> str:
        who = f"{self.source}/{self.station_id}" if self.station_id else self.source
        return f"{who} {self.check}: {self.detail}"


@dataclass
class QualityResult:
    run_id: str
    checked_at: datetime
    checks: list[Check] = field(default_factory=list)
    written: int = 0
    failures: list[str] = field(default_factory=list)  # the run itself failed

    @property
    def failing(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]


# --- SQL (kept here, not in storage/sql.py) --------------------------------------------------

UPSERT_CHECK = """
INSERT INTO data_quality_checks
    (checked_at, source, station_id, check_name, status, value, threshold, detail)
VALUES (
    %(checked_at)s, %(source)s, %(station_id)s, %(check_name)s, %(status)s, %(value)s,
    %(threshold)s, %(detail)s
)
ON CONFLICT (checked_at, source, station_id, check_name)
DO UPDATE SET status = excluded.status, value = excluded.value,
              threshold = excluded.threshold, detail = excluded.detail
"""

SELECT_RECENT_OBSERVATIONS = """
SELECT station_id, ts, pm25, source FROM observations
WHERE ts >= %(since)s AND ts <= %(now)s
"""

# Usable = finite and in range; NaN and infinity compare above PM25_MAX in PostgreSQL.
SELECT_NEWEST_USABLE_OBSERVATION = """
SELECT station_id, max(ts) FROM observations
WHERE source = %(source)s AND pm25 IS NOT NULL AND pm25 >= %(lo)s AND pm25 <= %(hi)s
  AND ts <= %(now)s
GROUP BY station_id
"""

SELECT_NEWEST_FORECAST = """
SELECT station_id, max(issue_time) FROM predictions
WHERE model_version = %(model_version)s AND issue_time <= %(now)s
GROUP BY station_id
"""


# --- Pure checks -----------------------------------------------------------------------------


def floor_hour(moment: datetime) -> datetime:
    return sql.require_utc(moment).replace(minute=0, second=0, microsecond=0)


def usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and PM25_MIN <= value <= PM25_MAX


def freshness_check(
    source: str, station_id: str, newest: datetime | None, now: datetime, limit: Limit, what: str
) -> Check:
    if newest is None:
        status = FAIL if limit.fail is not None else WARN
        threshold = limit.fail if limit.fail is not None else limit.warn
        return Check(source, station_id, "freshness", status, None, threshold, f"no {what} found")
    age = (sql.require_utc(now) - sql.require_utc(newest)) / timedelta(hours=1)
    if limit.fail is not None and age > limit.fail:
        status, threshold = FAIL, limit.fail
    elif age > limit.warn:
        status, threshold = WARN, limit.warn
    else:
        status, threshold = OK, limit.warn
    detail = f"newest {what} is {age:.1f} h old (limit {threshold:g} h)"
    return Check(source, station_id, "freshness", status, round(age, 3), threshold, detail)


def missing_hours_check(
    source: str, station_id: str, rows: Iterable[ObsRow], now: datetime
) -> Check:
    end = floor_hour(now)
    slots = {end - timedelta(hours=i) for i in range(1, WINDOW_H + 1)}
    have = {floor_hour(r.ts) for r in rows if usable(r.pm25)} & slots
    missing = len(slots) - len(have)
    status = WARN if missing >= MISSING_HOURS_WARN else OK
    detail = f"{missing} of {WINDOW_H} hours without a usable reading"
    return Check(
        source, station_id, "missing_hours", status, float(missing), MISSING_HOURS_WARN, detail
    )


def out_of_range_check(source: str, station_id: str, rows: Iterable[ObsRow]) -> Check:
    bad = sum(1 for r in rows if r.pm25 is not None and not usable(r.pm25))
    detail = f"{bad} reading(s) below {PM25_MIN:g}, above {PM25_MAX:g} or non-finite"
    return Check(source, station_id, "out_of_range", FAIL if bad else OK, float(bad), 0.0, detail)


def expected_station_check(source: str, station_id: str, rows: Sequence[ObsRow]) -> Check:
    n = len(rows)
    status = WARN if n < EXPECTED_STATION_MIN_ROWS else OK
    detail = f"{n} row(s) in the last {WINDOW_H} h"
    return Check(
        source,
        station_id,
        "expected_station",
        status,
        float(n),
        float(EXPECTED_STATION_MIN_ROWS),
        detail,
    )


def null_rate_check(source: str, station_id: str, rows: Sequence[ObsRow]) -> Check | None:
    if not rows:
        return None  # nothing to rate; expected_station covers an empty station
    rate = sum(r.pm25 is None for r in rows) / len(rows)
    status = WARN if rate > NULL_RATE_WARN else OK
    detail = f"{rate:.0%} of {len(rows)} rows have no pm25"
    return Check(source, station_id, "null_rate", status, round(rate, 4), NULL_RATE_WARN, detail)


def observation_checks(
    rows: Sequence[ObsRow],
    newest: dict[tuple[str, str], datetime],
    expected: Sequence[str],
    now: datetime,
) -> list[Check]:
    """All observation checks. air4thai covers every expected station; other sources only the
    stations they actually have data for (OpenAQ is a gap filler, absence is not a defect)."""
    window: dict[tuple[str, str], list[ObsRow]] = {}
    for r in rows:
        window.setdefault((r.source, r.station_id), []).append(r)
    keys = set(window) | set(newest) | {(AIR4THAI_SOURCE, s) for s in expected}
    out: list[Check] = []
    for source, station in sorted(keys):
        mine = window.get((source, station), [])
        limit = OBSERVATION_FRESHNESS.get(source)
        if limit is not None:
            found = newest.get((source, station))
            out.append(freshness_check(source, station, found, now, limit, "usable observation"))
        out.append(out_of_range_check(source, station, mine))
        if (rate := null_rate_check(source, station, mine)) is not None:
            out.append(rate)
        if source == AIR4THAI_SOURCE:
            out.append(missing_hours_check(source, station, mine, now))
            out.append(expected_station_check(source, station, mine))
    return out


def forecast_checks(
    newest: dict[tuple[str, str], datetime],
    models: Sequence[str],
    expected: Sequence[str],
    now: datetime,
) -> list[Check]:
    out = []
    for model in models:
        is_cams = model == CAMS_MODEL_VERSION
        limit = CAMS_FRESHNESS if is_cams else PREDICTION_FRESHNESS
        what = "vintage" if is_cams else "forecast"
        out.extend(
            freshness_check(model, station, newest.get((model, station)), now, limit, what)
            for station in expected
        )
    return out


def _validated[M: BaseModel](model: type[M], rows: Sequence[Sequence[Any]], names: Sequence[str]):
    good: list[M] = []
    for row in rows:
        try:
            good.append(model.model_validate(dict(zip(names, row, strict=True))))
        except ValidationError:
            logger.warning("%s: dropped a malformed row", model.__name__)
    return good


def alert_text(failing: Sequence[Check], limit: int = 15) -> str:
    """The one alert body: fail items only, no secrets (ids, numbers and check names)."""
    shown = "; ".join(c.label() for c in failing[:limit])
    more = f" (+{len(failing) - limit} more)" if len(failing) > limit else ""
    return f"{len(failing)} data quality check(s) failing: {shown}{more}"


def run(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> QualityResult:
    """One run. Driver connection errors propagate for `run_entrypoint` to map."""
    now = sql.require_utc(clock())
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    try:
        expected = sorted({sid for sid, _lat, _lon in settings.station_location_list})
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if not expected:
        raise ConfigError("STATION_LOCATIONS is not set")

    models = [PERSISTENCE_MODEL_VERSION, CAMS_MODEL_VERSION]
    challenger = settings.challenger_model_version
    if challenger is not None and challenger not in models:
        models.append(challenger)
    checked_at = floor_hour(now)
    result = QualityResult(run_id=default_run_id(now), checked_at=checked_at)

    with open_db(connect, settings.database_url.get_secret_value()) as con:
        obs = _validated(
            ObsRow,
            con.execute(
                SELECT_RECENT_OBSERVATIONS,
                {"since": checked_at - timedelta(hours=WINDOW_H), "now": now},
            ).fetchall(),
            ("station_id", "ts", "pm25", "source"),
        )
        newest_obs: dict[tuple[str, str], datetime] = {}
        for source in OBSERVATION_FRESHNESS:
            raw = con.execute(
                SELECT_NEWEST_USABLE_OBSERVATION,
                {"source": source, "lo": PM25_MIN, "hi": PM25_MAX, "now": now},
            ).fetchall()
            for n in _validated(Newest, raw, ("station_id", "newest")):
                newest_obs[(source, n.station_id)] = n.newest
        newest_fc: dict[tuple[str, str], datetime] = {}
        for model in models:
            raw = con.execute(
                SELECT_NEWEST_FORECAST, {"model_version": model, "now": now}
            ).fetchall()
            for n in _validated(Newest, raw, ("station_id", "newest")):
                newest_fc[(model, n.station_id)] = n.newest
        result.checks = observation_checks(obs, newest_obs, expected, now) + forecast_checks(
            newest_fc, models, expected, now
        )
        result.written = sql.write_rows(
            con, UPSERT_CHECK, [c.row(checked_at) for c in result.checks]
        )

    by_status = {s: sum(c.status == s for c in result.checks) for s in (OK, WARN, FAIL)}
    log_summary(
        logger,
        "quality",
        {
            "run_id": result.run_id,
            "status": "failed" if result.failures else "ok",
            "checked_at": checked_at,
            "checks": len(result.checks),
            "by_status": by_status,
            "failing": [c.label() for c in result.failing],
            "failures": result.failures,
        },
    )
    return result


def main() -> int:
    configure_logging()
    alert = make_alert(None, prefix="[pm25 quality]")
    try:
        settings = get_settings()
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook, prefix="[pm25 quality]")
        result, code = run_entrypoint(
            lambda: run(settings), alert, connection_errors=db.CONNECTION_ERRORS
        )
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        safe_alert(alert, f"configuration error: {exc}")
        log_summary(logger, "quality", {"status": "config_error"})
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.exception("quality crashed")
        safe_alert(alert, f"quality crashed: {type(exc).__name__}")
        log_summary(logger, "quality", {"status": "crashed", "error": type(exc).__name__})
        return EXIT_FAILED
    if isinstance(result, RunSkipped):  # DB down: the guard has already alerted
        log_summary(logger, "quality", {"status": "db_unreachable"})
        return code
    if result.failing:  # one alert per run, fail items only
        safe_alert(alert, alert_text(result.failing))
    if result.failures:
        safe_alert(alert, "run completed with failures: " + "; ".join(result.failures))
        return EXIT_FAILED
    return code


if __name__ == "__main__":
    sys.exit(main())
