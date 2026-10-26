"""Hourly collector (KAN-18): ingest only. Run with ``python -m pm25_forecast.collector``.

Per run: fetch Air4Thai latest readings and upsert them in their own transaction; then fetch the
Open-Meteo CAMS PM2.5 forecast per station and, in a second transaction, append it as
`predictions` rows (model_version ``cams_global``, ``issue_time`` = retrieval time). A failure in
the CAMS half therefore never rolls back observations.

Design points (the GitHub schedule is best effort: runs are late, skipped or doubled up):

* Data timestamps, never run time, are the keys. Re-running or double-running is idempotent.
* Air4Thai's ``getNewAQI_JSON`` returns only each station's latest reading, so a missed hour
  cannot be recovered from it (maximum recoverable gap: 0 h). The gap check compares each
  station's newest stored ``ts`` with the new reading, logs and alerts on every missing hour,
  and never invents data.
* CAMS refreshes about twice a day, so most hourly snapshots repeat the previous run. A snapshot
  whose values equal the latest stored vintage on every overlapping valid_time (and adds no
  later hours) is not stored.
* Air4Thai is polled every 20 minutes (workflow draft) because each poll only sees the latest
  reading; upserts are idempotent and CAMS repeats are deduped, so extra polls are cheap.
* Weather is off by default (``COLLECT_WEATHER``): the KAN-17 schema has no weather table.
  When enabled it is fetched, validated and raw-archived only, and its failures are not counted.
* Fetching has a per-host circuit breaker (the first transport failure for a host skips the
  remaining stations for that host) and an overall deadline well under the job timeout.
* DB down: no fallback. The run is skipped, an alert is sent and the exit code is non-zero.
* Content from the APIs is data. Nothing here builds requests or SQL from it.
"""

import gzip
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import psycopg

from pm25_forecast.ingest.air4thai import (
    SOURCE as AIR4THAI_SOURCE,
)
from pm25_forecast.ingest.air4thai import (
    Air4ThaiClient,
    Observation,
    gzip_file_archive,
)
from pm25_forecast.ingest.open_meteo import (
    AIR_QUALITY_URL,
    CAMS_MODEL_VERSION,
    FORECAST_URL,
    OpenMeteoClient,
    PredictionRow,
    RawResponse,
    StationLocation,
    _RetryableError,
    to_prediction_rows,
)
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

logger = logging.getLogger(__name__)

STEP = timedelta(hours=1)  # Air4Thai publishes one reading per station per hour
MAX_FUTURE_SKEW = timedelta(hours=1)  # a reading further ahead than this is a bad timestamp
STALE_AFTER = timedelta(hours=3)
STALE_ALERT_WINDOW = timedelta(minutes=20)  # one poll interval: alert as a station crosses
FETCH_DEADLINE_S = 240.0  # whole fetch phase; the workflow timeout is 10 minutes
EXIT_FAILED, EXIT_CONFIG = 1, 2


class ConfigError(Exception):
    """Missing or malformed configuration (never contains secret values)."""


@dataclass(frozen=True)
class Gap:
    station_id: str
    last_stored: datetime
    next_seen: datetime
    missing_hours: int

    def __str__(self) -> str:
        return (
            f"station {self.station_id}: {self.missing_hours} h missing between "
            f"{self.last_stored.isoformat()} and {self.next_seen.isoformat()} (not recoverable)"
        )


@dataclass
class CollectorResult:
    run_id: str
    observations_written: int = 0
    observations_dropped: int = 0
    predictions_written: int = 0
    cams_duplicates_skipped: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    stale_stations: list[str] = field(default_factory=list)
    newly_stale_stations: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


# --- pure helpers ----------------------------------------------------------------------------


def as_utc(value: Any) -> datetime:
    """Coerce a DB value (datetime, or ISO string from the SQLite test double) to aware UTC."""
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    return sql.require_utc(moment)


def find_gaps(
    last_stored: Mapping[str, datetime], observations: Iterable[Observation]
) -> list[Gap]:
    """Missing whole hours between each station's newest stored ts and its new reading.

    A station with nothing stored yet (first run) has no gap. Equal or consecutive hours are
    not gaps. This reports; it never fills.
    """
    gaps = []
    for obs in observations:
        previous = last_stored.get(obs.station_id)
        if previous is None:
            continue
        missing = round((obs.ts - previous) / STEP) - 1
        if missing >= 1:
            gaps.append(Gap(obs.station_id, previous, obs.ts, missing))
    return gaps


def is_duplicate_vintage(
    new_rows: Sequence[PredictionRow], previous: Mapping[datetime, float]
) -> bool:
    """True if `new_rows` repeats the previous vintage: same values on every overlapping
    valid_time, at least one overlap, and no valid_time later than the previous vintage had."""
    new = {row.valid_time: row.pm25_pred for row in new_rows}
    overlap = new.keys() & previous.keys()
    if not overlap or max(new) > max(previous):
        return False
    return all(math.isclose(new[k], previous[k], rel_tol=1e-9, abs_tol=1e-9) for k in overlap)


# --- IO helpers ------------------------------------------------------------------------------


def gzip_open_meteo_archive(directory: Path) -> Callable[[RawResponse], None]:
    """Archive hook: ``<dir>/<source>-<UTC stamp>-<lat>_<lon>.json.gz`` (never committed)."""

    def archive(raw: RawResponse) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = raw.retrieved_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        lat = raw.params.get("latitude", "x")
        lon = raw.params.get("longitude", "x")
        path = directory / f"{raw.source}-{stamp}-{lat}_{lon}.json.gz"
        path.write_bytes(gzip.compress(raw.model_dump_json().encode()))

    return archive


def make_alert(webhook_url: str | None) -> Callable[[str], None]:
    """Alert function posting ``{"text": ...}`` to the webhook; without one it logs at ERROR.

    The webhook URL is a secret: failures are re-raised without the URL (httpx puts it in its
    messages) so the guard's traceback logging cannot leak it.
    """

    def alert(message: str) -> None:
        logger.error("ALERT: %s", message)
        if not webhook_url:
            return
        try:
            httpx.post(webhook_url, json={"text": f"[pm25 collector] {message}"}, timeout=10.0)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"alert webhook failed: {type(exc).__name__}") from None

    return alert


def default_run_id(now: datetime) -> str:
    github_run = os.environ.get("GITHUB_RUN_ID")
    if github_run:
        return f"gh-{github_run}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    return f"local-{now:%Y%m%dT%H%M%SZ}"


# --- the run ---------------------------------------------------------------------------------


def fetch_air4thai(
    air4thai: Air4ThaiClient,
    station_ids: Sequence[str],
    now: datetime,
    result: CollectorResult,
) -> list[Observation]:
    """Fetch and filter Air4Thai readings. A failure is recorded, never raised."""
    try:
        fetched = air4thai.fetch(station_ids)
    except Exception as exc:  # any failure of this source must not stop the CAMS half
        result.failures.append(f"air4thai fetch failed: {type(exc).__name__}")
        logger.error("air4thai fetch failed: %s", type(exc).__name__)
        return []
    observations: list[Observation] = []
    for obs in fetched:
        if obs.pm25 is None:
            result.observations_dropped += 1  # no value: store nothing rather than a hollow row
        elif obs.ts > now + MAX_FUTURE_SKEW:
            result.observations_dropped += 1
            logger.warning("air4thai: %s reading dated in the future, dropped", obs.station_id)
        else:
            observations.append(obs)
    if not observations:  # a 200 with nothing usable is schema drift or an outage, not success
        result.failures.append("air4thai returned no usable observations")
        logger.error("air4thai returned no usable observations (%d fetched)", len(fetched))
    return observations


def fetch_cams(
    open_meteo: OpenMeteoClient,
    stations: Sequence[StationLocation],
    run_id: str,
    result: CollectorResult,
    *,
    collect_weather: bool = False,
    deadline: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, list[PredictionRow]]:
    """CAMS rows per station (and optionally weather, validated and archived only).

    Per-host circuit breaker: once a host fails at transport level (timeouts, connection errors,
    429/5xx after the client's own retries) the remaining stations skip it and one failure is
    recorded. Past `deadline` (a `monotonic` reading) the remaining stations are skipped.
    Weather problems are logged but never counted as run failures.
    """
    jobs = [("cams", httpx.URL(AIR_QUALITY_URL).host, open_meteo.cams_forecast)]
    if collect_weather:
        jobs.append(("weather", httpx.URL(FORECAST_URL).host, open_meteo.weather_forecast))
    tripped: set[str] = set()
    cams: dict[str, list[PredictionRow]] = {}
    for station in stations:
        if monotonic() > deadline:
            result.failures.append("fetch deadline exceeded; remaining stations skipped")
            logger.error("fetch deadline exceeded before station %s", station.station_id)
            break
        for what, host, call in jobs:
            if host in tripped:
                continue
            try:
                forecast = call(station.latitude, station.longitude)
            except Exception as exc:
                logger.error(
                    "open-meteo %s failed for %s: %s", what, station.station_id, type(exc).__name__
                )
                if isinstance(exc, _RetryableError):
                    tripped.add(host)
                    logger.error("open-meteo host %s tripped; skipping remaining stations", host)
                    if what == "cams":
                        result.failures.append(
                            f"open-meteo cams unreachable ({type(exc).__name__})"
                        )
                elif what == "cams":
                    result.failures.append(f"open-meteo cams failed for {station.station_id}")
                continue
            if what == "cams":
                cams[station.station_id] = to_prediction_rows(forecast, station.station_id, run_id)
            else:
                logger.info("weather %s: %d hourly rows", station.station_id, len(forecast.rows))
    return cams


def write_observations(
    con: Any,
    observations: Sequence[Observation],
    station_ids: Sequence[str],
    now: datetime,
    result: CollectorResult,
) -> None:
    """Gap and stale checks against what is stored, then the idempotent upsert."""
    last_stored = {
        station_id: as_utc(ts)
        for station_id, ts in con.execute(
            sql.SELECT_LAST_OBSERVATION_TS, {"source": AIR4THAI_SOURCE}
        ).fetchall()
        if ts is not None
    }
    result.gaps = find_gaps(last_stored, observations)
    newest = dict(last_stored)
    for obs in observations:
        newest[obs.station_id] = max(obs.ts, newest.get(obs.station_id, obs.ts))
    for sid in station_ids:
        if sid not in newest:
            continue  # never seen: nothing to be stale relative to
        age = now - newest[sid]
        if age > STALE_AFTER:
            result.stale_stations.append(sid)
            if age <= STALE_AFTER + STALE_ALERT_WINDOW:  # only as it crosses the threshold
                result.newly_stale_stations.append(sid)
    for gap in result.gaps:
        logger.warning("GAP %s", gap)
    for sid in result.stale_stations:
        logger.warning("station %s has no reading newer than %s", sid, STALE_AFTER)
    result.observations_written = sql.write_rows(
        con, sql.UPSERT_OBSERVATION, [o.model_dump() for o in observations]
    )


def write_cams(
    con: Any, cams: Mapping[str, Sequence[PredictionRow]], result: CollectorResult
) -> None:
    for station_id, rows in cams.items():
        if not rows:
            logger.warning("cams %s: no usable rows", station_id)
            continue
        previous = {
            as_utc(valid): value
            for _issue, valid, value in con.execute(
                sql.SELECT_LATEST_VINTAGE,
                {"station_id": station_id, "model_version": CAMS_MODEL_VERSION},
            ).fetchall()
        }
        if is_duplicate_vintage(rows, previous):
            result.cams_duplicates_skipped.append(station_id)
            logger.info("cams %s: repeats the stored vintage, not stored", station_id)
            continue
        result.predictions_written += sql.write_rows(
            con, sql.INSERT_PREDICTION, [r.model_dump() for r in rows]
        )


def open_db(connect: Callable[[str], Any], url: str) -> Any:
    try:
        return connect(url)
    except psycopg.ProgrammingError:  # malformed conninfo; its message may echo the URL
        raise ConfigError("DATABASE_URL is malformed") from None


def _safe_alert(alert: Callable[[str], None] | None, message: str) -> None:
    if alert is None:
        return
    try:
        alert(message)
    except Exception:
        logger.exception("alert failed: %s", message)


def run(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic: Callable[[], float] = time.monotonic,
    http: httpx.Client | None = None,
    alert: Callable[[str], None] | None = None,
) -> CollectorResult:
    """One collector run. Driver connection errors propagate for `run_entrypoint` to map."""
    now = clock()
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    try:
        stations = [
            StationLocation(station_id=i, latitude=a, longitude=o)
            for i, a, o in settings.station_location_list
        ]
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if not stations:
        raise ConfigError("STATION_LOCATIONS is not set (format: ID=lat,lon;ID=lat,lon)")

    url = settings.database_url.get_secret_value()
    station_ids = [s.station_id for s in stations]
    run_id = default_run_id(now)
    result = CollectorResult(run_id=run_id)
    deadline = monotonic() + FETCH_DEADLINE_S
    archive_dir = Path(settings.raw_archive_dir or "data/raw")

    # Transaction 1: Air4Thai only, committed before CAMS is even fetched.
    air4thai = Air4ThaiClient(client=http, archive=gzip_file_archive(archive_dir / "air4thai"))
    try:
        observations = fetch_air4thai(air4thai, station_ids, now, result)
    finally:
        air4thai.close()
    with open_db(connect, url) as con:
        write_observations(con, observations, station_ids, now, result)
    for gap in result.gaps:
        _safe_alert(alert, f"gap in Air4Thai data: {gap}")
    if result.newly_stale_stations:
        _safe_alert(
            alert,
            f"stations with no reading for over {STALE_AFTER}: "
            + ", ".join(result.newly_stale_stations),
        )

    # Transaction 2: CAMS vintages.
    open_meteo = OpenMeteoClient(
        http, archive=gzip_open_meteo_archive(archive_dir / "open_meteo"), now=clock
    )
    try:
        cams = fetch_cams(
            open_meteo,
            stations,
            run_id,
            result,
            collect_weather=settings.collect_weather,
            deadline=deadline,
            monotonic=monotonic,
        )
    finally:
        open_meteo.close()
    if cams:
        with open_db(connect, url) as con:
            write_cams(con, cams, result)

    logger.info(
        "run %s: %d observations written (%d dropped), %d predictions written, "
        "%d CAMS repeats skipped, %d gaps, %d failures",
        run_id,
        result.observations_written,
        result.observations_dropped,
        result.predictions_written,
        len(result.cams_duplicates_skipped),
        len(result.gaps),
        len(result.failures),
    )
    return result


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines log full request URLs
    settings = get_settings()
    webhook = settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
    alert = make_alert(webhook)
    try:
        result, code = run_entrypoint(
            lambda: run(settings, alert=alert), alert, connection_errors=db.CONNECTION_ERRORS
        )
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.exception("collector crashed")
        _safe_alert(alert, f"collector crashed: {type(exc).__name__}")
        return EXIT_FAILED
    if isinstance(result, RunSkipped):
        return code
    if result.failures:
        _safe_alert(alert, "run completed with failures: " + "; ".join(result.failures))
        return EXIT_FAILED
    return code


if __name__ == "__main__":
    sys.exit(main())
