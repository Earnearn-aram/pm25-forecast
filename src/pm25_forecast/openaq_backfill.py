"""KAN-49: restartable OpenAQ hourly history ingestion, with KAN-52 sub-window retries
for sensors that fail a whole chunk; see docs/openaq-backfill.md."""

import argparse
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import AliasPath, AwareDatetime, BaseModel, Field, ValidationError

from pm25_forecast.backfill import write_station
from pm25_forecast.collector import ConfigError, open_db
from pm25_forecast.ingest.air4thai import Observation
from pm25_forecast.ingest.probe import (
    MATCH_RADIUS_M,
    OPENAQ_API_URL,
    OPENAQ_PAGE_LIMIT,
    TARGET_STATION_IDS,
    Station,
    parse_openaq_locations,
    parse_openaq_sensors,
    rank_openaq_candidates,
)
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db
from pm25_forecast.storage.sql import require_utc

logger = logging.getLogger(__name__)
CHUNK_DAYS = 31
SUBCHUNK_DAYS = 1  # KAN-52: retry unit for a sensor whose whole-chunk fetch fails
MAX_PAGES = 32
MAX_ATTEMPTS = 4
MAX_WAIT = 60.0


class OpenAQError(Exception):
    """Sanitized API failure, safe to log (never response bodies or exception text)."""


class Page(BaseModel):
    results: list[Any]


class Hour(BaseModel):
    ts: AwareDatetime = Field(validation_alias=AliasPath("period", "datetimeFrom", "utc"))
    value: float = Field(ge=0, le=1000, allow_inf_nan=False)


class OpenAQObservation(Observation):
    pm25: float = Field(ge=0, le=1000, allow_inf_nan=False)
    source: str = "openaq"


class Report(BaseModel):
    station_id: str
    location_id: int | None = None
    sensor_ids: list[int] = Field(default_factory=list)
    chunks: int = 0
    rows_usable: int = 0
    rows_upserted: int = 0
    rows_new: int = 0
    failures: list[str] = Field(default_factory=list)


def retry_wait(response: httpx.Response, attempt: int) -> float:
    """Honor numeric or HTTP-date Retry-After, with a bounded wait and exponential fallback."""
    raw = response.headers.get("Retry-After", "")
    try:
        wait = float(raw)
    except ValueError:
        try:
            wait = (parsedate_to_datetime(raw) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            wait = 2.0**attempt
    return min(MAX_WAIT, max(1.0, wait)) if math.isfinite(wait) else MAX_WAIT


class OpenAQClient:
    def __init__(self, http: httpx.Client, key: str, sleep: Callable[[float], None]):
        self.http, self.key, self.sleep = http, key, sleep

    def get(self, path: str, params: dict[str, Any]) -> Page:
        for attempt in range(MAX_ATTEMPTS):
            self.sleep(1.0)  # pace every request, including pagination and retries
            try:
                response = self.http.get(
                    OPENAQ_API_URL + path, params=params, headers={"X-API-Key": self.key}
                )
            except httpx.TransportError:
                if attempt + 1 == MAX_ATTEMPTS:
                    raise OpenAQError("transport failure after bounded retries") from None
                self.sleep(min(MAX_WAIT, 2.0**attempt))
                continue
            if (response.status_code == 429 or 500 <= response.status_code < 600) and (
                attempt + 1 < MAX_ATTEMPTS
            ):
                self.sleep(retry_wait(response, attempt))
                continue
            if response.status_code != 200:
                raise OpenAQError(f"HTTP {response.status_code}")
            try:
                return Page.model_validate(response.json())
            except (ValueError, ValidationError):
                raise OpenAQError("invalid response payload") from None
        raise OpenAQError("retry limit reached")  # defensive

    def pages(self, path: str, params: dict[str, Any]):
        for page in range(1, MAX_PAGES + 1):
            result = self.get(path, {**params, "page": page, "limit": OPENAQ_PAGE_LIMIT})
            yield result
            if len(result.results) < OPENAQ_PAGE_LIMIT:
                return
        raise OpenAQError("pagination limit reached; narrow the requested window")

    def sensors(self, station: Station) -> tuple[int, list[int]]:
        locations = []
        for page in self.pages(
            "/locations",
            {
                "coordinates": f"{station.lat},{station.lon}",
                "radius": int(MATCH_RADIUS_M),
                "parameters_id": 2,
            },
        ):
            locations.extend(parse_openaq_locations(page.model_dump()))
        ranked = rank_openaq_candidates(station, locations)
        if not ranked:
            raise OpenAQError("no location within 1000 m")
        best = ranked[0]
        sensors = best.sensor_ids
        if not sensors:  # reuse main's embedded sensor logic, with its compatibility fallback
            for page in self.pages(f"/locations/{best.location_id}/sensors", {}):
                sensors.extend(parse_openaq_sensors(page.model_dump()))
        if not sensors:
            raise OpenAQError("no PM2.5 sensors")
        return best.location_id, sorted(set(sensors))

    def hours(self, sensor: int, start: datetime, end: datetime) -> dict[datetime, float]:
        series = {}
        for page in self.pages(
            f"/sensors/{sensor}/hours",
            {"datetime_from": start.isoformat(), "datetime_to": end.isoformat()},
        ):
            for raw in page.results:
                try:
                    hour = Hour.model_validate(raw)
                except ValidationError:
                    continue
                stamp = hour.ts.astimezone(UTC)
                if stamp.minute or stamp.second or stamp.microsecond:
                    continue
                if start <= stamp < end:
                    series[stamp] = hour.value
        return series


def chunk_series(
    client: OpenAQClient,
    station_id: str,
    sensors: Sequence[int],
    chunk_start: datetime,
    chunk_end: datetime,
) -> tuple[dict[datetime, float], list[str], list[tuple[datetime, datetime]]]:
    """One station's merged chunk series, failure messages and failed sub-window spans.

    Sensors merge in ascending id order, so the highest sensor id wins each non-null
    overlapping hour. A sensor whose whole-chunk fetch fails (KAN-52: the API can HTTP 500
    on a large window that it serves fine in smaller pieces) is retried alone over
    SUBCHUNK_DAYS sub-windows and every successful sub-window is kept. Failed sub-windows
    are returned as spans; their hours are excluded from the merged series so a partial
    replay can never displace previously stored higher-priority sensor data there.
    """
    series: dict[datetime, float] = {}
    failures: list[str] = []
    failed_spans: list[tuple[datetime, datetime]] = []
    for sensor in sensors:
        try:
            # Fetch each sensor atomically so a failed later page cannot publish a
            # partial series that would alter precedence on a subsequent rerun.
            series.update(client.hours(sensor, chunk_start, chunk_end))
            continue
        except OpenAQError as exc:
            logger.warning(
                "station %s chunk %s sensor %d: %s; retrying in %d-day sub-windows",
                station_id,
                chunk_start.isoformat(),
                sensor,
                exc,
                SUBCHUNK_DAYS,
            )
        sub_start = chunk_start
        while sub_start < chunk_end:
            sub_end = min(chunk_end, sub_start + timedelta(days=SUBCHUNK_DAYS))
            try:
                series.update(client.hours(sensor, sub_start, sub_end))
            except OpenAQError as sub_exc:
                failed_spans.append((sub_start, sub_end))
                failures.append(
                    f"{sub_start.isoformat()}..{sub_end.isoformat()} sensor {sensor}: {sub_exc}"
                )
            sub_start = sub_end
    if failed_spans:
        series = {
            stamp: value
            for stamp, value in series.items()
            if not any(lo <= stamp < hi for lo, hi in failed_spans)
        }
    return series, failures, failed_spans


def run(
    settings: Settings,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    station_ids: Sequence[str] = TARGET_STATION_IDS,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Report]:
    # Intentionally read ONLY the environment: no CLI key, settings override or .env key.
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not key:
        raise ConfigError("OPENAQ_API_KEY is not set in the environment")
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    try:
        end = (
            require_utc(end)
            if end
            else require_utc(clock()).replace(minute=0, second=0, microsecond=0)
        )
        start = require_utc(start) if start else end - timedelta(days=730)
        coordinates = {sid: (lat, lon) for sid, lat, lon in settings.station_location_list}
    except ValueError:
        raise ConfigError("invalid UTC window or STATION_LOCATIONS") from None
    if start >= end or any((d.minute or d.second or d.microsecond) for d in (start, end)):
        raise ConfigError("window must be increasing, with whole UTC hours")
    ids = list(dict.fromkeys(station_ids))
    if not ids or any(sid not in TARGET_STATION_IDS or sid not in coordinates for sid in ids):
        raise ConfigError("each selected target station requires STATION_LOCATIONS coordinates")
    stations = []
    for sid in ids:
        lat, lon = coordinates[sid]
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not (-90 <= lat <= 90 and -180 <= lon <= 180)
        ):
            raise ConfigError("invalid station coordinates")
        stations.append(Station(station_id=sid, lat=lat, lon=lon))
    owned_http = http is None
    http = http or httpx.Client(timeout=30.0, follow_redirects=False)
    client = OpenAQClient(http, key, sleep)
    reports = []
    try:
        for station in stations:
            report = Report(station_id=station.station_id)
            reports.append(report)
            try:
                report.location_id, report.sensor_ids = client.sensors(station)
            except OpenAQError as exc:
                report.failures.append(str(exc))
                logger.error("station %s: %s", station.station_id, exc)
                continue
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(end, chunk_start + timedelta(days=CHUNK_DAYS))
                series, chunk_failures, failed_spans = chunk_series(
                    client, station.station_id, report.sensor_ids, chunk_start, chunk_end
                )
                report.failures.extend(chunk_failures)
                if failed_spans:
                    # Store what succeeded; failed sub-windows stay gaps for a targeted
                    # replay (KAN-52). With nothing storable, leave the chunk untouched.
                    logger.error(
                        "station %s chunk %s: %d sub-window failures, %d rows storable;"
                        " replay required for the failed sub-windows",
                        station.station_id,
                        chunk_start.isoformat(),
                        len(failed_spans),
                        len(series),
                    )
                    if not series:
                        chunk_start = chunk_end
                        continue
                rows = [
                    OpenAQObservation(station_id=station.station_id, ts=stamp, pm25=value)
                    for stamp, value in sorted(series.items())
                ]
                # One durable transaction per station/chunk; failed chunks can be replayed.
                with open_db(connect, settings.database_url.get_secret_value()) as con:
                    affected, new = write_station(con, station.station_id, rows)
                report.rows_usable += len(rows)
                report.rows_upserted += affected
                report.rows_new += new
                report.chunks += 1
                logger.info(
                    "station %s location %s: %s .. %s, %d affected (%d new), %d failures",
                    station.station_id,
                    report.location_id,
                    chunk_start.isoformat(),
                    chunk_end.isoformat(),
                    affected,
                    new,
                    len(report.failures),
                )
                chunk_start = chunk_end
    finally:
        if owned_http:
            http.close()
    return reports


def utc_arg(raw: str) -> datetime:
    try:
        return require_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError:
        raise argparse.ArgumentTypeError("use a timezone-aware ISO timestamp") from None


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", type=utc_arg, help="inclusive UTC hour (default: end - 730 days)"
    )
    parser.add_argument(
        "--end", type=utc_arg, help="exclusive UTC hour (default: current full hour)"
    )
    parser.add_argument("--stations", default=",".join(TARGET_STATION_IDS))
    args = parser.parse_args(argv)
    try:
        reports = run(
            get_settings(), start=args.start, end=args.end, station_ids=args.stations.split(",")
        )
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except Exception as exc:
        logger.error("backfill failed: %s", type(exc).__name__)
        return 1
    for report in reports:
        for failure in report.failures:
            logger.error("station %s: %s", report.station_id, failure)
    return int(any(r.failures or not r.rows_usable for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
