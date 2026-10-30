"""Air4Thai (PCD Thailand) ingestion client: ``getNewAQI_JSON`` (latest) and history (KAN-12/34).

The API is undocumented, so everything it returns is untrusted data: payloads are validated with
pydantic, unknown fields are ignored, missing or malformed readings become ``None`` instead of
failing the whole batch, and stations that fail validation are logged and skipped (never silently).

Timestamps: Air4Thai reports local Bangkok time (UTC+7) as separate ``date`` and ``time`` strings.
They are converted to timezone-aware UTC here. Note ``24:00`` style times are tolerated.

History (``getHistoryData.php``) only keeps a rolling window (KAN-9: every station starts
2026-08-01 00:00 Bangkok time), so older chunks come back as an empty ``data`` list. That is
normal and is reported as an empty chunk, not an error.
"""

from __future__ import annotations

import gzip
import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from pm25_forecast.ingest.tls import build_verify, is_tls_verify_error

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_DIR = Path("data/raw/air4thai")
AIR4THAI_URL = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
AIR4THAI_HISTORY_URL = "http://air4thai.com/forweb/getHistoryData.php"
HISTORY_CHUNK_DAYS = 31  # local calendar days per history request (both ends inclusive)
HISTORY_PAUSE_S = 2.0  # pause between history requests, to be polite to the host
SOURCE = "air4thai"
BANGKOK_TZ = timezone(timedelta(hours=7), name="Asia/Bangkok")
# Physically plausible bounds for hourly PM2.5 in ug/m3; anything else is treated as a gap.
PM25_MIN, PM25_MAX = 0.0, 1000.0
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
MAX_ATTEMPTS = 4


class Air4ThaiError(Exception):
    """The response could not be used at all (bad JSON or an unrecognised top-level shape)."""


def _to_float(value: Any) -> float | None:
    """Parse a reading that may be a number, a numeric string, ``""``, ``"-"`` or ``"-1"``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if not math.isfinite(number) or number < PM25_MIN or number > PM25_MAX:
        return None  # includes the -1 / -999 "no data" sentinels
    return number


class Pollutant(BaseModel):
    """One pollutant block, e.g. ``"PM25": {"color_id": "1", "aqi": "20", "value": "15"}``."""

    model_config = ConfigDict(extra="ignore")

    value: float | None = None
    aqi: float | None = None

    @field_validator("value", "aqi", mode="before")
    @classmethod
    def _tolerant_number(cls, v: Any) -> float | None:
        return _to_float(v)


class AQILast(BaseModel):
    """The latest reading of a station (local Bangkok date and time as strings)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    date: str
    time: str
    pm25: Pollutant | None = Field(default=None, alias="PM25")

    @field_validator("pm25", mode="before")
    @classmethod
    def _pm25_not_a_dict(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else None


class Station(BaseModel):
    """A station entry; ``AQILast`` is optional because stations sometimes report nothing."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    station_id: str = Field(alias="stationID", min_length=1)
    name_en: str | None = Field(default=None, alias="nameEN")
    lat: float | None = None
    lon: float | None = Field(default=None, alias="long")
    aqi_last: AQILast | None = Field(default=None, alias="AQILast")

    @field_validator("lat", "lon", mode="before")
    @classmethod
    def _coord(cls, v: Any) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("aqi_last", mode="before")
    @classmethod
    def _aqi_last_not_a_dict(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else None


class Observation(BaseModel):
    """A parsed PM2.5 observation. Natural key ``(station_id, ts)``; ``ts`` is UTC (KAN-17)."""

    model_config = ConfigDict(frozen=True)

    station_id: str
    ts: datetime
    pm25: float | None
    source: str = SOURCE

    @field_validator("ts")
    @classmethod
    def _utc_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() != timedelta(0):
            raise ValueError("ts must be timezone-aware UTC")
        return v


def parse_local_timestamp(date: str, time: str) -> datetime:
    """Convert Air4Thai local (Bangkok) ``YYYY-MM-DD`` and ``HH:MM[:SS]`` strings to aware UTC.

    Hours 0-24 are accepted (``24:00`` is midnight at the end of the day); anything else raises.
    """
    day = datetime.strptime(date.strip(), "%Y-%m-%d")  # noqa: DTZ007 - made aware below
    parts = time.strip().split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"unrecognised time {time!r}")
    hh, mm = int(parts[0]), int(parts[1])
    ss = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= hh <= 24 and 0 <= mm < 60 and 0 <= ss < 60):
        raise ValueError(f"time out of range: {time!r}")
    local = day.replace(tzinfo=BANGKOK_TZ) + timedelta(hours=hh, minutes=mm, seconds=ss)
    return local.astimezone(UTC)


def parse_response(payload: Any) -> list[Observation]:
    """Validate a decoded ``getNewAQI_JSON`` payload into observations.

    Raises ``Air4ThaiError`` only if the top-level shape is unusable. Per-station problems are
    logged and the station is skipped; stations without a reading are skipped quietly.
    Schema drift is not silent: if stations were returned but none parsed, ``Air4ThaiError`` is
    raised, and if every parsed PM2.5 value is ``None`` an ERROR is logged so the collector
    (KAN-18) can alert on it.
    """
    stations = payload.get("stations") if isinstance(payload, dict) else None
    if not isinstance(stations, list):
        raise Air4ThaiError("response has no 'stations' list")

    observations: list[Observation] = []
    for index, raw in enumerate(stations):
        try:
            station = Station.model_validate(raw)
            last = station.aqi_last
            if last is None:
                logger.info("air4thai: station %s has no AQILast, skipping", station.station_id)
                continue
            observations.append(
                Observation(
                    station_id=station.station_id,
                    ts=parse_local_timestamp(last.date, last.time),
                    pm25=last.pm25.value if last.pm25 else None,
                )
            )
        except (ValidationError, ValueError) as exc:
            logger.warning("air4thai: invalid station entry #%d skipped: %s", index, exc)
    if stations and not observations:
        raise Air4ThaiError(f"none of {len(stations)} station entries could be parsed")
    if observations and all(o.pm25 is None for o in observations):
        logger.error(
            "air4thai: all %d parsed stations have no PM2.5 value (possible schema drift)",
            len(observations),
        )
    return observations


# --- history (KAN-12 part 1 / KAN-34) ------------------------------------------------------


class HistoryRow(BaseModel):
    """One hourly history row, e.g. ``{"DATETIMEDATA": "2026-10-25 00:00:00", "PM25": 12.2}``.

    The time key has changed between observations (``DATETIMEDATA`` in recordings, ``DATETIC``
    in older notes), so several spellings are accepted. PM2.5 may be a number, a string, null
    or a sentinel; anything implausible becomes ``None``.
    """

    model_config = ConfigDict(extra="ignore")

    local_time: str = Field(
        validation_alias=AliasChoices("DATETIMEDATA", "DATETIC", "DATETIME", "datetime")
    )
    pm25: float | None = Field(default=None, validation_alias=AliasChoices("PM25", "pm25"))

    @field_validator("pm25", mode="before")
    @classmethod
    def _tolerant_number(cls, v: Any) -> float | None:
        if isinstance(v, dict):
            v = v.get("value")
        return _to_float(v)


def parse_history_timestamp(value: str) -> datetime:
    """``YYYY-MM-DD HH:MM[:SS]`` local Bangkok time (``T`` separator tolerated) to aware UTC."""
    day, sep, clock = value.strip().replace("T", " ").partition(" ")
    if not sep:
        raise ValueError(f"unrecognised history timestamp {value!r}")
    return parse_local_timestamp(day, clock)


def parse_history_response(payload: Any, station_id: str) -> list[Observation]:
    """Validate a decoded ``getHistoryData.php`` payload into observations for one station.

    Raises ``Air4ThaiError`` if the top-level shape is unusable or ``result`` is not ``OK``.
    An empty ``data`` list is a normal empty chunk (outside the retention window) and returns
    ``[]``. Rows with an unparseable time are skipped and counted in one WARNING; rows with no
    usable PM2.5 are kept with ``pm25=None`` (the caller decides whether to store gaps).
    Entries for other station ids are ignored.
    """
    if not isinstance(payload, dict):
        raise Air4ThaiError("history response is not an object")
    result = payload.get("result")
    if result is not None and str(result).upper() != "OK":
        raise Air4ThaiError(f"history result is {str(result)[:40]!r}")
    stations = payload.get("stations")
    if not isinstance(stations, list):
        raise Air4ThaiError("history response has no 'stations' list")

    observations: dict[datetime, Observation] = {}
    skipped = 0
    for entry in stations:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("stationID")
        if sid is not None and str(sid).strip() != station_id:
            continue
        rows = entry.get("data")
        for raw in rows if isinstance(rows, list) else []:
            try:
                row = HistoryRow.model_validate(raw)
                ts = parse_history_timestamp(row.local_time)
            except (ValidationError, ValueError, TypeError):
                skipped += 1
                continue
            observations[ts] = Observation(station_id=station_id, ts=ts, pm25=row.pm25)
    if skipped:
        logger.warning("air4thai history %s: %d malformed rows skipped", station_id, skipped)
    return sorted(observations.values(), key=lambda o: o.ts)


def history_chunks(
    start: date, end: date, days: int = HISTORY_CHUNK_DAYS
) -> list[tuple[date, date]]:
    """Split the inclusive local date range ``start..end`` into non-overlapping inclusive chunks
    of at most ``days`` days, oldest first."""
    if days < 1:
        raise ValueError("days must be >= 1")
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        last = min(cur + timedelta(days=days - 1), end)
        chunks.append((cur, last))
        cur = last + timedelta(days=1)
    return chunks


@dataclass
class HistoryResult:
    """Outcome of one station's history fetch. Chunks are ``(sdate, edate)`` in local dates."""

    station_id: str
    observations: list[Observation] = field(default_factory=list)
    chunks: list[tuple[date, date]] = field(default_factory=list)
    empty_chunks: list[tuple[date, date]] = field(default_factory=list)
    failed_chunks: list[tuple[date, date]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # True if a chunk failed with a retryable transport error (timeout, connection, 429/5xx)
    # after the client's own retries: the host is unhealthy, callers should stop asking it.
    transport_failure: bool = False

    @property
    def total_failure(self) -> bool:
        return bool(self.chunks) and len(self.failed_chunks) == len(self.chunks)


# --- hooks ---------------------------------------------------------------------------------

# Raw-response archive hook: called with the raw body and the UTC fetch time before parsing.
RawArchive = Callable[[bytes, datetime], None]
# Storage hook: the collector (KAN-18) supplies an upsert on (station_id, ts); see KAN-17.
ObservationSink = Callable[[Sequence[Observation]], None]
# History archive hook (KAN-74): raw body, UTC fetch time and the station id it was asked for.
HistoryArchive = Callable[[bytes, datetime, str], None]


def gzip_file_archive(directory: Path) -> RawArchive:
    """Return an archive hook writing each raw payload as ``<dir>/air4thai-<UTC stamp>.json.gz``."""

    def archive(body: bytes, fetched_at: datetime) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = fetched_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = directory / f"air4thai-{stamp}.json.gz"
        path.write_bytes(gzip.compress(body))

    return archive


def gzip_history_archive(directory: Path) -> HistoryArchive:
    """Archive hook writing each raw history body as
    ``<dir>/air4thai-history-<station>-<UTC stamp>.json.gz`` (public data, no request URL)."""

    def archive(body: bytes, fetched_at: datetime, station_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = fetched_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", station_id)[:40]
        path = directory / f"air4thai-history-{safe}-{stamp}.json.gz"
        path.write_bytes(gzip.compress(body))

    return archive


# --- client --------------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    if is_tls_verify_error(exc):  # a certificate problem will not fix itself on retry
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.TransportError)  # timeouts, connection errors


class Air4ThaiClient:
    """Fetch and parse Air4Thai readings.

    Station ids are passed in (configurable, see ``Settings.air4thai_station_ids``); with no ids
    all stations returned by the endpoint are kept. One request is made per call to be polite.

    ``ca_bundle`` (``Settings.air4thai_ca_bundle``, env ``AIR4THAI_CA_BUNDLE``) is an optional PEM
    file of extra CAs, e.g. an intermediate the server fails to send. Unset means the default
    trust store. Certificate verification is never disabled.

    ``timeout`` and ``ca_bundle`` apply only to a client created here; an injected ``client``
    keeps its own settings and is not closed by ``close()``. The endpoint is plain ``http://``,
    so responses can be tampered with in transit: payloads are range-validated, and
    ``AIR4THAI_URL`` should move to https if the host ever supports it. Raw archives default to
    ``data/raw/air4thai`` (git-ignored, see ``DEFAULT_ARCHIVE_DIR``); ``*.json.gz`` must never be
    committed.
    """

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        url: str = AIR4THAI_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        ca_bundle: str | Path | None = None,
        archive: RawArchive | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        wait_max: float = 30.0,
        history_url: str = AIR4THAI_HISTORY_URL,
        history_archive: HistoryArchive | None = None,
    ) -> None:
        if client is not None and ca_bundle:
            raise ValueError("ca_bundle applies only to a client created by Air4ThaiClient")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, verify=build_verify(ca_bundle)
        )
        self._url = url
        self._history_url = history_url
        self._archive = archive
        self._history_archive = history_archive
        self._fetch = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=1, max=wait_max),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(self._fetch_once)

    def _fetch_once(self, url: str | None = None, params: dict[str, str] | None = None) -> bytes:
        response = self._client.get(url or self._url, params=params)
        response.raise_for_status()
        return response.content

    def fetch(self, station_ids: Iterable[str] = ()) -> list[Observation]:
        """Fetch the latest readings, archive the raw body, and return validated observations.

        Warning (KAN-74): the ``PM25.value`` of this feed is PCD's 24 h rolling average, not an
        hourly value. Do not store it as ``observations.pm25``; use ``fetch_history`` for that.
        """
        body = self._fetch()
        fetched_at = datetime.now(UTC)
        if self._archive is not None:
            try:
                self._archive(body, fetched_at)
            except Exception:  # an archive problem must never lose the observations
                logger.exception("air4thai: raw archive failed; continuing")
        try:
            payload = httpx.Response(200, content=body).json()
        except ValueError as exc:
            raise Air4ThaiError("response is not valid JSON") from exc
        observations = parse_response(payload)
        wanted = {s.strip() for s in station_ids if s.strip()}
        if wanted:
            missing = wanted - {o.station_id for o in observations}
            if missing:
                logger.warning("air4thai: no reading for stations %s", sorted(missing))
            observations = [o for o in observations if o.station_id in wanted]
        return observations

    def ingest(self, sink: ObservationSink, station_ids: Iterable[str] = ()) -> int:
        """Fetch and hand observations to the storage hook; returns the number delivered."""
        observations = self.fetch(station_ids)
        if observations:
            sink(observations)
        return len(observations)

    def fetch_history(
        self,
        station_id: str,
        start: datetime,
        end: datetime,
        *,
        chunk_days: int = HISTORY_CHUNK_DAYS,
        pause: float = HISTORY_PAUSE_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> HistoryResult:
        """Hourly PM2.5 history for one station between aware ``start`` and ``end`` (inclusive).

        The UTC range is converted to Bangkok local dates and requested in chunks of at most
        ``chunk_days`` days (``stime=00``, ``etime=23``), pausing ``pause`` seconds between
        requests. Each chunk gets the client's retries; a chunk that still fails is recorded in
        ``failed_chunks`` and the remaining chunks are still fetched. Returned observations are
        deduplicated on ``ts``, limited to ``start..end`` and sorted; ``pm25`` may be ``None``.
        Raw bodies go to ``history_archive`` when one is set (the ``archive`` hook is for the
        latest-reading feed).
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        result = HistoryResult(station_id=station_id)
        result.chunks = history_chunks(
            start.astimezone(BANGKOK_TZ).date(), end.astimezone(BANGKOK_TZ).date(), chunk_days
        )
        by_ts: dict[datetime, Observation] = {}
        for index, (sdate, edate) in enumerate(result.chunks):
            if index and pause > 0:
                sleep(pause)
            params = {
                "stationID": station_id,
                "param": "PM25",
                "type": "hr",
                "sdate": sdate.isoformat(),
                "edate": edate.isoformat(),
                "stime": "00",
                "etime": "23",
            }
            try:
                body = self._fetch(self._history_url, params)
                if self._history_archive is not None:
                    try:
                        self._history_archive(body, datetime.now(UTC), station_id)
                    except Exception:  # an archive problem must never lose the observations
                        logger.exception("air4thai: history archive failed; continuing")
                try:
                    payload = httpx.Response(200, content=body).json()
                except ValueError as exc:
                    raise Air4ThaiError("history response is not valid JSON") from exc
                chunk = parse_history_response(payload, station_id)
            except (httpx.HTTPError, Air4ThaiError) as exc:
                # Type and our own message only: httpx messages carry the full request URL.
                detail = str(exc) if isinstance(exc, Air4ThaiError) else type(exc).__name__
                result.failed_chunks.append((sdate, edate))
                if _is_retryable(exc):
                    result.transport_failure = True
                result.errors.append(f"{sdate}..{edate}: {detail}")
                logger.error(
                    "air4thai history %s %s..%s failed: %s", station_id, sdate, edate, detail
                )
                continue
            if not chunk:
                result.empty_chunks.append((sdate, edate))
                logger.info("air4thai history %s %s..%s: empty chunk", station_id, sdate, edate)
            for obs in chunk:
                if start <= obs.ts <= end:
                    by_ts[obs.ts] = obs
        result.observations = sorted(by_ts.values(), key=lambda o: o.ts)
        return result

    def close(self) -> None:
        """Close the HTTP client only if this object created it."""
        if self._owns_client:
            self._client.close()
