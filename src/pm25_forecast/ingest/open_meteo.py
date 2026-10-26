"""Open-Meteo client: weather forecast, historical weather and CAMS PM2.5 forecast.

Data attribution: Open-Meteo (https://open-meteo.com/), CC BY 4.0. The free tier is for
non-commercial use only; keep request volume low (one request per location per run).

Everything returned by the API is untrusted data. It is validated, never executed or used to
build requests, and gaps, renamed or missing variables and sentinel values are tolerated.
All timestamps are UTC and timezone-aware.

Forecast vintage: the API response does not say which model run produced it. For forecast
endpoints ``issue_time`` is therefore the time we retrieved the response. That is an upper bound
on the true run time, so features filtered on ``issue_time <= prediction_time`` cannot leak
future information (it may only make a forecast look slightly older than it is).
"""

import contextlib
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError
from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter
from tenacity.retry import retry_if_exception_type
from tenacity.wait import wait_base

logger = logging.getLogger(__name__)

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0, non-commercial use)."

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

DEFAULT_WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",
)
CAMS_VARIABLE = "pm2_5"
# Bangkok is outside the CAMS Europe domain, so ask for the global model explicitly.
CAMS_DOMAIN = "cams_global"

_VARIABLE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_SENTINEL_AT_OR_BELOW = -999.0


class OpenMeteoError(Exception):
    """Request failed or the response could not be used."""


class OpenMeteoSchemaError(OpenMeteoError):
    """The response was not in the expected shape (not retried)."""


class _RetryableError(OpenMeteoError):
    """Transport error, timeout, HTTP 429 or 5xx (retried with backoff)."""


class RawResponse(BaseModel):
    """A raw API response handed to the archive hook before parsing."""

    source: str
    url: str
    params: dict[str, str]
    retrieved_at: AwareDatetime
    body: str


class HourlyRow(BaseModel):
    valid_time: AwareDatetime
    values: dict[str, float | None]


class HourlyForecast(BaseModel):
    """Validated hourly series for one location."""

    source: str
    latitude: float
    longitude: float
    retrieved_at: AwareDatetime
    issue_time: AwareDatetime | None = None  # forecast endpoints only; see module docstring
    rows: list[HourlyRow]
    missing_variables: list[str] = []


class CamsSnapshotRow(BaseModel):
    """One append-only CAMS forecast value: never overwrite a past (issue_time, valid_time)."""

    model_config = ConfigDict(frozen=True)

    location_id: str
    forecast_issue_time: AwareDatetime
    valid_time: AwareDatetime
    variable: str
    value: float


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: bool = False
    reason: str | None = None
    hourly: dict[str, Any] | None = None


def _parse_time(value: Any) -> datetime | None:
    """Unix seconds or ISO 8601 string to aware UTC; naive strings are taken as UTC."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(value, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def _clean_value(value: Any) -> float | None:
    """Numbers only; null, NaN/inf, strings and -999-style sentinels become None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= _SENTINEL_AT_OR_BELOW:
        return None
    return number


def parse_hourly(payload: Any, variables: Sequence[str]) -> tuple[list[HourlyRow], list[str]]:
    """Validate an Open-Meteo payload; return (rows, requested variables that are missing)."""
    try:
        envelope = _Envelope.model_validate(payload)
    except ValidationError as exc:
        raise OpenMeteoSchemaError(
            f"unexpected response shape: {exc.error_count()} errors"
        ) from exc
    if envelope.error:
        reason = (envelope.reason or "unknown")[:200]
        raise OpenMeteoError(f"API reported an error: {reason}")
    hourly = envelope.hourly
    times = hourly.get("time") if hourly else None
    if not isinstance(times, list):
        raise OpenMeteoSchemaError("response has no hourly.time list")

    columns: dict[str, list[Any]] = {}
    missing: list[str] = []
    for name in variables:
        column = hourly.get(name) if hourly else None
        if not isinstance(column, list):
            missing.append(name)
            continue
        if len(column) != len(times):
            logger.warning("length mismatch for %s: %d vs %d times", name, len(column), len(times))
            column = (column + [None] * len(times))[: len(times)]
        columns[name] = column
    if missing:
        logger.warning("variables missing from response: %s", ", ".join(missing))

    rows: list[HourlyRow] = []
    for i, raw_time in enumerate(times):
        valid_time = _parse_time(raw_time)
        if valid_time is None:
            logger.warning("dropping row %d with unparseable time", i)
            continue
        rows.append(
            HourlyRow(
                valid_time=valid_time,
                values={name: _clean_value(col[i]) for name, col in columns.items()},
            )
        )
    rows.sort(key=lambda r: r.valid_time)
    return rows, missing


def to_snapshot_rows(
    forecast: HourlyForecast, location_id: str, variable: str = CAMS_VARIABLE
) -> list[CamsSnapshotRow]:
    """Turn a forecast into append-only (location, issue_time, valid_time, value) rows."""
    if forecast.issue_time is None:
        raise ValueError("forecast has no issue_time; only forecast endpoints can be snapshotted")
    issue_time = forecast.issue_time
    return [
        CamsSnapshotRow(
            location_id=location_id,
            forecast_issue_time=issue_time,
            valid_time=row.valid_time,
            variable=variable,
            value=value,
        )
        for row in forecast.rows
        if (value := row.values.get(variable)) is not None
    ]


class OpenMeteoClient:
    """Thin, defensive client. One instance per run; call `close()` if it owns the HTTP client."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        timeout: float = 15.0,
        max_attempts: int = 4,
        retry_wait: wait_base | None = None,
        archive: Callable[[RawResponse], None] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=timeout)
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._wait = retry_wait or wait_exponential_jitter(initial=1, max=30)
        self._archive = archive
        self._now = now

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def weather_forecast(
        self,
        latitude: float,
        longitude: float,
        variables: Sequence[str] = DEFAULT_WEATHER_VARIABLES,
        *,
        past_days: int = 0,
        forecast_days: int = 2,
    ) -> HourlyForecast:
        extra = {"past_days": str(past_days), "forecast_days": str(forecast_days)}
        return self._fetch("weather_forecast", FORECAST_URL, latitude, longitude, variables, extra)

    def cams_forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        past_days: int = 0,
        forecast_days: int = 2,
    ) -> HourlyForecast:
        """CAMS PM2.5 forecast; `issue_time` is set for the vintage snapshot."""
        extra = {
            "domains": CAMS_DOMAIN,
            "past_days": str(past_days),
            "forecast_days": str(forecast_days),
        }
        return self._fetch(
            "cams_forecast", AIR_QUALITY_URL, latitude, longitude, [CAMS_VARIABLE], extra
        )

    def historical_weather(
        self,
        latitude: float,
        longitude: float,
        start: date,
        end: date,
        variables: Sequence[str] = DEFAULT_WEATHER_VARIABLES,
    ) -> HourlyForecast:
        extra = {"start_date": start.isoformat(), "end_date": end.isoformat()}
        return self._fetch(
            "historical_weather",
            HISTORICAL_WEATHER_URL,
            latitude,
            longitude,
            variables,
            extra,
            is_forecast=False,
        )

    def historical_forecast(
        self,
        latitude: float,
        longitude: float,
        start: date,
        end: date,
        variables: Sequence[str],
        *,
        url: str = HISTORICAL_FORECAST_URL,
    ) -> HourlyForecast:
        """Historical Forecast API (also reachable with `previous_runs`; used for coverage)."""
        extra = {"start_date": start.isoformat(), "end_date": end.isoformat()}
        return self._fetch("historical_forecast", url, latitude, longitude, variables, extra, False)

    def previous_runs(
        self,
        latitude: float,
        longitude: float,
        start: date,
        end: date,
        variables: Sequence[str],
    ) -> HourlyForecast:
        """Previous Runs API; ask for names like `temperature_2m_previous_day1`."""
        return self.historical_forecast(
            latitude, longitude, start, end, variables, url=PREVIOUS_RUNS_URL
        )

    def _fetch(
        self,
        source: str,
        url: str,
        latitude: float,
        longitude: float,
        variables: Sequence[str],
        extra: Mapping[str, str],
        is_forecast: bool = True,
    ) -> HourlyForecast:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("latitude/longitude out of range")
        if not variables or not all(_VARIABLE_RE.match(v) for v in variables):
            raise ValueError("variables must be a non-empty list of simple snake_case names")
        params = {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "hourly": ",".join(variables),
            "timezone": "UTC",
            "timeformat": "unixtime",
            **extra,
        }
        retrieved_at = self._now()
        payload = self._get_json(source, url, params, retrieved_at)
        rows, missing = parse_hourly(payload, variables)
        return HourlyForecast(
            source=source,
            latitude=latitude,
            longitude=longitude,
            retrieved_at=retrieved_at,
            issue_time=retrieved_at if is_forecast else None,
            rows=rows,
            missing_variables=missing,
        )

    def _get_json(
        self, source: str, url: str, params: dict[str, str], retrieved_at: datetime
    ) -> Any:
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception_type(_RetryableError),
            reraise=True,
        )
        body = retrying(self._send, url, params)
        if self._archive is not None:
            try:
                self._archive(
                    RawResponse(
                        source=source, url=url, params=params, retrieved_at=retrieved_at, body=body
                    )
                )
            except Exception:  # the archive must never break ingestion
                logger.exception("raw-response archive hook failed for %s", source)
        try:
            return httpx.Response(200, content=body).json()
        except ValueError as exc:
            raise OpenMeteoSchemaError("response is not valid JSON") from exc

    def _send(self, url: str, params: dict[str, str]) -> str:
        try:
            response = self._http.get(url, params=params, timeout=self._timeout)
        except httpx.TransportError as exc:  # includes timeouts
            raise _RetryableError(f"transport error: {type(exc).__name__}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            reason = ""
            with contextlib.suppress(ValueError, AttributeError):
                reason = str(response.json().get("reason", ""))[:200]
            raise OpenMeteoError(f"HTTP {response.status_code}: {reason}")
        return response.text
