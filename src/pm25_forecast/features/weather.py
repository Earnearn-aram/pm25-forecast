"""KAN-88 (E11) weather-forecast features: an add-on feature group, shared by training and serving.

Source: Open-Meteo Previous Runs API, ONE model throughout (``ecmwf_ifs025``), the lead-specific
variables ``<var>_previous_day1`` (forecast of that hour from the run about 24 h earlier) and, for
a sensitivity only, ``_previous_day2`` (about 48 h). ERA5 or any reanalysis/analysis value is never
a feature (leakage: it is not what a live forecast can see).

As-of rule (docs/experiments/protocol.md section 2): the value for valid hour ``u`` with lead ``L``
days has the NOMINAL issue time ``u - 24 h * L``. A value is used for origin ``t`` only when that
nominal issue time is ``<= t``. Caveat (not enforced here, stated in every report): a real run is
published some hours after its init time, so ``previous_day1`` at the target hour of a +24 h origin
(nominal issue time == origin) is a slightly optimistic stand-in for what is live at ``t``;
``previous_day2`` (issued 24 h before the origin) is the strictly available variant.

Boundary-layer height is requested but Open-Meteo returns null for it in previous-runs data, so
it is not a feature; ``shortwave_radiation`` (driver of daytime mixing) stands in. Missing
inputs give ``None``; nothing is imputed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from pm25_forecast.models.baselines import require_utc

HOUR = timedelta(hours=1)
MODEL = "ecmwf_ifs025"
VARIABLES: tuple[str, ...] = (
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "temperature_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
)
LEADS: tuple[str, ...] = ("previous_day1", "previous_day2")
LEAD_DAYS = {"previous_day1": 1, "previous_day2": 2}
SUM_WINDOW_H = 24

WEATHER_FEATURE_NAMES: tuple[str, ...] = (
    "wx_wind_speed",
    "wx_wind_dir_sin",
    "wx_wind_dir_cos",
    "wx_rh",
    "wx_temp",
    "wx_precip",
    "wx_pressure",
    "wx_cloud",
    "wx_shortwave",
    "wx_precip_sum_24h",
    "wx_wind_speed_mean_24h",
    "wx_shortwave_sum_24h",
)
WEATHER_FEATURE_SETS: dict[str, tuple[str, ...]] = {"wx_v1": WEATHER_FEATURE_NAMES}


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hourly: dict[str, list[float | None]]


class WeatherSeries:
    """station -> lead -> UTC hour -> variable -> value (None = missing)."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[datetime, dict[str, float | None]]]] = {}

    def add(self, station: str, payload: Mapping[str, object]) -> None:
        """Validate one Open-Meteo response (``timeformat=unixtime``) and index it."""
        try:
            parsed = _Payload.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"bad Open-Meteo payload for {station}: {exc}") from exc
        times = parsed.hourly.get("time")
        if not times or any(t is None for t in times):
            raise ValueError(f"payload for {station} has no usable 'time' axis")
        hours = [datetime.fromtimestamp(int(t), UTC) for t in times]  # type: ignore[arg-type]
        for lead in LEADS:
            by_hour: dict[datetime, dict[str, float | None]] = {h: {} for h in hours}
            for var in VARIABLES:
                column = parsed.hourly.get(f"{var}_{lead}")
                if column is None or len(column) != len(hours):
                    continue  # tolerate a missing/odd column: its features become None
                for h, v in zip(hours, column, strict=True):
                    by_hour[h][var] = v if v is not None and math.isfinite(v) else None
            self._data.setdefault(station, {})[lead] = by_hour

    def value(self, station: str, lead: str, hour: datetime, var: str) -> float | None:
        return self._data.get(station, {}).get(lead, {}).get(hour, {}).get(var)

    @property
    def stations(self) -> tuple[str, ...]:
        return tuple(self._data)


def load_weather_cache(directory: str | Path) -> WeatherSeries:
    """Read ``<station>.json`` files written by ``experiments.kan88_fetch_weather``."""
    series = WeatherSeries()
    for path in sorted(Path(directory).expanduser().glob("*.json")):
        series.add(path.stem, json.loads(path.read_text()))
    if not series.stations:
        raise FileNotFoundError(f"no weather cache files in {directory}")
    return series


def weather_features(
    series: WeatherSeries,
    station_id: str,
    prediction_time: datetime,
    valid_time: datetime,
    lead: str = "previous_day1",
) -> dict[str, float | None]:
    """The ``wx_v1`` features for one station, origin and target hour (all keys always present)."""
    origin, valid = require_utc(prediction_time), require_utc(valid_time)
    valid = valid.replace(minute=0, second=0, microsecond=0)
    lag = LEAD_DAYS[lead] * 24 * HOUR

    def at(hour: datetime, var: str) -> float | None:
        if hour - lag > origin:  # nominal issue time after the origin: not available
            return None
        return series.value(station_id, lead, hour, var)

    out: dict[str, float | None] = dict.fromkeys(WEATHER_FEATURE_NAMES)
    out["wx_wind_speed"] = at(valid, "wind_speed_10m")
    direction = at(valid, "wind_direction_10m")
    if direction is not None:
        out["wx_wind_dir_sin"] = math.sin(math.radians(direction))
        out["wx_wind_dir_cos"] = math.cos(math.radians(direction))
    out["wx_rh"] = at(valid, "relative_humidity_2m")
    out["wx_temp"] = at(valid, "temperature_2m")
    out["wx_precip"] = at(valid, "precipitation")
    out["wx_pressure"] = at(valid, "surface_pressure")
    out["wx_cloud"] = at(valid, "cloud_cover")
    out["wx_shortwave"] = at(valid, "shortwave_radiation")

    def window(var: str) -> list[float] | None:  # the 24 h ending at valid, all present or None
        vals = [at(valid - k * HOUR, var) for k in range(SUM_WINDOW_H)]
        return None if any(v is None for v in vals) else vals  # type: ignore[return-value]

    if (w := window("precipitation")) is not None:
        out["wx_precip_sum_24h"] = sum(w)
    if (w := window("wind_speed_10m")) is not None:
        out["wx_wind_speed_mean_24h"] = sum(w) / len(w)
    if (w := window("shortwave_radiation")) is not None:
        out["wx_shortwave_sum_24h"] = sum(w)
    return out


def weather_feature_set(name: str) -> tuple[str, ...]:
    try:
        return WEATHER_FEATURE_SETS[name]
    except KeyError:
        raise KeyError(
            f"unknown weather set {name!r}; known: {sorted(WEATHER_FEATURE_SETS)}"
        ) from None
