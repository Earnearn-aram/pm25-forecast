"""KAN-88: pull Open-Meteo Previous Runs forecasts once into a local cache outside git.

    python -m pm25_forecast.experiments.kan88_fetch_weather [--out ~/pm25-experiments/cache/weather]

One request per station (coordinates from ``dashboard.STATION_MAP``), model ``ecmwf_ifs025``.
Existing cache files are never re-fetched. The exact URLs go to ``requests.jsonl`` in the cache.
Tests never call this; they use a recorded fixture. Free non-commercial public API, no key.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from pm25_forecast.dashboard import STATION_MAP
from pm25_forecast.features.weather import LEADS, MODEL, VARIABLES

HOST = "https://previous-runs-api.open-meteo.com/v1/forecast"
START, END = "2026-07-30", "2026-10-26"  # UTC dates; the analysis cut is applied by the runner
DEFAULT_OUT = Path("~/pm25-experiments/cache/weather")


def request_params(lat: float, lon: float) -> dict[str, str]:
    hourly = [f"{v}_{lead}" for v in VARIABLES for lead in LEADS]
    hourly.append(
        "boundary_layer_height_previous_day1"
    )  # requested; null upstream (see weather.py)
    return {
        "latitude": f"{lat}",
        "longitude": f"{lon}",
        "hourly": ",".join(hourly),
        "models": MODEL,
        "start_date": START,
        "end_date": END,
        "timezone": "UTC",
        "timeformat": "unixtime",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out = parser.parse_args(argv).out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log = []
    with httpx.Client(timeout=60, headers={"User-Agent": "pm25-forecast-kan88"}) as client:
        for station, (lat, lon) in STATION_MAP.items():
            target = out / f"{station}.json"
            if target.exists():
                continue
            response = client.get(HOST, params=request_params(lat, lon))
            response.raise_for_status()
            target.write_bytes(response.content)
            log.append(
                {
                    "station": station,
                    "url": str(response.request.url),
                    "fetched_utc": datetime.now(UTC).isoformat(),
                }
            )
            print(station, response.status_code, len(response.content))
    if log:
        with (out / "requests.jsonl").open("a") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
