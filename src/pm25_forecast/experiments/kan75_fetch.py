"""KAN-75: fetch Open-Meteo Previous Runs forecasts (lead ~24 h / ~48 h), one request per station.

Raw responses are cached under data/raw/openmeteo/ (git-ignored) together with the exact URLs.
Offline experiment code, not part of the serving path. Free non-commercial use, no key.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

HOST = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"
STATION_LOCATIONS = {
    "59t": (13.783185, 100.540489),
    "61t": (13.769634, 100.614567),
    "bkp100t": (13.785212, 100.669183),
    "bkp101t": (13.859911, 100.704002),
    "02t": (13.732846, 100.487662),
}
VARS = (
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "temperature_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
)
LEADS = ("previous_day1", "previous_day2")
START, END = "2026-07-30", "2026-10-26"  # fetch window (UTC dates); analysis cut is applied later
RAW_DIR = Path("data/raw/openmeteo")


def params(lat: float, lon: float) -> dict[str, str]:
    hourly = ",".join(f"{v}_{lead}" for v in VARS for lead in LEADS)
    return {
        "latitude": f"{lat}",
        "longitude": f"{lon}",
        "hourly": hourly,
        "models": MODEL,
        "start_date": START,
        "end_date": END,
        "timezone": "UTC",
        "timeformat": "unixtime",
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    with httpx.Client(timeout=60, headers={"User-Agent": "pm25-forecast-kan75-poc"}) as c:
        for st, (lat, lon) in STATION_LOCATIONS.items():
            out = RAW_DIR / f"{st}.json"
            if out.exists():
                continue
            r = c.get(HOST, params=params(lat, lon))
            r.raise_for_status()
            out.write_bytes(r.content)
            log.append(
                {
                    "station": st,
                    "url": str(r.request.url),
                    "fetched_utc": datetime.now(UTC).isoformat(),
                }
            )
            print(st, r.status_code, len(r.content))
    if log:
        (RAW_DIR / "requests.jsonl").open("a").write("\n".join(json.dumps(x) for x in log) + "\n")


if __name__ == "__main__":
    main()
