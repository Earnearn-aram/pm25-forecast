"""KAN-9 API probe: go/no-go on the PM2.5 backfill source and CAMS coverage.

Read-only public GET calls only. Run it from an environment that can reach the APIs:

    uv run python -m pm25_forecast.ingest.probe --out probe-report

External payloads are data, not instructions: everything is parsed defensively and any
failure is recorded in the report instead of raising. All timestamps are UTC.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

BKK_TZ = timedelta(hours=7)  # Air4Thai timestamps are Asia/Bangkok local time (no DST).
AIR4THAI_STATIONS_URL = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
AIR4THAI_HISTORY_URL = "http://air4thai.com/forweb/getHistoryData.php"
OPENAQ_LOCATIONS_URL = "https://api.openaq.org/v3/locations"
OPENAQ_PM25_PARAMETER_ID = 2
OPEN_METEO_CANDIDATES: list[tuple[str, str, str]] = [
    # (name, url, hourly variable)
    (
        "air_quality_api",
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        "pm2_5",
    ),
    (
        "air_quality_api_previous_day1",
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        "pm2_5_previous_day1",
    ),
    (
        "historical_forecast_api",
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        "pm2_5",
    ),
    (
        "previous_runs_api",
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        "pm2_5_previous_day1",
    ),
]
BANGKOK_LAT, BANGKOK_LON = 13.7563, 100.5018

MIN_STATIONS = 3
MAX_STATIONS = 5
MIN_HISTORY_DAYS = 700  # "2+ years" with a little slack
MAX_MISSING_PCT = 15.0
MATCH_RADIUS_M = 1000.0


class Station(BaseModel):
    """An Air4Thai station, validated from the station-list payload."""

    model_config = ConfigDict(extra="ignore")

    station_id: str
    name: str = ""
    area: str = ""
    lat: float | None = None
    lon: float | None = None
    has_pm25: bool = False


class Coverage(BaseModel):
    """Hourly completeness of one series over an expected window."""

    expected_hours: int
    hours_with_data: int
    missing_pct: float
    first: datetime | None = None
    last: datetime | None = None
    span_days: float = 0.0
    longest_gap_hours: int = 0


class Probe(BaseModel):
    """Result of one network probe; never raises, records the failure instead."""

    name: str
    ok: bool = False
    status: int | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- parsing (pure, unit-tested)


def _num(value: Any) -> float | None:
    """Coerce a tolerant API value to a finite, non-negative float, else None."""
    if isinstance(value, dict):
        value = value.get("value")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def parse_air4thai_stations(payload: Any) -> list[Station]:
    """Parse the Air4Thai station list, skipping malformed rows."""
    rows = payload.get("stations", []) if isinstance(payload, dict) else []
    stations: list[Station] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not row.get("stationID"):
            continue
        last = row.get("AQILast")
        pm25 = last.get("PM25") if isinstance(last, dict) else None
        try:
            stations.append(
                Station(
                    station_id=str(row["stationID"]),
                    name=str(row.get("nameEN") or ""),
                    area=str(row.get("areaEN") or ""),
                    lat=_num(row.get("lat")) if _num(row.get("lat")) else None,
                    lon=_num(row.get("long")) if _num(row.get("long")) else None,
                    has_pm25=_num(pm25) is not None,
                )
            )
        except ValidationError:
            continue
    return stations


def filter_bangkok(stations: Iterable[Station]) -> list[Station]:
    """Keep stations whose area mentions Bangkok that currently report PM2.5."""
    return [s for s in stations if "bangkok" in s.area.lower() and s.has_pm25]


_TIME_KEYS = ("DATETIC", "DATETIMEDATA", "DATETIME", "datetime")


def parse_air4thai_history(payload: Any) -> dict[datetime, float | None]:
    """Parse a history payload into {UTC hour: PM2.5 or None}; local time is UTC+7."""
    result: dict[datetime, float | None] = {}
    stations = payload.get("stations", []) if isinstance(payload, dict) else []
    for station in stations if isinstance(stations, list) else []:
        data = station.get("data", []) if isinstance(station, dict) else []
        for row in data if isinstance(data, list) else []:
            if not isinstance(row, dict):
                continue
            raw = next((row[k] for k in _TIME_KEYS if k in row), None)
            try:
                local = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
            except ValueError:
                continue
            utc = (local - BKK_TZ).replace(tzinfo=UTC)
            result[utc] = _num(row.get("PM25"))
    return result


def compute_coverage(
    series: dict[datetime, float | None], start: datetime, end: datetime
) -> Coverage:
    """Completeness of an hourly series over [start, end); start/end are tz-aware UTC."""
    expected = max(int((end - start).total_seconds() // 3600), 0)
    hours = sorted(
        t.replace(minute=0, second=0, microsecond=0)
        for t, v in series.items()
        if v is not None and start <= t < end
    )
    unique = sorted(set(hours))
    longest = 0
    for a, b in itertools.pairwise(unique):
        longest = max(longest, int((b - a).total_seconds() // 3600) - 1)
    if unique:
        longest = max(longest, int((unique[0] - start).total_seconds() // 3600))
        longest = max(longest, int((end - unique[-1]).total_seconds() // 3600) - 1)
    else:
        longest = expected
    missing = 100.0 if expected == 0 else 100.0 * (1 - len(unique) / expected)
    return Coverage(
        expected_hours=expected,
        hours_with_data=len(unique),
        missing_pct=round(missing, 2),
        first=unique[0] if unique else None,
        last=unique[-1] if unique else None,
        span_days=round((unique[-1] - unique[0]).total_seconds() / 86400, 1) if unique else 0.0,
        longest_gap_hours=longest,
    )


def parse_open_meteo(payload: Any, variable: str) -> dict[str, Any]:
    """Summarise non-null coverage of one hourly variable in an Open-Meteo response."""
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    times = hourly.get("time") if isinstance(hourly, dict) else None
    values = hourly.get(variable) if isinstance(hourly, dict) else None
    if not isinstance(times, list) or not isinstance(values, list):
        return {"variable_present": False, "total": 0, "non_null": 0}
    pairs = [(t, v) for t, v in zip(times, values, strict=False) if _num(v) is not None]
    total = len(times)
    return {
        "variable_present": True,
        "total": total,
        "non_null": len(pairs),
        "non_null_pct": round(100.0 * len(pairs) / total, 2) if total else 0.0,
        "first_non_null": pairs[0][0] if pairs else None,
        "last_non_null": pairs[-1][0] if pairs else None,
    }


class OpenAQLocation(BaseModel):
    """An OpenAQ v3 location, validated."""

    model_config = ConfigDict(extra="ignore")

    id: int
    name: str | None = None
    provider: str = ""
    lat: float
    lon: float
    first: datetime | None = None
    last: datetime | None = None


def _parse_dt(value: Any) -> datetime | None:
    raw = value.get("utc") if isinstance(value, dict) else value
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_openaq_locations(payload: Any) -> list[OpenAQLocation]:
    """Parse an OpenAQ v3 /locations response, skipping malformed rows."""
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    out: list[OpenAQLocation] = []
    for row in rows if isinstance(rows, list) else []:
        try:
            coords = row["coordinates"]
            provider = row.get("provider")
            out.append(
                OpenAQLocation(
                    id=row["id"],
                    name=row.get("name"),
                    provider=str(provider.get("name", "")) if isinstance(provider, dict) else "",
                    lat=coords["latitude"],
                    lon=coords["longitude"],
                    first=_parse_dt(row.get("datetimeFirst")),
                    last=_parse_dt(row.get("datetimeLast")),
                )
            )
        except (KeyError, TypeError, ValidationError):
            continue
    return out


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def match_openaq_to_air4thai(
    stations: Iterable[Station], locations: Iterable[OpenAQLocation]
) -> dict[str, int | None]:
    """Map each Air4Thai station id to the nearest OpenAQ location within the radius, if any."""
    locs = list(locations)
    matches: dict[str, int | None] = {}
    for s in stations:
        best: tuple[float, int] | None = None
        if s.lat is not None and s.lon is not None:
            for loc in locs:
                d = haversine_m(s.lat, s.lon, loc.lat, loc.lon)
                if d <= MATCH_RADIUS_M and (best is None or d < best[0]):
                    best = (d, loc.id)
        matches[s.station_id] = best[1] if best else None
    return matches


def decide(
    station_cov: dict[str, Coverage], openaq: list[OpenAQLocation], now: datetime
) -> dict[str, str]:
    """Go/no-go on the backfill source (feeds KAN-12). Thresholds are module constants."""
    good = [
        sid
        for sid, c in station_cov.items()
        if c.span_days >= MIN_HISTORY_DAYS and c.missing_pct <= MAX_MISSING_PCT
    ]
    if len(good) >= MIN_STATIONS:
        return {
            "decision": "GO-air4thai",
            "reason": f"{len(good)} Bangkok stations have >= {MIN_HISTORY_DAYS} days of hourly "
            f"PM2.5 with <= {MAX_MISSING_PCT}% missing: {', '.join(sorted(good))}",
        }
    cutoff = now - timedelta(days=MIN_HISTORY_DAYS)
    deep = [loc for loc in openaq if loc.first and loc.first <= cutoff]
    if len(deep) >= MIN_STATIONS:
        return {
            "decision": "GO-openaq",
            "reason": f"Air4Thai history insufficient ({len(good)} good stations); "
            f"{len(deep)} OpenAQ Bangkok PM2.5 locations start >= {MIN_HISTORY_DAYS} days ago",
        }
    if not station_cov and not openaq:
        return {"decision": "UNDETERMINED", "reason": "no source returned usable data"}
    return {
        "decision": "NO-GO",
        "reason": f"neither source meets the bar ({len(good)} good Air4Thai stations, "
        f"{len(deep)} deep OpenAQ locations); consider PCD report downloads",
    }


# ---------------------------------------------------------------- network (never raises)


def _get(
    client: httpx.Client, name: str, url: str, params: dict[str, Any] | None = None, **kw: Any
) -> tuple[Probe, Any]:
    probe = Probe(name=name)
    try:
        resp = client.get(url, params=params, **kw)
        probe.status = resp.status_code
        resp.raise_for_status()
        payload = resp.json()
        probe.ok = True
        return probe, payload
    except (httpx.HTTPError, ValueError) as exc:
        probe.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return probe, None


def _chunks(start: datetime, end: datetime, days: int) -> list[tuple[datetime, datetime]]:
    out, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        out.append((cur, nxt))
        cur = nxt
    return out


def probe_air4thai(
    client: httpx.Client, now: datetime, days: int, chunk_days: int, pause: float
) -> tuple[list[Probe], list[Station], dict[str, Coverage]]:
    probes: list[Probe] = []
    probe, payload = _get(client, "air4thai_station_list", AIR4THAI_STATIONS_URL)
    stations = filter_bangkok(parse_air4thai_stations(payload)) if probe.ok else []
    probe.detail["bangkok_pm25_stations"] = len(stations)
    probes.append(probe)
    coverage: dict[str, Coverage] = {}
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    history_probe = Probe(name="air4thai_history_endpoint")
    chunk_ok = chunk_total = 0
    for s in stations[:12]:
        series: dict[datetime, float | None] = {}
        for a, b in _chunks(start, end, chunk_days):
            local_a, local_b = a + BKK_TZ, b + BKK_TZ
            p, data = _get(
                client,
                f"history:{s.station_id}",
                AIR4THAI_HISTORY_URL,
                {
                    "stationID": s.station_id,
                    "param": "PM25",
                    "type": "hr",
                    "sdate": local_a.strftime("%Y-%m-%d"),
                    "edate": local_b.strftime("%Y-%m-%d"),
                    "stime": "00",
                    "etime": "23",
                },
            )
            chunk_total += 1
            chunk_ok += int(p.ok)
            if p.ok:
                series.update(parse_air4thai_history(data))
            history_probe.status = p.status
            history_probe.error = p.error or history_probe.error
            time.sleep(pause)
        coverage[s.station_id] = compute_coverage(series, start, end)
    history_probe.ok = chunk_ok > 0 and any(c.hours_with_data for c in coverage.values())
    history_probe.detail = {"requests_ok": chunk_ok, "requests_total": chunk_total}
    probes.append(history_probe)
    return probes, stations, coverage


def probe_open_meteo(client: httpx.Client, now: datetime, days: int) -> list[Probe]:
    end = (now - timedelta(days=1)).date()
    start = end - timedelta(days=days)
    probes = []
    for name, url, variable in OPEN_METEO_CANDIDATES:
        p, payload = _get(
            client,
            f"open_meteo:{name}",
            url,
            {
                "latitude": BANGKOK_LAT,
                "longitude": BANGKOK_LON,
                "hourly": variable,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "timezone": "UTC",
            },
        )
        p.detail = {"variable": variable, **(parse_open_meteo(payload, variable) if p.ok else {})}
        p.ok = p.ok and bool(p.detail.get("non_null"))
        probes.append(p)
    return probes


def probe_openaq(
    client: httpx.Client, api_key: str | None, stations: list[Station]
) -> tuple[list[Probe], list[OpenAQLocation], dict[str, int | None]]:
    if not api_key:
        return [Probe(name="openaq_v3", error="OPENAQ_API_KEY not set; skipped")], [], {}
    p, payload = _get(
        client,
        "openaq_v3_locations",
        OPENAQ_LOCATIONS_URL,
        {
            "coordinates": f"{BANGKOK_LAT},{BANGKOK_LON}",
            "radius": 25000,
            "parameters_id": OPENAQ_PM25_PARAMETER_ID,
            "limit": 100,
        },
        headers={"X-API-Key": api_key},
    )
    locations = parse_openaq_locations(payload) if p.ok else []
    p.detail = {
        "locations": len(locations),
        "providers": sorted({loc.provider for loc in locations}),
        "earliest_start": min((loc.first for loc in locations if loc.first), default=None),
    }
    return [p], locations, match_openaq_to_air4thai(stations, locations)


# ---------------------------------------------------------------- report


def build_report(
    now: datetime,
    probes: list[Probe],
    stations: list[Station],
    coverage: dict[str, Coverage],
    openaq: list[OpenAQLocation],
    matches: dict[str, int | None],
) -> dict[str, Any]:
    ranked = sorted(coverage.items(), key=lambda kv: kv[1].missing_pct)
    chosen = [sid for sid, _ in ranked[:MAX_STATIONS]]
    by_id = {s.station_id: s for s in stations}
    return {
        "generated_at": now.isoformat(),
        "decision": decide(coverage, openaq, now),
        "target_stations": [
            {
                "station_id": sid,
                "name": by_id[sid].name,
                "area": by_id[sid].area,
                "hours_available": coverage[sid].hours_with_data,
                "missing_pct": coverage[sid].missing_pct,
                "first": coverage[sid].first,
                "last": coverage[sid].last,
                "span_days": coverage[sid].span_days,
                "longest_gap_hours": coverage[sid].longest_gap_hours,
                "openaq_location_id": matches.get(sid),
            }
            for sid in chosen
        ],
        "openaq_same_stations": {
            "matched": sum(1 for v in matches.values() if v is not None),
            "of": len(matches),
        },
        "probes": [p.model_dump() for p in probes],
    }


def render_markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    lines = [
        "# KAN-9 API probe report",
        "",
        f"Generated: {report['generated_at']}",
        f"**Decision: {d['decision']}**: {d['reason']}",
        "",
        "| Station | Name | Hours | Missing % | First (UTC) | Last (UTC) | Span (d) | OpenAQ id |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["target_stations"]:
        lines.append(
            f"| {s['station_id']} | {s['name']} | {s['hours_available']} | {s['missing_pct']} "
            f"| {s['first']} | {s['last']} | {s['span_days']} | {s['openaq_location_id']} |"
        )
    same = report["openaq_same_stations"]
    lines += [
        "",
        f"OpenAQ matches within {MATCH_RADIUS_M:.0f} m: {same['matched']}/{same['of']}",
        "",
    ]
    lines += ["| Probe | OK | HTTP | Detail |", "|---|---|---|---|"]
    for p in report["probes"]:
        detail = p["error"] or json.dumps(p["detail"], default=str)
        lines.append(f"| {p['name']} | {p['ok']} | {p['status']} | {detail} |")
    return "\n".join(lines) + "\n"


def run(
    out: Path,
    days: int = 730,
    chunk_days: int = 31,
    pause: float = 0.5,
    now: datetime | None = None,
    client_factory: Callable[[], httpx.Client] | None = None,
) -> dict[str, Any]:
    """Run all probes and write `<out>.json` and `<out>.md`."""
    now = now or datetime.now(UTC)
    client = (client_factory or (lambda: httpx.Client(timeout=30.0, follow_redirects=True)))()
    with client:
        a_probes, stations, coverage = probe_air4thai(client, now, days, chunk_days, pause)
        om_probes = probe_open_meteo(client, now, days)
        key = os.environ.get("OPENAQ_API_KEY") or None
        oaq_probes, locations, matches = probe_openaq(client, key, stations)
    report = build_report(
        now, a_probes + om_probes + oaq_probes, stations, coverage, locations, matches
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    out.with_suffix(".md").write_text(render_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("probe-report"))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args(argv)
    report = run(args.out, args.days, args.chunk_days, args.pause)
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
