"""KAN-14: conservative, read-only rolling-origin evaluation. No network or fitting magic."""

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean

from pm25_forecast.models.baselines import HORIZONS_H, persistence_forecast, require_utc

HOUR = timedelta(hours=1)
BASELINES = ("persistence-24h", "seasonal-naive", "cams")


@dataclass(frozen=True)
class Observation:
    station_id: str
    ts: datetime
    value: float
    available_at: datetime


@dataclass(frozen=True)
class Vintage:
    station_id: str
    issue_time: datetime
    valid_time: datetime
    value: float
    available_at: datetime
    model: str = "cams"


@dataclass(frozen=True)
class Snapshot:
    """Only inputs that existed at prediction_time; training labels must also be mature."""

    prediction_time: datetime
    station_id: str
    observations: tuple[Observation, ...]
    vintages: tuple[Vintage, ...]

    def __post_init__(self):
        for row in self.observations:
            if row.ts > self.prediction_time or row.available_at > self.prediction_time:
                raise ValueError("observation/label after prediction_time")
        for row in self.vintages:
            if row.issue_time > self.prediction_time or row.available_at > self.prediction_time:
                raise ValueError("feature/vintage after prediction_time")


# Called afresh at every origin, with only mature training labels and as-of features.
# A fitting challenger must fit inside this callback, never prefit on the full evaluation window.
Predictor = Callable[[Snapshot, int], float | None]


def _value(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("PM2.5 must be finite and nonnegative")


def _validate(observations: Sequence[Observation], vintages: Sequence[Vintage]) -> None:
    seen = set()
    for row in observations:
        require_utc(row.ts)
        require_utc(row.available_at)
        _value(row.value)
        key = (row.station_id, row.ts)
        if key in seen:
            raise ValueError("duplicate observation: revision history is not supported")
        seen.add(key)
    seen.clear()
    for row in vintages:
        require_utc(row.issue_time)
        require_utc(row.valid_time)
        require_utc(row.available_at)
        _value(row.value)
        key = (row.station_id, row.issue_time, row.valid_time, row.model)
        if key in seen:
            raise ValueError("duplicate forecast vintage: preserve the first write")
        seen.add(key)


def _metrics(rows: list[dict], model: str, breaks: Sequence[float]) -> dict:
    if not rows:
        return {"n": 0, "mae": None, "rmse": None, "aqi_category_accuracy": None}
    errors = [r[model] - r["actual"] for r in rows]
    category = lambda v: sum(v > b for b in breaks)  # noqa: E731
    return {
        "n": len(rows),
        "mae": fmean(abs(e) for e in errors),
        "rmse": math.sqrt(fmean(e * e for e in errors)),
        "aqi_category_accuracy": fmean(category(r[model]) == category(r["actual"]) for r in rows),
    }


def evaluate(
    observations: Sequence[Observation],
    vintages: Sequence[Vintage],
    *,
    start: datetime,
    end: datetime,
    as_of: datetime,
    stations: Sequence[str],
    aqi_breakpoints: Sequence[float],
    challengers: dict[str, Predictor] | None = None,
    horizons: Sequence[int] = HORIZONS_H,
) -> dict:
    """Hourly expanding-window origins in [start,end); evaluation labels mature by as_of.

    Metrics use the SAME (station, prediction_time, valid_time, horizon) intersection for all
    models. Missing inputs are never imputed. AQI thresholds are caller-supplied concentration
    bands; hourly category accuracy is a proxy, not an official daily AQI calculation.
    """
    start, end, as_of = map(require_utc, (start, end, as_of))
    if start >= end or any(t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError("start/end must define a nonempty hourly interval")
    if not stations or len(set(stations)) != len(stations):
        raise ValueError("provide a fixed, unique station cohort")
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(type(h) is not int or h <= 0 for h in horizons)
    ):
        raise ValueError("horizons must be positive integer hours")
    if not aqi_breakpoints or any(not math.isfinite(b) or b < 0 for b in aqi_breakpoints):
        raise ValueError("provide finite nonnegative AQI concentration breakpoints")
    if list(aqi_breakpoints) != sorted(set(aqi_breakpoints)):
        raise ValueError("AQI breakpoints must be strictly increasing")
    challengers = challengers or {}
    if set(challengers) & set(BASELINES):
        raise ValueError("challenger names must not replace baselines")
    _validate(observations, vintages)
    models = (*BASELINES, *challengers)
    actuals = {(o.station_id, o.ts): o for o in observations if o.available_at <= as_of}
    rows: list[dict] = []
    missing: dict[str, int] = defaultdict(int)
    eligible = 0
    origin = start
    while origin < end:
        for station in stations:
            snapshot = Snapshot(
                origin,
                station,
                tuple(
                    o
                    for o in observations
                    if o.station_id == station and o.ts <= origin and o.available_at <= origin
                ),
                tuple(
                    v
                    for v in vintages
                    if v.station_id == station
                    and v.issue_time <= origin
                    and v.available_at <= origin
                ),
            )
            persisted = persistence_forecast(
                [(o.ts, o.value) for o in snapshot.observations], origin, horizons
            )
            persistence = {f.horizon_h: f.pm25_pred for f in persisted}
            history = {o.ts: o.value for o in snapshot.observations}
            for horizon in horizons:
                target = origin + horizon * HOUR
                actual = actuals.get((station, target))
                if target > as_of or actual is None:
                    missing["actual"] += 1
                    continue
                eligible += 1
                # Same hour on the previous day (or repeat 24h cycle for longer horizons).
                seasonal_time = target
                while seasonal_time > origin:
                    seasonal_time -= 24 * HOUR
                predictions = {
                    "persistence-24h": persistence.get(horizon),
                    "seasonal-naive": history.get(seasonal_time),
                }
                cams = [
                    v for v in snapshot.vintages if v.model == "cams" and v.valid_time == target
                ]
                predictions["cams"] = max(cams, key=lambda v: v.issue_time).value if cams else None
                predictions.update(
                    {name: fn(snapshot, horizon) for name, fn in challengers.items()}
                )
                for name, value in predictions.items():
                    if value is None:
                        missing[name] += 1
                    else:
                        _value(value)
                if any(v is None for v in predictions.values()):
                    continue
                rows.append(
                    {
                        "station": station,
                        "origin": origin,
                        "target": target,
                        "horizon": horizon,
                        "actual": actual.value,
                        **predictions,
                    }
                )
        origin += HOUR

    by_horizon = {
        str(h): {
            m: _metrics([r for r in rows if r["horizon"] == h], m, aqi_breakpoints) for m in models
        }
        for h in horizons
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['station']}/{row['target']:%Y-%m}/+{row['horizon']}h"].append(row)
    return {
        "status": "ok" if rows else "insufficient_data",
        "window": {"start": start.isoformat(), "end": end.isoformat(), "as_of": as_of.isoformat()},
        "stations": list(stations),
        "eligible_actual_pairs": eligible,
        "paired_rows": len(rows),
        "missing": dict(missing),
        "aqi_breakpoints": list(aqi_breakpoints),
        "metrics_by_horizon": by_horizon,
        "mae_by_station_month_horizon": {
            key: {m: _metrics(group, m, aqi_breakpoints)["mae"] for m in models}
            for key, group in sorted(grouped.items())
        },
        "beats_cams": {m: _beats(rows, m, start, end, stations, as_of) for m in challengers},
        "limitations": [
            "No historical CAMS is synthesized; missing as-of vintages reduce shared coverage.",
            "Observation revisions require an external append-only availability archive; the live "
            "upsert table cannot reconstruct original historical values.",
            "AQI-category accuracy uses supplied hourly concentration bands; "
            "it is not official daily AQI.",
        ],
    }


def _beats(rows, model, start, end, stations, as_of):
    """Paired daily block bootstrap: 28 days, >=22 paired hours/station/day at +24h."""
    rule = "28 UTC days; +24h; >=22 pairs/station/day; upper one-sided 95% daily-block CI < 0"
    result = {"status": "insufficient_data", "beats_cams": None, "rule": rule}
    if end - start != timedelta(days=28) or start.hour or as_of < end + 23 * HOUR:
        return result
    daily = defaultdict(list)
    counts = defaultdict(int)
    for r in rows:
        if r["horizon"] == 24:
            day = r["origin"].date()
            daily[day].append(abs(r[model] - r["actual"]) - abs(r["cams"] - r["actual"]))
            counts[(day, r["station"])] += 1
    days = [(start + timedelta(days=n)).date() for n in range(28)]
    if any(counts[(day, station)] < 22 for day in days for station in stations):
        return result
    # Resample entire origin days, retaining all station/hour pairs within each block.
    rng = random.Random(14)
    samples = sorted(
        sum(sum(daily[d]) for d in sample) / sum(len(daily[d]) for d in sample)
        for sample in (rng.choices(days, k=28) for _ in range(2000))
    )
    delta = fmean(v for day in days for v in daily[day])
    upper = samples[math.ceil(0.95 * len(samples)) - 1]
    return {
        **result,
        "status": "ok",
        "beats_cams": delta < 0 and upper < 0,
        "mae_delta": delta,
        "upper_95": upper,
        "bootstrap_replicates": 2000,
        "paired_days": 28,
        "paired_rows": sum(len(daily[d]) for d in days),
    }


def stored_predictor(model: str) -> Predictor:
    """Replay archived forecasts using the latest already-available issue for this target."""

    def predict(snapshot: Snapshot, horizon: int) -> float | None:
        target = snapshot.prediction_time + horizon * HOUR
        candidates = [
            row for row in snapshot.vintages if row.model == model and row.valid_time == target
        ]
        return max(candidates, key=lambda row: row.issue_time).value if candidates else None

    return predict


def main() -> int:
    """Offline JSON input only: no production data or credentials written by the harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="private JSON input; see docs/backtest.md")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    timestamp = lambda value: require_utc(datetime.fromisoformat(value))  # noqa: E731
    observations = [
        Observation(r["station_id"], timestamp(r["ts"]), r["value"], timestamp(r["available_at"]))
        for r in payload["observations"]
    ]
    vintages = [
        Vintage(
            r["station_id"],
            timestamp(r["issue_time"]),
            timestamp(r["valid_time"]),
            r["value"],
            timestamp(r["available_at"]),
            r.get("model", "cams"),
        )
        for r in payload["vintages"]
    ]
    report = evaluate(
        observations,
        vintages,
        start=timestamp(payload["start"]),
        end=timestamp(payload["end"]),
        as_of=timestamp(payload["as_of"]),
        stations=payload["stations"],
        aqi_breakpoints=payload["aqi_breakpoints"],
        challengers={name: stored_predictor(name) for name in payload.get("challenger_models", [])},
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
