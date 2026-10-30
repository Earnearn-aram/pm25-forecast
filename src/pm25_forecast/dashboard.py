"""Public dashboard builder (KAN-25). Run with ``python -m pm25_forecast.dashboard --out site/``.

Reads the database (read-only SELECTs, one read-only connection for the whole build) and writes
a static site:
``index.html`` (server-rendered, inline SVG charts), ``data.json`` (the same numbers), plus
``dashboard.css`` and ``dashboard.js`` (language, station and animation controls).

* Every section is independent. A query that fails (missing table, bad rows, DB hiccup) turns that
  section into ``"unavailable"`` in ``data.json`` and on the page; it never stops the build.
  If every database section fails the build exits 1 and writes nothing, so a good deployed site is
  not replaced by an empty one.
* Empty data renders as "waiting", never as zero.
* Nothing from the environment (DATABASE_URL, webhook URLs) is ever put in the output.
* All times are UTC; the header also shows Bangkok time (fixed UTC+7, no DST).

Exit codes: 0 built, 1 every database section failed, 2 config error (no DATABASE_URL).
"""

import argparse
import html
import json
import logging
import math
import shutil
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, TypeAdapter, ValidationError

from pm25_forecast.collector import ConfigError, open_db
from pm25_forecast.drift import PSI_ALERT
from pm25_forecast.models.baselines import HORIZONS_H, PERSISTENCE_MODEL_VERSION
from pm25_forecast.monitoring.logs import configure_logging
from pm25_forecast.score import ScoreRow, rolling_metrics
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql

logger = logging.getLogger(__name__)

EXIT_FAILED, EXIT_CONFIG = 1, 2
ASSETS = Path(__file__).parent / "dashboard_assets"
REPO_URL = "https://github.com/Earnearn-aram/pm25-forecast"
WINDOW = timedelta(days=7)
BANGKOK = timezone(timedelta(hours=7))
REPORTING_H = 3  # a station that reported within this many hours counts as reporting
PIPELINE_HOURS = 24
PIPELINE_GRACE = timedelta(minutes=45)  # the newest cell is the last hour whose run is due
HEADLINE_H = 24
PSI_BAR_MAX = 2 * PSI_ALERT  # the alert tick sits in the middle of the track
SKILL_BAR_MAX = 0.5
AIR4THAI = "air4thai"
# Optional display names, "id": "Name". Left empty rather than guessed; the page then shows the id.
STATION_NAMES: dict[str, str] = {
    "59t": "Government Public Relations Department",
    "61t": "Bodindecha School",
    "bkp100t": "Bueng Kum District Office",
    "bkp101t": "Khlong Sam Wa District Office",
    "02t": "Bansomdejchaopraya Rajabhat University",
}
STATION_MAP: dict[str, tuple[float, float]] = {
    "59t": (13.783185, 100.540489),
    "61t": (13.769634, 100.614567),
    "bkp100t": (13.785212, 100.669183),
    "bkp101t": (13.859911, 100.704002),
    "02t": (13.732846, 100.487662),
}
SECTIONS = ("scores", "observations", "chart", "quality", "drift", "models")

Status = Literal["ok", "warn", "fail"]
RANK = {"ok": 0, "warn": 1, "fail": 2}


class ObsRow(BaseModel):
    station_id: str
    ts: AwareDatetime
    pm25: float


class PredRow(BaseModel):
    station_id: str
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    model_version: str
    pm25_pred: float


class DqRow(BaseModel):
    checked_at: AwareDatetime
    source: str
    station_id: str
    check_name: str
    status: Status
    value: float | None = None


class DriftRow(BaseModel):
    checked_at: AwareDatetime
    model_version: str
    feature: str
    psi: float | None = None
    status: str


class ChampionRow(BaseModel):
    model_version: str
    promoted_at: AwareDatetime


class RetrainRow(BaseModel):
    model_version: str
    created_at: AwareDatetime


SELECT_OBS = """
SELECT station_id, ts, pm25 FROM observations
WHERE ts >= %(since)s AND ts <= %(now)s AND pm25 IS NOT NULL AND pm25 >= 0
"""
SELECT_PREDS = """
SELECT station_id, issue_time, valid_time, model_version, pm25_pred FROM predictions
WHERE model_version IN (%(m1)s, %(m2)s) AND valid_time >= %(since)s AND valid_time <= %(now)s
"""
SELECT_FUTURE = """
SELECT station_id, issue_time, valid_time, model_version, pm25_pred FROM predictions
WHERE model_version IN (%(m1)s, %(m2)s) AND issue_time <= %(now)s
AND issue_time >= %(since)s AND valid_time > %(now)s AND valid_time <= %(until)s
"""
SELECT_DQ = """
SELECT checked_at, source, station_id, check_name, status, value FROM data_quality_checks
WHERE checked_at >= %(since)s AND checked_at <= %(now)s
"""
SELECT_DRIFT = """
SELECT checked_at, model_version, feature, psi, status FROM drift_checks
WHERE checked_at >= %(since)s AND checked_at <= %(now)s
"""
SELECT_CHAMPION = "SELECT model_version, promoted_at FROM model_champion WHERE slot = 'production'"
SELECT_RETRAIN = """
SELECT model_version, created_at FROM models WHERE model_version LIKE %(prefix)s
ORDER BY created_at DESC LIMIT 1
"""
SELECT_FIRST_ISSUE = """
SELECT issue_time FROM predictions
WHERE model_version = %(m)s AND issue_time >= %(since)s ORDER BY issue_time LIMIT 1
"""
SET_READ_ONLY = "SET TRANSACTION READ ONLY"
SET_TIMEOUT = "SET LOCAL statement_timeout = '30s'"
SAVEPOINT = "dashboard_section"


class Reader:
    """One read-only connection for the whole build; never commits.

    ``SET TRANSACTION READ ONLY`` is the first statement, so the database itself rejects any write.
    Each query runs inside a SAVEPOINT: a failed statement aborts only that savepoint (rolled
    back to), so the next section still runs on the same connection and transaction.
    """

    def __init__(self, connect: Callable[[str], Any], url: str):
        self._connect, self._url = connect, url
        self._con: Any = None

    def open(self) -> None:
        con = open_db(self._connect, self._url)
        try:
            con.execute(SET_READ_ONLY, {})
            con.execute(SET_TIMEOUT, {})
        except Exception:
            con.close()
            raise
        self._con = con

    def close(self) -> None:
        con, self._con = self._con, None
        if con is None:
            return
        try:
            con.rollback()  # read-only transaction: nothing to keep, never commit
            con.close()
        except Exception as exc:  # a dropped connection must not replace the collected data
            logger.warning("dashboard: closing connection failed (%s)", type(exc).__name__)

    def query(self, statement: str, params: dict[str, Any]) -> list[Sequence[Any]]:
        if self._con is None:
            raise RuntimeError("database unavailable")
        self._con.execute(f"SAVEPOINT {SAVEPOINT}", {})
        try:
            rows = list(self._con.execute(statement, params).fetchall())
        except Exception:
            self._con.execute(f"ROLLBACK TO SAVEPOINT {SAVEPOINT}", {})
            raise
        self._con.execute(f"RELEASE SAVEPOINT {SAVEPOINT}", {})
        return rows


def _rows[M: BaseModel](model: type[M], raw: Sequence[Sequence[Any]]) -> list[M]:
    names = list(model.model_fields)
    good: list[M] = []
    for row in raw:
        try:
            good.append(model.model_validate(dict(zip(names, row, strict=True))))
        except (ValidationError, ValueError):
            logger.warning("%s: dropped a malformed row", model.__name__)
    return good


def _num(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _hour(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _station_label(station_id: str) -> str:
    name = STATION_NAMES.get(station_id)
    return name if name else station_id


# --- headline ------------------------------------------------------------------------------------


def headline(skill: float | None) -> dict[str, Any]:
    """The plain-language answer. skill = 1 - MAE_challenger / MAE_persistence at +24 h."""
    if skill is None or not math.isfinite(skill):
        return {"state": "waiting", "text": "Not scored yet", "skill_pct": None}
    pct = skill * 100
    if skill > 0:
        return {"state": "yes", "text": f"Yes, by {pct:.1f}%", "skill_pct": round(pct, 1)}
    sign = "\u2212" if skill < 0 else ""
    return {
        "state": "not_yet",
        "text": f"Not yet ({sign}{abs(pct):.1f}%)",
        "skill_pct": round(pct, 1),
    }


# --- section builders ----------------------------------------------------------------------------


def _scores(rd: Reader, challenger: str | None, now: datetime) -> dict[str, Any]:
    models = [PERSISTENCE_MODEL_VERSION]
    if challenger and challenger != PERSISTENCE_MODEL_VERSION:
        models.append(challenger)
    by_model: dict[str, list[ScoreRow]] = {}
    for model in models:
        raw = rd.query(
            sql.SELECT_SCORES_BETWEEN, {"model_version": model, "since": now - WINDOW, "now": now}
        )
        by_model[model] = _rows(ScoreRow, raw)
    metrics = rolling_metrics(by_model)
    out: dict[str, Any] = {"persistence": {}, "challenger": {}, "first_scores_from": None}
    for key, model in (("persistence", PERSISTENCE_MODEL_VERSION), ("challenger", challenger)):
        for h in HORIZONS_H:
            m = metrics.get(model or "", {}).get(h)
            out[key][str(h)] = (
                None
                if m is None
                else {
                    "n": m.n,
                    "mae": _num(m.mae),
                    "rmse": _num(m.rmse),
                    "skill": _num(m.skill_vs_persistence, 4),
                    "n_matched": m.n_matched,
                }
            )
    if challenger and out["challenger"][str(HEADLINE_H)] is None:
        raw = rd.query(
            SELECT_FIRST_ISSUE, {"m": challenger, "since": now - WINDOW - timedelta(hours=24)}
        )
        first = raw[0][0] if raw else None
        if first is not None:
            issue = TypeAdapter(AwareDatetime).validate_python(first)
            out["first_scores_from"] = _iso(issue + timedelta(hours=HEADLINE_H))
    return out


def _observations(rd: Reader, now: datetime) -> list[ObsRow]:
    return _rows(ObsRow, rd.query(SELECT_OBS, {"since": now - WINDOW, "now": now}))


def _station_summary(obs: list[ObsRow], now: datetime) -> dict[str, Any]:
    newest: dict[str, datetime] = {}
    for o in obs:
        if o.station_id not in newest or o.ts > newest[o.station_id]:
            newest[o.station_id] = o.ts
    ages = {s: (now - ts) / timedelta(minutes=1) for s, ts in newest.items()}
    reporting = [s for s, age in ages.items() if age <= REPORTING_H * 60]
    return {
        "newest": {s: _iso(ts) for s, ts in sorted(newest.items())},
        "freshness_min": _num(min(ages.values()), 0) if ages else None,
        "stations_total": len(newest),
        "stations_reporting": len(reporting),
    }


def _chart(rd: Reader, obs: list[ObsRow], challenger: str | None, now: datetime) -> dict[str, Any]:
    since = _hour(now - WINDOW)
    raw = rd.query(
        SELECT_PREDS,
        {"m1": PERSISTENCE_MODEL_VERSION, "m2": challenger or PERSISTENCE_MODEL_VERSION}
        | {"since": since, "now": now},
    )
    future_raw = rd.query(
        SELECT_FUTURE,
        {
            "m1": PERSISTENCE_MODEL_VERSION,
            "m2": challenger or PERSISTENCE_MODEL_VERSION,
            "since": now - timedelta(hours=3),
            "now": now,
            "until": _hour(now) + timedelta(hours=24),
        },
    )
    future_rows = _rows(PredRow, future_raw)
    latest_issue: dict[tuple[str, str], datetime] = {}
    for prediction in future_rows:
        if prediction.valid_time - prediction.issue_time not in [
            timedelta(hours=h) for h in HORIZONS_H
        ]:
            continue
        key = (prediction.station_id, prediction.model_version)
        latest_issue[key] = max(latest_issue.get(key, prediction.issue_time), prediction.issue_time)
    future: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for prediction in future_rows:
        key = (prediction.station_id, prediction.model_version)
        if prediction.issue_time == latest_issue.get(key):
            future.setdefault(key, []).append(
                {
                    "issue_time": _iso(prediction.issue_time),
                    "valid_time": _iso(prediction.valid_time),
                    "value": _num(prediction.pm25_pred, 1),
                }
            )
    actual: dict[str, dict[datetime, float]] = {}
    newest_observations: dict[str, ObsRow] = {}
    hourly_observations: dict[tuple[str, datetime], ObsRow] = {}
    for o in obs:
        if not math.isfinite(o.pm25):
            continue
        bucket = (o.station_id, _hour(o.ts))
        if bucket not in hourly_observations or o.ts > hourly_observations[bucket].ts:
            hourly_observations[bucket] = o
        if o.station_id not in newest_observations or o.ts > newest_observations[o.station_id].ts:
            newest_observations[o.station_id] = o
    for (sid, hour), observation in hourly_observations.items():
        actual.setdefault(sid, {})[hour] = observation.pm25
    series: dict[tuple[str, str], dict[datetime, float]] = {}
    for p in _rows(PredRow, raw):
        if p.valid_time - p.issue_time != timedelta(hours=HEADLINE_H):
            continue
        series.setdefault((p.station_id, p.model_version), {})[_hour(p.valid_time)] = p.pm25_pred
    hours = []
    t = since
    while t <= _hour(now):
        hours.append(t)
        t += timedelta(hours=1)
    stations = []
    has_challenger = False
    for sid in sorted(actual):
        pers = series.get((sid, PERSISTENCE_MODEL_VERSION), {})
        chal = series.get((sid, challenger), {}) if challenger else {}
        has_challenger = has_challenger or bool(chal)
        points = [
            {
                "t": _iso(h),
                "actual": _num(actual[sid].get(h), 1),
                "persistence": _num(pers.get(h), 1),
                "challenger": _num(chal.get(h), 1),
            }
            for h in hours
        ]
        latest = newest_observations[sid]
        stations.append(
            {
                "id": sid,
                "label": _station_label(sid),
                "points": points,
                "future_persistence": sorted(
                    future.get((sid, PERSISTENCE_MODEL_VERSION), []), key=lambda p: p["valid_time"]
                ),
                "future_challenger": sorted(
                    future.get((sid, challenger or ""), []), key=lambda p: p["valid_time"]
                ),
                "latest_value": _num(latest.pm25, 1),
                "latest_at": _iso(latest.ts),
            }
        )
    return {
        "stations": stations,
        "has_challenger": has_challenger,
        "start": _iso(since),
        "end": _iso(_hour(now)),
    }


def _worst(statuses: Sequence[str]) -> str:
    return max(statuses, key=lambda s: RANK.get(s, 0)) if statuses else "ok"


def _quality(rd: Reader, now: datetime) -> dict[str, Any]:
    rows = _rows(DqRow, rd.query(SELECT_DQ, {"since": now - timedelta(hours=48), "now": now}))
    latest: dict[tuple[str, str, str], DqRow] = {}
    for r in rows:
        key = (r.source, r.station_id, r.check_name)
        if key not in latest or r.checked_at > latest[key].checked_at:
            latest[key] = r
    stations: dict[str, dict[str, Any]] = {}
    for (source, sid, check), r in latest.items():
        if source != AIR4THAI or not sid:
            continue
        s = stations.setdefault(sid, {"statuses": [], "missing_h": None})
        s["statuses"].append(r.status)
        if check == "missing_hours" and r.value is not None:
            s["missing_h"] = _num(r.value, 0)
    table = {
        sid: {"status": _worst(s["statuses"]), "missing_h": s["missing_h"]}
        for sid, s in sorted(stations.items())
    }
    # Pipeline: one cell per UTC hour, worst status of that hour's checks; no row = missed.
    end = _hour(now - PIPELINE_GRACE)
    hours = [end - timedelta(hours=i) for i in range(PIPELINE_HOURS - 1, -1, -1)]
    lines = {
        "Collect": lambda r: r.source == AIR4THAI,
        "Predict": lambda r: r.source == PERSISTENCE_MODEL_VERSION,
        "Quality": lambda r: True,
    }
    pipeline = []
    for name, pick in lines.items():
        per_hour: dict[datetime, list[str]] = {}
        for r in rows:
            if pick(r):
                per_hour.setdefault(_hour(r.checked_at), []).append(r.status)
        cells = [_worst(per_hour[h]) if h in per_hour else "missed" for h in hours]
        pipeline.append({"name": name, "cells": cells, "ok": cells.count("ok")})
    return {
        "stations": table,
        "worst": _worst([r.status for r in latest.values()]) if latest else None,
        "has_checks": bool(latest),
        "pipeline": pipeline,
        "pipeline_end": _iso(end),
    }


def _drift(rd: Reader, challenger: str | None, now: datetime) -> dict[str, Any]:
    rows = _rows(DriftRow, rd.query(SELECT_DRIFT, {"since": now - timedelta(days=3), "now": now}))
    models = sorted({r.model_version for r in rows})
    model = challenger if challenger in models else (models[0] if models else None)
    latest: dict[str, DriftRow] = {}
    for r in rows:
        if r.model_version == model and (
            r.feature not in latest or r.checked_at > latest[r.feature].checked_at
        ):
            latest[r.feature] = r
    features = [
        {
            "name": f,
            "psi": _num(r.psi),
            "alert": r.status == "alert",
            "status": r.status,
        }
        for f, r in sorted(latest.items())
    ]
    psis = [f for f in features if f["psi"] is not None]
    worst = max(psis, key=lambda f: f["psi"]) if psis else None
    return {
        "model": model,
        "features": features,
        "worst_psi": None if worst is None else worst["psi"],
        "worst_feature": None if worst is None else worst["name"],
    }


def _models(rd: Reader, challenger: str | None) -> dict[str, Any]:
    champ = _rows(ChampionRow, rd.query(SELECT_CHAMPION, {}))
    retrain = _rows(RetrainRow, rd.query(SELECT_RETRAIN, {"prefix": "retrain-%"}))
    return {
        "champion": champ[0].model_version if champ else PERSISTENCE_MODEL_VERSION,
        "champion_registered": bool(champ),
        "challenger": challenger,
        "last_retrain": (
            {"version": retrain[0].model_version, "at": _iso(retrain[0].created_at)}
            if retrain
            else None
        ),
        "promotion": "Review unless auto-promotion is enabled",
    }


# --- assembling ----------------------------------------------------------------------------------


def _overall(data: dict[str, Any]) -> str:
    q, obs = data["quality"], data["freshness"]
    total = (obs or {}).get("stations_total", 0)
    reporting = (obs or {}).get("stations_reporting", 0)
    unavailable = any(v == "unavailable" for v in data["sections"].values())
    worst = (q or {}).get("worst")
    if worst == "fail" or (total > 0 and reporting == 0):
        return "failed"
    if worst == "warn" or unavailable or reporting < total:
        return "degraded"
    if total == 0 and not (q or {}).get("has_checks"):
        return "waiting"
    return "healthy"


def collect(
    connect: Callable[[str], Any], url: str, challenger: str | None, now: datetime
) -> dict[str, Any]:
    """Query every section; a failing section is recorded as unavailable and the rest go on."""
    rd = Reader(connect, url)
    try:
        rd.open()
    except ConfigError:
        raise
    except Exception as exc:  # type only: driver messages can echo the connection string
        logger.warning("dashboard: database unavailable (%s)", type(exc).__name__)
    try:
        return _collect(rd, challenger, now)
    finally:
        rd.close()


def _collect(rd: Reader, challenger: str | None, now: datetime) -> dict[str, Any]:
    sections: dict[str, str] = {}

    def section(name: str, fn: Callable[[], Any]) -> Any:
        try:
            value = fn()
        except Exception as exc:  # type only: driver messages can echo the connection string
            logger.warning("dashboard section %s unavailable (%s)", name, type(exc).__name__)
            sections[name] = "unavailable"
            return None
        sections[name] = "ok"
        return value

    scores = section("scores", lambda: _scores(rd, challenger, now))
    obs = section("observations", lambda: _observations(rd, now))
    freshness = None if obs is None else _station_summary(obs, now)

    def chart_section() -> dict[str, Any]:
        if obs is None:
            raise RuntimeError("observations unavailable")
        return _chart(rd, obs, challenger, now)

    chart = section("chart", chart_section)
    quality = section("quality", lambda: _quality(rd, now))
    drift = section("drift", lambda: _drift(rd, challenger, now))
    models = section("models", lambda: _models(rd, challenger))

    chal24 = ((scores or {}).get("challenger") or {}).get(str(HEADLINE_H))
    pers24 = ((scores or {}).get("persistence") or {}).get(str(HEADLINE_H))
    data: dict[str, Any] = {
        "generated_at": _iso(now),
        "generated_at_bangkok": now.astimezone(BANGKOK).isoformat(),
        "sections": sections,
        "challenger_version": challenger,
        "headline": headline(chal24["skill"] if chal24 else None),
        "kpi": {
            "skill_24h": chal24["skill"] if chal24 else None,
            "mae_challenger_24h": chal24["mae"] if chal24 else None,
            "mae_persistence_24h": pers24["mae"] if pers24 else None,
            "freshness_min": (freshness or {}).get("freshness_min"),
            "stations_reporting": (freshness or {}).get("stations_reporting"),
            "stations_total": (freshness or {}).get("stations_total"),
            "worst_psi": (drift or {}).get("worst_psi"),
            "worst_psi_feature": (drift or {}).get("worst_feature"),
        },
        "scores": scores,
        "freshness": freshness,
        "chart": chart,
        "quality": quality,
        "drift": drift,
        "models": models,
    }
    data["status"] = _overall(data)
    return data


# --- rendering -----------------------------------------------------------------------------------

e = html.escape


def _fmt_time(iso: str) -> str:
    dt = datetime.fromisoformat(iso)
    return f"{dt.astimezone(UTC):%H:%M} UTC \u00b7 {dt.astimezone(BANGKOK):%H:%M} Bangkok"


def _val(text: str | None, good: bool = False) -> str:
    if text is None:
        return '<p class="val waiting">waiting</p>'
    return f'<p class="val{" good" if good else ""}">{e(text)}</p>'


def _unavailable(what: str) -> str:
    return f'<p class="muted small">{e(what)} unavailable right now.</p>'


def _nice_max(vmax: float) -> float:
    for step in (5, 10, 20, 25, 50, 100, 200):
        if step * 4 >= vmax:
            return float(step * 4)
    return float(math.ceil(vmax / 400) * 400)


def _lines(points: list[dict[str, Any]], key: str, ymax: float, css: str, width: int = 400) -> str:
    h, pad = 220, 8
    n = max(len(points) - 1, 1)
    segments: list[list[str]] = [[]]
    for i, p in enumerate(points):
        v = p[key]
        if v is None:
            segments.append([])
            continue
        y = h - pad - min(v, ymax) / ymax * (h - 2 * pad)
        segments[-1].append(f"{i / n * width:.1f},{y:.1f}")
    return "".join(
        f'<polyline class="ln {css}" vector-effect="non-scaling-stroke" points="{" ".join(s)}"/>'
        for s in segments
        if len(s) > 1
    )


def _chart_block(st: dict[str, Any], chart: dict[str, Any], first: bool) -> str:
    # Show the latest 24 hours beside the next 24 hours on equal time scales.
    points = st["points"][-25:]
    keys = ["actual", "persistence"] + (["challenger"] if chart["has_challenger"] else [])
    has_future = any(
        p["value"] is not None
        for key in ("future_persistence", "future_challenger")
        for p in st.get(key, [])
    )
    past_width = 400 if has_future else 800
    values = [p[k] for p in points for k in keys if p[k] is not None]
    values += [
        p["value"]
        for key in ("future_persistence", "future_challenger")
        for p in st.get(key, [])
        if p["value"] is not None
    ]
    if not values:
        body = '<p class="muted small">No hourly data for this station yet.</p>'
    else:
        ymax = _nice_max(max(values))
        grid = "".join(
            f'<line class="grid-line" x1="0" x2="800" y1="{y:.1f}" y2="{y:.1f}"/>'
            for y in (8 + i * 204 / 4 for i in range(5))
        )
        labels = "".join(
            f'<span class="ylab" style="top:{(8 + i * 204 / 4) / 220 * 100:.2f}%">'
            f"{int(ymax - i * ymax / 4)}</span>"
            for i in range(5)
        )
        series = _lines(points, "persistence", ymax, "ln-pers", past_width)
        if chart["has_challenger"]:
            series += _lines(points, "challenger", ymax, "ln-chal", past_width)
        series += _lines(points, "actual", ymax, "ln-actual", past_width)
        actual_points = [
            (i, point) for i, point in enumerate(points) if point["actual"] is not None
        ]
        if actual_points:
            i, point = actual_points[-1]
            x = i / max(len(points) - 1, 1) * past_width
            y = 212 - min(point["actual"], ymax) / ymax * 204
            description = f"Actual: {point['actual']:g} µg/m³ at {point['t']}"
            series += (
                f'<circle class="actual-dot" cx="{x:.1f}" cy="{y:.1f}" r="4">'
                f"<title>{e(description)}</title></circle>"
            )
        forecast_origin = datetime.fromisoformat(chart["end"])
        for key, css in (("future_persistence", "ln-pers"), ("future_challenger", "ln-chal")):
            coords = []
            for forecast in st.get(key, []):
                if forecast["value"] is None:
                    continue
                t = datetime.fromisoformat(forecast["valid_time"])
                x = 400 + (t - forecast_origin).total_seconds() / 86400 * 400
                y = 212 - min(forecast["value"], ymax) / ymax * 204
                coords.append((x, y, forecast))
            if len(coords) > 1:
                pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
                series += (
                    f'<polyline class="ln {css} future-line" '
                    f'vector-effect="non-scaling-stroke" points="{pairs}"/>'
                )
            for x, y, forecast in coords:
                description = (
                    f"{key.replace('future_', '').capitalize()} forecast: "
                    f"{forecast['value']:g} µg/m³, "
                    f"valid {forecast['valid_time']}; issued {forecast['issue_time']}"
                )
                series += (
                    f'<circle class="forecast-dot {css}" cx="{x:.1f}" cy="{y:.1f}" '
                    f'r="4"><title>{e(description)}</title></circle>'
                )
        aria = (
            f"Station {st['label']}, last 24 hours in UTC: actual PM2.5 in \u00b5g/m\u00b3 against "
            "persistence and challenger forecasts made 24 h earlier"
        )
        end = datetime.fromisoformat(chart["end"])
        start = end - timedelta(hours=24)
        future_html = []
        for key, name in (
            ("future_persistence", "Persistence forecast"),
            ("future_challenger", "Challenger forecast"),
        ):
            for forecast in st.get(key, []):
                if forecast["value"] is None:
                    continue
                issue = datetime.fromisoformat(forecast["issue_time"])
                valid = datetime.fromisoformat(forecast["valid_time"])
                hours_ahead = int((valid - issue) / timedelta(hours=1))
                future_html.append(
                    f'<span class="forecast-point"><b>{e(name)}</b> +{hours_ahead} h · '
                    f"{forecast['value']:g} µg/m³ <small>({valid:%d %b %H:%M} UTC)</small></span>"
                )
        future_block = (
            '<div class="future-forecasts"><strong>Upcoming forecasts</strong>'
            + (
                "".join(future_html)
                if future_html
                else '<span class="forecast-empty">Waiting for future forecasts.</span>'
            )
            + "</div>"
        )
        future_background = (
            '<rect class="forecast-region" x="400" y="0" width="400" height="220"/>'
            '<line class="forecast-boundary" x1="400" x2="400" y1="0" y2="220"/>'
            if has_future
            else ""
        )
        axis_end = (
            f'<span class="axis-now" title="{end:%d %b %H:%M} UTC">Now</span><span>+24 h</span>'
            if has_future
            else f'<span title="{end:%d %b %H:%M} UTC">Now</span>'
        )
        body = (
            '<div class="plotwrap"><div class="ytitle">\u00b5g/m\u00b3</div><div class="plot">'
            f'{labels}<svg viewBox="-8 0 816 220" preserveAspectRatio="none" role="img" '
            f'aria-label="{e(aria)}">{future_background}{grid}{series}</svg></div>'
            f'<div class="xlabs"><span>{start:%d %b %H:%M} UTC</span>'
            f"{axis_end}</div></div>" + future_block
        )
    hidden = "" if first else " hidden"
    return f'<div class="chart" data-station="{e(st["id"])}"{hidden}>{body}</div>'


def _section_chart(data: dict[str, Any]) -> str:
    chart = data["chart"]
    chal = "Challenger" if (chart or {}).get("has_challenger") else "Challenger (not scored yet)"
    legend = (
        '<div class="legend"><span><i class="sw actual"></i>Actual</span>'
        f'<span><i class="sw chal"></i>{chal}</span>'
        '<span><i class="sw pers"></i>Persistence (latest reading)</span></div>'
    )
    if chart is None:
        inner = _unavailable("The chart")
        pills = ""
        title = "Forecast vs actual \u00b7 +24 h"
    elif not chart["stations"]:
        inner = '<p class="muted small">Waiting for the first observations.</p>'
        pills = ""
        title = "Forecast vs actual \u00b7 +24 h"
    else:
        sts = chart["stations"]
        title = f"Forecast vs actual \u00b7 +24 h \u00b7 {sts[0]['label']}"
        pills = "".join(
            f'<button type="button" class="pill" data-station="{e(s["id"])}" '
            f'data-label="{e(s["label"])}" aria-label="{e(s["label"])}" '
            f'aria-pressed="{"true" if i == 0 else "false"}">{e(s["label"])}</button>'
            for i, s in enumerate(sts)
        )
        inner = "".join(_chart_block(s, chart, i == 0) for i, s in enumerate(sts))
    return (
        '<figure class="card wide" style="margin:0"><div class="cardhead">'
        f'<h2 id="chart-title">{e(title)}</h2>'
        f'<div class="pills" role="group" aria-label="Station">{pills}</div></div>'
        f"{legend}{inner}"
        '<figcaption class="muted small">Each point: the forecast made 24 h before that hour. '
        "Last 24 hours, hourly, UTC. Future dots are forecasts; dashed lines connect available "
        "points and do not imply intermediate predictions.</figcaption></figure>"
    )


def _section_map(data: dict[str, Any]) -> str:
    stations = (data.get("chart") or {}).get("stations", [])
    pins = []
    for st in stations:
        coords = STATION_MAP.get(st["id"])
        if not coords:
            continue
        lat, lon = coords
        value = st.get("latest_value")
        if value is None:
            continue
        label = _station_label(st["id"])
        latest_at = datetime.fromisoformat(st["latest_at"])
        generated_at = datetime.fromisoformat(data["generated_at"])
        age = int((generated_at - latest_at).total_seconds() // 60)
        stale = age > REPORTING_H * 60
        pins.append(
            f'<button type="button" class="map-pin {"stale" if stale else ""}" '
            f'data-age="{age}" data-stale="{str(stale).lower()}" '
            f'data-time="{e(st["latest_at"])}" data-station="{e(st["id"])}" '
            f'data-lat="{lat}" data-lon="{lon}" '
            f'aria-label="{e(label)}: {value:g} µg/m³" title="{e(label)}: {value:g} µg/m³">'
            f"{value:g}</button>"
        )
    return (
        '<section class="card map-card"><div class="cardhead"><div>'
        '<h2>Explore the map</h2><p class="muted small">Select a station · latest PM2.5 (µg/m³)</p>'
        '</div></div><div class="map-frame"><div id="station-map" class="station-map">'
        f'<div class="map-tiles" aria-hidden="true"></div>{"".join(pins)}'
        '<p class="map-fallback" role="status" hidden '
        'data-en="Map tiles could not load. Use the station list instead." '
        'data-th="โหลดแผนที่ไม่สำเร็จ ใช้รายชื่อสถานีแทนได้">'
        "Map tiles could not load. Use the station list instead.</p>"
        '<span class="map-credit">© <a href="https://www.openstreetmap.org/copyright" '
        'rel="noopener" referrerpolicy="strict-origin-when-cross-origin">'
        "OpenStreetMap contributors</a></span></div></div>"
        '<p id="selected-map-reading" class="muted small" aria-live="polite"></p>'
        '<p class="muted small">Numbers are PM2.5 concentrations, not AQI. '
        "Use the station list if the map is unavailable.</p></section>"
    )


def _section_horizons(data: dict[str, Any]) -> str:
    scores = data["scores"]
    if scores is None:
        body = _unavailable("Skill by horizon")
    else:
        rows = []
        for h in HORIZONS_H:
            m = scores["challenger"].get(str(h))
            skill = None if m is None else m["skill"]
            if skill is None:
                value, bar = "waiting", ""
            else:
                width = min(abs(skill) / SKILL_BAR_MAX, 1) * 100
                value = f"{skill * 100:+.1f}%".replace("-", "\u2212")
                cls = "fill" if skill > 0 else "fill neg"
                bar = f'<div class="{cls}" style="width:{width:.0f}%"></div>'
            rows.append(
                f'<div class="hrow"><div class="top"><span>+{h} h</span>'
                f'<span class="mono">{value}</span></div><div class="track">{bar}</div></div>'
            )
        body = "".join(rows)
    return (
        '<div class="card side"><h2>Skill by horizon</h2>'
        f"{body}"
        '<p class="muted small">1 \u2212 MAE challenger \u00f7 MAE persistence, on the same '
        "station-hours. Above 0 means the model beats \u201ctomorrow = today\u201d.</p></div>"
    )


def _section_pipeline(data: dict[str, Any]) -> str:
    q = data["quality"]
    if q is None:
        body = _unavailable("Pipeline history")
    elif not q["has_checks"]:
        body = '<p class="muted small">Waiting for the first hourly quality run.</p>'
    else:
        rows = []
        for line in q["pipeline"]:
            counts = {c: line["cells"].count(c) for c in ("ok", "warn", "fail", "missed")}
            aria = (
                f"{line['name']}: {counts['ok']} of {PIPELINE_HOURS} hours ok, "
                f"{counts['warn']} warning, {counts['fail']} failed, {counts['missed']} missed"
            )
            cells = "".join(
                f'<span class="cell {"miss" if c == "missed" else c}"></span>'
                for c in line["cells"]
            )
            details = {
                "Collect": (
                    "Air4Thai quality checks; Python and Neon Postgres. "
                    "These checks do not prove workflow completion."
                ),
                "Predict": (
                    "Persistence prediction quality checks; Python and Neon Postgres. "
                    "These checks do not prove workflow completion."
                ),
                "Quality": (
                    "Worst recorded quality-check status across sources; Python and Neon Postgres."
                ),
            }
            rows.append(
                f'<abbr tabindex="0" title="{e(details[line["name"]])}">{e(line["name"])}</abbr>'
                f'<div class="cells" role="img" aria-label="{e(aria)}">{cells}</div>'
                f'<span class="mono muted">{line["ok"]}/{PIPELINE_HOURS} ok</span>'
            )
        body = f'<div class="pipe">{"".join(rows)}</div>'
    legend = (
        '<p class="legend"><span><i class="sq" style="background:#2dd4bf"></i>checks OK</span>'
        '<span><i class="sq" style="background:#f5a524"></i>warning</span>'
        '<span><i class="sq" style="background:#f87171"></i>failed</span>'
        '<span><i class="sq" style="background:#6b7480;height:5px"></i>no check recorded '
        "(half height)</span></p>"
    )
    return (
        '<section class="card"><div class="cardhead"><h2>Quality checks \u00b7 last 24 hours</h2>'
        '<span class="muted small">one cell per UTC hour \u00b7 newest on the right</span></div>'
        f"{body}{legend}</section>"
    )


def _section_quality(data: dict[str, Any]) -> str:
    q, fresh = data["quality"], data["freshness"]
    if q is None and fresh is None:
        body = _unavailable("Data quality")
    else:
        newest = (fresh or {}).get("newest", {})
        ids = sorted(set(newest) | set((q or {}).get("stations", {})))
        if not ids:
            body = '<p class="muted small">Waiting for the first station readings.</p>'
        else:
            now = datetime.fromisoformat(data["generated_at"])
            counts = dict.fromkeys(("ok", "warn", "fail", "none"), 0)
            for station_id in ids:
                info = (q or {}).get("stations", {}).get(station_id)
                counts[info["status"] if info else "none"] += 1
            summary = (
                '<div class="quality-summary">'
                f"<span>Stations <strong>{len(ids)}</strong></span>"
                f"<span>OK <strong>{counts['ok']}</strong></span>"
                f"<span>Needs attention <strong>{counts['warn'] + counts['fail']}</strong></span>"
                f"<span>Waiting <strong>{counts['none']}</strong></span></div>"
            )
            rows = []
            for sid in ids:
                info = (q or {}).get("stations", {}).get(sid)
                age = (
                    f"{(now - datetime.fromisoformat(newest[sid])) / timedelta(minutes=1):.0f} min"
                    if sid in newest
                    else "\u2013"
                )
                gap = info["missing_h"] if info else None
                missing = "\u2013" if gap is None else f"{gap:.0f} h"
                status = info["status"] if info else "none"
                word = {"ok": "OK", "warn": "Warn", "fail": "Fail"}.get(status, "Waiting")
                rows.append(
                    f'<tr data-quality-status="{status}"><td>{e(_station_label(sid))}</td>'
                    f"<td>{age}</td><td>{missing}</td>"
                    f'<td><span class="badge {status}">{word}</span></td></tr>'
                )
            body = (
                summary + '<div class="quality-tools" hidden><label>Search stations '
                '<input id="quality-search" type="search"></label> '
                '<label>Status <select id="quality-filter"><option value="attention">'
                'Needs attention</option><option value="all">All</option>'
                '<option value="ok">OK</option><option value="warn">Warn</option>'
                '<option value="fail">Fail</option><option value="none">Waiting</option>'
                '</select></label></div><div class="tablewrap" tabindex="0" role="region" '
                'aria-label="Data quality by station, scrollable"><table>'
                "<caption>Latest hourly check per station</caption><thead><tr>"
                '<th scope="col">Station</th><th scope="col">Newest reading</th>'
                '<th scope="col">Missing, 24 h</th><th scope="col">Status</th></tr></thead>'
                f"<tbody>{''.join(rows)}</tbody></table></div>"
                '<div class="quality-pagination" hidden><span id="quality-count"></span>'
                '<button id="quality-prev" type="button">Previous</button>'
                '<button id="quality-next" type="button">Next</button></div>'
            )
    return f'<div class="card grow"><h2>Data quality by station</h2>{body}</div>'


def _section_drift(data: dict[str, Any]) -> str:
    d = data["drift"]
    if d is None:
        body = _unavailable("Drift")
    elif not d["features"]:
        body = '<p class="muted small">Waiting for the first daily drift check.</p>'
    else:
        rows = []
        for f in d["features"]:
            psi = f["psi"]
            if psi is None:
                bar, text = "", "n/a"
            else:
                width = min(psi / PSI_BAR_MAX, 1) * 100
                cls = "fill alert" if f["alert"] else "fill plain"
                bar = f'<div class="{cls}" style="width:{width:.0f}%"></div>'
                suffix = " Alert" if f["alert"] else " Warning" if f["status"] == "warn" else ""
                text = f"{psi:.2f}" + suffix
            tick = '<div class="tick"></div>'
            rows.append(
                f'<div class="drow"><span class="name">{e(f["name"])}</span>'
                f'<div class="track">{bar}{tick}</div><span class="psi">{e(text)}</span></div>'
            )
        body = "".join(rows)
    return (
        '<div class="card grow"><h2>Drift \u00b7 last 7 days vs training data</h2>'
        f"{body}"
        '<p class="muted small">PSI measures how far recent inputs have moved from what the '
        "model was trained on. The threshold is 0.2; an alert requires two consecutive "
        "daily breaches."
        "</p></div>"
    )


def _section_models(data: dict[str, Any]) -> str:
    m = data["models"]
    if m is None:
        body = _unavailable("Model registry")
    else:
        retrain = m["last_retrain"]
        last = "none yet"
        if retrain is not None:
            last = f"{retrain['version']} \u00b7 {retrain['at'][:10]}"
        chal = m["challenger"] or "none configured"

        def item(label: str, value: str, mono: bool = True) -> str:
            cls = ' class="mono"' if mono else ""
            return f'<div><span class="muted">{label}</span><span{cls}>{e(value)}</span></div>'

        body = (
            '<div class="models">'
            + item(
                "Registered champion",
                m["champion"] if m["champion_registered"] else "none registered",
            )
            + item("Configured challenger (shadow)", chal)
            + item("Last retrain", last)
            + item("Promotion", m["promotion"], mono=False)
            + "</div>"
        )
    return f'<section class="card"><h2>Models</h2>{body}</section>'


STEPS = (
    ("Trigger", "cron-job.org starts the run at :07, :27, :47"),
    ("Collect", "Air4Thai and CAMS; OpenAQ backfill is manual"),
    ("Predict", "persistence + challenger (shadow), +6/+12/+24 h"),
    ("Score", "forecasts vs actuals once they arrive"),
    ("Monitor", "data quality, drift (PSI), alerts"),
    ("Retrain", "weekly; evaluate candidate before promotion"),
)


def _section_how() -> str:
    items = "".join(
        f'<li style="--i:{i}"><button type="button" class="arch-node" data-step="{i}" '
        f'aria-expanded="false" aria-controls="step-tooltip" style="--i:{i}">'
        f'<span class="arch-dot"></span><span class="step">{i + 1:02d}</span>'
        f'<span class="t">{e(t)}</span><span class="b">{e(b)}</span></button></li>'
        for i, (t, b) in enumerate(STEPS)
    )
    return (
        '<section class="card" aria-labelledby="how"><div class="archhead">'
        '<h2 id="how">How it works</h2>'
        '<span class="muted small">every 20 minutes, unattended \u00b7 $0/month</span>'
        '<button type="button" id="pause" aria-pressed="false">Pause animation</button></div>'
        f'<ol class="arch" id="arch" style="list-style:none">{items}</ol>'
        '<div id="step-tooltip" class="step-tooltip" role="tooltip" hidden></div>'
        '<div class="row muted small"><span>Storage: Neon Postgres (observations, predictions, '
        "scores, models)</span><span>Alerts: Discord \u00b7 daily heartbeat</span>"
        "<span>A candidate is served only after configuration and review</span></div></section>"
    )


def _section_story() -> str:
    return """<section class="story card">
<h2 data-en="Behind the forecast" data-th="เบื้องหลังการพยากรณ์">Behind the forecast</h2>
<div class="story-grid">
<article><h3 data-en="Data sources" data-th="แหล่งข้อมูล">Data sources</h3>
<p data-en="Air4Thai provides readings; OpenAQ backfills history; Open-Meteo provides CAMS forecasts." data-th="Air4Thai ให้ค่าตรวจวัด OpenAQ เติมข้อมูลย้อนหลัง และ Open-Meteo ให้ค่าพยากรณ์ CAMS">Air4Thai provides readings; OpenAQ backfills history; Open-Meteo provides CAMS forecasts.</p>
<div class="source-list"><details><summary>Air4Thai<span class="source-unmeasured" data-en="Coverage not measured" data-th="ยังไม่วัดความครบ">Coverage not measured</span></summary><p data-en="Station PM2.5 readings. Collection runs up to every 20 minutes; Air4Thai backfill runs daily." data-th="ข้อมูล PM2.5 จากสถานี เก็บข้อมูลสูงสุดทุก 20 นาที และเติมข้อมูลย้อนหลังทุกวัน">Station PM2.5 readings. Collection runs up to every 20 minutes; Air4Thai backfill runs daily.</p></details><details><summary>OpenAQ<span class="source-unmeasured" data-en="Coverage not measured" data-th="ยังไม่วัดความครบ">Coverage not measured</span></summary><p data-en="Historical PM2.5 backfill, run manually. Stored source labels do not prove complete source coverage." data-th="เติมข้อมูล PM2.5 ย้อนหลังด้วยตนเอง ป้ายแหล่งข้อมูลในฐานข้อมูลไม่ได้ยืนยันความครบของข้อมูลต้นทาง">Historical PM2.5 backfill, run manually. Stored source labels do not prove complete source coverage.</p></details><details><summary>Open-Meteo / CAMS<span class="source-unmeasured" data-en="Coverage not measured" data-th="ยังไม่วัดความครบ">Coverage not measured</span></summary><p data-en="CAMS forecast vintages collected up to every 20 minutes. Provider update schedules can differ." data-th="เก็บชุดพยากรณ์ CAMS สูงสุดทุก 20 นาที รอบอัปเดตของผู้ให้บริการอาจแตกต่างกัน">CAMS forecast vintages collected up to every 20 minutes. Provider update schedules can differ.</p></details></div></article>
<article><h3 data-en="Time-safe inputs" data-th="ข้อมูลเข้า ณ เวลาพยากรณ์">Time-safe inputs</h3>
<p data-en="Features use only information available by forecast time. Training and serving share the same feature builder." data-th="ใช้เฉพาะข้อมูลที่มี ณ เวลาพยากรณ์ และใช้วิธีสร้างตัวแปรเดียวกันทั้งตอนฝึกและใช้งาน">Features use only information available by forecast time. Training and serving share the same feature builder.</p><div class="feature-timeline" role="img" aria-label="Past features are available at forecast time; predictions extend to six, twelve and twenty-four hours"><div class="feature-point"><i></i><b data-en="Past" data-th="อดีต">Past</b><small data-en="Available inputs" data-th="ข้อมูลที่มี">Available inputs</small></div><div class="feature-point"><i></i><b data-en="Now" data-th="ตอนนี้">Now</b><small data-en="Issue time" data-th="เวลาพยากรณ์">Issue time</small></div><div class="feature-point"><i></i><b>+6h</b></div><div class="feature-point"><i></i><b>+12h</b></div><div class="feature-point"><i></i><b>+24h</b></div></div><p class="story-note" data-en="Up to every 20 minutes; issue times are hourly. Model horizons vary." data-th="ทำงานสูงสุดทุก 20 นาที เวลาออกพยากรณ์เป็นรายชั่วโมง ระยะพยากรณ์ขึ้นอยู่กับโมเดล">Up to every 20 minutes; issue times are hourly. Model horizons vary.</p></article>
<article><h3 data-en="Why five stations?" data-th="ทำไมเลือกห้าสถานี?">Why five stations?</h3>
<p data-en="The pilot selected the five stations with the fewest missing hourly readings in the Air4Thai probe." data-th="โครงการนำร่องเลือกห้าสถานีที่มีข้อมูลรายชั่วโมงขาดน้อยที่สุดจากการตรวจ Air4Thai">The pilot selected the five stations with the fewest missing hourly readings in the Air4Thai probe.</p>
<div class="selection-chart"><div class="selection-chart-head"><span data-en="SELECTED STATIONS · HISTORICAL PROBE" data-th="สถานีที่เลือก · ผลตรวจย้อนหลัง">SELECTED STATIONS · HISTORICAL PROBE</span><span>~87 days</span></div><div class="selection-row"><span data-en="Government Public Relations Dept." data-th="กรมประชาสัมพันธ์">Government Public Relations Dept.</span><i></i><b>~100%</b></div><div class="selection-row"><span data-en="Bodindecha School" data-th="โรงเรียนบดินทรเดชา">Bodindecha School</span><i></i><b>~100%</b></div><div class="selection-row"><span data-en="Bueng Kum District Office" data-th="สำนักงานเขตบึงกุ่ม">Bueng Kum District Office</span><i></i><b>~100%</b></div><div class="selection-row"><span data-en="Khlong Sam Wa District Office" data-th="สำนักงานเขตคลองสามวา">Khlong Sam Wa District Office</span><i></i><b>~100%</b></div><div class="selection-row"><span data-en="Bansomdejchaopraya Rajabhat Univ." data-th="มหาวิทยาลัยราชภัฏบ้านสมเด็จเจ้าพระยา">Bansomdejchaopraya Rajabhat Univ.</span><i></i><b>~100%</b></div><small data-en="Rounded values within the available ~87-day window in the 27 Oct 2026 probe. The full ranking was not retained." data-th="ค่าปัดในช่วงที่มีข้อมูลประมาณ 87 วัน จากการตรวจวันที่ 27 ต.ค. 2026 รายงานเดิมไม่ได้เก็บอันดับทั้งหมด">Rounded values within the available ~87-day window in the 27 Oct 2026 probe. The full ranking was not retained.</small></div><details class="story-details"><summary data-en="How to expand" data-th="แนวทางขยาย">How to expand</summary><p data-en="Check additional public stations and historical coverage before scaling ingestion, evaluation and monitoring." data-th="ตรวจสถานีสาธารณะและข้อมูลย้อนหลังเพิ่มเติมก่อนขยายการเก็บข้อมูล การประเมิน และการเฝ้าระวัง">Check additional public stations and historical coverage before scaling ingestion, evaluation and monitoring.</p></details></article>
<article><h3 data-en="Persistence baseline" data-th="วิธีพื้นฐาน Persistence">Persistence baseline</h3>
<p data-en="The latest usable reading is carried forward. A reading older than three hours is not used." data-th="ใช้ค่าล่าสุดที่ใช้ได้เป็นค่าพยากรณ์ หากข้อมูลเก่ากว่าสามชั่วโมงจะไม่ใช้">The latest usable reading is carried forward. A reading older than three hours is not used.</p>
<small data-en="Illustrative example" data-th="ตัวอย่างประกอบ">Illustrative example</small><svg class="explain-chart" viewBox="0 0 300 105" role="img" aria-label="Persistence example: the latest reading is carried forward to the forecast horizons."><polyline class="explain-actual" points="5,75 40,60 75,68 110,42 145,55"/><path class="explain-forecast" d="M145 55 H290"/><circle class="explain-current" cx="145" cy="55" r="4"/><circle class="explain-predicted" cx="195" cy="55" r="4"/><circle class="explain-predicted" cx="240" cy="55" r="4"/><circle class="explain-predicted" cx="290" cy="55" r="4"/><text x="132" y="93">Now</text><text x="183" y="93">+6h</text><text x="225" y="93">+12h</text><text x="273" y="93">+24h</text></svg>
<a href="https://otexts.com/fpp3/toolbox.html" data-en="Read about naïve forecasting ↗" data-th="อ่านการพยากรณ์แบบ naïve ↗">Read about naïve forecasting ↗</a></article>
<article><h3 data-en="LightGBM challenger" data-th="โมเดลท้าชิง LightGBM">LightGBM challenger</h3>
<p data-en="Decision trees learn corrections. The configured challenger runs in shadow and is evaluated against persistence." data-th="ต้นไม้ตัดสินใจเรียนรู้เพื่อแก้ความคลาดเคลื่อน โมเดลท้าชิงทำงานคู่ขนานและประเมินเทียบวิธีพื้นฐาน">Decision trees learn corrections. The configured challenger runs in shadow and is evaluated against persistence.</p>
<div class="boost-visual" role="img" aria-label="Illustrative gradient boosting: sequential decision trees combine corrections"><div class="boost-tree"><svg viewBox="0 0 52 45" aria-hidden="true"><path d="M26 6 L12 22 M26 6 L40 22 M12 22 L6 36 M12 22 L18 36 M40 22 L34 36 M40 22 L46 36"/><circle cx="26" cy="6" r="3"/><circle cx="12" cy="22" r="3"/><circle cx="40" cy="22" r="3"/><circle cx="6" cy="36" r="2.5"/><circle cx="18" cy="36" r="2.5"/><circle cx="34" cy="36" r="2.5"/><circle cx="46" cy="36" r="2.5"/></svg><span data-en="Tree 1" data-th="ต้นไม้ 1">Tree 1</span></div><b>+</b><div class="boost-tree"><svg viewBox="0 0 52 45" aria-hidden="true"><path d="M26 6 L12 22 M26 6 L40 22 M12 22 L6 36 M12 22 L18 36 M40 22 L34 36 M40 22 L46 36"/><circle cx="26" cy="6" r="3"/><circle cx="12" cy="22" r="3"/><circle cx="40" cy="22" r="3"/><circle cx="6" cy="36" r="2.5"/><circle cx="18" cy="36" r="2.5"/><circle cx="34" cy="36" r="2.5"/><circle cx="46" cy="36" r="2.5"/></svg><span data-en="Tree 2" data-th="ต้นไม้ 2">Tree 2</span></div><b>+</b><div class="boost-tree"><svg viewBox="0 0 52 45" aria-hidden="true"><path d="M26 6 L12 22 M26 6 L40 22 M12 22 L6 36 M12 22 L18 36 M40 22 L34 36 M40 22 L46 36"/><circle cx="26" cy="6" r="3"/><circle cx="12" cy="22" r="3"/><circle cx="40" cy="22" r="3"/><circle cx="6" cy="36" r="2.5"/><circle cx="18" cy="36" r="2.5"/><circle cx="34" cy="36" r="2.5"/><circle cx="46" cy="36" r="2.5"/></svg><span data-en="Tree 3" data-th="ต้นไม้ 3">Tree 3</span></div><b>=</b><span data-en="Forecast" data-th="พยากรณ์">Forecast</span></div>
<a href="https://lightgbm.readthedocs.io/" data-en="LightGBM documentation ↗" data-th="เอกสาร LightGBM ↗">LightGBM documentation ↗</a></article>
<article><h3 data-en="How we evaluate" data-th="วิธีประเมินผล">How we evaluate</h3>
<p data-en="MAE is average absolute error. Skill compares challenger and baseline MAE on matched station-hours; above zero favors the challenger." data-th="MAE คือค่าเฉลี่ยความคลาดเคลื่อนสัมบูรณ์ Skill เปรียบเทียบ MAE ของสองโมเดลในสถานีและเวลาเดียวกัน ค่าบวกหมายถึงโมเดลท้าชิงแม่นกว่า">MAE is average absolute error. Skill compares challenger and baseline MAE on matched station-hours; above zero favors the challenger.</p></article>
</div></section>"""  # noqa: E501


def render_html(data: dict[str, Any]) -> str:
    k, head = data["kpi"], data["headline"]
    chal = data["challenger_version"]
    status = data["status"]
    label = {
        "healthy": "Healthy",
        "degraded": "Degraded",
        "failed": "Failing",
        "waiting": "Waiting for data",
    }[status]
    unav = [n for n, v in data["sections"].items() if v == "unavailable"]
    note = ""
    if chal and head["state"] == "waiting":
        first = (data["scores"] or {}).get("first_scores_from")
        when = (
            f"from <strong>{datetime.fromisoformat(first):%d %b, %H:%M} UTC</strong>"
            if first
            else "once the first forecasts have matured"
        )
        note = (
            '<p class="note">The challenger\u2019s first +24 h forecasts are scored '
            f"{when}. Until then, its numbers below read \u201cwaiting\u201d.</p>"
        )
    elif not chal:
        note = '<p class="note">No challenger model is configured, so only the baseline runs.</p>'
    if unav:
        note += (
            f'<p class="note">Some data is unavailable right now: {e(", ".join(sorted(unav)))}.</p>'
        )

    skill = k["skill_24h"]
    skill_txt = None if skill is None else f"{skill * 100:+.1f}%".replace("-", "\u2212")
    mae_c, mae_p = k["mae_challenger_24h"], k["mae_persistence_24h"]
    if mae_c is not None:
        mae_txt = f"{mae_c:.1f}"
        mae_sub = "challenger" if mae_p is None else f"challenger \u00b7 persistence {mae_p:.1f}"
    else:
        mae_txt = None if mae_p is None else f"{mae_p:.1f}"
        mae_sub = "persistence only \u00b7 challenger not scored yet"
    fresh = k["freshness_min"]
    fresh_sub = (
        "no readings yet"
        if k["stations_total"] in (None, 0)
        else f"newest reading \u00b7 {k['stations_reporting']} of {k['stations_total']} "
        "stations reporting"
    )
    psi = k["worst_psi"]
    psi_sub = (
        "warning above 0.2"
        if psi is None
        else f"worst: {k['worst_psi_feature']} \u00b7 warning above 0.2"
    )

    def card(title: str, value: str, sub: str) -> str:
        return (
            f'<div class="card"><p class="muted small">{e(title)}</p>{value}'
            f'<p class="muted small">{e(sub)}</p></div>'
        )

    cards = "".join(
        [
            card(
                "Beats the baseline by",
                _val(skill_txt, good=skill is not None and skill > 0),
                "challenger vs persistence \u00b7 +24 h \u00b7 7 days",
            ),
            card("Typical error (MAE, \u00b5g/m\u00b3)", _val(mae_txt), mae_sub),
            card(
                "Data freshness",
                _val(None if fresh is None else f"{fresh:.0f} min"),
                fresh_sub,
            ),
            card("Input drift (PSI)", _val(None if psi is None else f"{psi:.2f}"), psi_sub),
        ]
    )
    answer_cls = {"yes": "yes", "not_yet": "notyet", "waiting": "waiting"}[head["state"]]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bangkok PM2.5 forecast \u00b7 live MLOps dashboard</title>
<link rel="stylesheet" href="dashboard.css?v=kan73-chart-layout-v2">
<script src="dashboard.js?v=kan73-chart-layout-v2" defer></script>
</head>
<body>
<main>
<header>
<div class="title">
<h1>Bangkok PM2.5 \u00b7 24 h forecast</h1>
<p class="muted">A live MLOps loop on free tiers: collect \u2192 predict \u2192 score \u2192 monitor \u2192 retrain</p>
<p class="headline">Is it beating the baseline of keeping the latest reading? <span class="answer {answer_cls}">{e(head["text"])}</span></p>
</div>
<div class="row" style="align-items:center">
<p class="status"><span class="dot {status}" aria-hidden="true"></span>{label} \u00b7 updated {_fmt_time(data["generated_at"])}</p>
<a href="{REPO_URL}">Source on GitHub</a>
<div id="language-switch" class="language-switch" role="group" aria-label="Dashboard language" hidden>
<button type="button" data-language="en" lang="en" aria-pressed="true">English</button>
<button type="button" data-language="th" lang="th" aria-pressed="false">ไทย</button>
</div>
</div>
</header>
{note}
<section aria-label="Key numbers" class="kpis">{cards}</section>
<section class="dashboard-main">{_section_map(data)}{_section_chart(data)}</section>
<section aria-label="Model performance" class="row">{_section_horizons(data)}</section>
{_section_pipeline(data)}
<section class="row">{_section_quality(data)}{_section_drift(data)}</section>
{_section_models(data)}
{_section_how()}
{_section_story()}
<footer>GitHub Actions \u00b7 Neon Postgres \u00b7 cron-job.org \u00b7 $0/month. Data: Air4Thai (Pollution Control Department, Thailand), OpenAQ, CAMS via Open-Meteo.</footer>
</main>
</body>
</html>
"""  # noqa: E501


def write_site(data: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False)
    # Still valid JSON (parses back identically) but never carries a literal <, > or & that could
    # be mistaken for markup if the file is ever served or embedded as HTML.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (out / "data.json").write_text(payload + "\n")
    (out / "index.html").write_text(render_html(data), encoding="utf-8")
    for asset in ("dashboard.css", "dashboard.js"):
        shutil.copyfile(ASSETS / asset, out / asset)
    shutil.copytree(ASSETS / "fonts", out / "fonts", dirs_exist_ok=True)  # self-hosted, OFL


def run(
    settings: Settings,
    out: Path,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    now = sql.require_utc(clock())
    data = collect(
        connect, settings.database_url.get_secret_value(), settings.challenger_model_version, now
    )
    if all(data["sections"].get(name) == "unavailable" for name in SECTIONS):
        logger.error("dashboard: every database section failed; nothing written")
        return EXIT_FAILED
    write_site(data, out)
    logger.info("dashboard: wrote %s (status %s)", out, data["status"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build the static PM2.5 dashboard")
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args(argv)
    try:
        return run(get_settings(), args.out)
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
