"""Public dashboard builder (KAN-25). Run with ``python -m pm25_forecast.dashboard --out site/``.

Reads the database (read-only SELECTs, one short connection per query) and writes a static site:
``index.html`` (server-rendered, inline SVG charts), ``data.json`` (the same numbers), plus
``dashboard.css`` and ``dashboard.js`` (station pills and the animation pause only).

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
STATION_NAMES: dict[str, str] = {}
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
SELECT_FIRST_ISSUE = "SELECT min(issue_time) FROM predictions WHERE model_version = %(m)s"


class Reader:
    """One short connection per query, so a failed query cannot poison the next section."""

    def __init__(self, connect: Callable[[str], Any], url: str):
        self._connect, self._url = connect, url

    def query(self, statement: str, params: dict[str, Any]) -> list[Sequence[Any]]:
        with open_db(self._connect, self._url) as con:
            return list(con.execute(statement, params).fetchall())


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
    return f"{name} ({station_id})" if name else station_id


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
        raw = rd.query(SELECT_FIRST_ISSUE, {"m": challenger})
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
    actual: dict[str, dict[datetime, float]] = {}
    for o in obs:
        actual.setdefault(o.station_id, {})[_hour(o.ts)] = o.pm25
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
        stations.append({"id": sid, "label": _station_label(sid), "points": points})
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
            "alert": r.psi is not None and math.isfinite(r.psi) and r.psi > PSI_ALERT,
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
        "promotion": "Manual, after 1\u20132 weeks of scores",
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


def _lines(points: list[dict[str, Any]], key: str, ymax: float, css: str) -> str:
    w, h, pad = 800, 220, 8
    n = max(len(points) - 1, 1)
    segments: list[list[str]] = [[]]
    for i, p in enumerate(points):
        v = p[key]
        if v is None:
            segments.append([])
            continue
        y = h - pad - min(v, ymax) / ymax * (h - 2 * pad)
        segments[-1].append(f"{i / n * w:.1f},{y:.1f}")
    return "".join(
        f'<polyline class="ln {css}" vector-effect="non-scaling-stroke" points="{" ".join(s)}"/>'
        for s in segments
        if len(s) > 1
    )


def _chart_block(st: dict[str, Any], chart: dict[str, Any], first: bool) -> str:
    points = st["points"]
    keys = ["actual", "persistence"] + (["challenger"] if chart["has_challenger"] else [])
    values = [p[k] for p in points for k in keys if p[k] is not None]
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
        series = _lines(points, "persistence", ymax, "ln-pers")
        if chart["has_challenger"]:
            series += _lines(points, "challenger", ymax, "ln-chal")
        series += _lines(points, "actual", ymax, "ln-actual")
        aria = (
            f"Station {st['label']}, last 7 days in UTC: actual PM2.5 in \u00b5g/m\u00b3 against "
            "persistence and challenger forecasts made 24 h earlier"
        )
        start, end = datetime.fromisoformat(chart["start"]), datetime.fromisoformat(chart["end"])
        body = (
            '<div class="plotwrap"><div class="ytitle">\u00b5g/m\u00b3</div><div class="plot">'
            f'{labels}<svg viewBox="0 0 800 220" preserveAspectRatio="none" role="img" '
            f'aria-label="{e(aria)}">{grid}{series}</svg></div>'
            f'<div class="xlabs"><span>{start:%d %b %H:%M}</span><span>UTC, hourly</span>'
            f"<span>{end:%d %b %H:%M}</span></div></div>"
        )
    hidden = "" if first else " hidden"
    return f'<div class="chart" data-station="{e(st["id"])}"{hidden}>{body}</div>'


def _section_chart(data: dict[str, Any]) -> str:
    chart = data["chart"]
    chal = "Challenger" if (chart or {}).get("has_challenger") else "Challenger (not scored yet)"
    legend = (
        '<div class="legend"><span><i class="sw actual"></i>Actual</span>'
        f'<span><i class="sw chal"></i>{chal}</span>'
        '<span><i class="sw pers"></i>Persistence (tomorrow = today)</span></div>'
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
            f'aria-pressed="{"true" if i == 0 else "false"}">{e(s["id"])}</button>'
            for i, s in enumerate(sts)
        )
        inner = "".join(_chart_block(s, chart, i == 0) for i, s in enumerate(sts))
    return (
        '<figure class="card wide" style="margin:0"><div class="cardhead">'
        f'<h2 id="chart-title">{e(title)}</h2>'
        f'<div class="pills" role="group" aria-label="Station">{pills}</div></div>'
        f"{legend}{inner}"
        '<figcaption class="muted small">Each point: the forecast made 24 h before that hour. '
        "Last 7 days, hourly, UTC.</figcaption></figure>"
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
            rows.append(
                f"<span>{e(line['name'])}</span>"
                f'<div class="cells" role="img" aria-label="{e(aria)}">{cells}</div>'
                f'<span class="mono muted">{line["ok"]}/{PIPELINE_HOURS} ok</span>'
            )
        body = f'<div class="pipe">{"".join(rows)}</div>'
    legend = (
        '<p class="legend"><span><i class="sq" style="background:#2dd4bf"></i>ran ok</span>'
        '<span><i class="sq" style="background:#f5a524"></i>warning</span>'
        '<span><i class="sq" style="background:#f87171"></i>failed</span>'
        '<span><i class="sq" style="background:#6b7480;height:5px"></i>no run recorded '
        "(half height)</span></p>"
    )
    return (
        '<section class="card"><div class="cardhead"><h2>Pipeline \u00b7 last 24 hours</h2>'
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
                    f"<tr><td>{e(_station_label(sid))}</td><td>{age}</td><td>{missing}</td>"
                    f'<td><span class="badge {status}">{word}</span></td></tr>'
                )
            body = (
                '<div class="tablewrap" tabindex="0" role="region" '
                'aria-label="Data quality by station, scrollable"><table>'
                "<caption>Latest hourly check per station</caption><thead><tr>"
                '<th scope="col">Station</th><th scope="col">Newest reading</th>'
                '<th scope="col">Missing, 24 h</th><th scope="col">Status</th></tr></thead>'
                f"<tbody>{''.join(rows)}</tbody></table></div>"
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
                text = f"{psi:.2f}" + (" Alert" if f["alert"] else "")
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
        "model was trained on. Tick = 0.2 alert line; bars above it turn amber and say Alert."
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
            + item("Champion (serves)", m["champion"])
            + item("Challenger (shadow)", chal)
            + item("Last retrain", last)
            + item("Promotion", m["promotion"], mono=False)
            + "</div>"
        )
    return f'<section class="card"><h2>Models</h2>{body}</section>'


STEPS = (
    ("Trigger", "cron-job.org starts the run at :07, :27, :47"),
    ("Collect", "Air4Thai, OpenAQ and CAMS into Postgres"),
    ("Predict", "persistence + challenger (shadow), +6/+12/+24 h"),
    ("Score", "forecasts vs actuals once they arrive"),
    ("Monitor", "data quality, drift (PSI), alerts"),
    ("Retrain", "weekly; promote only if it beats the champion"),
)


def _section_how() -> str:
    items = "".join(
        f'<li style="--i:{i}"><div class="arch-node" style="--i:{i}"><span class="arch-dot"></span>'
        f'<span class="step">{i + 1:02d}</span><span class="t">{e(t)}</span>'
        f'<span class="b">{e(b)}</span></div></li>'
        for i, (t, b) in enumerate(STEPS)
    )
    return (
        '<section class="card" aria-labelledby="how"><div class="archhead">'
        '<h2 id="how">How it works</h2>'
        '<span class="muted small">every 20 minutes, unattended \u00b7 $0/month</span>'
        '<button type="button" id="pause" aria-pressed="false">Pause animation</button></div>'
        f'<ol class="arch" id="arch" style="list-style:none">{items}</ol>'
        '<div class="row muted small"><span>Storage: Neon Postgres (observations, predictions, '
        "scores, models)</span><span>Alerts: Discord \u00b7 daily heartbeat</span>"
        "<span>Weekly retrain feeds a new challenger back into Predict</span></div></section>"
    )


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
        "alert above 0.2"
        if psi is None
        else f"worst: {k['worst_psi_feature']} \u00b7 alert above 0.2"
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap">
<link rel="stylesheet" href="dashboard.css">
<script src="dashboard.js" defer></script>
</head>
<body>
<main>
<header>
<div class="title">
<h1>Bangkok PM2.5 \u00b7 24 h forecast</h1>
<p class="muted">A live MLOps loop on free tiers: collect \u2192 predict \u2192 score \u2192 monitor \u2192 retrain</p>
<p class="headline">Is it beating the naive guess \u201ctomorrow = today\u201d? <span class="answer {answer_cls}">{e(head["text"])}</span></p>
</div>
<div class="row" style="align-items:center">
<p class="status"><span class="dot {status}" aria-hidden="true"></span>{label} \u00b7 updated {_fmt_time(data["generated_at"])}</p>
<a href="{REPO_URL}">Source on GitHub</a>
</div>
</header>
{note}
<section aria-label="Key numbers" class="kpis">{cards}</section>
<section class="row">{_section_chart(data)}{_section_horizons(data)}</section>
{_section_pipeline(data)}
<section class="row">{_section_quality(data)}{_section_drift(data)}</section>
{_section_models(data)}
{_section_how()}
<footer>GitHub Actions \u00b7 Neon Postgres \u00b7 cron-job.org \u00b7 $0/month. Data: Air4Thai (Pollution Control Department, Thailand), OpenAQ, CAMS via Open-Meteo.</footer>
</main>
</body>
</html>
"""  # noqa: E501


def write_site(data: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (out / "index.html").write_text(render_html(data), encoding="utf-8")
    for asset in ("dashboard.css", "dashboard.js"):
        shutil.copyfile(ASSETS / asset, out / asset)


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
