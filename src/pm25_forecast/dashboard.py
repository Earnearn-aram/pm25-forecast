"""Public dashboard builder (KAN-25). Run with ``python -m pm25_forecast.dashboard --out site/``.

Reads the database (read-only SELECTs, one read-only connection for the whole build) and writes
a static site:
``index.html`` (server-rendered, inline SVG charts), ``data.json`` (the same numbers), plus
``dashboard.css`` and ``dashboard.js`` (pills, pause, Thai / English switch).

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
from pm25_forecast.dashboard_i18n import Markup, attr, attr_of, date_hm, pair, t, tr
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


def _val(text: Markup | None, good: bool = False) -> str:
    if text is None:
        return f'<p class="val waiting">{t("waiting")}</p>'
    return f'<p class="val{" good" if good else ""}">{text}</p>'


def _plain(text: str) -> Markup:
    """A value that is the same in both languages (numbers, ids), escaped."""
    return Markup(e(text))


def _unavailable(what: str) -> str:
    text = pair(lambda L: tr(L, "{what} unavailable right now.", what=tr(L, what)))
    return f'<p class="muted small">{text}</p>'


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
        body = f'<p class="muted small">{t("No hourly data for this station yet.")}</p>'
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
        aria = attr(
            "aria-label",
            "Station {label}, last 7 days in UTC: actual PM2.5 in µg/m³ against "
            "persistence and challenger forecasts made 24 h earlier",
            label=st["label"],
        )
        start, end = datetime.fromisoformat(chart["start"]), datetime.fromisoformat(chart["end"])
        body = (
            '<div class="plotwrap"><div class="ytitle">µg/m³</div><div class="plot">'
            f'{labels}<svg viewBox="0 0 800 220" preserveAspectRatio="none" role="img" '
            f"{aria}>{grid}{series}</svg></div>"
            f'<div class="xlabs"><span>{pair(lambda L: date_hm(start, L))}</span>'
            f"<span>{t('UTC, hourly')}</span>"
            f"<span>{pair(lambda L: date_hm(end, L))}</span></div></div>"
        )
    hidden = "" if first else " hidden"
    return f'<div class="chart" data-station="{e(st["id"])}"{hidden}>{body}</div>'


def _section_chart(data: dict[str, Any]) -> str:
    chart = data["chart"]
    chal = (
        t("Challenger") if (chart or {}).get("has_challenger") else t("Challenger (not scored yet)")
    )
    legend = (
        f'<div class="legend"><span><i class="sw actual"></i>{t("Actual")}</span>'
        f'<span><i class="sw chal"></i>{chal}</span>'
        f'<span><i class="sw pers"></i>{t("Persistence (tomorrow = today)")}</span></div>'
    )
    if chart is None:
        inner = _unavailable("The chart")
        pills = ""
        title = t("Forecast vs actual · +24 h")
    elif not chart["stations"]:
        inner = f'<p class="muted small">{t("Waiting for the first observations.")}</p>'
        pills = ""
        title = t("Forecast vs actual · +24 h")
    else:
        sts = chart["stations"]
        station = Markup(f'<span class="chart-station">{e(sts[0]["label"])}</span>')
        title = t("Forecast vs actual · +24 h · {station}", station=station)
        pills = "".join(
            f'<button type="button" class="pill" data-station="{e(s["id"])}" '
            f'data-label="{e(s["label"])}" aria-label="{e(s["label"])}" '
            f'aria-pressed="{"true" if i == 0 else "false"}">{e(s["id"])}</button>'
            for i, s in enumerate(sts)
        )
        inner = "".join(_chart_block(s, chart, i == 0) for i, s in enumerate(sts))
    cap = t("Each point: the forecast made 24 h before that hour. Last 7 days, hourly, UTC.")
    return (
        '<figure class="card wide" style="margin:0"><div class="cardhead">'
        f'<h2 id="chart-title">{title}</h2>'
        f'<div class="pills" role="group" {attr("aria-label", "Station")}>{pills}</div></div>'
        f'{legend}{inner}<figcaption class="muted small">{cap}</figcaption></figure>'
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
                value, bar = t("waiting"), ""
            else:
                width = min(abs(skill) / SKILL_BAR_MAX, 1) * 100
                value = f"{skill * 100:+.1f}%".replace("-", "\u2212")
                cls = "fill" if skill > 0 else "fill neg"
                bar = f'<div class="{cls}" style="width:{width:.0f}%"></div>'
            rows.append(
                f'<div class="hrow"><div class="top"><span>{t("+{h} h", h=h)}</span>'
                f'<span class="mono">{value}</span></div><div class="track">{bar}</div></div>'
            )
        body = "".join(rows)
    note = t(
        "1 \u2212 MAE challenger ÷ MAE persistence, on the same station-hours. "
        "Above 0 means the model beats “tomorrow = today”."
    )
    return (
        f'<div class="card side"><h2>{t("Skill by horizon")}</h2>{body}'
        f'<p class="muted small">{note}</p></div>'
    )


def _section_pipeline(data: dict[str, Any]) -> str:
    q = data["quality"]
    if q is None:
        body = _unavailable("Pipeline history")
    elif not q["has_checks"]:
        body = f'<p class="muted small">{t("Waiting for the first hourly quality run.")}</p>'
    else:
        rows = []
        for line in q["pipeline"]:
            counts = {c: line["cells"].count(c) for c in ("ok", "warn", "fail", "missed")}
            aria = attr_of(
                "aria-label",
                lambda L, line=line, counts=counts: tr(
                    L,
                    "{name}: {ok} of {total} hours ok, {warn} warning, {fail} failed, "
                    "{missed} missed",
                    name=tr(L, line["name"]),
                    total=PIPELINE_HOURS,
                    **counts,
                ),
            )
            cells = "".join(
                f'<span class="cell {"miss" if c == "missed" else c}"></span>'
                for c in line["cells"]
            )
            rows.append(
                f"<span>{t(line['name'])}</span>"
                f'<div class="cells" role="img" {aria}>{cells}</div>'
                f'<span class="mono muted">'
                f"{t('{ok}/{total} ok', ok=line['ok'], total=PIPELINE_HOURS)}</span>"
            )
        body = f'<div class="pipe">{"".join(rows)}</div>'
    legend = (
        f'<p class="legend"><span><i class="sq" style="background:#2dd4bf"></i>{t("ran ok")}</span>'
        f'<span><i class="sq" style="background:#f5a524"></i>{t("warning")}</span>'
        f'<span><i class="sq" style="background:#f87171"></i>{t("failed")}</span>'
        '<span><i class="sq" style="background:#6b7480;height:5px"></i>'
        f"{t('no run recorded (half height)')}</span></p>"
    )
    return (
        f'<section class="card"><div class="cardhead"><h2>{t("Pipeline · last 24 hours")}</h2>'
        f'<span class="muted small">{t("one cell per UTC hour · newest on the right")}</span>'
        f"</div>{body}{legend}</section>"
    )


def _section_quality(data: dict[str, Any]) -> str:
    q, fresh = data["quality"], data["freshness"]
    if q is None and fresh is None:
        body = _unavailable("Data quality")
    else:
        newest = (fresh or {}).get("newest", {})
        ids = sorted(set(newest) | set((q or {}).get("stations", {})))
        if not ids:
            body = f'<p class="muted small">{t("Waiting for the first station readings.")}</p>'
        else:
            now = datetime.fromisoformat(data["generated_at"])
            rows = []
            for sid in ids:
                info = (q or {}).get("stations", {}).get(sid)
                minutes = (
                    (now - datetime.fromisoformat(newest[sid])) / timedelta(minutes=1)
                    if sid in newest
                    else None
                )
                age = "\u2013" if minutes is None else t("{n} min", n=f"{minutes:.0f}")
                gap = info["missing_h"] if info else None
                missing = "\u2013" if gap is None else t("{n} h", n=f"{gap:.0f}")
                status = info["status"] if info else "none"
                word = t({"ok": "OK", "warn": "Warn", "fail": "Fail"}.get(status, "Waiting"))
                rows.append(
                    f"<tr><td>{e(_station_label(sid))}</td><td>{age}</td><td>{missing}</td>"
                    f'<td><span class="badge {status}">{word}</span></td></tr>'
                )
            heads = "".join(
                f'<th scope="col">{t(k)}</th>'
                for k in ("Station", "Newest reading", "Missing, 24 h", "Status")
            )
            body = (
                '<div class="tablewrap" tabindex="0" role="region" '
                f"{attr('aria-label', 'Data quality by station, scrollable')}><table>"
                f"<caption>{t('Latest hourly check per station')}</caption>"
                f"<thead><tr>{heads}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            )
    return f'<div class="card grow"><h2>{t("Data quality by station")}</h2>{body}</div>'


def _section_drift(data: dict[str, Any]) -> str:
    d = data["drift"]
    if d is None:
        body = _unavailable("Drift")
    elif not d["features"]:
        body = f'<p class="muted small">{t("Waiting for the first daily drift check.")}</p>'
    else:
        rows = []
        for f in d["features"]:
            psi = f["psi"]
            if psi is None:
                bar, text = "", t("n/a")
            else:
                width = min(psi / PSI_BAR_MAX, 1) * 100
                cls = "fill alert" if f["alert"] else "fill plain"
                bar = f'<div class="{cls}" style="width:{width:.0f}%"></div>'
                text = (
                    pair(lambda L, psi=psi: f"{psi:.2f} {tr(L, 'Alert')}")
                    if f["alert"]
                    else _plain(f"{psi:.2f}")
                )
            tick = '<div class="tick"></div>'
            rows.append(
                f'<div class="drow"><span class="name">{e(f["name"])}</span>'
                f'<div class="track">{bar}{tick}</div><span class="psi">{text}</span></div>'
            )
        body = "".join(rows)
    note = t(
        "PSI measures how far recent inputs have moved from what the model was trained on. "
        "Tick = 0.2 alert line; bars above it turn amber and say Alert."
    )
    return (
        f'<div class="card grow"><h2>{t("Drift · last 7 days vs training data")}</h2>'
        f'{body}<p class="muted small">{note}</p></div>'
    )


def _section_models(data: dict[str, Any]) -> str:
    m = data["models"]
    if m is None:
        body = _unavailable("Model registry")
    else:
        retrain = m["last_retrain"]
        last = (
            t("none yet")
            if retrain is None
            else _plain(f"{retrain['version']} · {retrain['at'][:10]}")
        )
        chal = _plain(m["challenger"]) if m["challenger"] else t("none configured")

        def item(label: str, value: Markup, mono: bool = True) -> str:
            cls = ' class="mono"' if mono else ""
            return f'<div><span class="muted">{t(label)}</span><span{cls}>{value}</span></div>'

        body = (
            '<div class="models">'
            + item("Champion (serves)", _plain(m["champion"]))
            + item("Challenger (shadow)", chal)
            + item("Last retrain", last)
            + item("Promotion", t(m["promotion"]), mono=False)
            + "</div>"
        )
    return f'<section class="card"><h2>{t("Models")}</h2>{body}</section>'


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
        f'<span class="step">{i + 1:02d}</span><span class="t">{t(title)}</span>'
        f'<span class="b">{t(body)}</span></div></li>'
        for i, (title, body) in enumerate(STEPS)
    )
    return (
        '<section class="card" aria-labelledby="how"><div class="archhead">'
        f'<h2 id="how">{t("How it works")}</h2>'
        f'<span class="muted small">{t("every 20 minutes, unattended · $0/month")}</span>'
        f'<button type="button" id="pause" aria-pressed="false">{t("Pause animation")}</button>'
        f'</div><ol class="arch" id="arch" style="list-style:none">{items}</ol>'
        '<div class="row muted small">'
        f"<span>{t('Storage: Neon Postgres (observations, predictions, scores, models)')}</span>"
        f"<span>{t('Alerts: Discord · daily heartbeat')}</span>"
        f"<span>{t('Weekly retrain feeds a new challenger back into Predict')}</span>"
        "</div></section>"
    )


STATUS_LABELS = {
    "healthy": "Healthy",
    "degraded": "Degraded",
    "failed": "Failing",
    "waiting": "Waiting for data",
}


def _headline_text(head: dict[str, Any]) -> Markup:
    """The headline answer in both languages, from the state and rounded percent in data.json."""
    state, pct = head["state"], head["skill_pct"]
    if state == "waiting" or pct is None:
        return t("Not scored yet")
    if state == "yes":
        return t("Yes, by {pct}%", pct=f"{pct:.1f}")
    sign = "\u2212" if pct < 0 else ""
    return t("Not yet ({sign}{pct}%)", sign=sign, pct=f"{abs(pct):.1f}")


def _note(data: dict[str, Any]) -> str:
    chal, head = data["challenger_version"], data["headline"]
    note = ""
    if chal and head["state"] == "waiting":
        first = (data["scores"] or {}).get("first_scores_from")

        def when(lang: str) -> Markup:
            if not first:
                return tr(lang, "once the first forecasts have matured")
            dt = date_hm(datetime.fromisoformat(first), lang, comma=True)
            return tr(lang, "from {dt}", dt=Markup(f"<strong>{dt} UTC</strong>"))

        text = pair(
            lambda L: tr(
                L,
                "The challenger\u2019s first +24 h forecasts are scored {when}. "
                "Until then, its numbers below read “waiting”.",
                when=when(L),
            )
        )
        note = f'<p class="note">{text}</p>'
    elif not chal:
        note = (
            f'<p class="note">'
            f"{t('No challenger model is configured, so only the baseline runs.')}</p>"
        )
    unav = sorted(n for n, v in data["sections"].items() if v == "unavailable")
    if unav:
        text = pair(
            lambda L: tr(
                L,
                "Some data is unavailable right now: {names}.",
                names=", ".join(tr(L, n) for n in unav),
            )
        )
        note += f'<p class="note">{text}</p>'
    return note


def _cards(k: dict[str, Any]) -> str:
    skill = k["skill_24h"]
    skill_txt = None if skill is None else _plain(f"{skill * 100:+.1f}%".replace("-", "\u2212"))
    mae_c, mae_p = k["mae_challenger_24h"], k["mae_persistence_24h"]
    if mae_c is not None:
        mae_txt = _plain(f"{mae_c:.1f}")
        mae_sub = (
            t("challenger")
            if mae_p is None
            else t("challenger · persistence {mae}", mae=f"{mae_p:.1f}")
        )
    else:
        mae_txt = None if mae_p is None else _plain(f"{mae_p:.1f}")
        mae_sub = t("persistence only · challenger not scored yet")
    fresh = k["freshness_min"]
    fresh_sub = (
        t("no readings yet")
        if k["stations_total"] in (None, 0)
        else t(
            "newest reading · {up} of {total} stations reporting",
            up=k["stations_reporting"],
            total=k["stations_total"],
        )
    )
    psi = k["worst_psi"]
    psi_sub = (
        t("alert above 0.2")
        if psi is None
        else t("worst: {feature} · alert above 0.2", feature=k["worst_psi_feature"])
    )

    def card(title: Markup, value: str, sub: Markup) -> str:
        return (
            f'<div class="card"><p class="muted small">{title}</p>{value}'
            f'<p class="muted small">{sub}</p></div>'
        )

    return "".join(
        [
            card(
                t("Beats the baseline by"),
                _val(skill_txt, good=skill is not None and skill > 0),
                t("challenger vs persistence · +24 h · 7 days"),
            ),
            card(t("Typical error (MAE, µg/m³)"), _val(mae_txt), mae_sub),
            card(
                t("Data freshness"),
                _val(None if fresh is None else t("{n} min", n=f"{fresh:.0f}")),
                fresh_sub,
            ),
            card(
                t("Input drift (PSI)"),
                _val(None if psi is None else _plain(f"{psi:.2f}")),
                psi_sub,
            ),
        ]
    )


def _lang_toggle() -> str:
    return (
        f'<div class="lang" role="group" {attr("aria-label", "Language")}>'
        '<button type="button" class="langbtn" data-lang="th" lang="th" aria-pressed="false" '
        'aria-label="TH \u2013 ภาษาไทย">TH</button>'
        '<button type="button" class="langbtn" data-lang="en" lang="en" aria-pressed="true" '
        'aria-label="EN \u2013 English">EN</button></div>'
    )


# Runs before first paint (no flash of English): saved choice, else the browser language if Thai.
LANG_INIT = (
    "(function(){var l='en';try{var s=localStorage.getItem('pm25-lang');"
    "if(s==='th'||s==='en'){l=s}else if(/^th/i.test(navigator.language||'')){l='th'}}"
    "catch(x){if(/^th/i.test(navigator.language||'')){l='th'}}"
    "document.documentElement.lang=l})();"
)


def render_html(data: dict[str, Any]) -> str:
    k, head = data["kpi"], data["headline"]
    status = data["status"]
    label = pair(
        lambda L: tr(
            L,
            "{label} · updated {when}",
            label=tr(L, STATUS_LABELS[status]),
            when=tr(
                L,
                "{utc} UTC · {bkk} Bangkok",
                utc=f"{datetime.fromisoformat(data['generated_at']).astimezone(UTC):%H:%M}",
                bkk=f"{datetime.fromisoformat(data['generated_at']).astimezone(BANGKOK):%H:%M}",
            ),
        )
    )
    answer_cls = {"yes": "yes", "not_yet": "notyet", "waiting": "waiting"}[head["state"]]
    title_key = "Bangkok PM2.5 forecast · live MLOps dashboard"
    title_en, title_th = tr("en", title_key), tr("th", title_key)
    footer = t(
        "GitHub Actions · Neon Postgres · cron-job.org · $0/month. Data: Air4Thai "
        "(Pollution Control Department, Thailand), OpenAQ, CAMS via Open-Meteo."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title data-en="{e(title_en)}" data-th="{e(title_th)}">{e(title_en)}</title>
<script>{LANG_INIT}</script>
<link rel="stylesheet" href="dashboard.css">
<script src="dashboard.js" defer></script>
</head>
<body>
<main>
<header>
<div class="title">
<h1>{t("Bangkok PM2.5 · 24 h forecast")}</h1>
<p class="muted">{t("A live MLOps loop on free tiers: collect → predict → score → monitor → retrain")}</p>
<p class="headline">{t("Is it beating the naive guess “tomorrow = today”?")} <span class="answer {answer_cls}">{_headline_text(head)}</span></p>
</div>
<div class="row" style="align-items:center">
{_lang_toggle()}
<p class="status"><span class="dot {status}" aria-hidden="true"></span>{label}</p>
<a href="{REPO_URL}">{t("Source on GitHub")}</a>
</div>
</header>
{_note(data)}
<section {attr("aria-label", "Key numbers")} class="kpis">{_cards(k)}</section>
<section class="row">{_section_chart(data)}{_section_horizons(data)}</section>
{_section_pipeline(data)}
<section class="row">{_section_quality(data)}{_section_drift(data)}</section>
{_section_models(data)}
{_section_how()}
<footer>{footer}</footer>
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
