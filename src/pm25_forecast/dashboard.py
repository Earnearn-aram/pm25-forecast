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
from pm25_forecast.drift import DATA_SOURCE_BREAK, PSI_ALERT
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
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
SECTIONS = ("scores", "observations", "chart", "quality", "drift", "models", "serving")
OPENAQ = "openaq"
COVERAGE_HOURS = (24, 168)  # the two windows (last 24 h, last 7 days), in hours

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


class CovObsRow(BaseModel):
    station_id: str
    ts: AwareDatetime


class CovNewest(BaseModel):
    ts: AwareDatetime | None = None


class CovOpenaqRow(BaseModel):
    station_id: str
    n: int


class DriftRow(BaseModel):
    checked_at: AwareDatetime
    model_version: str
    feature: str
    psi: float | None = None
    status: str


class ChampionRow(BaseModel):
    model_version: str
    promoted_at: AwareDatetime


class ServingChoiceRow(BaseModel):
    issue_time: AwareDatetime
    station_id: str
    valid_time: AwareDatetime
    model_version: str
    reason: str


class ServedScoreRow(BaseModel):
    model_version: str
    station_id: str
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    abs_error: float


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
SELECT_SERVING = """
SELECT issue_time, station_id, valid_time, model_version, reason FROM serving_forecasts
WHERE issue_time = (SELECT max(issue_time) FROM serving_forecasts WHERE issue_time <= %(now)s)
ORDER BY station_id, valid_time
"""
SERVED_REFERENCE = "__served_persistence_reference__"
SELECT_RETRAIN = """
SELECT model_version, created_at FROM models WHERE model_version LIKE %(prefix)s
ORDER BY created_at DESC LIMIT 1
"""
SELECT_FIRST_ISSUE = """
SELECT issue_time FROM predictions
WHERE model_version = %(m)s AND issue_time >= %(since)s ORDER BY issue_time LIMIT 1
"""
SELECT_COV_OBS = """
SELECT station_id, ts FROM observations
WHERE source = %(source)s AND ts >= %(since)s AND ts <= %(now)s AND pm25 IS NOT NULL
"""
SELECT_COV_NEWEST_OBS = """
SELECT max(ts) FROM observations
WHERE source = %(source)s AND ts <= %(now)s AND pm25 IS NOT NULL
"""
SELECT_COV_OPENAQ = """
SELECT station_id, count(*) FROM observations
WHERE source = %(source)s AND ts < %(until)s AND pm25 IS NOT NULL GROUP BY station_id
"""
SELECT_COV_OPENAQ_EDGE = """
SELECT ts FROM observations
WHERE source = %(source)s AND ts < %(until)s AND pm25 IS NOT NULL
AND station_id IN ({stations})
ORDER BY ts {order} LIMIT 1
"""
SELECT_COV_FIRST_OBS = """
SELECT ts FROM observations
WHERE source = %(source)s AND ts <= %(now)s AND pm25 IS NOT NULL ORDER BY ts LIMIT 1
"""
SELECT_COV_FIRST_VINTAGE = """
SELECT issue_time FROM predictions
WHERE model_version = %(m)s AND issue_time <= %(now)s ORDER BY issue_time LIMIT 1
"""
SELECT_COV_VINTAGES = """
SELECT issue_time FROM predictions
WHERE model_version = %(m)s AND issue_time >= %(since)s AND issue_time <= %(now)s
GROUP BY issue_time
"""
SELECT_COV_NEWEST_VINTAGE = """
SELECT max(issue_time) FROM predictions WHERE model_version = %(m)s AND issue_time <= %(now)s
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


# --- source coverage (KAN-102) -------------------------------------------------------------------


def _pct(covered: int, expected: int) -> float | None:
    """Percent to one decimal, rounded down so that a gap never displays as 100%."""
    if expected <= 0:
        return None
    return math.floor(min(covered / expected, 1.0) * 1000) / 10


def _newest(rd: Reader, statement: str, params: dict[str, Any], now: datetime) -> dict[str, Any]:
    rows = _rows(CovNewest, rd.query(statement, params))
    ts = rows[0].ts if rows else None
    age = None if ts is None else max(0, int((now - ts).total_seconds() // 60))
    return {"newest_at": _iso(ts), "newest_age_min": age}


def _cov_unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def _windows(end: datetime, first: datetime | None) -> dict[int, tuple[datetime, int]]:
    """Per window: (start, hours). The start never precedes the first stored hour, so a source
    that started collecting recently is measured since then, not against hours it never had."""
    out: dict[int, tuple[datetime, int]] = {}
    for hours in COVERAGE_HOURS:
        start = end - timedelta(hours=hours - 1)
        if first is not None:
            start = max(start, _hour(first))
        out[hours] = (start, max(0, int((end - start).total_seconds() // 3600) + 1))
    return out


def _first_ts(rd: Reader, statement: str, params: dict[str, Any]) -> datetime | None:
    rows = _rows(CovNewest, rd.query(statement, params))
    return rows[0].ts if rows else None


def _cov_air4thai(rd: Reader, now: datetime) -> dict[str, Any]:
    """Expected = every one of the pilot stations has a reading in every UTC hour of the window.

    The window ends at the last hour whose reading is due (PIPELINE_GRACE after the hour); a
    reading is a stored non-null PM2.5 value. Several rows in one hour count once.
    """
    end = _hour(now - PIPELINE_GRACE)
    longest = max(COVERAGE_HOURS)
    since = end - timedelta(hours=longest - 1)
    rows = _rows(
        CovObsRow,
        rd.query(SELECT_COV_OBS, {"source": AIR4THAI, "since": since, "now": now}),
    )
    newest = _newest(rd, SELECT_COV_NEWEST_OBS, {"source": AIR4THAI, "now": now}, now)
    if not rows and newest["newest_at"] is None:
        return _cov_unavailable("no_rows")
    first = _first_ts(rd, SELECT_COV_FIRST_OBS, {"source": AIR4THAI, "now": now})
    cells = {(r.station_id, _hour(r.ts)) for r in rows if r.station_id in STATION_MAP}
    out: dict[str, Any] = {"available": True, "stations": len(STATION_MAP), **newest}
    for hours, (start, n) in _windows(end, first).items():
        got = sum(1 for _sid, h in cells if start <= h <= end)
        out[f"pct_{hours}h"] = _pct(got, n * len(STATION_MAP))
    out["since"] = _iso(None if first is None else _hour(first))
    out["clamped"] = first is not None and _hour(first) > end - timedelta(
        hours=max(COVERAGE_HOURS) - 1
    )
    return out


def _cov_openaq(rd: Reader, now: datetime) -> dict[str, Any]:
    """Expected = every pilot station has a row in every hour from its first OpenAQ hour to
    the Air4Thai era start (exclusive). OpenAQ is a history backfill only; nothing live."""
    params = {"source": OPENAQ, "until": DATA_SOURCE_BREAK}
    rows = [
        r
        for r in _rows(CovOpenaqRow, rd.query(SELECT_COV_OPENAQ, params))
        if r.station_id in STATION_MAP
    ]
    if not rows:
        return _cov_unavailable("no_rows")
    ids = {f"s{i}": sid for i, sid in enumerate(STATION_MAP)}
    marks = ", ".join(f"%({k})s" for k in ids)  # only our own placeholders are formatted in
    edges = [
        _rows(
            CovNewest,
            rd.query(SELECT_COV_OPENAQ_EDGE.format(order=o, stations=marks), {**params, **ids}),
        )
        for o in ("ASC", "DESC")
    ]
    if not edges[0] or not edges[1] or edges[0][0].ts is None or edges[1][0].ts is None:
        return _cov_unavailable("no_rows")
    first_ts, last_ts = edges[0][0].ts, edges[1][0].ts
    first = _hour(first_ts)
    span_h = int((DATA_SOURCE_BREAK - first).total_seconds() // 3600)
    # count(*) equals distinct station-hours: the primary key is (station_id, ts) and OpenAQ
    # rows are hour-aligned. _pct clamps at 100% in case that ever stops holding.
    return {
        "available": True,
        "pct": _pct(sum(r.n for r in rows), span_h * len(STATION_MAP)),
        "stations": len(STATION_MAP),
        "first": _iso(first),
        "last": _iso(last_ts),
        "era_start": _iso(DATA_SOURCE_BREAK),
    }


CAMS_CYCLE_H = 12  # the repo (collector.py) says CAMS refreshes about twice a day
CAMS_OVERDUE_H = 24  # newest vintage older than this = a cycle was missed (quality.py warns too)


def _slot(t: datetime) -> datetime:
    """Start of the 12 h UTC forecast cycle slot that contains t."""
    t = t.astimezone(UTC)
    return t.replace(hour=t.hour - t.hour % CAMS_CYCLE_H, minute=0, second=0, microsecond=0)


def _cov_cams(rd: Reader, now: datetime) -> dict[str, Any]:
    """Expected = one stored CAMS vintage for every forecast cycle of the window.

    CAMS refreshes about twice a day and the collector skips snapshots that repeat the previous
    vintage, so a healthy pipeline stores about one vintage per cycle, not one per hour. A cycle
    is a 12 h UTC slot (00-12, 12-24); it is covered when any vintage was stored in it. The
    current slot is open: it counts only once it has a vintage, never as a miss. Windows are 2
    (24 h) and 14 (7 d) cycles, clamped to the cycle of the first stored vintage.
    """
    end = _hour(now - PIPELINE_GRACE)
    cycle = timedelta(hours=CAMS_CYCLE_H)
    current = _slot(end)
    since = current - cycle * (max(COVERAGE_HOURS) // CAMS_CYCLE_H)
    params = {"m": CAMS_MODEL_VERSION, "since": since, "now": now}
    stored = {r.ts for r in _rows(CovNewest, rd.query(SELECT_COV_VINTAGES, params))}
    newest = _newest(rd, SELECT_COV_NEWEST_VINTAGE, {"m": CAMS_MODEL_VERSION, "now": now}, now)
    if not stored and newest["newest_at"] is None:
        return _cov_unavailable("no_rows")
    first = _first_ts(rd, SELECT_COV_FIRST_VINTAGE, {"m": CAMS_MODEL_VERSION, "now": now})
    slots = {_slot(t) for t in stored}
    out: dict[str, Any] = {"available": True, "cycle_hours": CAMS_CYCLE_H, **newest}
    age = newest["newest_age_min"]
    out["overdue"] = age is None or age > CAMS_OVERDUE_H * 60
    for hours in COVERAGE_HOURS:
        start = current - cycle * (hours // CAMS_CYCLE_H)
        if first is not None:
            start = max(start, _slot(first))
        due = int((current - start) / cycle)  # completed cycles; the open one is not a miss
        got = sum(1 for t in slots if start <= t < current)
        got_open = 1 if current in slots else 0
        out[f"pct_{hours}h"] = _pct(got + got_open, due + got_open)
    out["since"] = _iso(None if first is None else _slot(first))
    out["clamped"] = first is not None and _slot(first) > since
    return out


def _coverage(rd: Reader, now: datetime) -> dict[str, Any]:
    """Each source on its own: a failing query gives that source 'not available', not a crash."""
    result: dict[str, Any] = {}
    for name, fn in (("air4thai", _cov_air4thai), ("openaq", _cov_openaq), ("cams", _cov_cams)):
        try:
            result[name] = fn(rd, now)
        except Exception as exc:  # type only: driver messages can echo the connection string
            logger.warning("dashboard coverage %s unavailable (%s)", name, type(exc).__name__)
            result[name] = _cov_unavailable("query_failed")
    return result


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


def _serving(rd: Reader, now: datetime) -> dict[str, Any]:
    """Summarize only recorded serving choices; absent records are not forecasts."""
    raw = rd.query(SELECT_SERVING, {"now": now})
    parsed = _rows(ServingChoiceRow, raw)
    choices = [
        row
        for row in parsed
        if row.station_id in STATION_MAP
        and row.reason in ("champion", "baseline_fallback")
        and row.valid_time - row.issue_time in (timedelta(hours=h) for h in HORIZONS_H)
    ]
    invalid = len(raw) - len(choices)
    counts: dict[tuple[str, str], int] = {}
    for row in choices:
        key = (row.model_version, row.reason)
        counts[key] = counts.get(key, 0) + 1
    try:
        accuracy = _serving_accuracy(rd, now)
    except Exception as exc:  # keep recorded choice visibility if score history is unavailable
        logger.warning("dashboard served accuracy unavailable (%s)", type(exc).__name__)
        accuracy = {"status": "unavailable", "window_days": WINDOW.days, "models": {}}
    return {
        "issue_time": _iso(choices[0].issue_time) if choices else None,
        "is_current_hour": bool(choices and _hour(choices[0].issue_time) == _hour(now)),
        "recorded": len(choices),
        "expected": len(STATION_MAP) * len(HORIZONS_H),
        "invalid": invalid,
        "choices": [
            {"model_version": version, "reason": reason, "station_horizons": count}
            for (version, reason), count in sorted(counts.items())
        ],
        "accuracy": accuracy,
    }


def _serving_accuracy(rd: Reader, now: datetime) -> dict[str, Any]:
    """Accuracy only for ledger-selected forecasts, paired with persistence on the same keys."""
    since = now - WINDOW
    raw_served = rd.query(sql.SELECT_SERVED_SCORES_BETWEEN, {"since": since, "now": now})
    parsed_served = _rows(ServedScoreRow, raw_served)
    served_rows = [
        row for row in parsed_served if math.isfinite(row.abs_error) and row.abs_error >= 0
    ]
    raw_reference = rd.query(
        sql.SELECT_SERVED_PERSISTENCE_SCORES_BETWEEN,
        {"model_version": PERSISTENCE_MODEL_VERSION, "since": since, "now": now},
    )
    parsed_reference = _rows(ScoreRow, raw_reference)
    reference_rows = [
        row for row in parsed_reference if math.isfinite(row.abs_error) and row.abs_error >= 0
    ]
    malformed = len(raw_served) - len(served_rows) + len(raw_reference) - len(reference_rows)
    scores: dict[str, list[ScoreRow]] = {SERVED_REFERENCE: reference_rows}
    for row in served_rows:
        scores.setdefault(row.model_version, []).append(row)
    metrics = rolling_metrics(scores, reference=SERVED_REFERENCE)
    metrics.pop(SERVED_REFERENCE, None)
    models = {
        model: {
            str(horizon): {
                "n": metric.n,
                "mae": _num(metric.mae),
                "rmse": _num(metric.rmse),
                "skill": _num(metric.skill_vs_persistence, 4),
                "n_matched": metric.n_matched,
            }
            for horizon, metric in sorted(by_horizon.items())
        }
        for model, by_horizon in sorted(metrics.items())
    }
    return {
        "status": "degraded" if malformed else "ok" if models else "waiting",
        "window_days": WINDOW.days,
        "malformed": malformed,
        "models": models,
    }


# --- assembling ----------------------------------------------------------------------------------


def _overall(data: dict[str, Any]) -> str:
    q, obs = data["quality"], data["freshness"]
    total = (obs or {}).get("stations_total", 0)
    reporting = (obs or {}).get("stations_reporting", 0)
    unavailable = any(v in ("unavailable", "degraded") for v in data["sections"].values())
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
    serving = section("serving", lambda: _serving(rd, now))
    if serving and (
        serving["invalid"] or serving["accuracy"]["status"] in ("degraded", "unavailable")
    ):
        sections["serving"] = "degraded"
    coverage = section("coverage", lambda: _coverage(rd, now))
    if coverage and any(c.get("reason") == "query_failed" for c in coverage.values()):
        sections["coverage"] = "degraded"

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
        "serving": serving,
        "coverage": coverage,
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


CHART_TITLE = "PM2.5 \u00b7 last 24 h and next 24 h"

# Chart geometry (viewBox units). The SVG scales with its container and keeps text proportional.
CH_W, CH_H = 820, 290
CH_X0, CH_X1, CH_Y0, CH_Y1 = 60, 804, 16, 236


def _bkk(dt: datetime) -> str:
    return f"{dt.astimezone(BANGKOK):%d %b %H:%M}"


def _x_ticks(start: datetime, stop: datetime) -> list[datetime]:
    """Every 6 h on Bangkok-local 00/06/12/18 within [start, stop]."""
    t = start.astimezone(BANGKOK).replace(minute=0, second=0, microsecond=0)
    if t < start.astimezone(BANGKOK):
        t += timedelta(hours=1)
    while t.hour % 6:
        t += timedelta(hours=1)
    ticks = []
    while t <= stop.astimezone(BANGKOK):
        ticks.append(t)
        t += timedelta(hours=6)
    return ticks


THAI_MONTHS = (
    "ม.ค.",
    "ก.พ.",
    "มี.ค.",
    "เม.ย.",
    "พ.ค.",
    "มิ.ย.",
    "ก.ค.",
    "ส.ค.",
    "ก.ย.",
    "ต.ค.",
    "พ.ย.",
    "ธ.ค.",
)


def _stale_age_min(st: dict[str, Any], generated_at: datetime | None) -> int | None:
    """Minutes since the newest reading when it is older than REPORTING_H, else None (KAN-108)."""
    if generated_at is None or not st.get("latest_at"):
        return None
    age = int((generated_at - datetime.fromisoformat(st["latest_at"])).total_seconds() // 60)
    return age if age > REPORTING_H * 60 else None


def _stale_note(st: dict[str, Any], age_min: int) -> tuple[str, str]:
    """(English, Thai) reason a stale station has no forecast, computed from the data."""
    hours = round(age_min / 60)
    when = datetime.fromisoformat(st["latest_at"]).astimezone(BANGKOK)
    last = f"{when:%d %b %H:%M}"
    last_th = f"{when.day:02d} {THAI_MONTHS[when.month - 1]} {when:%H:%M}"
    en = (
        f"No forecast: no new reading from this station for {hours} h "
        f"(last reading {last} Bangkok). Forecasts resume when readings return."
    )
    th = (
        f"ไม่มีค่าพยากรณ์: สถานีนี้ไม่มีค่าที่ตรวจวัดใหม่มา {hours} ชั่วโมง "
        f"(ค่าล่าสุดเวลา {last_th} น. เวลากรุงเทพฯ) ค่าพยากรณ์จะกลับมาเมื่อมีค่าตรวจวัดใหม่"
    )
    return en, th


def _bilingual(en: str, th: str, cls: str) -> str:
    return f'<span class="{cls}" data-en="{e(en)}" data-th="{e(th)}">{e(en)}</span>'


def _chart_block(
    st: dict[str, Any], chart: dict[str, Any], first: bool, generated_at: datetime | None = None
) -> str:
    # Past 24 h of measurements, then forecasts that start at the latest measured point.
    points = st["points"][-25:]
    future_keys = (
        ("future_persistence", "ln-pers", "Persistence"),
        ("future_challenger", "ln-chal", "Challenger"),
    )
    has_future = any(p["value"] is not None for key, _, _ in future_keys for p in st.get(key, []))
    values = [p["actual"] for p in points if p["actual"] is not None]
    values += [
        p["value"] for key, _, _ in future_keys for p in st.get(key, []) if p["value"] is not None
    ]
    stale_age = _stale_age_min(st, generated_at)
    if not values:
        body = '<p class="muted small">No hourly data for this station yet.</p>'
        hidden = "" if first else " hidden"
        return f'<div class="chart" data-station="{e(st["id"])}"{hidden}>{body}</div>'

    end = datetime.fromisoformat(chart["end"])
    start = end - timedelta(hours=24)
    stop = end + timedelta(hours=24 if has_future else 0)
    span = (stop - start).total_seconds()
    ymax = _nice_max(max(values))

    def px(t: datetime) -> float:
        return CH_X0 + (t - start).total_seconds() / span * (CH_X1 - CH_X0)

    def py(v: float) -> float:
        return CH_Y1 - min(v, ymax) / ymax * (CH_Y1 - CH_Y0)

    parts: list[str] = []
    if has_future:
        xn = px(end)
        parts.append(
            f'<rect class="forecast-region" x="{xn:.1f}" y="{CH_Y0}" '
            f'width="{CH_X1 - xn:.1f}" height="{CH_Y1 - CH_Y0}"/>'
        )
    for i in range(5):  # y axis: 4 equal steps -> 5 round ticks
        y = CH_Y1 - i * (CH_Y1 - CH_Y0) / 4
        parts.append(
            f'<line class="grid-line" x1="{CH_X0}" x2="{CH_X1}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<text class="ax-text" x="{CH_X0 - 8}" y="{y + 5:.1f}" text-anchor="end">'
            f"{ymax * i / 4:g}</text>"
        )
    parts.append(
        f'<text class="ax-title" transform="translate(14 {(CH_Y0 + CH_Y1) / 2:.0f}) rotate(-90)" '
        'text-anchor="middle">PM2.5 (µg/m³)</text>'
    )
    for t in _x_ticks(start, stop):
        x = px(t.astimezone(UTC))
        date = f'<tspan x="{x:.1f}" dy="17">{t:%d %b}</tspan>' if t.hour == 0 else ""
        parts.append(
            f'<line class="tick" x1="{x:.1f}" x2="{x:.1f}" y1="{CH_Y1}" y2="{CH_Y1 + 5}"/>'
            f'<text class="ax-text" x="{x:.1f}" y="{CH_Y1 + 22}" text-anchor="middle">'
            f"{t:%H}:00{date}</text>"
        )
    if has_future:
        xn = px(end)
        parts.append(
            f'<line class="forecast-boundary" x1="{xn:.1f}" x2="{xn:.1f}" y1="{CH_Y0}" '
            f'y2="{CH_Y1}"/><text class="axis-now" x="{xn + 6:.1f}" y="{CH_Y0 + 14}" '
            'text-anchor="start">Now</text>'
        )

    def path(coords: list[tuple[float, float]], css: str) -> str:
        if len(coords) < 2:
            return ""
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        return f'<path class="ln {css}" d="{d}"/>'

    # Past: measured values only; a data gap breaks the line.
    segments: list[list[tuple[float, float]]] = [[]]
    hover: list[str] = []
    for p in points:
        if p["actual"] is None:
            segments.append([])
            continue
        t = datetime.fromisoformat(p["t"])
        xy = (px(t), py(p["actual"]))
        segments[-1].append(xy)
        hover.append(
            f'<circle class="hover-pt" cx="{xy[0]:.1f}" cy="{xy[1]:.1f}" r="6">'
            f"<title>Measured: {p['actual']:g} µg/m³ at {_bkk(t)}</title></circle>"
        )
    anchor: tuple[float, float] | None = None
    actual_pts = [p for p in points if p["actual"] is not None]
    if actual_pts:
        last = actual_pts[-1]
        anchor = (px(datetime.fromisoformat(last["t"])), py(last["actual"]))

    # Future series start at the latest measured point so they connect to the actual line.
    future_parts: list[str] = []
    for key, css, _ in future_keys:
        coords: list[tuple[float, float, dict[str, Any]]] = []
        for forecast in st.get(key, []):
            if forecast["value"] is not None:
                t = datetime.fromisoformat(forecast["valid_time"])
                coords.append((px(t), py(forecast["value"]), forecast))
        if not coords:
            continue
        start_xy = [anchor] if anchor else []
        future_parts.append(path(start_xy + [(x, y) for x, y, _ in coords], css + " future-line"))
    parts.extend(future_parts)
    for seg in segments:
        parts.append(path(seg, "ln-actual"))
    if anchor and actual_pts:
        last = actual_pts[-1]
        t = datetime.fromisoformat(last["t"])
        parts.append(
            f'<circle class="actual-dot" cx="{anchor[0]:.1f}" cy="{anchor[1]:.1f}" r="5">'
            f"<title>Measured: {last['actual']:g} µg/m³ at {_bkk(t)}</title></circle>"
            f'<text class="pt-label latest" x="{anchor[0] - 9:.1f}" '
            f'y="{max(anchor[1] - 10, CH_Y0 + 10):.1f}" text-anchor="end">'
            f"Latest {last['actual']:g}</text>"
        )
    for key, css, _ in future_keys:
        for forecast in st.get(key, []):
            if forecast["value"] is None:
                continue
            valid = datetime.fromisoformat(forecast["valid_time"])
            issue = datetime.fromisoformat(forecast["issue_time"])
            ahead = int((valid - issue) / timedelta(hours=1))
            x, y = px(valid), py(forecast["value"])
            description = (
                f"{key.replace('future_', '').capitalize()} forecast: "
                f"{forecast['value']:g} µg/m³, valid {_bkk(valid)}; issued {_bkk(issue)}"
            )
            circle = (
                f'<circle class="forecast-dot {css}" cx="{x:.1f}" cy="{y:.1f}" r="4.5">'
                f"<title>{e(description)}</title></circle>"
            )
            label = ""
            if css == "ln-chal":  # only the model carries value labels
                edge = x > CH_X1 - 40
                lx = x - 4 if edge else x
                ly = min(max(y - 11, CH_Y0 + 10), CH_Y1 - 4)
                label = (
                    f'<text class="pt-label chal" x="{lx:.1f}" y="{ly:.1f}" '
                    f'text-anchor="{"end" if edge else "middle"}">'
                    f"+{ahead}h {forecast['value']:g}</text>"
                )
            parts.append(circle + label)
    parts.extend(hover)

    aria = (
        f"Station {st['label']}: PM2.5 in µg/m³, measured over the last 24 hours and "
        "forecast for the next 24 hours, Bangkok time"
    )
    future_html = []
    for key, _, label in future_keys:
        for forecast in st.get(key, []):
            if forecast["value"] is None:
                continue
            issue = datetime.fromisoformat(forecast["issue_time"])
            valid = datetime.fromisoformat(forecast["valid_time"])
            hours_ahead = int((valid - issue) / timedelta(hours=1))
            future_html.append(
                f'<span class="forecast-point"><b>{label} forecast</b> +{hours_ahead} h · '
                f"{forecast['value']:g} µg/m³ <small>({_bkk(valid)})</small></span>"
            )
    if future_html:
        empty = ""
    elif stale_age is not None:
        empty = _bilingual(*_stale_note(st, stale_age), "forecast-empty stale-note")
    else:
        empty = '<span class="forecast-empty">Waiting for future forecasts.</span>'
    future_block = (
        '<div class="future-forecasts"><strong>Upcoming forecasts</strong>'
        + ("".join(future_html) or empty)
        + "</div>"
    )
    body = (
        f'<div class="plot"><svg viewBox="0 0 {CH_W} {CH_H}" role="img" '
        f'aria-label="{e(aria)}">{"".join(parts)}</svg></div>' + future_block
    )
    hidden = "" if first else " hidden"
    return f'<div class="chart" data-station="{e(st["id"])}"{hidden}>{body}</div>'


def _pill(st: dict[str, Any], first: bool, generated_at: datetime) -> str:
    label = st["label"]
    pressed = "true" if first else "false"
    stale = ""
    name = label
    if _stale_age_min(st, generated_at) is not None:
        en, th = "no recent reading", "ไม่มีค่าตรวจวัดล่าสุด"
        name = f"{label}, {en}"
        stale = f' data-stale-en="{e(en)}" data-stale-th="{e(th)}"'
    dot = f'<span class="stale-dot" aria-hidden="true" title="{e(en)}"></span>' if stale else ""
    return (
        f'<button type="button" class="pill" data-station="{e(st["id"])}" '
        f'data-label="{e(label)}" aria-label="{e(name)}"{stale} '
        f'aria-pressed="{pressed}">{e(label)}{dot}</button>'
    )


def _section_chart(data: dict[str, Any]) -> str:
    chart = data["chart"]
    legend = (
        '<div class="legend"><span><i class="sw actual"></i>Measured</span>'
        '<span><i class="sw pers"></i>Baseline forecast (tomorrow = today)</span>'
        '<span><i class="sw chal"></i>Model forecast</span></div>'
    )
    if chart is None:
        inner = _unavailable("The chart")
        pills = ""
        title = CHART_TITLE
    elif not chart["stations"]:
        inner = '<p class="muted small">Waiting for the first observations.</p>'
        pills = ""
        title = CHART_TITLE
    else:
        sts = chart["stations"]
        title = f"{CHART_TITLE} \u00b7 {sts[0]['label']}"
        generated_at = datetime.fromisoformat(data["generated_at"])
        pills = "".join(_pill(s, i == 0, generated_at) for i, s in enumerate(sts))
        inner = "".join(_chart_block(s, chart, i == 0, generated_at) for i, s in enumerate(sts))
    return (
        '<figure class="card wide" style="margin:0"><div class="cardhead">'
        f'<h2 id="chart-title">{e(title)}</h2>'
        f'<div class="pills" role="group" aria-label="Station">{pills}</div></div>'
        f"{legend}{inner}"
        '<p class="chart-note muted small">Accuracy of past forecasts: see Typical error and '
        "Skill by horizon above.</p>"
        '<figcaption class="muted small">Times in Bangkok time (UTC+7). Forecast lines start at '
        "the latest measured value and join only the horizons available."
        "</figcaption></figure>"
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
        '<p class="legend"><span><i class="sq" style="background:#0f766e"></i>checks OK</span>'
        '<span><i class="sq" style="background:#b45309"></i>warning</span>'
        '<span><i class="sq" style="background:#b91c1c"></i>failed</span>'
        '<span><i class="sq" style="background:#6b7280;height:5px"></i>no check recorded '
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


DRIFT_SHORT = ("Not comparable yet", "ยังเปรียบเทียบไม่ได้")
DRIFT_ERA = (
    "Not comparable yet: this model was trained on data from before the current data source "
    f"({DATA_SOURCE_BREAK:%d %b %Y}), so there is no matching reference. "
    "It will show once a model trained on the new data is live.",
    f"ยังเปรียบเทียบไม่ได้: โมเดลนี้ฝึกจากข้อมูลก่อนเปลี่ยนแหล่งข้อมูลปัจจุบัน ({DATA_SOURCE_BREAK:%Y-%m-%d} UTC) "
    "จึงไม่มีข้อมูลอ้างอิงที่ตรงกัน จะแสดงผลเมื่อมีโมเดลที่ฝึกด้วยข้อมูลใหม่ใช้งานอยู่",
)
DRIFT_CAMS = (
    "cams_pm25: not comparable yet, the CAMS history is still too short for a reference.",
    "cams_pm25: ยังเปรียบเทียบไม่ได้ เพราะประวัติ CAMS ยังสั้นเกินไปสำหรับข้อมูลอ้างอิง",
)
DRIFT_NO_PSI = (
    "Not comparable yet: too little data in the reference or recent window.",
    "ยังเปรียบเทียบไม่ได้: ข้อมูลในช่วงอ้างอิงหรือช่วงล่าสุดน้อยเกินไป",
)


def _drift_reason(f: dict[str, Any]) -> tuple[str, str]:
    if f["status"] != "insufficient":
        return DRIFT_NO_PSI
    return DRIFT_CAMS if f["name"] == "cams_pm25" else DRIFT_ERA


KPI_TIP_TITLES = {
    "skill": ("Beats the baseline by", "แม่นกว่าวิธีพื้นฐาน"),
    "mae": ("Typical error (MAE, \u00b5g/m\u00b3)", "ความคลาดเคลื่อนเฉลี่ย (MAE, \u00b5g/m\u00b3)"),
    "fresh": ("Data freshness", "ความใหม่ของข้อมูล"),
    "psi": ("Input drift (PSI)", "การเปลี่ยนแปลงของข้อมูล (PSI)"),
}


def _kpi_tip_button(key: str, title: str) -> str:
    """Disclosure button (tap, Enter, Space); JS reveals it and toggles the panel."""
    en, th = KPI_TIP_TITLES[key]
    label = _bi(f"About: {en}", f"อธิบาย: {th}", cls="sr-only")
    return (
        f'<button type="button" class="tip-btn" data-tip="{key}" aria-expanded="false" '
        f'aria-controls="tip-{key}" hidden><span aria-hidden="true">i</span>{label}</button>'
    )


def _kpi_tip_texts(key: str, data: dict[str, Any]) -> list[tuple[str, str]]:
    """(English, Thai) paragraphs. Every number and rule here mirrors score.py, quality.py,
    collector.py, drift.py and the helpers above; dynamic values come from the page data."""
    if key == "skill":
        s24 = ((data.get("scores") or {}).get("challenger") or {}).get(str(HEADLINE_H)) or {}
        n = s24.get("n_matched")
        n_en = f" Right now n = {n}." if n else " There are no matched forecasts yet."
        n_th = f" ขณะนี้ n = {n}." if n else " ยังไม่มีค่าพยากรณ์ที่จับคู่ได้."
        chal = data["challenger_version"]
        model_en = (
            f"Model: {chal} (the shadow challenger). The model changes when the challenger does."
            if chal
            else "No challenger is configured, so there is nothing to compare yet."
        )
        model_th = (
            f"โมเดล: {chal} (โมเดลท้าชิงที่ทำงานคู่ขนาน) และจะเปลี่ยนเมื่อโมเดลท้าชิงเปลี่ยน"
            if chal
            else "ยังไม่ได้ตั้งค่าโมเดลท้าชิง จึงยังไม่มีอะไรให้เปรียบเทียบ"
        )
        return [
            (
                "Idea: how much better the model\u2019s +24 h forecast is than the simplest "
                "baseline, persistence (tomorrow equals now).",
                "แนวคิด: ค่าพยากรณ์ล่วงหน้า 24 ชั่วโมงของโมเดลแม่นกว่าวิธีพื้นฐานที่สุด "
                "คือ persistence (พรุ่งนี้เท่ากับตอนนี้) มากแค่ไหน",
            ),
            (
                "Calculation: (MAE of persistence \u2212 MAE of the model) \u00f7 MAE of "
                "persistence, over the last 7 days at +24 h, only on matched station-hours "
                f"where both forecasts exist.{n_en}",
                "วิธีคำนวณ: (MAE ของ persistence \u2212 MAE ของโมเดล) \u00f7 MAE ของ persistence "
                "ในช่วง 7 วันล่าสุดที่ +24 ชม. เฉพาะสถานีและชั่วโมงที่มีค่าพยากรณ์ของทั้งสองวิธี"
                f"{n_th}",
            ),
            (
                "Reading: above 0 is better than persistence, 0 is the same, below 0 is worse. "
                "With a small n the number is noisy. It is not a significance test.",
                "การอ่านค่า: มากกว่า 0 คือแม่นกว่า persistence, 0 คือเท่ากัน, น้อยกว่า 0 คือแย่กว่า "
                "ถ้า n น้อยค่านี้จะแกว่งมาก และไม่ใช่การทดสอบนัยสำคัญทางสถิติ",
            ),
            (model_en, model_th),
        ]
    if key == "mae":
        return [
            (
                "Idea: the average size of a forecast miss, in \u00b5g/m\u00b3 of PM2.5.",
                "แนวคิด: ขนาดเฉลี่ยของความคลาดเคลื่อนของค่าพยากรณ์ มีหน่วยเป็น \u00b5g/m\u00b3 ของ PM2.5",
            ),
            (
                "Calculation: the mean of |forecast \u2212 actual| over scored forecasts, last "
                "7 days at +24 h, for the model and for persistence (shown under the number). "
                "Each average uses all of that method\u2019s scored forecasts, so it can cover "
                "slightly different hours than the skill above.",
                "วิธีคำนวณ: ค่าเฉลี่ยของ |ค่าพยากรณ์ \u2212 ค่าจริง| จากค่าพยากรณ์ที่ประเมินแล้ว "
                "ใน 7 วันล่าสุดที่ +24 ชม. ของโมเดลและของ persistence (แสดงใต้ตัวเลข) "
                "แต่ละค่าเฉลี่ยใช้ค่าพยากรณ์ที่ประเมินแล้วทั้งหมดของวิธีนั้น "
                "จึงอาจครอบคลุมชั่วโมงไม่ตรงกับค่า Skill ด้านบนทั้งหมด",
            ),
            (
                "Reading: lower is better. Compare it with persistence. Large misses count in "
                "proportion to their size (not squared).",
                "การอ่านค่า: ยิ่งต่ำยิ่งดี ให้เทียบกับ persistence "
                "ความคลาดเคลื่อนที่ใหญ่นับตามขนาดจริง (ไม่ยกกำลังสอง)",
            ),
        ]
    if key == "fresh":
        return [
            (
                "Idea: how long ago the newest reading arrived, and how many stations are "
                "still reporting.",
                "แนวคิด: ค่าตรวจวัดล่าสุดเข้ามานานเท่าไรแล้ว และมีกี่สถานีที่ยังส่งข้อมูลอยู่",
            ),
            (
                "Calculation: minutes since the newest reading across the stations that sent "
                "data in the last 7 days. A station counts as reporting if its newest reading "
                "is at most 3 hours old.",
                "วิธีคำนวณ: จำนวนนาทีนับจากค่าตรวจวัดล่าสุดของทุกสถานีที่ส่งข้อมูลใน 7 วันล่าสุด "
                "สถานีนับว่ายังส่งข้อมูลถ้าค่าล่าสุดเก่าไม่เกิน 3 ชั่วโมง",
            ),
            (
                "Reading: Air4Thai updates hourly, so under about 60\u201390 minutes is normal "
                "(a rule of thumb). "
                "The collector alerts when a station crosses 3 hours without a reading, and "
                "forecasts for that station are skipped (it is also a data quality warning). "
                "Over 6 hours "
                "is a data quality failure. The minutes show the newest station only; the "
                "station count shows the rest.",
                "การอ่านค่า: Air4Thai อัปเดตทุกชั่วโมง จึงปกติถ้าน้อยกว่าประมาณ 60\u201390 นาที "
                "(เป็นเกณฑ์คร่าว ๆ) "
                "ตัวเก็บข้อมูลจะแจ้งเตือนเมื่อสถานีไม่มีค่าตรวจวัดเกิน 3 ชั่วโมง "
                "และข้ามการพยากรณ์ของสถานีนั้น (และเป็นคำเตือนด้านคุณภาพข้อมูลด้วย) "
                "เกิน 6 ชั่วโมงถือว่าคุณภาพข้อมูลไม่ผ่าน "
                "ตัวเลขนาทีแสดงเฉพาะสถานีที่ใหม่ที่สุด ส่วนจำนวนสถานีบอกภาพรวมที่เหลือ",
            ),
        ]
    reasons = []
    d = data.get("drift")
    if data["kpi"]["worst_psi"] is None and d and d.get("features"):
        seen: dict[tuple[str, str], None] = {}
        for f in d["features"]:
            if f["psi"] is None:
                seen.setdefault(_drift_reason(f), None)
        reasons = list(seen)
    waiting = (
        "Waiting means not comparable yet.",
        "รอข้อมูล หมายถึงยังเปรียบเทียบไม่ได้",
    )
    return [
        (
            "Idea: whether recent inputs still look like the data the model learned from.",
            "แนวคิด: ข้อมูลนำเข้าช่วงล่าสุดยังคล้ายกับข้อมูลที่โมเดลใช้เรียนรู้หรือไม่",
        ),
        (
            "Calculation: Population Stability Index (PSI) of the last 7 days against "
            "same-season training data, using 5 quantile bins, for the inputs and the target. "
            "The number is the worst feature, named under it, from the latest daily drift run "
            "(up to about 3 days old).",
            "วิธีคำนวณ: ค่า Population Stability Index (PSI) ของ 7 วันล่าสุดเทียบกับข้อมูลฝึกช่วงฤดูเดียวกัน "
            "แบ่งเป็น 5 ช่วงตามควอนไทล์ ครอบคลุมทั้งข้อมูลนำเข้าและค่าเป้าหมาย "
            "ตัวเลขคือค่าของตัวแปรที่แย่ที่สุด ซึ่งระบุชื่อไว้ด้านล่าง จากการตรวจความเปลี่ยนแปลงรายวันครั้งล่าสุด "
            "(เก่าได้ถึงประมาณ 3 วัน)",
        ),
        (
            "Reading: below 0.1 means no meaningful change, 0.1 to 0.2 a moderate change, "
            "above 0.2 a warning; an alert needs 2 consecutive daily breaches. It says the "
            "data changed, not that the model got worse.",
            "การอ่านค่า: ต่ำกว่า 0.1 คือไม่เปลี่ยนอย่างมีนัยสำคัญ, 0.1\u20130.2 คือเปลี่ยนปานกลาง, "
            "มากกว่า 0.2 คือเตือน และจะแจ้งเตือนจริงเมื่อเกินติดต่อกัน 2 วัน "
            "ค่านี้บอกว่าข้อมูลเปลี่ยน ไม่ได้บอกว่าโมเดลแย่ลง",
        ),
        waiting,
        *reasons,
    ]


def _kpi_tip_panel(key: str, data: dict[str, Any]) -> str:
    paras = "".join(_bi(en, th, tag="p") for en, th in _kpi_tip_texts(key, data))
    return f'<div id="tip-{key}" class="kpi-tip" role="note" hidden>{paras}</div>'


def _section_drift(data: dict[str, Any]) -> str:
    d = data["drift"]
    if d is None:
        body = _unavailable("Drift")
    elif not d["features"]:
        body = '<p class="muted small">Waiting for the first daily drift check.</p>'
    else:
        rows = []
        reasons: dict[tuple[str, str], None] = {}
        for f in d["features"]:
            psi = f["psi"]
            if psi is None:
                bar, text = "", None
            else:
                width = min(psi / PSI_BAR_MAX, 1) * 100
                cls = "fill alert" if f["alert"] else "fill plain"
                bar = f'<div class="{cls}" style="width:{width:.0f}%"></div>'
                suffix = " Alert" if f["alert"] else " Warning" if f["status"] == "warn" else ""
                text = f"{psi:.2f}" + suffix
            tick = '<div class="tick"></div>'
            shown = _bi(*DRIFT_SHORT) if text is None else e(text)
            rows.append(
                f'<div class="drow"><span class="name">{e(f["name"])}</span>'
                f'<div class="track">{bar}{tick}</div><span class="psi">{shown}</span></div>'
            )
            if text is None:
                reasons.setdefault(_drift_reason(f), None)
        body = "".join(rows) + "".join(
            _bi(en, th, tag="p", cls="muted small") for en, th in reasons
        )
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
    serving = data.get("serving")
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
    if serving is None:
        serving_body = _bi("Serving history unavailable", "ไม่สามารถโหลดประวัติโมเดลที่ใช้")
    elif not serving["choices"]:
        if serving["invalid"]:
            serving_body = _bi(
                "Serving records are malformed; coverage is unavailable.",
                "ข้อมูลโมเดลที่ใช้มีรูปแบบไม่ถูกต้อง จึงไม่ทราบความครอบคลุม",
            )
        else:
            serving_body = _bi(
                f"No serving choices recorded yet; current UTC hour: 0/{serving['expected']} "
                "pilot station-horizons recorded. Missing records have unknown cause.",
                f"ยังไม่มีบันทึกการเลือกโมเดล ชั่วโมง UTC ปัจจุบัน: 0/{serving['expected']} "
                "คู่สถานีและระยะพยากรณ์ ยังไม่ทราบสาเหตุที่ไม่มีบันทึก",
            )
    else:
        timestamp = datetime.fromisoformat(serving["issue_time"])
        utc = timestamp.astimezone(UTC).strftime("%d %b %Y %H:%M UTC")
        local = timestamp.astimezone(BANGKOK).strftime("%d %b %Y %H:%M Bangkok")
        items = []
        for choice in serving["choices"]:
            role_en, role_th = {
                "champion": ("champion", "โมเดลหลัก"),
                "baseline_fallback": ("baseline fallback", "ใช้ baseline สำรอง"),
            }.get(choice["reason"], ("other", "อื่น ๆ"))
            version = choice["model_version"]
            count = choice["station_horizons"]
            items.append(
                "<li>"
                + _bi(
                    f"{version}: {count} station-horizons ({role_en})",
                    f"{version}: {count} คู่สถานีและระยะพยากรณ์ ({role_th})",
                )
                + "</li>"
            )
        current_count = serving["recorded"] if serving["is_current_hour"] else 0
        expected = serving["expected"]
        coverage = _bi(
            f"Current UTC hour: {current_count}/{expected} pilot station-horizons recorded; "
            "missing records have unknown cause.",
            f"ชั่วโมง UTC ปัจจุบัน: บันทึกแล้ว {current_count}/{expected} คู่สถานีและระยะพยากรณ์ "
            "รายการที่ไม่พบบันทึกยังไม่ทราบสาเหตุ",
        )
        incomplete = (
            _bi(
                f"{serving['invalid']} records could not be interpreted; "
                "coverage may be incomplete.",
                f"อ่านข้อมูลไม่ได้ {serving['invalid']} รายการ ความครอบคลุมอาจไม่ครบ",
            )
            if serving["invalid"]
            else ""
        )
        serving_body = (
            _bi(f"Latest recorded issue: {utc} · {local}", f"รอบที่บันทึกล่าสุด: {utc} · {local}")
            + "<br>"
            + coverage
            + "<br>"
            + incomplete
            + f"<ul>{''.join(items)}</ul>"
            + _bi(
                "Accuracy below uses mature forecasts marked as served and compares with "
                "persistence on matched cases. The chart still shows baseline and any "
                "configured shadow forecasts.",
                "ค่าความแม่นยำด้านล่างคำนวณจากพยากรณ์ที่บันทึกว่าใช้งานและมีผลจริงแล้ว "
                "โดยเทียบกับ persistence ในกรณีเดียวกัน กราฟยังแสดง baseline และ shadow ที่ตั้งค่าไว้",
            )
        )
    accuracy = (
        serving.get("accuracy", {"status": "waiting", "window_days": WINDOW.days, "models": {}})
        if serving is not None
        else {"status": "unavailable", "window_days": WINDOW.days, "models": {}}
    )
    if accuracy["status"] == "unavailable":
        accuracy_body = _bi(
            "Served forecast accuracy is unavailable; recorded model choices are still shown.",
            "ไม่สามารถโหลดค่าความแม่นยำของพยากรณ์ที่ใช้งานได้ แต่ยังแสดงรายการโมเดลที่บันทึกไว้",
        )
    else:
        accuracy_items = []
        for version, horizons in accuracy["models"].items():
            for horizon, metric in horizons.items():
                mae = "unavailable" if metric["mae"] is None else f"{metric['mae']:.1f}"
                skill_en, skill_th = (
                    ("matched persistence unavailable", "ไม่มีข้อมูล persistence ที่จับคู่กัน")
                    if metric["skill"] is None
                    else (
                        f"skill vs persistence {metric['skill'] * 100:+.1f}% "
                        f"(n={metric['n_matched']} matched)",
                        f"skill เทียบ persistence {metric['skill'] * 100:+.1f}% "
                        f"(จับคู่ {metric['n_matched']} รายการ)",
                    )
                )
                accuracy_items.append(
                    "<li>"
                    + _bi(
                        f"{version} +{horizon}h: MAE {mae} µg/m³, {skill_en}; n={metric['n']}",
                        f"{version} +{horizon}h: MAE {mae} µg/m³, {skill_th}; n={metric['n']}",
                    )
                    + "</li>"
                )
        accuracy_body = ""
        if accuracy["status"] == "degraded":
            accuracy_body += _bi(
                f"{accuracy['malformed']} score rows were unusable; "
                "these results may be incomplete.",
                f"มีข้อมูลคะแนน {accuracy['malformed']} รายการที่ใช้ไม่ได้ ผลลัพธ์นี้อาจไม่ครบถ้วน",
            )
            accuracy_body += "<br>"
        if accuracy_items:
            accuracy_body += f"<ul>{''.join(accuracy_items)}</ul>"
        elif accuracy["status"] == "degraded":
            accuracy_body += _bi(
                "No valid served scores are available in this window.",
                "ไม่มีคะแนนที่ใช้ได้ของพยากรณ์ที่ใช้งานในช่วงเวลานี้",
            )
        else:
            accuracy_body += _bi(
                f"No current-valid served scores are available in the last "
                f"{accuracy['window_days']} days; actuals may be pending or "
                "scores may need refresh.",
                f"ยังไม่มีคะแนนพยากรณ์ที่ใช้งานซึ่งตรงกับค่าจริงล่าสุดใน "
                f"{accuracy['window_days']} วันที่ผ่านมา ข้อมูลจริงอาจยังไม่มาหรือรอคำนวณใหม่",
            )
    serving_body += (
        '<div class="served-accuracy"><strong>'
        + _bi(
            f"Served forecast accuracy · last {accuracy['window_days']} days",
            f"ความแม่นยำของพยากรณ์ที่ใช้งาน · {accuracy['window_days']} วันล่าสุด",
        )
        + f"</strong>{accuracy_body}</div>"
    )
    return (
        f'<section class="card"><h2>Models</h2>{body}<div class="muted small">'
        f"{serving_body}</div></section>"
    )


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


SOURCE_INFO = (
    (
        "air4thai",
        "Air4Thai",
        "Station PM2.5 readings. Collection runs up to every 20 minutes; "
        "Air4Thai backfill runs daily.",
        "ข้อมูล PM2.5 จากสถานี เก็บข้อมูลสูงสุดทุก 20 นาที และเติมข้อมูลย้อนหลังทุกวัน",
    ),
    (
        "openaq",
        "OpenAQ",
        "Historical PM2.5 backfill, run manually. History only, not live.",
        "เติมข้อมูล PM2.5 ย้อนหลังด้วยตนเอง เป็นข้อมูลย้อนหลังเท่านั้น ไม่ใช่ข้อมูลสด",
    ),
    (
        "cams",
        "Open-Meteo / CAMS",
        "CAMS forecast vintages collected up to every 20 minutes. "
        "Provider update schedules can differ.",
        "เก็บชุดพยากรณ์ CAMS สูงสุดทุก 20 นาที รอบอัปเดตของผู้ให้บริการอาจแตกต่างกัน",
    ),
)
REASONS = {
    "no_rows": ("no rows stored yet", "ยังไม่มีข้อมูลในฐานข้อมูล"),
    "query_failed": ("the database query failed", "คำสั่งค้นฐานข้อมูลล้มเหลว"),
    "section": ("the coverage query failed", "คำสั่งค้นความครบของข้อมูลล้มเหลว"),
}


def _age_text(minutes: int | None) -> tuple[str, str]:
    if minutes is None:
        return "no reading", "ไม่มีข้อมูล"
    if minutes < 60:
        return f"{minutes} min ago", f"{minutes} นาทีก่อน"
    if minutes < 48 * 60:
        h, m = divmod(minutes, 60)
        return f"{h} h {m} min ago", f"{h} ชม. {m} นาทีก่อน"
    return f"{minutes // 1440} days ago", f"{minutes // 1440} วันก่อน"


def _pct_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _day(iso: str | None) -> str:
    return "?" if iso is None else f"{datetime.fromisoformat(iso):%Y-%m-%d %H:%M} UTC"


def _bi(en: str, th: str, tag: str = "span", cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<{tag}{c} data-en="{e(en)}" data-th="{e(th)}">{e(en)}</{tag}>'


def _coverage_texts(key: str, c: dict[str, Any]) -> tuple[tuple[str, str], tuple[str, str]]:
    """(headline, definition-and-details) as (English, Thai) pairs for one source."""
    if not c.get("available"):
        en, th = REASONS.get(c.get("reason", ""), REASONS["query_failed"])
        return (f"not available ({en})", f"ไม่พร้อมใช้งาน ({th})"), ("", "")
    if key == "openaq":
        pct = _pct_text(c["pct"])
        span = f"{_day(c['first'])} to {_day(c['last'])}"
        span_th = f"{_day(c['first'])} ถึง {_day(c['last'])}"
        return (
            (f"{pct} of history", f"{pct} ของช่วงประวัติ"),
            (
                f" Coverage: share of station-hours with an OpenAQ row, for {c['stations']} "
                f"stations, from the first OpenAQ hour to {_day(c['era_start'])} (when "
                f"Air4Thai takes over). Range covered: {span}. OpenAQ is history only, "
                "not live.",
                f" ความครบ: สัดส่วนสถานี-ชั่วโมงที่มีข้อมูล OpenAQ ของ {c['stations']} สถานี "
                f"ตั้งแต่ชั่วโมงแรกของ OpenAQ ถึง {_day(c['era_start'])} (ก่อน Air4Thai เริ่ม) "
                f"ช่วงที่ครอบคลุม: {span_th} OpenAQ เป็นข้อมูลย้อนหลังเท่านั้น ไม่ใช่ข้อมูลสด",
            ),
        )
    a24, a7 = _pct_text(c["pct_24h"]), _pct_text(c["pct_168h"])
    age_en, age_th = _age_text(c["newest_age_min"])
    head = (f"{a24} last 24 h · {a7} last 7 d", f"{a24} 24 ชม. · {a7} 7 วัน")
    if key == "air4thai":
        n = c["stations"]
        since_en = f" Collected since {_day(c['since'])}." if c.get("clamped") else ""
        since_th = f" เก็บตั้งแต่ {_day(c['since'])}" if c.get("clamped") else ""
        return head, (
            (
                f" Coverage: share of expected station-hours that have a reading. Expected = "
                f"{n} stations x 1 reading per UTC hour (24 h = {24 * n}, "
                f"a full 7 days = {168 * n}); "
                f"the window ends at the last hour already due.{since_en} "
                f"Newest reading: {age_en}.",
                f" ความครบ: สัดส่วนสถานี-ชั่วโมงที่คาดไว้ซึ่งมีค่าตรวจวัด ที่คาดไว้ = {n} สถานี "
                f"x 1 ค่าต่อชั่วโมง UTC (24 ชม. = {24 * n}, 7 วัน = {168 * n}) "
                f"ช่วงสิ้นสุดที่ชั่วโมงล่าสุดที่ถึงกำหนดแล้ว{since_th} ค่าล่าสุด: {age_th}",
            )
        )
    over_en = " Overdue: the newest vintage is more than 24 h old." if c.get("overdue") else ""
    over_th = " ล่าช้า: ชุดล่าสุดเก่ากว่า 24 ชม." if c.get("overdue") else ""
    since_en = f" Collected since {_day(c['since'])}." if c.get("clamped") else ""
    since_th = f" เก็บตั้งแต่ {_day(c['since'])}" if c.get("clamped") else ""
    n = c.get("cycle_hours", CAMS_CYCLE_H)
    return head, (
        (
            f" Coverage: share of forecast cycles that stored a CAMS vintage. CAMS refreshes "
            f"about twice a day, so a cycle is a {n} h UTC slot (00-12, 12-24). 100% means every "
            f"cycle in the window has at least one vintage (24 h = 2 cycles, 7 days = 14); the "
            f"current cycle is not counted as missed until it has ended.{since_en} "
            f"Newest vintage: {age_en} (expected every ~{n} h).{over_en}",
            f" ความครบ: สัดส่วนรอบพยากรณ์ที่มีชุดพยากรณ์ CAMS ที่บันทึกไว้ CAMS อัปเดตประมาณวันละ 2 ครั้ง "
            f"หนึ่งรอบคือช่วง {n} ชม. (UTC) (00-12, 12-24) "
            f"100% หมายถึงทุกรอบในช่วงเวลามีชุดพยากรณ์อย่างน้อยหนึ่งชุด "
            f"(24 ชม. = 2 รอบ, 7 วัน = 14 รอบ) รอบปัจจุบันยังไม่นับว่าขาดจนกว่าจะจบรอบ{since_th} "
            f"ชุดล่าสุด: {age_th} (คาดว่าทุก ~{n} ชม.){over_th}",
        )
    )


def _source_list(coverage: dict[str, Any] | None) -> str:
    items = []
    for key, name, desc_en, desc_th in SOURCE_INFO:
        c = (coverage or {}).get(key) or _cov_unavailable("section")
        (h_en, h_th), (d_en, d_th) = _coverage_texts(key, c)
        items.append(
            f"<details><summary>{e(name)}{_bi(h_en, h_th, cls='source-coverage')}</summary>"
            f"{_bi(desc_en + d_en, desc_th + d_th, tag='p')}</details>"
        )
    return '<div class="source-list">' + "".join(items) + "</div>"


def _section_story(data: dict[str, Any]) -> str:
    return """<section class="story card">
<h2 data-en="Behind the forecast" data-th="เบื้องหลังการพยากรณ์">Behind the forecast</h2>
<div class="story-grid">
<article><h3 data-en="Data sources" data-th="แหล่งข้อมูล">Data sources</h3>
<p data-en="Air4Thai provides readings; OpenAQ backfills history; Open-Meteo provides CAMS forecasts." data-th="Air4Thai ให้ค่าตรวจวัด OpenAQ เติมข้อมูลย้อนหลัง และ Open-Meteo ให้ค่าพยากรณ์ CAMS">Air4Thai provides readings; OpenAQ backfills history; Open-Meteo provides CAMS forecasts.</p>
@@SOURCES@@</div></article>
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
</div></section>""".replace("@@SOURCES@@", _source_list(data.get("coverage")))  # noqa: E501


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

    def card(title: str, value: str, sub: str, tip_key: str) -> str:
        return (
            f'<div class="card"><div class="kpi-head"><p class="muted small">{e(title)}</p>'
            f"{_kpi_tip_button(tip_key, title)}</div>{value}"
            f'<p class="muted small">{e(sub)}</p>{_kpi_tip_panel(tip_key, data)}</div>'
        )

    cards = "".join(
        [
            card(
                "Beats the baseline by",
                _val(skill_txt, good=skill is not None and skill > 0),
                "challenger vs persistence \u00b7 +24 h \u00b7 7 days",
                "skill",
            ),
            card("Typical error (MAE, \u00b5g/m\u00b3)", _val(mae_txt), mae_sub, "mae"),
            card(
                "Data freshness",
                _val(None if fresh is None else f"{fresh:.0f} min"),
                fresh_sub,
                "fresh",
            ),
            card(
                "Input drift (PSI)",
                _val(None if psi is None else f"{psi:.2f}"),
                psi_sub,
                "psi",
            ),
        ]
    )
    answer_cls = {"yes": "yes", "not_yet": "notyet", "waiting": "waiting"}[head["state"]]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bangkok PM2.5 forecast \u00b7 live MLOps dashboard</title>
<link rel="stylesheet" href="dashboard.css?v=kan73-chart-v2-kan103">
<script src="dashboard.js?v=kan73-chart-v2-kan103" defer></script>
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
{_section_story(data)}
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
