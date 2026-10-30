"""Daily drift check (KAN-22). Run with ``python -m pm25_forecast.drift``.

For the shadow challenger (``CHALLENGER_MODEL_VERSION``) and the production champion, if one is
registered: compare the last ``RECENT_WINDOW`` of hourly model inputs with the inputs of the
model's ``training_window`` (from its manifest) using the Population Stability Index (PSI), per
feature in ``FEATURE_NAMES``, plus the shift of the observed PM2.5 itself (``target_pm25``).

* Both sides are rebuilt from the database (stored observations and CAMS vintages) through the
  shared ``build_features`` module, so training and serving feature logic is not duplicated. One
  hourly origin per station and per horizon the model was trained for (manifest ``metrics`` keys,
  default 24 h), observations from the 24 h before the origin up to the model's
  ``assumed_reporting_lag_hours`` before it (as in training). Origins without any feature value
  are skipped, as in training.
* The reference side (bin edges and shares) is computed once per model_version from the training
  window and cached in ``drift_reference``; a daily run reads only the recent 7 days.
* Status: PSI below ``PSI_WARN`` is ``ok``, up to ``PSI_ALERT`` ``warn``, above it ``alert`` only
  if the previous daily run was also above ``PSI_ALERT`` (otherwise ``warn``). ``insufficient``
  (stored without a PSI, never an alert): fewer than ``MIN_RECENT`` recent or ``MIN_REFERENCE``
  reference points, fewer than ``MIN_BINS`` distinct reference bins, or ``cams_pm25`` covering
  under ``CAMS_MIN_COVERAGE`` of the reference origins. See docs/drift.md.
* Results are upserted into ``drift_checks`` per (UTC day, model_version, feature): a re-run the
  same day updates the rows. One alert per run lists the features in ``alert`` status.
* ``latest_drift_alert(con, model_version=None)`` tells the retrain job (KAN-24) whether the latest
  run had an alert; a run older than ``STALE_AFTER`` counts as no alert.
* Reads observations, predictions (CAMS vintages) and the model registry; writes only
  ``drift_checks``. DB down: ``run_entrypoint`` skips the run, alerts, exits non-zero.

Exit codes: 0 ok (drift alerts are reported, not failures), 1 failed / DB down, 2 config error.
"""

import json
import logging
import math
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from pydantic import AwareDatetime, BaseModel, ValidationError

from pm25_forecast.collector import ConfigError, default_run_id, open_db
from pm25_forecast.features import FEATURE_NAMES, build_features
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
from pm25_forecast.monitoring.alerts import make_alert, safe_alert
from pm25_forecast.monitoring.logs import configure_logging, log_summary
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

logger = logging.getLogger(__name__)

EXIT_FAILED, EXIT_CONFIG = 1, 2

PSI_WARN = 0.1  # below: ok
PSI_ALERT = 0.2  # above: alert; in between: warn
MIN_RECENT = 150  # fewer recent points: insufficient (a full week is 168 per station and horizon)
MIN_REFERENCE = 200  # fewer reference points: insufficient
MIN_BINS = 3  # fewer distinct reference bins (near-constant or integer-heavy data): insufficient
CAMS_MIN_COVERAGE = 0.5  # cams_pm25 needs CAMS for at least this share of the reference origins
RECENT_WINDOW = timedelta(days=7)
PSI_BINS = 5
PSI_EPSILON = 1e-4  # share used for an empty bin, so the logarithm is defined
TARGET_FEATURE = "target_pm25"
CAMS_FEATURE = "cams_pm25"
DEFAULT_HORIZONS = (24,)  # used when the manifest metrics do not name horizons
HISTORY = timedelta(hours=24)  # observation history given to build_features, as in training
CONSECUTIVE_WITHIN = timedelta(days=2)  # the previous run counts as "consecutive" within this
STALE_AFTER = timedelta(hours=36)  # latest_drift_alert ignores a run older than this

OK, WARN, ALERT, INSUFFICIENT = "ok", "warn", "alert", "insufficient"

SELECT_MANIFEST = "SELECT manifest FROM models WHERE model_version = %(model_version)s"
SELECT_CHAMPION_VERSION = "SELECT model_version FROM model_champion WHERE slot = 'production'"
SELECT_OBSERVATIONS = """
SELECT station_id, ts, pm25 FROM observations
WHERE ts >= %(since)s AND ts <= %(until)s AND pm25 IS NOT NULL
ORDER BY station_id, ts
"""
SELECT_CAMS_VINTAGES = """
SELECT station_id, issue_time, valid_time, pm25_pred FROM predictions
WHERE model_version = %(model_version)s AND valid_time >= %(since)s AND valid_time <= %(until)s
"""
UPSERT_DRIFT_CHECK = """
INSERT INTO drift_checks (checked_at, model_version, feature, psi, status, ref_start, ref_end,
                          recent_start, recent_end, n_ref, n_recent, computed_at)
VALUES (%(checked_at)s, %(model_version)s, %(feature)s, %(psi)s, %(status)s, %(ref_start)s,
        %(ref_end)s, %(recent_start)s, %(recent_end)s, %(n_ref)s, %(n_recent)s, %(computed_at)s)
ON CONFLICT (checked_at, model_version, feature) DO UPDATE SET
    psi = excluded.psi, status = excluded.status, ref_start = excluded.ref_start,
    ref_end = excluded.ref_end, recent_start = excluded.recent_start,
    recent_end = excluded.recent_end, n_ref = excluded.n_ref, n_recent = excluded.n_recent,
    computed_at = excluded.computed_at
"""
SELECT_REFERENCE = """
SELECT feature, edges, shares, n_ref, n_origins, usable, ref_start, ref_end FROM drift_reference
WHERE model_version = %(model_version)s
"""
UPSERT_REFERENCE = """
INSERT INTO drift_reference (model_version, feature, edges, shares, n_ref, n_origins, usable,
                             computed_at, ref_start, ref_end)
VALUES (%(model_version)s, %(feature)s, %(edges)s, %(shares)s, %(n_ref)s, %(n_origins)s,
        %(usable)s, %(computed_at)s, %(ref_start)s, %(ref_end)s)
ON CONFLICT (model_version, feature) DO UPDATE SET
    edges = excluded.edges, shares = excluded.shares, n_ref = excluded.n_ref,
    n_origins = excluded.n_origins, usable = excluded.usable, computed_at = excluded.computed_at,
    ref_start = excluded.ref_start, ref_end = excluded.ref_end
"""
SELECT_PREVIOUS_PSI = """
SELECT psi FROM drift_checks
WHERE model_version = %(model_version)s AND feature = %(feature)s
  AND checked_at < %(day)s AND checked_at >= %(since)s
ORDER BY checked_at DESC LIMIT 1
"""
# Latest run = latest checked_at day (optionally for one model); its age is its newest computed_at.
SELECT_LATEST_RUN = """
SELECT max(computed_at) FROM drift_checks
WHERE checked_at = (SELECT max(checked_at) FROM drift_checks)
"""
COUNT_LATEST_ALERTS = """
SELECT count(*) FROM drift_checks
WHERE status = 'alert' AND checked_at = (SELECT max(checked_at) FROM drift_checks)
"""
SELECT_LATEST_RUN_MODEL = """
SELECT max(computed_at) FROM drift_checks
WHERE model_version = %(model_version)s AND checked_at = (
    SELECT max(checked_at) FROM drift_checks WHERE model_version = %(model_version)s)
"""
COUNT_LATEST_ALERTS_MODEL = """
SELECT count(*) FROM drift_checks
WHERE status = 'alert' AND model_version = %(model_version)s AND checked_at = (
    SELECT max(checked_at) FROM drift_checks WHERE model_version = %(model_version)s)
"""


# --- pure statistics ---------------------------------------------------------------------------


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _shares(values: list[float], edges: Sequence[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect_left(edges, v)] += 1  # bins are (e[i-1], e[i]]
    return [max(c / len(values), PSI_EPSILON) for c in counts]


def reference_profile(
    reference: Iterable[Any], bins: int = PSI_BINS
) -> tuple[list[float], list[float]]:
    """Bin edges (interior reference quantiles, ties collapsed) and the reference bin shares."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    ref = _finite(reference)
    if not ref:
        raise ValueError("psi needs at least one finite value on each side")
    edges = sorted(set(np.quantile(ref, [k / bins for k in range(1, bins)]).tolist()))
    return edges, _shares(ref, edges)


def psi_vs_profile(
    edges: Sequence[float], ref_shares: Sequence[float], recent: Iterable[Any]
) -> float:
    """PSI of ``recent`` against a stored reference profile."""
    rec = _finite(recent)
    if not rec:
        raise ValueError("psi needs at least one finite value on each side")
    shares = _shares(rec, edges)
    return sum((r - e) * math.log(r / e) for e, r in zip(ref_shares, shares, strict=True))


def psi(reference: Iterable[Any], recent: Iterable[Any], bins: int = 10) -> float:
    """Population Stability Index of ``recent`` against ``reference``.

    Bin edges are the reference's interior quantiles (``bins`` equal-share bins; ties collapse
    duplicate edges, so there may be fewer). Open-ended outer bins catch values outside the
    reference range. A bin share below ``PSI_EPSILON`` (empty bin) is raised to it. Non-finite
    values (None, NaN, inf) are dropped. Raises ValueError if either side has no finite value.
    """
    edges, ref_shares = reference_profile(reference, bins)
    return psi_vs_profile(edges, ref_shares, recent)


def classify(value: float | None, n_ref: int, n_recent: int) -> str:
    if n_ref < MIN_REFERENCE or n_recent < MIN_RECENT or value is None:
        return INSUFFICIENT
    if value < PSI_WARN:
        return OK
    return WARN if value <= PSI_ALERT else ALERT


# --- reading and rebuilding inputs -------------------------------------------------------------


class Window(BaseModel):
    start: AwareDatetime
    end: AwareDatetime


class Manifest(BaseModel):
    training_window: Window
    assumed_reporting_lag_hours: float | None = None
    metrics: dict[str, Any] = {}  # keyed by horizon in hours ("24", or "6", "12", "24")

    @property
    def horizons(self) -> tuple[int, ...]:
        found = sorted({int(k) for k in self.metrics if str(k).isdigit() and int(k) > 0})
        return tuple(found) or DEFAULT_HORIZONS


class ObsRow(BaseModel):
    station_id: str
    ts: AwareDatetime
    pm25: float


class CamsRow(BaseModel):
    station_id: str
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    pm25_pred: float


@dataclass(frozen=True)
class Check:
    model_version: str
    feature: str
    psi: float | None
    status: str
    ref: Window
    recent: Window
    n_ref: int
    n_recent: int


@dataclass
class DriftResult:
    run_id: str
    checked_at: datetime  # the UTC day
    checks: list[Check] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def alerts(self) -> list[Check]:
        return [c for c in self.checks if c.status == ALERT]


def floor_to_hour(moment: datetime) -> datetime:
    return sql.require_utc(moment).replace(minute=0, second=0, microsecond=0)


def _validated[M: BaseModel](model: type[M], rows: Sequence[Sequence[Any]], names: Sequence[str]):
    good: list[M] = []
    for row in rows:
        try:
            good.append(model.model_validate(dict(zip(names, row, strict=True))))
        except ValidationError:
            logger.warning("%s: dropped a malformed row", model.__name__)
    return good


def read_manifest(con: Any, model_version: str) -> Manifest:
    row = con.execute(SELECT_MANIFEST, {"model_version": model_version}).fetchone()
    if row is None:
        raise LookupError(f"model version not found: {model_version}")
    raw = json.loads(row[0]) if isinstance(row[0], str | bytes) else row[0]
    try:
        manifest = Manifest.model_validate(raw)
    except ValidationError:
        raise ValueError(f"{model_version}: manifest has no valid training_window") from None
    if manifest.training_window.start >= manifest.training_window.end:
        raise ValueError(f"{model_version}: training_window start is not before end")
    return manifest


class Inputs:
    """Stored observations and CAMS vintages for a time range, indexed for hourly rebuilds."""

    def __init__(
        self,
        con: Any,
        start: datetime,
        end: datetime,
        station_ids: set[str] | None,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
    ):
        obs_raw = con.execute(
            SELECT_OBSERVATIONS, {"since": start - HISTORY, "until": end}
        ).fetchall()
        cams_raw = con.execute(
            SELECT_CAMS_VINTAGES,
            {
                "model_version": CAMS_MODEL_VERSION,
                "since": start + timedelta(hours=min(horizons)),
                "until": end + timedelta(hours=max(horizons)),
            },
        ).fetchall()
        obs = _validated(ObsRow, obs_raw, ("station_id", "ts", "pm25"))
        cams = _validated(
            CamsRow, cams_raw, ("station_id", "issue_time", "valid_time", "pm25_pred")
        )

        def keep(station: str) -> bool:
            return not station_ids or station in station_ids

        self.times: dict[str, list[datetime]] = defaultdict(list)
        self.values: dict[str, list[float]] = defaultdict(list)
        for o in sorted(obs, key=lambda r: (r.station_id, r.ts)):
            if keep(o.station_id) and o.pm25 >= 0 and math.isfinite(o.pm25):
                self.times[o.station_id].append(o.ts)
                self.values[o.station_id].append(o.pm25)
        self.cams: dict[tuple[str, datetime], list[tuple[Any, ...]]] = defaultdict(list)
        for c in cams:
            if keep(c.station_id):
                self.cams[(c.station_id, c.valid_time)].append(
                    (c.issue_time, c.valid_time, c.pm25_pred, CAMS_MODEL_VERSION)
                )

    def target_values(self, start: datetime, end: datetime) -> list[float]:
        out: list[float] = []
        for station, times in self.times.items():
            lo, hi = bisect_left(times, start), bisect_right(times, end)
            out.extend(self.values[station][lo:hi])
        return out

    def feature_values(
        self, start: datetime, end: datetime, lag: timedelta, horizons: Sequence[int]
    ) -> tuple[dict[str, list[float]], int]:
        """Hourly features for origins in [start, end], through the shared ``build_features``.

        One feature set per station, origin and horizon (as ``predict`` builds them). The history
        passed is the 24 h up to ``origin - lag``, as in ``train_model``. ``pm25_latest`` does not
        depend on the horizon and is counted once per origin. Also returns the number of
        (station, origin, horizon) combinations that had any feature value.
        """
        out: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
        n_origins = 0
        first = floor_to_hour(start)
        if first < start:
            first += timedelta(hours=1)
        origin_count = int((end - first) / timedelta(hours=1)) + 1
        for station, times in self.times.items():
            values = self.values[station]
            for i in range(max(origin_count, 0)):
                origin = first + timedelta(hours=i)
                lo = bisect_left(times, origin - HISTORY)
                hi = bisect_right(times, origin - lag)
                history = list(zip(times[lo:hi], values[lo:hi], strict=True))
                for j, horizon in enumerate(horizons):
                    target = origin + timedelta(hours=horizon)
                    features = build_features(
                        history, self.cams.get((station, target), ()), origin, target
                    )
                    if all(features[name] is None for name in FEATURE_NAMES):
                        continue
                    n_origins += 1
                    for name in FEATURE_NAMES:
                        if features[name] is not None and (j == 0 or name != "pm25_latest"):
                            out[name].append(features[name])
        return out, n_origins


@dataclass(frozen=True)
class Reference:
    """Stored reference distribution of one feature (see ``drift_reference``)."""

    edges: list[float]
    shares: list[float]
    n_ref: int
    n_origins: int
    usable: bool


def make_reference(feature: str, values: list[float], n_origins: int) -> Reference:
    finite = _finite(values)
    if len(finite) < MIN_REFERENCE:
        return Reference([], [], len(finite), n_origins, False)
    edges, shares = reference_profile(finite, PSI_BINS)
    usable = len(edges) + 1 >= MIN_BINS
    if feature == CAMS_FEATURE and len(finite) < CAMS_MIN_COVERAGE * n_origins:
        usable = False
    return Reference(edges, shares, len(finite), n_origins, usable)


def load_reference(
    con: Any, model_version: str, window: Window | None = None
) -> dict[str, Reference] | None:
    """The cached reference for every feature and the target, or None if not (fully) cached.

    With ``window`` (the manifest's training window), a reference built from a different window,
    or one that stored no window (written before KAN-24), counts as not cached and is rebuilt.
    """
    rows = con.execute(SELECT_REFERENCE, {"model_version": model_version}).fetchall()
    found: dict[str, Reference] = {}
    for feature, edges, shares, n_ref, n_origins, usable, ref_start, ref_end in rows:
        if window is not None and (
            as_utc(ref_start) != as_utc(window.start) or as_utc(ref_end) != as_utc(window.end)
        ):
            return None
        try:
            found[feature] = Reference(
                [float(x) for x in json.loads(edges)],
                [float(x) for x in json.loads(shares)],
                int(n_ref),
                int(n_origins),
                bool(usable),
            )
        except (TypeError, ValueError):
            return None  # unreadable cache: rebuild it
    return found if {*FEATURE_NAMES, TARGET_FEATURE} <= found.keys() else None


def save_reference(
    con: Any,
    model_version: str,
    reference: dict[str, Reference],
    now: datetime,
    window: Window | None = None,
) -> None:
    rows = [
        {
            "model_version": model_version,
            "feature": feature,
            "edges": json.dumps(ref.edges),
            "shares": json.dumps(ref.shares),
            "n_ref": ref.n_ref,
            "n_origins": ref.n_origins,
            "usable": ref.usable,
            "computed_at": now,
            "ref_start": None if window is None else window.start,
            "ref_end": None if window is None else window.end,
        }
        for feature, ref in reference.items()
    ]
    sql.write_rows(con, UPSERT_REFERENCE, rows)


def build_reference(
    con: Any, manifest: Manifest, stations: set[str] | None
) -> dict[str, Reference]:
    """Rebuild the reference from the training window (once per model_version, then cached)."""
    window = manifest.training_window
    lag = timedelta(hours=manifest.assumed_reporting_lag_hours or 0)
    inputs = Inputs(con, window.start, window.end, stations, manifest.horizons)
    features, n_origins = inputs.feature_values(window.start, window.end, lag, manifest.horizons)
    reference = {name: make_reference(name, features[name], n_origins) for name in FEATURE_NAMES}
    target = inputs.target_values(window.start, window.end)
    reference[TARGET_FEATURE] = make_reference(TARGET_FEATURE, target, len(target))
    return reference


def previous_psi(con: Any, model_version: str, feature: str, day: datetime) -> float | None:
    row = con.execute(
        SELECT_PREVIOUS_PSI,
        {
            "model_version": model_version,
            "feature": feature,
            "day": day,
            "since": day - CONSECUTIVE_WITHIN,
        },
    ).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def model_checks(
    model_version: str,
    manifest: Manifest,
    con: Any,
    now: datetime,
    stations: set[str] | None,
    day: datetime | None = None,
) -> list[Check]:
    day = day or now.replace(hour=0, minute=0, second=0, microsecond=0)
    ref = manifest.training_window
    recent_end = floor_to_hour(now)
    recent = Window(start=recent_end - RECENT_WINDOW, end=recent_end)
    lag = timedelta(hours=manifest.assumed_reporting_lag_hours or 0)

    reference = load_reference(con, model_version, ref)
    if reference is None:  # not cached, or built from a window other than the manifest's
        reference = build_reference(con, manifest, stations)
        save_reference(con, model_version, reference, now, ref)

    recent_inputs = Inputs(con, recent.start, recent.end, stations, manifest.horizons)
    recent_features, _ = recent_inputs.feature_values(
        recent.start, recent.end, lag, manifest.horizons
    )
    recent_series = {
        **recent_features,
        TARGET_FEATURE: recent_inputs.target_values(recent.start, recent.end),
    }

    checks: list[Check] = []
    for name, recent_values in recent_series.items():
        ref_profile = reference[name]
        n_recent = len(_finite(recent_values))
        value = None
        if ref_profile.usable and n_recent >= MIN_RECENT:
            value = psi_vs_profile(ref_profile.edges, ref_profile.shares, recent_values)
        status = classify(value, ref_profile.n_ref, n_recent)
        if not ref_profile.usable:
            status = INSUFFICIENT
        if status == ALERT:
            previous = previous_psi(con, model_version, name, day)
            if previous is None or previous <= PSI_ALERT:
                status = WARN  # needs two consecutive daily runs above the threshold
        checks.append(
            Check(model_version, name, value, status, ref, recent, ref_profile.n_ref, n_recent)
        )
    return checks


def as_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def latest_drift_run(con: Any, model_version: str) -> datetime | None:
    """When the latest drift run for ``model_version`` was computed (UTC), or None if none."""
    row = con.execute(SELECT_LATEST_RUN_MODEL, {"model_version": model_version}).fetchone()
    return as_utc(row[0]) if row else None


def latest_drift_alert(
    con: Any, model_version: str | None = None, *, now: datetime | None = None
) -> bool:
    """True if the latest drift run (latest UTC day) has a check in ``alert`` status.

    Optionally only for one ``model_version``. A latest run computed more than ``STALE_AFTER``
    ago counts as no alert, so a stale alert cannot trigger a retrain again. ``now`` is for tests.
    """
    now = sql.require_utc(now or datetime.now(UTC))
    params = {} if model_version is None else {"model_version": model_version}
    latest = as_utc(
        con.execute(
            SELECT_LATEST_RUN if model_version is None else SELECT_LATEST_RUN_MODEL, params
        ).fetchone()[0]
    )
    if latest is None or now - latest > STALE_AFTER:
        return False
    row = con.execute(
        COUNT_LATEST_ALERTS if model_version is None else COUNT_LATEST_ALERTS_MODEL, params
    ).fetchone()
    return bool(row and row[0] > 0)


def alert_message(result: DriftResult) -> str | None:
    parts: list[str] = []
    if result.alerts:
        listed = ", ".join(
            f"{c.model_version}/{c.feature} (PSI {c.psi:.2f})" for c in result.alerts
        )
        parts.append(f"drift alert (PSI > {PSI_ALERT}): {listed}")
    if result.failures:
        parts.append("run completed with failures: " + "; ".join(result.failures))
    return " | ".join(parts) or None


def run(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> DriftResult:
    """One drift run. Driver connection errors propagate for `run_entrypoint` to map."""
    now = sql.require_utc(clock())
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    try:
        stations = {i for i, _lat, _lon in settings.station_location_list} or None
    except ValueError as exc:
        raise ConfigError(str(exc)) from None

    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    result = DriftResult(run_id=default_run_id(now), checked_at=day)
    with open_db(connect, settings.database_url.get_secret_value()) as con:
        versions: list[str] = []
        if settings.challenger_model_version:
            versions.append(settings.challenger_model_version)
        champion = con.execute(SELECT_CHAMPION_VERSION, {}).fetchone()
        if champion and champion[0] not in versions:
            versions.append(champion[0])
        if not versions:
            result.failures.append("no challenger configured and no champion registered")
        for model_version in versions:
            try:
                manifest = read_manifest(con, model_version)
                result.checks += model_checks(model_version, manifest, con, now, stations, day)
            except (LookupError, ValueError) as exc:
                result.failures.append(f"{model_version}: {exc}")
        rows = [
            {
                "checked_at": day,
                "model_version": c.model_version,
                "feature": c.feature,
                "psi": None if c.psi is None else round(c.psi, 6),
                "status": c.status,
                "ref_start": c.ref.start,
                "ref_end": c.ref.end,
                "recent_start": c.recent.start,
                "recent_end": c.recent.end,
                "n_ref": c.n_ref,
                "n_recent": c.n_recent,
                "computed_at": now,
            }
            for c in result.checks
        ]
        sql.write_rows(con, UPSERT_DRIFT_CHECK, rows)

    log_summary(
        logger,
        "drift",
        {
            "run_id": result.run_id,
            "status": "failed" if result.failures else "ok",
            "checked_at": day,
            "checks": [
                {
                    "model": c.model_version,
                    "feature": c.feature,
                    "psi": None if c.psi is None else round(c.psi, 4),
                    "status": c.status,
                    "n_ref": c.n_ref,
                    "n_recent": c.n_recent,
                }
                for c in result.checks
            ],
            "alerts": [f"{c.model_version}/{c.feature}" for c in result.alerts],
            "failures": result.failures,
        },
    )
    return result


def main() -> int:
    configure_logging()
    alert = make_alert(None, prefix="[pm25 drift]")
    try:
        settings = get_settings()
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook, prefix="[pm25 drift]")
        result, code = run_entrypoint(
            lambda: run(settings), alert, connection_errors=db.CONNECTION_ERRORS
        )
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        safe_alert(alert, f"configuration error: {exc}")
        log_summary(logger, "drift", {"status": "config_error"})
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.exception("drift crashed")
        safe_alert(alert, f"drift crashed: {type(exc).__name__}")
        log_summary(logger, "drift", {"status": "crashed", "error": type(exc).__name__})
        return EXIT_FAILED
    if isinstance(result, RunSkipped):  # DB down: the guard has already alerted
        log_summary(logger, "drift", {"status": "db_unreachable"})
        return code
    message = alert_message(result)  # one alert per run
    if message:
        safe_alert(alert, message)
    return EXIT_FAILED if result.failures else code


if __name__ == "__main__":
    sys.exit(main())
