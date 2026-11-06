"""Hourly scoring step (KAN-20). Run with ``python -m pm25_forecast.score``.

Per run, for the baseline ``persistence-24h`` and every shadow challenger
(``CHALLENGER_MODEL_VERSION`` when set; a comma-separated list since KAN-104): join forecasts
whose valid_time has passed with the actual for that station and hour, write one row per forecast
to ``scores``, then report rolling metrics by model and horizon.

* Matured = ``valid_time <= run time``. A matured forecast with no usable actual yet is *waiting*,
  counted in the summary, never an error and never filled in.
* Forecasts are (re)scored over a lookback of ``RESCORE_LOOKBACK`` of valid_time, so a late or
  corrected observation updates its score. Only new or changed scores are written (upsert), so a
  re-run writes nothing.
* ``--since <UTC datetime>`` re-scores from that valid_time instead (one-off repair after
  observations were corrected, KAN-74); it is idempotent and does not change the metrics window.
* CAMS rows are not scored: CAMS is out of the plan (decided 2026-10-29).
* Metrics over ``METRICS_WINDOW`` of valid_time, per model and horizon (any whole number of hours
  present in the scores, KAN-104: a ridge serves +1..+24 h; +6/+12/+24 h stay the headline and are
  always listed in the run summary, as ``{"n": 0}`` when empty): n, MAE,
  RMSE and skill vs persistence = 1 - MAE_model / MAE_persistence on matched pairs (same station,
  issue_time and valid_time). Positive skill = better than persistence. ``n`` counts all of the
  model's scores; ``n_matched`` the pairs behind the skill. A model or horizon with no scores is
  reported as ``{"n": 0}``; skill is null when there are no matched pairs.
* A score whose actual is later corrected to unusable (NULL/negative) leaves the metrics; the
  row is kept, not deleted.
* Reads and writes only the database. Never touches ``predictions``, so a scoring failure cannot
  cost a forecast. DB down: ``run_entrypoint`` skips the run, alerts, exits non-zero.

Exit codes: 0 ok (waiting rows are reported, not failures), 1 failed / DB down, 2 config error.
"""

import argparse
import logging
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import AwareDatetime, BaseModel, ValidationError

from pm25_forecast.collector import ConfigError, default_run_id, open_db
from pm25_forecast.models.baselines import HORIZONS_H, PERSISTENCE_MODEL_VERSION
from pm25_forecast.monitoring.alerts import make_alert, safe_alert
from pm25_forecast.monitoring.logs import configure_logging, log_summary
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

logger = logging.getLogger(__name__)

EXIT_FAILED, EXIT_CONFIG = 1, 2
RESCORE_LOOKBACK = timedelta(hours=72)
METRICS_WINDOW = timedelta(days=7)
SERVED_REFERENCE = "__served_persistence_reference__"
REFERENCE_MODEL_VERSION = PERSISTENCE_MODEL_VERSION

Key = tuple[str, datetime, datetime]  # (station_id, issue_time, valid_time)


class MaturedRow(BaseModel):
    station_id: str
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    pm25_pred: float
    pm25_obs: float | None
    stored_obs: float | None


class ServedMaturedRow(MaturedRow):
    model_version: str


class ScoreRow(BaseModel):
    station_id: str
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    abs_error: float


class ServedScoreRow(ScoreRow):
    model_version: str


@dataclass(frozen=True)
class Metric:
    n: int
    mae: float
    rmse: float
    skill_vs_persistence: float | None = None
    n_matched: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mae": round(self.mae, 3),
            "rmse": round(self.rmse, 3),
            "skill_vs_persistence": None
            if self.skill_vs_persistence is None
            else round(self.skill_vs_persistence, 4),
            "n_matched": self.n_matched,
        }


@dataclass
class ScoreResult:
    run_id: str
    scored_at: datetime
    matured: dict[str, int] = field(default_factory=dict)
    waiting: dict[str, int] = field(default_factory=dict)
    written: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, dict[int, Metric]] = field(default_factory=dict)
    served_metrics: dict[str, dict[int, Metric]] = field(default_factory=dict)
    served_matured: int = 0
    served_waiting: int = 0
    served_status: str = "unavailable"
    failures: list[str] = field(default_factory=list)


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0


def _validated[M: BaseModel](model: type[M], rows: Sequence[Sequence[Any]], names: Sequence[str]):
    """Validate DB rows; a bad row is dropped and logged (type only), never used."""
    good: list[M] = []
    for row in rows:
        try:
            good.append(model.model_validate(dict(zip(names, row, strict=True))))
        except ValidationError:
            logger.warning("%s: dropped a malformed row", model.__name__)
    return good


def horizon_h(issue_time: datetime, valid_time: datetime) -> float:
    return (sql.require_utc(valid_time) - sql.require_utc(issue_time)) / timedelta(hours=1)


def score_rows(
    matured: Iterable[MaturedRow], model_version: str
) -> tuple[list[dict[str, Any]], int]:
    """Score rows to write (new or changed actual only) and the number still waiting."""
    rows: list[dict[str, Any]] = []
    waiting = 0
    for m in matured:
        if not _usable(m.pm25_obs) or not math.isfinite(m.pm25_pred):
            waiting += 1
            continue
        obs = float(m.pm25_obs)  # type: ignore[arg-type]  # checked by _usable
        if m.stored_obs is not None and m.stored_obs == obs:
            continue  # already scored against this actual
        rows.append(
            {
                "station_id": m.station_id,
                "issue_time": m.issue_time,
                "valid_time": m.valid_time,
                "model_version": model_version,
                "pm25_obs": obs,
                "abs_error": abs(m.pm25_pred - obs),
            }
        )
    return rows, waiting


def reported_horizons(*metrics: dict[str, dict[int, Metric]]) -> list[int]:
    """The headline horizons plus every other horizon that has metrics, ascending."""
    present = {h for m in metrics for per_h in m.values() for h in per_h}
    return sorted({*HORIZONS_H, *present})


def _mae_rmse(errors: Sequence[float]) -> tuple[float, float]:
    return sum(errors) / len(errors), math.sqrt(sum(e * e for e in errors) / len(errors))


def rolling_metrics(
    scores: dict[str, Sequence[ScoreRow]],
    reference: str = REFERENCE_MODEL_VERSION,
    horizons: Sequence[int] | None = None,
) -> dict[str, dict[int, Metric]]:
    """Per model and horizon: n, MAE, RMSE; skill vs ``reference`` on matched pairs only.

    ``horizons`` None (default, KAN-104) reports every whole-hour horizon found in the scores;
    pass a sequence to restrict to those hours.
    """
    by_model: dict[str, dict[int, dict[Key, float]]] = {}
    for model_version, rows in scores.items():
        per_h: dict[int, dict[Key, float]] = {} if horizons is None else {h: {} for h in horizons}
        for r in rows:
            h = horizon_h(r.issue_time, r.valid_time)
            if h != int(h) or h <= 0 or (horizons is not None and h not in per_h):
                continue  # not a whole forecast hour (or not asked for)
            key = (r.station_id, sql.require_utc(r.issue_time), sql.require_utc(r.valid_time))
            per_h.setdefault(int(h), {})[key] = r.abs_error
        by_model[model_version] = per_h

    out: dict[str, dict[int, Metric]] = {}
    ref = by_model.get(reference, {})
    for model_version, per_h in by_model.items():
        out[model_version] = {}
        for h, errs in per_h.items():
            if not errs:
                continue
            mae, rmse = _mae_rmse(list(errs.values()))
            skill, n_matched = None, 0
            if model_version != reference:
                common = errs.keys() & ref.get(h, {}).keys()
                n_matched = len(common)
                if common:
                    ref_mae = sum(ref[h][k] for k in common) / n_matched
                    model_mae = sum(errs[k] for k in common) / n_matched
                    skill = 1 - model_mae / ref_mae if ref_mae > 0 else None
            out[model_version][h] = Metric(len(errs), mae, rmse, skill, n_matched)
    return out


def run(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    since: datetime | None = None,
) -> ScoreResult:
    """One scoring run. Driver connection errors propagate for `run_entrypoint` to map.

    ``since`` (aware UTC, one-off re-score, KAN-74) replaces the ``RESCORE_LOOKBACK`` start so
    scores can be recomputed after observations were repaired. It only widens or moves the
    rescored valid_time range; the metrics window is unchanged.
    """
    now = sql.require_utc(clock())
    rescore_from = now - RESCORE_LOOKBACK if since is None else sql.require_utc(since)
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    models = [REFERENCE_MODEL_VERSION]
    models.extend(v for v in settings.challenger_model_versions if v not in models)

    result = ScoreResult(run_id=default_run_id(now), scored_at=now)
    scores: dict[str, list[ScoreRow]] = {}
    with open_db(connect, settings.database_url.get_secret_value()) as con:
        for model_version in models:
            raw = con.execute(
                sql.SELECT_MATURED_FORECASTS,
                {"model_version": model_version, "since": rescore_from, "now": now},
            ).fetchall()
            matured = _validated(
                MaturedRow,
                raw,
                ("station_id", "issue_time", "valid_time", "pm25_pred", "pm25_obs", "stored_obs"),
            )
            rows, waiting = score_rows(matured, model_version)
            result.matured[model_version] = len(matured)
            result.waiting[model_version] = waiting
            result.written[model_version] = sql.write_rows(con, sql.UPSERT_SCORE, rows)
        for model_version in models:
            raw = con.execute(
                sql.SELECT_SCORES_BETWEEN,
                {"model_version": model_version, "since": now - METRICS_WINDOW, "now": now},
            ).fetchall()
            scores[model_version] = _validated(
                ScoreRow, raw, ("station_id", "issue_time", "valid_time", "abs_error")
            )
        # Keep an absent/unreadable KAN-119 ledger from aborting legacy scoring. PostgreSQL
        # marks a transaction failed after a missing-table error, so isolate the optional path.
        con.execute("SAVEPOINT served_ledger_score", {})
        try:
            served_raw = con.execute(
                sql.SELECT_MATURED_SERVED_FORECASTS,
                {"since": rescore_from, "now": now},
            ).fetchall()
            served_rows = _validated(
                ServedMaturedRow,
                served_raw,
                (
                    "station_id",
                    "issue_time",
                    "valid_time",
                    "model_version",
                    "pm25_pred",
                    "pm25_obs",
                    "stored_obs",
                ),
            )
            result.served_matured = len(served_rows)
            served_scores: dict[str, list[ScoreRow]] = {}
            by_served_model: dict[str, list[ServedMaturedRow]] = {}
            for row in served_rows:
                by_served_model.setdefault(row.model_version, []).append(row)
            for model_version, rows_for_model in by_served_model.items():
                write_rows, waiting = score_rows(rows_for_model, model_version)
                result.served_waiting += waiting
                sql.write_rows(con, sql.UPSERT_SCORE, write_rows)
            for model_version, params in (
                ("served", {"since": now - METRICS_WINDOW, "now": now}),
                (
                    "reference",
                    {
                        "model_version": REFERENCE_MODEL_VERSION,
                        "since": now - METRICS_WINDOW,
                        "now": now,
                    },
                ),
            ):
                statement = (
                    sql.SELECT_SERVED_SCORES_BETWEEN
                    if model_version == "served"
                    else sql.SELECT_SERVED_PERSISTENCE_SCORES_BETWEEN
                )
                raw = con.execute(statement, params).fetchall()
                if model_version == "served":
                    served_score_rows = _validated(
                        ServedScoreRow,
                        raw,
                        ("model_version", "station_id", "issue_time", "valid_time", "abs_error"),
                    )
                    for row in served_score_rows:
                        served_scores.setdefault(row.model_version, []).append(row)
                else:
                    # This comparator covers all served keys, even where persistence was not
                    # selected; keep it under a private key so it cannot masquerade as served.
                    served_scores[SERVED_REFERENCE] = _validated(
                        ScoreRow,
                        raw,
                        ("station_id", "issue_time", "valid_time", "abs_error"),
                    )
            served_metrics = rolling_metrics(served_scores, reference=SERVED_REFERENCE)
            served_metrics.pop(SERVED_REFERENCE, None)
            result.served_metrics = served_metrics
            con.execute("RELEASE SAVEPOINT served_ledger_score", {})
            result.served_status = "ok"
        except Exception as exc:
            con.execute("ROLLBACK TO SAVEPOINT served_ledger_score", {})
            con.execute("RELEASE SAVEPOINT served_ledger_score", {})
            result.served_matured = 0
            result.served_waiting = 0
            result.served_metrics = {}
            logger.warning("served-ledger scoring unavailable (%s)", type(exc).__name__)
            result.served_status = f"unavailable: {type(exc).__name__}"
    result.metrics = rolling_metrics(scores)

    log_summary(
        logger,
        "score",
        {
            "run_id": result.run_id,
            "status": "failed" if result.failures else "ok",
            "scored_at": now,
            "models": models,
            "matured": result.matured,
            "waiting_for_actuals": result.waiting,
            "written": result.written,
            "metrics_window_days": METRICS_WINDOW.days,
            "metrics": {
                m: {
                    f"+{h}h": result.metrics[m][h].as_dict()
                    if h in result.metrics.get(m, {})
                    else {"n": 0}
                    for h in reported_horizons(result.metrics)
                }
                for m in models
            },
            "served_matured": result.served_matured,
            "served_waiting_for_actuals": result.served_waiting,
            "served_status": result.served_status,
            "served_rescore_window_hours": int(RESCORE_LOOKBACK.total_seconds() / 3600),
            "served_metrics_window_days": METRICS_WINDOW.days,
            "served_metrics": {
                m: {
                    f"+{h}h": result.served_metrics[m][h].as_dict()
                    if h in result.served_metrics.get(m, {})
                    else {"n": 0}
                    for h in reported_horizons(result.served_metrics)
                }
                for m in result.served_metrics
            },
            "failures": result.failures,
        },
    )
    return result


def parse_since(value: str) -> datetime:
    """``--since`` value: ISO datetime with an explicit UTC offset, e.g. 2026-10-27T00:00:00Z."""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO datetime: {value!r}") from None
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError("--since needs a UTC offset, e.g. 2026-10-27T00:00:00Z")
    return moment.astimezone(UTC)


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(
        prog="pm25_forecast.score", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--since",
        type=parse_since,
        default=None,
        help="one-off re-score of forecasts with valid_time from this UTC datetime "
        f"(default: the last {RESCORE_LOOKBACK})",
    )
    args = parser.parse_args(list(argv))
    configure_logging()
    alert = make_alert(None, prefix="[pm25 score]")
    try:
        try:
            settings = get_settings()
        except ValidationError as exc:
            # field names only, never values (KAN-104: an invalid challenger list is a config error)
            fields = ", ".join(sorted({".".join(map(str, e["loc"])) for e in exc.errors()}))
            raise ConfigError(f"invalid settings: {fields}") from None
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook, prefix="[pm25 score]")
        result, code = run_entrypoint(
            lambda: run(settings, since=args.since), alert, connection_errors=db.CONNECTION_ERRORS
        )
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        safe_alert(alert, f"configuration error: {exc}")
        log_summary(logger, "score", {"status": "config_error"})
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.exception("score crashed")
        safe_alert(alert, f"score crashed: {type(exc).__name__}")
        log_summary(logger, "score", {"status": "crashed", "error": type(exc).__name__})
        return EXIT_FAILED
    if isinstance(result, RunSkipped):  # DB down: the guard has already alerted
        log_summary(logger, "score", {"status": "db_unreachable"})
        return code
    if result.failures:
        safe_alert(alert, "run completed with failures: " + "; ".join(result.failures))
        return EXIT_FAILED
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
