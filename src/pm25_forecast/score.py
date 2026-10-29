"""Hourly scoring step (KAN-20). Run with ``python -m pm25_forecast.score``.

Per run, for the baseline ``persistence-24h`` and the shadow challenger
(``CHALLENGER_MODEL_VERSION`` when set): join forecasts whose valid_time has passed with the
actual for that station and hour, write one row per forecast to ``scores``, then report rolling
metrics by model and horizon.

* Matured = ``valid_time <= run time``. A matured forecast with no usable actual yet is *waiting*,
  counted in the summary, never an error and never filled in.
* Forecasts are (re)scored over a lookback of ``RESCORE_LOOKBACK`` of valid_time, so a late or
  corrected observation updates its score. Only new or changed scores are written (upsert), so a
  re-run writes nothing.
* CAMS rows are not scored: CAMS is out of the plan (decided 2026-10-29).
* Metrics over ``METRICS_WINDOW`` of valid_time, per model and horizon (+6/+12/+24 h): n, MAE,
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
REFERENCE_MODEL_VERSION = PERSISTENCE_MODEL_VERSION

Key = tuple[str, datetime, datetime]  # (station_id, issue_time, valid_time)


class MaturedRow(BaseModel):
    station_id: str
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    pm25_pred: float
    pm25_obs: float | None
    stored_obs: float | None


class ScoreRow(BaseModel):
    station_id: str
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    abs_error: float


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


def _mae_rmse(errors: Sequence[float]) -> tuple[float, float]:
    return sum(errors) / len(errors), math.sqrt(sum(e * e for e in errors) / len(errors))


def rolling_metrics(
    scores: dict[str, Sequence[ScoreRow]],
    reference: str = REFERENCE_MODEL_VERSION,
    horizons: Sequence[int] = HORIZONS_H,
) -> dict[str, dict[int, Metric]]:
    """Per model and horizon: n, MAE, RMSE; skill vs ``reference`` on matched pairs only."""
    by_model: dict[str, dict[int, dict[Key, float]]] = {}
    for model_version, rows in scores.items():
        per_h: dict[int, dict[Key, float]] = {h: {} for h in horizons}
        for r in rows:
            h = horizon_h(r.issue_time, r.valid_time)
            if h in per_h:
                key = (r.station_id, sql.require_utc(r.issue_time), sql.require_utc(r.valid_time))
                per_h[int(h)][key] = r.abs_error
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
) -> ScoreResult:
    """One scoring run. Driver connection errors propagate for `run_entrypoint` to map."""
    now = sql.require_utc(clock())
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    models = [REFERENCE_MODEL_VERSION]
    challenger = settings.challenger_model_version
    if challenger is not None and challenger != REFERENCE_MODEL_VERSION:
        models.append(challenger)

    result = ScoreResult(run_id=default_run_id(now), scored_at=now)
    scores: dict[str, list[ScoreRow]] = {}
    with open_db(connect, settings.database_url.get_secret_value()) as con:
        for model_version in models:
            raw = con.execute(
                sql.SELECT_MATURED_FORECASTS,
                {"model_version": model_version, "since": now - RESCORE_LOOKBACK, "now": now},
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
                    for h in HORIZONS_H
                }
                for m in models
            },
            "failures": result.failures,
        },
    )
    return result


def main() -> int:
    configure_logging()
    alert = make_alert(None, prefix="[pm25 score]")
    try:
        settings = get_settings()
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook, prefix="[pm25 score]")
        result, code = run_entrypoint(
            lambda: run(settings), alert, connection_errors=db.CONNECTION_ERRORS
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
    sys.exit(main())
