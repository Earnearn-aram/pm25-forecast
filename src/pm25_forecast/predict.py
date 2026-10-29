"""Hourly predict step (KAN-28). Run with ``python -m pm25_forecast.predict``.

Per run, for every station in STATION_LOCATIONS: read stored data, make baseline forecasts for
+6 h, +12 h and +24 h, and append them to ``predictions`` in one transaction.

* ``issue_time`` is the run time floored to the UTC hour, so a re-run or a doubled run in the same
  hour has the same natural key (station_id, issue_time, valid_time, model_version) and the
  insert is ``DO NOTHING``: the first vintage is kept, no duplicates.
* ``persistence-24h``: the latest observation at or before issue_time (at most 3 h old).
* ``cams``: pass-through of the collector's stored ``cams_global`` rows for the same valid_time,
  from the latest vintage issued at or before the run. There is no historical CAMS data (the
  collector started 2026-10-27), so early on some valid_times have no row: they are skipped.
* Missing inputs are skipped and logged, never filled in. They are listed in the run summary.
* DB down: no fallback to stale data. ``run_entrypoint`` skips the run, alerts, exits non-zero.
* Reads only the database: no HTTP calls, so ingest outages cannot block or fake a forecast.

Exit codes: 0 ok (skips are reported, not failures), 1 failed / DB down / nothing predicted,
2 configuration error.
"""

import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import AwareDatetime, BaseModel, ValidationError

from pm25_forecast.collector import ConfigError, default_run_id, open_db
from pm25_forecast.features import build_features
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
from pm25_forecast.models.baselines import (
    CAMS_PASSTHROUGH_MODEL_VERSION,
    HORIZONS_H,
    MAX_OBSERVATION_AGE,
    PERSISTENCE_MODEL_VERSION,
    Forecast,
    valid_times,
)
from pm25_forecast.monitoring.alerts import make_alert, safe_alert
from pm25_forecast.monitoring.logs import configure_logging, log_summary
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db, sql
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint

logger = logging.getLogger(__name__)

EXIT_FAILED, EXIT_CONFIG = 1, 2


class ObservationRow(BaseModel):
    ts: AwareDatetime
    pm25: float
    ingested_at: AwareDatetime


class ForecastRow(BaseModel):
    issue_time: AwareDatetime
    valid_time: AwareDatetime
    pm25_pred: float
    created_at: AwareDatetime


@dataclass
class PredictResult:
    run_id: str
    issue_time: datetime
    rows_built: int = 0
    rows_written: int = 0
    rows_ignored: int = 0  # already stored for this issue_time (idempotent re-run)
    per_model: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def floor_to_hour(moment: datetime) -> datetime:
    return sql.require_utc(moment).replace(minute=0, second=0, microsecond=0)


def _validated[M: BaseModel](model: type[M], rows: Sequence[Sequence[Any]], names: Sequence[str]):
    """Validate DB rows; a bad row is dropped and logged (type only), never used."""
    good: list[M] = []
    for row in rows:
        try:
            good.append(model.model_validate(dict(zip(names, row, strict=True))))
        except ValidationError:
            logger.warning("%s: dropped a malformed row", model.__name__)
    return good


def _row(
    station_id: str, issue_time: datetime, run_id: str, model_version: str, f: Forecast
) -> dict[str, Any]:
    return {
        "station_id": station_id,
        "issue_time": issue_time,
        "valid_time": f.valid_time,
        "model_version": model_version,
        "run_id": run_id,
        "pm25_pred": f.pm25_pred,
    }


def build_station_rows(
    con: Any,
    station_id: str,
    issue_time: datetime,
    now: datetime,
    run_id: str,
    result: PredictResult,
) -> list[dict[str, Any]]:
    """Read inputs for one station and build its rows; gaps are skipped, logged and reported."""
    rows: list[dict[str, Any]] = []

    obs_raw = con.execute(
        sql.SELECT_OBSERVATIONS_BETWEEN,
        {
            "station_id": station_id,
            "since": issue_time - timedelta(hours=24),
            "until": issue_time,
            "as_of": issue_time,
        },
    ).fetchall()
    obs = _validated(ObservationRow, obs_raw, ("ts", "pm25", "ingested_at"))
    targets = valid_times(issue_time)
    cams_raw = con.execute(
        sql.SELECT_FORECAST_ROWS,
        {
            "station_id": station_id,
            "model_version": CAMS_MODEL_VERSION,
            "as_of": issue_time,
            "valid_from": min(targets.values()),
            "valid_to": max(targets.values()),
        },
    ).fetchall()
    cams_rows = _validated(
        ForecastRow, cams_raw, ("issue_time", "valid_time", "pm25_pred", "created_at")
    )
    raw_observations = [(o.ts, o.pm25) for o in obs]
    raw_vintages = [
        (c.issue_time, c.valid_time, c.pm25_pred, CAMS_MODEL_VERSION) for c in cams_rows
    ]
    features = {
        horizon: build_features(raw_observations, raw_vintages, issue_time, target)
        for horizon, target in targets.items()
    }
    persistence = [
        Forecast(h, target, float(features[h]["pm25_latest"]))
        for h, target in targets.items()
        if features[h]["pm25_latest"] is not None
    ]
    if persistence:
        rows.extend(
            _row(station_id, issue_time, run_id, PERSISTENCE_MODEL_VERSION, f) for f in persistence
        )
    else:
        msg = (
            f"{station_id}: no observation within {MAX_OBSERVATION_AGE} of "
            f"{issue_time.isoformat()}, persistence skipped"
        )
        logger.warning(msg)
        result.skipped.append(msg)

    for h, target in targets.items():
        cams_pm25 = features[h]["cams_pm25"]
        if cams_pm25 is not None:
            forecast = Forecast(h, target, cams_pm25)
            rows.append(
                _row(station_id, issue_time, run_id, CAMS_PASSTHROUGH_MODEL_VERSION, forecast)
            )
            continue
        msg = f"{station_id}: no CAMS row for +{h}h ({targets[h].isoformat()}), skipped"
        logger.warning(msg)
        result.skipped.append(msg)
    return rows


def run(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> PredictResult:
    """One predict run. Driver connection errors propagate for `run_entrypoint` to map."""
    now = sql.require_utc(clock())
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    try:
        station_ids = [i for i, _lat, _lon in settings.station_location_list]
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if not station_ids:
        raise ConfigError("STATION_LOCATIONS is not set (format: ID=lat,lon;ID=lat,lon)")

    issue_time = floor_to_hour(now)
    run_id = default_run_id(now)
    result = PredictResult(run_id=run_id, issue_time=issue_time)

    with open_db(connect, settings.database_url.get_secret_value()) as con:
        rows: list[dict[str, Any]] = []
        for station_id in station_ids:
            rows += build_station_rows(con, station_id, issue_time, now, run_id, result)
        result.rows_built = len(rows)
        for model_version in (PERSISTENCE_MODEL_VERSION, CAMS_PASSTHROUGH_MODEL_VERSION):
            result.per_model[model_version] = sum(r["model_version"] == model_version for r in rows)
        result.rows_written = sql.write_rows(con, sql.INSERT_PREDICTION, rows)
        result.rows_ignored = len(rows) - result.rows_written
    if not rows:  # nothing to forecast from is an outage of the inputs, not a quiet success
        result.failures.append("no forecasts could be built for any station")

    log_summary(
        logger,
        "predict",
        {
            "run_id": run_id,
            "status": "failed" if result.failures else "ok",
            "issue_time": issue_time,
            "horizons_h": list(HORIZONS_H),
            "stations": station_ids,
            "rows_built": result.rows_built,
            "rows_written": result.rows_written,
            "rows_ignored": result.rows_ignored,
            "per_model": result.per_model,
            "skipped": result.skipped,
            "failures": result.failures,
        },
    )
    return result


def main() -> int:
    configure_logging()
    alert = make_alert(None, prefix="[pm25 predict]")
    try:
        settings = get_settings()
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook, prefix="[pm25 predict]")
        result, code = run_entrypoint(
            lambda: run(settings), alert, connection_errors=db.CONNECTION_ERRORS
        )
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        safe_alert(alert, f"configuration error: {exc}")
        log_summary(logger, "predict", {"status": "config_error"})
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.exception("predict crashed")
        safe_alert(alert, f"predict crashed: {type(exc).__name__}")
        log_summary(logger, "predict", {"status": "crashed", "error": type(exc).__name__})
        return EXIT_FAILED
    if isinstance(result, RunSkipped):  # DB down: the guard has already alerted
        log_summary(logger, "predict", {"status": "db_unreachable"})
        return code
    if result.failures:
        safe_alert(alert, "run completed with failures: " + "; ".join(result.failures))
        return EXIT_FAILED
    return code


if __name__ == "__main__":
    sys.exit(main())
