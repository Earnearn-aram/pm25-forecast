"""Hourly predict step (KAN-28). Run with ``python -m pm25_forecast.predict``.

Per run, for every station in STATION_LOCATIONS: read stored data, make the persistence baseline
for every hour +1..+24 (KAN-104; the headline horizons stay +6/+12/+24 h) and append it to
``predictions`` in one transaction.

* ``issue_time`` is the run time floored to the UTC hour, so a re-run or a doubled run in the same
  hour has the same natural key (station_id, issue_time, valid_time, model_version) and the
  insert is ``DO NOTHING``: the first vintage is kept, no duplicates.
* ``persistence-24h``: the latest observation at or before issue_time (at most 3 h old).
* No ``cams`` pass-through rows are written any more (KAN-69): nothing read them and KAN-20 skips
  them. Old ``cams`` rows stay and expire under the 90-day retention. The stored ``cams_global``
  vintages are still read, only to build the ``cams_pm25`` feature for the challenger.
* Missing inputs are skipped and logged, never filled in. They are listed in the run summary.
* Challengers (KAN-29, shadow mode; KAN-76: LightGBM, smoother or, KAN-104, ridge bundle): when
  ``CHALLENGER_MODEL_VERSION`` is set (KAN-104: a comma-separated list of up to 3 versions), each
  registered model is read from the registry once per run (content-verified, KAN-16) and its
  predictions are stored next to the baselines under its own model_version, for the horizons the
  model itself serves. Nothing reads them for decisions; KAN-20 scores them.
* Failure policy (KAN-104): every model is loaded and built on its own. A missing model, a bad
  bundle, a feature-spec mismatch or a prediction error drops that model's rows for this run only,
  is listed in ``failures`` and fails the run loudly (alert, exit 1), but the baselines and the
  other challengers are still written in the same transaction.
* History: observations are read back as far as the loaded models need (``model.lookback``: 24 h
  for LightGBM/smoother, 80 h for the ridge), never less than 24 h.
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
from pm25_forecast.features import MEAN_24H_NAME, build_features, mean_24h
from pm25_forecast.ingest.open_meteo import CAMS_MODEL_VERSION
from pm25_forecast.models.baselines import (
    CAMS_PASSTHROUGH_MODEL_VERSION,
    HORIZONS_H,
    HOURLY_HORIZONS_H,
    MAX_OBSERVATION_AGE,
    PERSISTENCE_MODEL_VERSION,
    Forecast,
    valid_times,
)
from pm25_forecast.models.serving import (
    DEFAULT_LOOKBACK,
    ModelLoadError,
    RidgeModel,
    ServingModel,
    SmootherModel,
    load_registered_model,
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


PredictModel = ServingModel | SmootherModel | RidgeModel


@dataclass
class StationInputs:
    """What one station's models are fed: the validated observations and the shared features."""

    observations: list[ObservationRow]
    features: dict[int, dict[str, float | None]]  # horizon -> shared feature mapping


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
    inputs_out: dict[str, StationInputs] | None = None,
    *,
    lookback: timedelta = DEFAULT_LOOKBACK,
    feature_horizons: Sequence[int] = HORIZONS_H,
) -> list[dict[str, Any]]:
    """Read inputs for one station and build its baseline rows; gaps are skipped and reported.

    ``lookback`` is how much observation history to read (the longest need of the loaded models)
    and ``feature_horizons`` the horizons the shared features are built for. Both are stored in
    ``inputs_out[station_id]`` when given, so every model reuses exactly the inputs the baseline
    saw (one read and one feature build per station). Persistence is written for +1..+24 h.
    """
    rows: list[dict[str, Any]] = []

    obs_raw = con.execute(
        sql.SELECT_OBSERVATIONS_BETWEEN,
        {
            "station_id": station_id,
            "since": issue_time - max(lookback, DEFAULT_LOOKBACK),
            "until": issue_time,
            "as_of": issue_time,
        },
    ).fetchall()
    obs = _validated(ObservationRow, obs_raw, ("ts", "pm25", "ingested_at"))
    targets = valid_times(issue_time, sorted({*HORIZONS_H, *feature_horizons}))
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
    mean24 = mean_24h(raw_observations, issue_time)  # smoother input (KAN-76), same for all h
    features = {
        horizon: {
            **build_features(raw_observations, raw_vintages, issue_time, target),
            MEAN_24H_NAME: mean24,
        }
        for horizon, target in targets.items()
    }
    if inputs_out is not None:
        inputs_out[station_id] = StationInputs(obs, features)
    latest = features[min(features)]["pm25_latest"]  # the same for every target hour
    if latest is not None:
        rows.extend(
            _row(station_id, issue_time, run_id, PERSISTENCE_MODEL_VERSION, Forecast(h, vt, latest))
            for h, vt in valid_times(issue_time, HOURLY_HORIZONS_H).items()
        )
    else:
        msg = (
            f"{station_id}: no observation within {MAX_OBSERVATION_AGE} of "
            f"{issue_time.isoformat()}, persistence skipped"
        )
        logger.warning(msg)
        result.skipped.append(msg)
    return rows


def _hours_label(hours: Sequence[int]) -> str:
    """``[6, 12]`` -> ``+6,12h``; runs of consecutive hours collapse: ``+1-24h``."""
    runs: list[list[int]] = []
    for h in sorted(hours):
        if runs and h == runs[-1][-1] + 1:
            runs[-1].append(h)
        else:
            runs.append([h])
    parts = [",".join(map(str, r)) if len(r) < 3 else f"{r[0]}-{r[-1]}" for r in runs]
    return "+" + ",".join(parts) + "h"


def build_challenger_rows(
    challenger: PredictModel,
    station_inputs: dict[str, StationInputs],
    issue_time: datetime,
    run_id: str,
    result: PredictResult,
    *,
    role: str = "challenger",
) -> list[dict[str, Any]]:
    """All-or-nothing rows for one model: any error drops all of its rows and fails the run loudly.

    Baseline rows and other models' rows are built separately and are unaffected. The model's own
    horizons are used (a ridge serves +1..+24 h); ``valid_time`` is ``issue_time + h``. Missing
    inputs are skipped, summarised in one line per station.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    try:
        for station_id, inputs in station_inputs.items():
            if isinstance(challenger, RidgeModel):
                preds = challenger.forecast(inputs.observations, station_id, issue_time)
            else:
                preds = {h: challenger.predict(h, inputs.features[h]) for h in challenger.horizons}
            for h, pred in preds.items():
                if pred is not None:
                    forecast = Forecast(h, issue_time + timedelta(hours=h), pred)
                    rows.append(
                        _row(station_id, issue_time, run_id, challenger.model_version, forecast)
                    )
            missing = [h for h, pred in preds.items() if pred is None]
            if missing:
                skipped.append(
                    f"{station_id}: no inputs for {challenger.model_version} "
                    f"{_hours_label(missing)}, skipped"
                )
    except Exception as exc:  # a challenger must never cost the baselines
        msg = f"{role} {challenger.model_version} dropped for this run: {exc!r}"
        logger.error(msg)
        result.failures.append(msg)
        return []
    for msg in skipped:
        logger.warning(msg)
    result.skipped.extend(skipped)
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

    challenger_versions = settings.challenger_model_versions
    if any(
        v in (PERSISTENCE_MODEL_VERSION, CAMS_PASSTHROUGH_MODEL_VERSION)
        for v in challenger_versions
    ):
        raise ConfigError("CHALLENGER_MODEL_VERSION must differ from the baseline model versions")

    with open_db(connect, settings.database_url.get_secret_value()) as con:
        decision_insert = con.execute(
            sql.INSERT_SERVING_DECISION,
            {"issue_time": issue_time, "run_id": run_id},
        )
        decision = con.execute(sql.SELECT_SERVING_DECISION, {"issue_time": issue_time}).fetchone()
        if decision is None:
            raise RuntimeError("serving decision was not recorded")
        champion_version = decision[0]
        champion = None
        if champion_version in (
            PERSISTENCE_MODEL_VERSION,
            CAMS_PASSTHROUGH_MODEL_VERSION,
            CAMS_MODEL_VERSION,
        ):
            result.failures.append(f"invalid champion version: {champion_version}")
        elif champion_version:
            try:
                champion = load_registered_model(con, champion_version)
            except (ModelLoadError, ValueError, TypeError, KeyError, UnicodeError) as exc:
                result.failures.append(f"champion {champion_version} not served: {exc}")
        shadow: list[PredictModel] = []
        for challenger_version in challenger_versions:
            if challenger_version == champion_version and champion is not None:
                continue  # already built as the champion; one set of rows per model
            try:
                shadow.append(load_registered_model(con, challenger_version))
            except (ModelLoadError, ValueError, TypeError, KeyError, UnicodeError) as exc:
                msg = f"challenger {challenger_version} not served: {exc}"
                logger.error(msg)
                result.failures.append(msg)
        loaded = [m for m in (champion, *shadow) if m is not None]
        lookback = max(
            [DEFAULT_LOOKBACK, *(getattr(m, "lookback", DEFAULT_LOOKBACK) for m in loaded)]
        )
        feature_horizons = sorted(
            {*HORIZONS_H, *(h for m in loaded if not isinstance(m, RidgeModel) for h in m.horizons)}
        )
        rows: list[dict[str, Any]] = []
        station_inputs: dict[str, StationInputs] = {}
        for station_id in station_ids:
            rows += build_station_rows(
                con,
                station_id,
                issue_time,
                now,
                run_id,
                result,
                station_inputs,
                lookback=lookback,
                feature_horizons=feature_horizons,
            )
        baseline_keys = {(r["station_id"], r["valid_time"]) for r in rows}
        champion_rows = []
        if champion is not None:
            champion_rows = build_challenger_rows(
                champion, station_inputs, issue_time, run_id, result, role="champion"
            )
            champion_rows = [
                r for r in champion_rows if (r["station_id"], r["valid_time"]) in baseline_keys
            ]
            rows += champion_rows
        for challenger in shadow:  # each model on its own: one failure cannot cost the others
            rows += build_challenger_rows(challenger, station_inputs, issue_time, run_id, result)
        result.rows_built = len(rows)
        result.rows_written = sql.write_rows(con, sql.INSERT_PREDICTION, rows)
        result.rows_ignored = len(rows) - result.rows_written
        champion_keys = {(r["station_id"], r["valid_time"]) for r in champion_rows}
        serving = []
        for station_id in station_ids:
            for target in valid_times(issue_time).values():
                key = (station_id, target)
                actual = (
                    champion_version
                    if key in champion_keys
                    else (PERSISTENCE_MODEL_VERSION if key in baseline_keys else None)
                )
                if actual is None:
                    continue  # A later run may recover this station/hour.
                stored = con.execute(
                    sql.SELECT_PREDICTION_RUN,
                    {
                        "station_id": station_id,
                        "issue_time": issue_time,
                        "valid_time": target,
                        "model_version": actual,
                    },
                ).fetchone()
                if stored is None:
                    raise RuntimeError("served forecast is missing from predictions")
                serving.append(
                    {
                        "station_id": station_id,
                        "issue_time": issue_time,
                        "valid_time": target,
                        "model_version": actual,
                        "reason": "champion"
                        if key in champion_keys
                        else ("baseline_fallback" if actual else "no_forecast"),
                        "run_id": stored[0],
                    }
                )
        sql.write_rows(con, sql.INSERT_SERVING_FORECAST, serving)
        served = [PERSISTENCE_MODEL_VERSION]
        if champion is not None:
            served.append(champion.model_version)
        served.extend(m.model_version for m in shadow)
        for model_version in served:
            result.per_model[model_version] = sum(r["model_version"] == model_version for r in rows)
        if not serving and decision_insert.rowcount == 1:
            con.execute(
                sql.DELETE_EMPTY_SERVING_DECISION, {"issue_time": issue_time, "run_id": run_id}
            )
    if not serving:  # Shadow-only outputs do not count as a served forecast.
        result.failures.append("no forecasts could be built for any station")

    log_summary(
        logger,
        "predict",
        {
            "run_id": run_id,
            "status": "failed" if result.failures else "ok",
            "issue_time": issue_time,
            "horizons_h": list(HORIZONS_H),
            "persistence_horizons_h": list(HOURLY_HORIZONS_H),
            "challengers": list(challenger_versions),
            "history_hours": int(lookback / timedelta(hours=1)),
            "stations": station_ids,
            "rows_built": result.rows_built,
            "rows_written": result.rows_written,
            "rows_ignored": result.rows_ignored,
            "per_model": result.per_model,
            "selected_champion": champion_version,
            "skipped": result.skipped,
            "failures": result.failures,
        },
    )
    return result


def main() -> int:
    configure_logging()
    alert = make_alert(None, prefix="[pm25 predict]")
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
