"""Weekly retrain with a champion/challenger comparison (KAN-23).

Run with ``python -m pm25_forecast.retrain``. One run:

1. Export the latest hosted history to DuckDB (``export_history``).
2. Train a candidate with ``train_model.train`` on a copy of the export cut at ``holdout_start``
   (now floored to the hour minus ``--holdout-days``, default 7). Nothing at or after
   ``holdout_start`` reaches training, so the held-out window is unseen by the candidate.
   Version ``retrain-YYYYMMDD`` (UTC date), ``-2``, ``-3`` ... for a same-day re-run.
3. Register the candidate with ``publish_model`` (always; registering changes nothing live).
4. Score the candidate, the current champion (if any), the current challenger
   (``CHALLENGER_MODEL_VERSION``, if set) and persistence-24h on the held-out window with the
   shared feature code, per horizon the bundle serves (+6/+12/+24 h), on matched
   (station, issue_time, valid_time) pairs only: MAE and RMSE.
5. ``decide`` (the single decision function) returns promote or keep.
6. Production changes only when ``AUTO_PROMOTE=true``. Otherwise a promote decision is logged and
   alerted as "recommend promote <version>" and the champion pointer is left alone.
7. Every outcome is written to ``model_events`` (KAN-24): ``promote`` (actor ``auto``, in the same
   transaction as the pointer change), ``recommend`` or ``keep`` (with the comparison metrics).

``--if-drift`` (KAN-24) makes the run conditional: it exits 0 without training unless the latest
drift run has an alert for the challenger (``CHALLENGER_MODEL_VERSION``) or the production
champion, and no retrain decision was logged since that drift run was computed (cooldown). A
triggered run is the normal run (still recommend-only unless AUTO_PROMOTE).

The RUN_SUMMARY line holds the whole comparison table and the decision. DB down: the run is
skipped, alerted, exit 1. Configuration error: exit 2. Failed run (e.g. not enough data): exit 1.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
from pydantic import ValidationError

from pm25_forecast.collector import ConfigError, default_run_id, open_db
from pm25_forecast.drift import latest_drift_alert
from pm25_forecast.export_history import export_history
from pm25_forecast.features import FEATURE_NAMES
from pm25_forecast.models.baselines import (
    HEADLINE_HORIZON_H,
    HORIZONS_H,
    PERSISTENCE_MODEL_VERSION,
    require_utc,
)
from pm25_forecast.models.serving import (
    ModelLoadError,
    ServingModel,
    load_serving_model,
)
from pm25_forecast.monitoring.alerts import make_alert, safe_alert
from pm25_forecast.monitoring.logs import configure_logging, log_summary
from pm25_forecast.settings import Settings, get_settings
from pm25_forecast.storage import db
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint
from pm25_forecast.storage.registry import (
    SELECT_CHAMPION,
    last_retrain_event_at,
    promote_model,
    publish_model,
    read_champion,
    read_model,
    record_event,
)
from pm25_forecast.train_model import build_samples, train

logger = logging.getLogger(__name__)

EXIT_FAILED, EXIT_CONFIG = 1, 2
HOLDOUT_DAYS = 7
MIN_PAIRS = 100
RETRAIN_MIN_GAP = timedelta(days=7)  # --if-drift: at most one retrain decision per this period
MIN_REL_GAIN = 0.02
_LATEST_INDEX = FEATURE_NAMES.index("pm25_latest")  # persistence = the latest observation


# --- decision (KAN-24 builds on this one function) --------------------------------------------


@dataclass(frozen=True)
class Evaluation:
    """One model on the matched held-out pairs at one horizon.

    ``persistence_mae`` is persistence-24h's MAE on the same pairs (None if unavailable).
    """

    version: str
    n: int
    mae: float
    rmse: float
    persistence_mae: float | None = None


@dataclass(frozen=True)
class Decision:
    action: str  # "promote" | "keep"
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


def decide(
    candidate: Evaluation,
    incumbent: Evaluation | None,
    min_pairs: int = MIN_PAIRS,
    min_rel_gain: float = MIN_REL_GAIN,
    *,
    other_horizons: Sequence[tuple[int, Evaluation, Evaluation]] = (),
) -> Decision:
    """Promote only if the candidate beats the incumbent's MAE by at least ``min_rel_gain``
    (relative) on at least ``min_pairs`` matched pairs AND beats persistence. Otherwise keep.

    ``incumbent`` is the current champion on the same pairs; None means there is no champion, so
    only the pairs and persistence checks apply. Boundaries: ``n == min_pairs`` passes, a relative
    gain of exactly ``min_rel_gain`` passes, equal to persistence does not beat it.

    ``other_horizons`` holds ``(horizon_h, candidate, incumbent)`` for the other horizons both
    models serve. A candidate worse than the incumbent by more than ``min_rel_gain`` (relative
    MAE) at any of them is kept out, even if +24 h wins.
    """
    gain = None
    if incumbent is not None and incumbent.mae > 0 and math.isfinite(incumbent.mae):
        gain = 1 - candidate.mae / incumbent.mae
    metrics = {
        "horizon_h": HEADLINE_HORIZON_H,
        "candidate": candidate.version,
        "incumbent": None if incumbent is None else incumbent.version,
        "n_pairs": candidate.n,
        "candidate_mae": candidate.mae,
        "incumbent_mae": None if incumbent is None else incumbent.mae,
        "persistence_mae": candidate.persistence_mae,
        "rel_gain_vs_incumbent": gain,
        "min_pairs": min_pairs,
        "min_rel_gain": min_rel_gain,
        "other_horizons": [
            {"horizon_h": h, "candidate_mae": c.mae, "incumbent_mae": i.mae, "n_pairs": c.n}
            for h, c, i in other_horizons
        ],
    }

    def keep(reason: str) -> Decision:
        return Decision("keep", reason, metrics)

    if candidate.n < min_pairs:
        return keep(f"only {candidate.n} matched pairs, need at least {min_pairs}")
    if not math.isfinite(candidate.mae):
        return keep("candidate MAE is not finite")
    if candidate.persistence_mae is None or not math.isfinite(candidate.persistence_mae):
        return keep("no persistence-24h reference on the matched pairs")
    if candidate.mae >= candidate.persistence_mae:
        return keep(
            f"candidate MAE {candidate.mae:.3f} does not beat persistence "
            f"{candidate.persistence_mae:.3f}"
        )
    if incumbent is None:
        return Decision(
            "promote",
            f"no champion; candidate beats persistence on {candidate.n} pairs",
            metrics,
        )
    if incumbent.n < min_pairs:
        return keep(f"incumbent has only {incumbent.n} matched pairs, need at least {min_pairs}")
    for horizon, cand, inc in other_horizons:
        if inc.mae > 0 and cand.mae / inc.mae - 1 > min_rel_gain + 1e-9:
            return keep(
                f"candidate is worse than champion {inc.version} at +{horizon}h "
                f"(MAE {cand.mae:.3f} vs {inc.mae:.3f}, more than {min_rel_gain:.0%})"
            )
    if gain is None:
        return keep("incumbent MAE is zero or not finite, no relative gain can be computed")
    if gain < min_rel_gain - 1e-9:  # tolerance: 1 - 0.98 is a hair under 0.02 in floating point
        return keep(
            f"relative MAE gain {gain:.2%} vs champion {incumbent.version} is below "
            f"{min_rel_gain:.0%}"
        )
    return Decision(
        "promote",
        f"relative MAE gain {gain:.2%} vs champion {incumbent.version} on {candidate.n} pairs "
        "and beats persistence",
        metrics,
    )


# --- held-out comparison ----------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """Matched pairs at one horizon and ``{model: (mae, rmse)}`` on exactly those pairs."""

    n: int
    metrics: dict[str, tuple[float, float]]


def _mae_rmse(pairs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    errors = [pred - actual for pred, actual in pairs]
    return (
        sum(abs(e) for e in errors) / len(errors),
        math.sqrt(sum(e * e for e in errors) / len(errors)),
    )


def compare(
    samples: Mapping[int, Sequence[tuple]],
    models: Mapping[str, ServingModel],
    holdout_start: datetime,
    horizons: Sequence[int] = HORIZONS_H,
) -> dict[int, Comparison]:
    """Score every model plus persistence on held-out samples (issue_time >= holdout_start).

    Only pairs scored by persistence and by every model that serves the horizon are kept, so all
    models are compared on identical (station, issue_time, valid_time) pairs. A model that does
    not serve a horizon is left out of that horizon.
    """
    result: dict[int, Comparison] = {}
    for horizon in horizons:
        rows = [s for s in samples.get(horizon, ()) if s[2] >= holdout_start]
        serving = {v: m for v, m in models.items() if horizon in m.boosters}
        truth: dict[tuple, float] = {}
        persistence: dict[tuple, float] = {}
        preds: dict[str, dict[tuple, float]] = {v: {} for v in serving}
        for vector, target, issue, _available, station in rows:
            key = (station, issue)  # valid_time is issue + horizon, fixed per horizon
            truth[key] = target
            if math.isfinite(vector[_LATEST_INDEX]):
                persistence[key] = vector[_LATEST_INDEX]
            features = {
                name: None if math.isnan(value) else value
                for name, value in zip(FEATURE_NAMES, vector, strict=True)
            }
            for version, model in serving.items():
                value = model.predict(horizon, features)
                if value is not None:
                    preds[version][key] = value
        common = set(truth) & set(persistence)
        for by_key in preds.values():
            common &= set(by_key)
        if not common or not serving:
            result[horizon] = Comparison(0, {})
            continue
        metrics = {
            version: _mae_rmse([(by_key[k], truth[k]) for k in common])
            for version, by_key in preds.items()
        }
        metrics[PERSISTENCE_MODEL_VERSION] = _mae_rmse([(persistence[k], truth[k]) for k in common])
        result[horizon] = Comparison(len(common), metrics)
    return result


def evaluation(comparison: Comparison, version: str) -> Evaluation | None:
    """The ``Evaluation`` of ``version`` on a horizon's matched pairs, or None if not scored."""
    if version not in comparison.metrics:
        return None
    mae, rmse = comparison.metrics[version]
    persistence = comparison.metrics.get(PERSISTENCE_MODEL_VERSION)
    reference = None if persistence is None else persistence[0]
    return Evaluation(version, comparison.n, mae, rmse, reference)


def comparison_table(comparisons: Mapping[int, Comparison]) -> dict[str, Any]:
    return {
        f"+{h}h": {
            "n_pairs": c.n,
            "models": {
                v: {"mae": round(mae, 3), "rmse": round(rmse, 3)}
                for v, (mae, rmse) in c.metrics.items()
            },
        }
        for h, c in comparisons.items()
    }


# --- plumbing ---------------------------------------------------------------------------------


def make_training_export(full: Path, destination: Path, cutoff: datetime) -> None:
    """Copy the export keeping only rows known before ``cutoff`` (the training window).

    A row must be dated before the cutoff AND have arrived before it (``available_at``), so a
    label or forecast vintage ingested late, after the cutoff, cannot leak into training.
    """
    source = str(full).replace("'", "''")
    with duckdb.connect(str(destination)) as con:
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"ATTACH '{source}' AS src (READ_ONLY)")
        con.execute(
            "CREATE TABLE observations AS "
            "SELECT * FROM src.observations WHERE ts < ? AND available_at < ?",
            [cutoff, cutoff],
        )
        con.execute(
            "CREATE TABLE forecast_vintages AS "
            "SELECT * FROM src.forecast_vintages WHERE issue_time < ? AND available_at < ?",
            [cutoff, cutoff],
        )
        con.execute("DETACH src")


def retrain_version(now: datetime, is_taken: Callable[[str], bool]) -> str:
    """``retrain-YYYYMMDD`` (UTC), then ``-2``, ``-3`` ... for a same-day re-run."""
    base = f"retrain-{require_utc(now):%Y%m%d}"
    if not is_taken(base):
        return base
    n = 2
    while is_taken(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def _registered(con: Any, version: str) -> bool:
    try:
        read_model(con, version)
    except LookupError:
        return False
    return True


def drift_targets(con: Any, settings: Settings) -> list[str]:
    """The models whose drift can trigger an ``--if-drift`` run: the challenger and the champion."""
    targets: list[str] = []
    if settings.challenger_model_version:
        targets.append(settings.challenger_model_version)
    row = con.execute(SELECT_CHAMPION, {}).fetchone()
    if row is not None and row[0] not in targets:
        targets.append(row[0])
    return targets


def drift_gate(con: Any, settings: Settings, now: datetime) -> tuple[list[str], str | None]:
    """Decide an ``--if-drift`` run: ``(alerted models, skip reason or None)``.

    Runs only if the latest drift run has an alert for the challenger OR the champion, and no
    retrain decision was logged within ``RETRAIN_MIN_GAP`` (7 days; cooldown). Persistent drift,
    a "keep" result, a same-day re-run or AUTO_PROMOTE off therefore cannot retrain every day.
    The last decision's ``at`` is the database clock and ``now`` the (UTC) job clock; both are
    timezone-aware UTC, so they differ only by clock skew, which is negligible against 7 days.
    """
    targets = drift_targets(con, settings)
    if not targets:
        return [], "no challenger or champion to check"
    alerted = [t for t in targets if latest_drift_alert(con, t, now=now)]
    if not alerted:
        return [], "no drift alert"
    last_retrain = last_retrain_event_at(con)
    if last_retrain is not None and now - last_retrain < RETRAIN_MIN_GAP:
        return alerted, (
            f"cooldown: a retrain was logged at {last_retrain.isoformat()}, less than "
            f"{RETRAIN_MIN_GAP.days} days ago"
        )
    return alerted, None


def _flag_overlap(
    result: RetrainResult, role: str, version: str, manifest: Mapping[str, Any], start: datetime
) -> None:
    """Warn when a model's training data can reach into the held-out window (optimistic score)."""
    window = manifest.get("training_window")
    try:
        end = require_utc(datetime.fromisoformat(window["end"]))
    except (KeyError, TypeError, ValueError):
        result.warnings.append(f"{role} {version}: training window unknown, overlap not ruled out")
        return
    if end + timedelta(hours=max(HORIZONS_H)) >= start:
        result.warnings.append(
            f"{role} {version} overlaps holdout (trained up to {end.isoformat()}); "
            "its scores may be optimistic"
        )


@dataclass
class RetrainResult:
    run_id: str
    version: str = ""
    status: str = "ok"
    decision: Decision | None = None
    comparison: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, Any] = field(default_factory=dict)
    challenger_comparison: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    artifact_bytes: int = 0
    promoted: bool = False
    recommended: bool = False
    failures: list[str] = field(default_factory=list)


def _summary(result: RetrainResult, settings: Settings, **extra: Any) -> None:
    decision = result.decision
    log_summary(
        logger,
        "retrain",
        {
            "run_id": result.run_id,
            "status": result.status,
            "version": result.version,
            "auto_promote": settings.auto_promote,
            "roles": result.roles,
            "comparison": result.comparison,
            "challenger_comparison": result.challenger_comparison,
            "candidate_artifact_bytes": result.artifact_bytes,
            "warnings": result.warnings,
            "decision": None
            if decision is None
            else {"action": decision.action, "reason": decision.reason, **decision.metrics},
            "promoted": result.promoted,
            "recommend_promote": result.recommended,
            "failures": result.failures,
            **extra,
        },
    )


def run(
    settings: Settings,
    *,
    connect: Callable[[str], Any] = db.connect,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    export: Callable[[Path, str], Any] | None = None,
    trainer: Callable[..., tuple[bytes, dict[str, Any]]] = train,
    holdout_days: int = HOLDOUT_DAYS,
    alert: Callable[[str], None] | None = None,
    workdir: Path | None = None,
    if_drift: bool = False,
) -> RetrainResult:
    """One retrain run. Driver connection errors propagate for `run_entrypoint` to map."""
    now = require_utc(clock())
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set")
    if holdout_days < 1:
        raise ConfigError("--holdout-days must be at least 1")
    url = settings.database_url.get_secret_value()
    if if_drift:
        with open_db(connect, url) as con:
            alerted, skip_reason = drift_gate(con, settings, now)
        if skip_reason is not None:
            skipped = RetrainResult(run_id=default_run_id(now), status="skipped")
            _summary(
                skipped, settings, if_drift=True, drift_alerted=alerted, skip_reason=skip_reason
            )
            return skipped
    do_export = export or (lambda path, u: export_history(path, u, connect=connect))

    holdout_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(days=holdout_days)
    result = RetrainResult(run_id=default_run_id(now))
    window = {"holdout_start": holdout_start.isoformat(), "holdout_days": holdout_days}

    with open_db(connect, url) as con:
        result.version = retrain_version(now, lambda v: _registered(con, v))

    with tempfile.TemporaryDirectory() as tmp:
        base = workdir or Path(tmp)
        base.mkdir(parents=True, exist_ok=True)
        full, train_copy = base / "retrain-history.duckdb", base / "retrain-train.duckdb"
        do_export(full, url)
        samples = build_samples(full)[0]
        make_training_export(full, train_copy, holdout_start)
        try:
            artifact, manifest = trainer(train_copy, result.version, as_of=holdout_start)
        except ValueError as exc:  # too little data before the held-out window
            result.status = "failed"
            result.failures.append(f"training rejected: {exc}")
            _summary(result, settings, **window)
            return result

    manifest = {**manifest, **window, "source": "retrain"}
    result.artifact_bytes = len(artifact)
    publish_model(url, result.version, artifact, manifest, connect=connect)
    candidate = load_serving_model(result.version, artifact, manifest)

    champion_version: str | None = None
    champion: ServingModel | None = None
    champion_error: str | None = None
    challenger_version = settings.challenger_model_version
    challenger: ServingModel | None = None
    challenger_error: str | None = None
    with open_db(connect, url) as con:
        try:
            champion_version, champion_artifact, champion_manifest = read_champion(con)
        except LookupError:
            champion_version = None  # no champion yet
        if champion_version is not None:
            try:
                champion = load_serving_model(
                    champion_version, champion_artifact, champion_manifest
                )
                _flag_overlap(
                    result, "champion", champion_version, champion_manifest, holdout_start
                )
            except ModelLoadError as exc:
                champion_error = f"champion {champion_version} cannot be loaded: {exc}"
        already_scored = {PERSISTENCE_MODEL_VERSION, champion_version}
        if challenger_version and challenger_version not in already_scored:
            try:
                challenger_artifact, challenger_manifest = read_model(con, challenger_version)
                challenger = load_serving_model(
                    challenger_version, challenger_artifact, challenger_manifest
                )
                _flag_overlap(
                    result, "challenger", challenger_version, challenger_manifest, holdout_start
                )
            except (LookupError, ValueError) as exc:  # ModelLoadError is a ValueError
                challenger_error = f"challenger {challenger_version} skipped: {exc}"
    result.roles = {
        "candidate": result.version,
        "champion": champion_version,
        "challenger": challenger_version,
        **({"challenger_error": challenger_error} if challenger_error else {}),
        **window,
    }

    # The decision set is candidate + champion + persistence pairs only. The challenger is
    # reported on its own pairs and can never shrink or change that set.
    decision_models = {result.version: candidate}
    if champion is not None and champion_version is not None:
        decision_models[champion_version] = champion
    comparisons = compare(samples, decision_models, holdout_start)
    result.comparison = comparison_table(comparisons)
    if challenger is not None and challenger_version is not None:
        pair = {result.version: candidate, challenger_version: challenger}
        result.challenger_comparison = comparison_table(compare(samples, pair, holdout_start))

    headline = comparisons.get(HEADLINE_HORIZON_H, Comparison(0, {}))
    candidate_eval = evaluation(headline, result.version)
    if candidate_eval is None:
        candidate_eval = Evaluation(result.version, 0, math.nan, math.nan)
    incumbent = None if champion_version is None else evaluation(headline, champion_version)
    if champion_version is not None and incumbent is None:
        # A champion exists but cannot be scored: never promote over it.
        reason = champion_error or (
            f"champion {champion_version} could not be scored at +{HEADLINE_HORIZON_H}h "
            "on the held-out pairs"
        )
        result.decision = Decision("keep", reason, {"candidate": result.version})
        result.failures.append(reason)
    else:
        others = []
        for horizon, comparison in comparisons.items():
            cand_h = evaluation(comparison, result.version)
            inc_h = None if champion_version is None else evaluation(comparison, champion_version)
            if horizon != HEADLINE_HORIZON_H and cand_h is not None and inc_h is not None:
                others.append((horizon, cand_h, inc_h))
        result.decision = decide(candidate_eval, incumbent, other_horizons=others)

    note = f" Warning: {'; '.join(result.warnings)}." if result.warnings else ""
    event_metrics = {**result.decision.metrics, "comparison": result.comparison}
    if result.decision.action == "promote":
        if settings.auto_promote:
            with open_db(connect, url) as con:
                # One transaction: the pointer change and its model_events row commit together.
                promote_model(
                    con,
                    result.version,
                    actor="auto",
                    reason=result.decision.reason,
                    metrics=event_metrics,
                )
                promoted_version, promoted_artifact, _ = read_champion(con)
            if promoted_version != result.version or promoted_artifact != artifact:
                raise ValueError("production champion readback does not match promoted model")
            result.promoted = True
            safe_alert(
                alert, f"AUTO_PROMOTE: {result.version} is now the production champion.{note}"
            )
        else:
            result.recommended = True
            with open_db(connect, url) as con:
                record_event(
                    con,
                    "recommend",
                    result.version,
                    previous_version=champion_version,
                    actor="retrain",
                    metrics=event_metrics,
                    reason=result.decision.reason,
                )
            safe_alert(
                alert,
                f"recommend promote {result.version}: {result.decision.reason}. Not applied "
                "(AUTO_PROMOTE is off). Run: python -m pm25_forecast.registry_cli promote "
                f"{result.version} --confirm.{note}",
            )
    else:
        with open_db(connect, url) as con:
            record_event(
                con,
                "keep",
                result.version,
                previous_version=champion_version,
                actor="retrain",
                metrics=event_metrics,
                reason=result.decision.reason,
            )
    if result.failures:
        result.status = "failed"
    _summary(result, settings)
    return result


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    alert = make_alert(None, prefix="[pm25 retrain]")
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
        parser.add_argument(
            "--if-drift",
            action="store_true",
            help="exit 0 without training unless the latest drift run has an alert",
        )
        args = parser.parse_args(argv)
        try:
            settings = get_settings()
        except ValidationError as exc:  # report field names only, never values
            fields = ", ".join(sorted({".".join(map(str, e["loc"])) for e in exc.errors()}))
            raise ConfigError(f"invalid settings: {fields}") from None
        webhook = (
            settings.alert_webhook_url.get_secret_value() if settings.alert_webhook_url else None
        )
        alert = make_alert(webhook, prefix="[pm25 retrain]")
        result, code = run_entrypoint(
            lambda: run(
                settings, holdout_days=args.holdout_days, alert=alert, if_drift=args.if_drift
            ),
            alert,
            connection_errors=db.CONNECTION_ERRORS,
        )
    except ConfigError as exc:  # messages never contain secret values
        logger.error("configuration error: %s", exc)
        safe_alert(alert, f"configuration error: {exc}")
        log_summary(logger, "retrain", {"status": "config_error"})
        return EXIT_CONFIG
    except Exception as exc:  # a crash must be red and alerted, with the type only
        logger.error("retrain crashed (%s)", type(exc).__name__)
        safe_alert(alert, f"retrain crashed: {type(exc).__name__}")
        log_summary(logger, "retrain", {"status": "crashed", "error": type(exc).__name__})
        return EXIT_FAILED
    if isinstance(result, RunSkipped):  # DB down: the guard has already alerted
        log_summary(logger, "retrain", {"status": "db_unreachable"})
        return code
    if result.failures:
        safe_alert(alert, "run completed with failures: " + "; ".join(result.failures))
        return EXIT_FAILED
    return code


if __name__ == "__main__":
    sys.exit(main())
