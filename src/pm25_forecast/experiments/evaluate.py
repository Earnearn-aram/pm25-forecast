"""Walk-forward evaluation runner (KAN-82, E04; docs/experiments/protocol.md sections 4-6, 11).

``python -m pm25_forecast.experiments.evaluate --prereg <yaml> --snapshot <dir> \
--model <module:factory>``

A model factory is ``fit(train: TrainSnapshot) -> predictor`` (optionally ``fit(train, params)``
to receive the frozen pre-registration ``params``). The predictor follows the
``backtest.Predictor`` contract: ``predictor(snapshot, horizon) -> float | None`` where
``snapshot`` holds only observations/vintages with ``ts``/``issue_time <= origin - 1 h`` and
``available_at <= origin``. It is refitted every week on data that had matured by then.

Persistence, seasonal-naive and the smoother are always computed on the same rows. The model
under test, persistence and smoother are scored on their common intersection of (station,
origin, horizon); seasonal-naive is reported on the part of those rows where it exists.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import math
import sys
import tempfile
import time
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from pm25_forecast.backtest import Observation, Predictor, Snapshot, Vintage
from pm25_forecast.experiments import folds as folds_mod
from pm25_forecast.experiments import stats, tracking
from pm25_forecast.experiments.dataset import load_snapshot
from pm25_forecast.experiments.prereg import Prereg, load_prereg
from pm25_forecast.features import mean_24h
from pm25_forecast.models.baselines import HORIZONS_H, latest_observation
from pm25_forecast.models.smoother import SMOOTHER_BETA

logger = logging.getLogger(__name__)

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
SPIKE_THRESHOLD = 37.5  # ug/m3, strictly above (protocol section 5)
MIN_SPIKE_PAIRS = 50
MIN_COVERAGE = 0.95
MIN_CONSIDERED_COVERAGE = 0.80  # station cohort: hourly coverage in the dev period
DEFAULT_TICKET = "KAN-82"
DEFAULT_AGENT = "claude-engineer"
BASELINE_NAMES = ("persistence", "seasonal_naive", "smoother")


class EvaluationError(RuntimeError):
    """The run cannot be done as the protocol requires (refusal, not a model result)."""


# ---- data handed to a model factory --------------------------------------------------------


@dataclass(frozen=True)
class TrainSnapshot:
    """Everything a model may fit on at refit time ``refit_time``: matured rows only.

    ``observations`` have ``ts <= refit_time`` and ``available_at <= refit_time``; ``vintages``
    have ``issue_time <= refit_time`` and ``available_at <= refit_time``. A training pair
    (origin ``o``, target ``o + h``) is usable only if ``o + h <= refit_time``. The refit time is
    ``fold start - max(horizons)``, so no horizon trains on data from after its own origin.
    """

    refit_time: datetime
    stations: tuple[str, ...]
    horizons: tuple[int, ...]
    observations: tuple[Observation, ...]
    vintages: tuple[Vintage, ...]


Factory = Callable[..., Predictor]


# ---- baselines (single implementation, used by the pipeline and the factories) -------------

ObsAt = Callable[[datetime], Observation | None]


def persistence_value(obs_at: ObsAt, origin: datetime) -> float | None:
    """Newest observation with ``ts`` in ``[origin-3h, origin-1h]`` (and available at origin)."""
    rows = []
    for k in (1, 2, 3):
        o = obs_at(origin - k * HOUR)
        if o is not None and o.available_at <= origin:
            rows.append((o.ts, o.value))
    return latest_observation(rows, origin - HOUR, timedelta(hours=2))


def smoother_value(obs_at: ObsAt, origin: datetime) -> float | None:
    """``latest + beta * (mean24 - latest)``; ``mean24`` None falls back to ``latest``."""
    latest = persistence_value(obs_at, origin)
    if latest is None:
        return None
    window = [o for k in range(1, 25) if (o := obs_at(origin - k * HOUR)) is not None]
    mean24 = mean_24h(window, origin)
    return latest if mean24 is None else latest + SMOOTHER_BETA * (mean24 - latest)


def seasonal_naive_value(obs_at: ObsAt, origin: datetime, horizon: int) -> float | None:
    """Value 24 h-multiples before the target: the nearest one with ``ts <= origin - 1 h``.

    Differs from ``backtest.evaluate`` (which allows ``ts == origin``): the serving lag of
    protocol section 2 forbids the origin-hour reading, so +24 h uses the day before yesterday's
    hour (``origin - 24 h``) instead.
    """
    target = origin + horizon * HOUR
    k = math.ceil((target - (origin - HOUR)) / DAY)
    o = obs_at(target - k * DAY)
    return o.value if o is not None and o.available_at <= origin else None


def _snapshot_lookup(snapshot: Snapshot) -> ObsAt:
    horizon_start = snapshot.prediction_time - 72 * HOUR
    recent: dict[datetime, Observation] = {}
    for o in reversed(snapshot.observations):  # sorted by ts; stop at the first old row
        if o.ts < horizon_start:
            break
        recent[o.ts] = o
    return recent.get


def persistence(train: TrainSnapshot) -> Predictor:
    """Built-in factory: persistence (newest observation within the 3 h age limit)."""
    return lambda snap, horizon: persistence_value(_snapshot_lookup(snap), snap.prediction_time)


def smoother(train: TrainSnapshot) -> Predictor:
    """Built-in factory: the KAN-76 smoother (beta 0.5), applied at every horizon."""
    return lambda snap, horizon: smoother_value(_snapshot_lookup(snap), snap.prediction_time)


def seasonal_naive(train: TrainSnapshot) -> Predictor:
    """Built-in factory: value one (or more) whole days before the target, serving-lag safe."""
    return lambda snap, horizon: seasonal_naive_value(
        _snapshot_lookup(snap), snap.prediction_time, horizon
    )


BUILTIN_FACTORIES: dict[str, Factory] = {
    "persistence": persistence,
    "seasonal_naive": seasonal_naive,
    "smoother": smoother,
}


def resolve_factory(spec: str) -> tuple[Factory, str]:
    """``module:attribute`` -> (callable, short model name)."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise EvaluationError(f"--model must look like module:factory, got {spec!r}")
    fn = getattr(importlib.import_module(module_name), attr, None)
    if not callable(fn):
        raise EvaluationError(f"{spec} is not a callable factory")
    return fn, attr


def _call_factory(
    fit: Factory, train: TrainSnapshot, params: Mapping[str, Any] | None
) -> Predictor:
    try:
        takes_params = "params" in inspect.signature(fit).parameters
    except (TypeError, ValueError):
        takes_params = False
    predictor = fit(train, params=dict(params or {})) if takes_params else fit(train)
    if not callable(predictor):
        raise EvaluationError("the model factory must return a callable predictor")
    return predictor


# ---- plan: split, folds, cohort ------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    split: str
    h0: datetime | None
    end: datetime  # E
    limit: datetime  # exclusive end of the targets (and of all data) this run may read
    window_start: datetime
    folds: tuple[tuple[datetime, datetime], ...]  # (target window start, end) per weekly fold
    dev_limit: datetime  # cohort period end: H0 or E


def make_plan(last_target: datetime, split: str) -> Plan:
    f = folds_mod.derive_folds(last_target)
    dev_limit = f.holdout_start or f.end
    if split == "holdout":
        if f.holdout_start is None:
            raise EvaluationError(
                "snapshot too short for a holdout (dev period < 49 d): no holdout exists"
            )
        starts, limit = list(f.holdout_folds), f.end
    elif split == "dev":
        starts, limit = list(f.dev_folds), dev_limit
    else:
        raise EvaluationError(f"unknown split {split!r}")
    if not starts:
        raise EvaluationError("no walk-forward folds fit in this snapshot")
    ends = [*starts[1:], limit]
    return Plan(
        split,
        f.holdout_start,
        f.end,
        limit,
        starts[0],
        tuple(zip(starts, ends, strict=True)),
        dev_limit,
    )


def select_cohort(observations: Sequence[Observation], start: datetime, end: datetime) -> list[str]:
    """Stations with at least 80% hourly coverage in ``[start, end)`` (the dev period)."""
    hours = (end - start) // HOUR
    seen: dict[str, set[datetime]] = defaultdict(set)
    for o in observations:
        if start <= o.ts < end:
            seen[o.station_id].add(o.ts)
    return sorted(
        s for s, ts in seen.items() if hours > 0 and len(ts) / hours >= MIN_CONSIDERED_COVERAGE
    )


# ---- walk-forward --------------------------------------------------------------------------


@dataclass
class _Rows:
    """Rows of one horizon; ``cand``/``seas`` are NaN where the forecast is missing."""

    station: list[str] = field(default_factory=list)
    origin: list[datetime] = field(default_factory=list)
    fold: list[int] = field(default_factory=list)
    actual: list[float] = field(default_factory=list)
    pers: list[float] = field(default_factory=list)
    smoo: list[float] = field(default_factory=list)
    seas: list[float] = field(default_factory=list)
    cand: list[float] = field(default_factory=list)


def _walk_forward(
    observations: Sequence[Observation],
    vintages: Sequence[Vintage],
    stations: Sequence[str],
    plan: Plan,
    horizons: Sequence[int],
    fit: Factory,
    params: Mapping[str, Any] | None,
) -> tuple[dict[int, _Rows], dict[str, int]]:
    by_station: dict[str, dict[datetime, Observation]] = {s: {} for s in stations}
    sorted_obs: dict[str, list[Observation]] = {s: [] for s in stations}
    for o in sorted(observations, key=lambda r: r.ts):
        if o.station_id in by_station:
            by_station[o.station_id][o.ts] = o
            sorted_obs[o.station_id].append(o)
    obs_ts = {s: [o.ts for o in rows] for s, rows in sorted_obs.items()}
    sorted_vin: dict[str, list[Vintage]] = {s: [] for s in stations}
    for v in sorted(vintages, key=lambda r: r.issue_time):
        if v.station_id in sorted_vin:
            sorted_vin[v.station_id].append(v)
    vin_ts = {s: [v.issue_time for v in rows] for s, rows in sorted_vin.items()}

    max_h, min_h = max(horizons), min(horizons)
    rows = {h: _Rows() for h in horizons}
    counters = {"clipped": 0, "candidate_none": 0}
    for fold_i, (fold_start, fold_end) in enumerate(plan.folds):
        refit = fold_start - max_h * HOUR
        train = TrainSnapshot(
            refit,
            tuple(stations),
            tuple(horizons),
            tuple(
                o
                for s in stations
                for o in sorted_obs[s][: bisect_right(obs_ts[s], refit)]
                if o.available_at <= refit
            ),
            tuple(
                v
                for s in stations
                for v in sorted_vin[s][: bisect_right(vin_ts[s], refit)]
                if v.available_at <= refit
            ),
        )
        predictor = _call_factory(fit, train, params)
        origin = refit
        last_origin = fold_end - min_h * HOUR
        while origin < last_origin:
            hs = [h for h in horizons if fold_start <= origin + h * HOUR < fold_end]
            for s in stations:
                lookup = by_station[s].get
                todo = [(h, a) for h in hs if (a := lookup(origin + h * HOUR)) is not None]
                if not todo:
                    continue
                pers = persistence_value(lookup, origin)
                smoo = smoother_value(lookup, origin)
                if pers is None or smoo is None:
                    continue  # not baseline-eligible: outside every intersection
                snap = Snapshot(
                    origin,
                    s,
                    tuple(
                        o
                        for o in sorted_obs[s][: bisect_right(obs_ts[s], origin - HOUR)]
                        if o.available_at <= origin
                    ),
                    tuple(
                        v
                        for v in sorted_vin[s][: bisect_right(vin_ts[s], origin)]
                        if v.available_at <= origin
                    ),
                )
                for h, actual in todo:
                    pred = predictor(snap, h)
                    if pred is None:
                        counters["candidate_none"] += 1
                        pred = math.nan
                    else:
                        pred = float(pred)
                        if not math.isfinite(pred):
                            raise EvaluationError(f"non-finite prediction at {s} {origin} +{h}h")
                        if pred < 0:
                            pred = 0.0
                            counters["clipped"] += 1
                    seas = seasonal_naive_value(lookup, origin, h)
                    r = rows[h]
                    r.station.append(s)
                    r.origin.append(origin)
                    r.fold.append(fold_i)
                    r.actual.append(actual.value)
                    r.pers.append(pers)
                    r.smoo.append(smoo)
                    r.seas.append(math.nan if seas is None else seas)
                    r.cand.append(pred)
            origin += HOUR
    return rows, counters


# ---- scoring -------------------------------------------------------------------------------


def _mae(e: np.ndarray) -> float:
    return float(np.abs(e).mean())


def _rmse(e: np.ndarray) -> float:
    return float(math.sqrt((e * e).mean()))


def _series(origin_idx: np.ndarray, diff: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.bincount(origin_idx, weights=diff, minlength=grid)
    n = np.bincount(origin_idx, minlength=grid).astype(np.float64)
    return s, n


def _compare(s: np.ndarray, n: np.ndarray, horizon: int, level: float) -> dict[str, Any]:
    """Pooled diff, 72 h block CI, sensitivities, fragile flag and DM for one paired series."""
    out: dict[str, Any] = {"diff": float(s.sum() / n.sum())}
    if not np.any(s):
        out["degenerate"] = "all paired differences are zero (same forecasts as the baseline)"
        return out
    lo, hi, _ = stats.moving_block_bootstrap_ci(s, n, block=stats.PRIMARY_BLOCK_H, level=level)
    out["ci"] = [lo, hi]
    sens = {}
    for block in stats.SENSITIVITY_BLOCKS_H:
        sl, sh, _ = stats.moving_block_bootstrap_ci(s, n, block=block, level=level)
        sens[str(block)] = [sl, sh]
    out["ci_sensitivity"] = sens
    out["fragile"] = bool(hi < 0 or lo > 0) and any((sl <= 0 <= sh) for sl, sh in sens.values())
    present = n > 0
    out["dm"] = {}
    if present.mean() < stats.DM_MIN_PRESENT:
        out["dm_note"] = f"only {present.mean():.1%} of grid hours present (< 95%): DM not computed"
        return out
    d = s[present] / n[present]
    for label, m in (("167", stats.DM_PRIMARY_BANDWIDTH), ("47", 47), ("h-1", horizon - 1)):
        res = stats.diebold_mariano(d, horizon, m)
        out["dm"][label] = (
            None
            if res is None
            else {"statistic": res.statistic, "pvalue": res.pvalue, "bandwidth": res.bandwidth}
        )
    if out["dm"]["167"] is None:
        out["dm_note"] = "variance <= 0: DM null, run invalid for gating"
    return out


def _spike(actual: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    obs, hit = actual > SPIKE_THRESHOLD, pred > SPIKE_THRESHOLD
    tp, fn, fp = int((obs & hit).sum()), int((obs & ~hit).sum()), int((~obs & hit).sum())
    enough = tp + fn >= MIN_SPIKE_PAIRS
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": tp / (tp + fn) if enough else None,
        "precision": tp / (tp + fp) if enough and tp + fp > 0 else None,
    }


def score_horizon(
    h: int, r: _Rows, plan: Plan, level: float, cohort: Sequence[str]
) -> dict[str, Any]:
    actual = np.array(r.actual)
    pers, smoo, seas, cand = (np.array(x) for x in (r.pers, r.smoo, r.seas, r.cand))
    eligible = len(actual)  # rows are only stored when persistence and smoother exist
    keep = np.isfinite(cand)
    if not keep.any():
        return {"n_pairs": 0, "eligible_pairs": eligible, "coverage": 0.0}
    actual, pers, smoo, seas, cand = (x[keep] for x in (actual, pers, smoo, seas, cand))
    stations = np.array(r.station)[keep]
    fold = np.array(r.fold)[keep]
    origins = np.array([(o - (plan.window_start - h * HOUR)) // HOUR for o in r.origin])[keep]
    grid = int((plan.limit - plan.window_start) // HOUR)
    err = {"model": cand - actual, "persistence": pers - actual, "smoother": smoo - actual}
    abs_err = {k: np.abs(v) for k, v in err.items()}
    out: dict[str, Any] = {
        "n_pairs": int(keep.sum()),
        "eligible_pairs": eligible,
        "coverage": float(keep.sum() / eligible),
        "n_origins": int(np.unique(origins).size),
        "mae": _mae(err["model"]),
        "rmse": _rmse(err["model"]),
        "baselines": {
            "persistence": {"mae": _mae(err["persistence"]), "rmse": _rmse(err["persistence"])},
            "smoother": {"mae": _mae(err["smoother"]), "rmse": _rmse(err["smoother"])},
        },
        "grid_hours": grid,
    }
    has_seas = np.isfinite(seas)
    out["baselines"]["seasonal_naive"] = {
        "n_pairs": int(has_seas.sum()),
        "mae": _mae(seas[has_seas] - actual[has_seas]) if has_seas.any() else None,
        "rmse": _rmse(seas[has_seas] - actual[has_seas]) if has_seas.any() else None,
        "note": "scored on the common rows where seasonal-naive exists",
    }
    out["skill_vs_persistence"] = 1 - out["mae"] / out["baselines"]["persistence"]["mae"]
    for name in ("persistence", "smoother"):
        s, n = _series(origins, abs_err["model"] - abs_err[name], grid)
        out[f"vs_{name}"] = _compare(s, n, h, level)
    out["spike"] = _spike(actual, cand)
    out["by_fold"] = {
        str(i): {
            "n_pairs": int((fold == i).sum()),
            "diff_vs_persistence": float(
                (abs_err["model"] - abs_err["persistence"])[fold == i].mean()
            ),
            "diff_vs_smoother": float((abs_err["model"] - abs_err["smoother"])[fold == i].mean()),
        }
        for i in sorted(set(fold.tolist()))
    }
    out["mae_by_station"] = {
        s: {
            "n_pairs": int((stations == s).sum()),
            **{k: float(v[stations == s].mean()) for k, v in abs_err.items()},
        }
        for s in sorted(set(stations.tolist()))
    }
    hours = np.array([(o + h * HOUR).hour for o in r.origin])[keep]
    out["mae_by_hour_utc"] = {
        f"{hh:02d}": {
            "n_pairs": int((hours == hh).sum()),
            **{k: float(v[hours == hh].mean()) for k, v in abs_err.items()},
        }
        for hh in sorted(set(hours.tolist()))
    }
    out["cohort_pairs"] = {s: int((stations == s).sum()) for s in cohort}
    return out


def validity_reasons(plan: Plan, scores: Mapping[int, dict[str, Any]]) -> list[str]:
    """Reasons the +24 h result is invalid for gating (protocol sections 1, 6, 7, 12)."""
    s = scores.get(24)
    if not s or not s.get("n_pairs"):
        return ["no +24h pairs"]
    reasons = []
    if s["coverage"] < MIN_COVERAGE:
        reasons.append(f"candidate covers {s['coverage']:.1%} of baseline-eligible pairs (< 95%)")
    if plan.split == "dev":
        big = sum(f["n_pairs"] >= folds_mod.MIN_DEV_FOLD_PAIRS for f in s["by_fold"].values())
        if big < 3:
            reasons.append(
                f"only {big} dev folds with >= {folds_mod.MIN_DEV_FOLD_PAIRS} pairs (< 3)"
            )
    else:
        if (
            s["n_pairs"] < folds_mod.MIN_HOLDOUT_PAIRS
            or s["n_origins"] < folds_mod.MIN_HOLDOUT_ORIGINS
        ):
            reasons.append("insufficient_data: holdout below 2100 pairs / 400 origins")
        low = [k for k, n in s["cohort_pairs"].items() if n < folds_mod.MIN_STATION_PAIRS]
        if low:
            reasons.append(f"insufficient_data: stations below 250 pairs: {', '.join(low)}")
    for name in ("persistence", "smoother"):
        cmp_ = s[f"vs_{name}"]
        if "dm_note" in cmp_ and "variance" in cmp_["dm_note"]:
            reasons.append(f"DM variance <= 0 vs {name}")
    return reasons


# ---- run -----------------------------------------------------------------------------------


def run_evaluation(
    observations: Sequence[Observation],
    vintages: Sequence[Vintage],
    *,
    fit: Factory,
    split: str = "dev",
    horizons: Sequence[int] = HORIZONS_H,
    ci_level: float = 0.95,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Walk-forward evaluation of ``fit`` on already-loaded data; returns the full result dict.

    Only data in ``[S0, TARGET_CUT)`` is used. A dev run never reads anything at or after H0.
    """
    started = time.monotonic()
    horizons = sorted(horizons)
    usable = [o for o in observations if folds_mod.S0 <= o.ts < folds_mod.TARGET_CUT]
    if not usable:
        raise EvaluationError("no usable observations in the air4thai era")
    plan = make_plan(max(o.ts for o in usable), split)
    data = [o for o in usable if o.ts < plan.limit]
    cohort = select_cohort(data, folds_mod.S0, plan.dev_limit)
    if not cohort:
        raise EvaluationError("no station has >= 80% hourly coverage in the dev period")
    kept_vin = [v for v in vintages if v.valid_time < plan.limit]
    rows, counters = _walk_forward(data, kept_vin, cohort, plan, horizons, fit, params)
    scores = {h: score_horizon(h, rows[h], plan, ci_level, cohort) for h in horizons}
    reasons = validity_reasons(plan, scores)
    return {
        "split": split,
        "ci_level": ci_level,
        "horizons": list(horizons),
        "stations": cohort,
        "plan": {
            "H0": plan.h0.isoformat() if plan.h0 else None,
            "E": plan.end.isoformat(),
            "fold_starts": [s.isoformat() for s, _ in plan.folds],
            "refit_time": "fold start - max(horizons)",
            "n_folds": len(plan.folds),
        },
        "scores": {str(h): v for h, v in scores.items()},
        "counters": counters,
        "invalid_reasons": reasons,
        "bootstrap": {
            "B": stats.BOOTSTRAP_B,
            "seed": stats.BOOTSTRAP_SEED,
            "primary_block_h": stats.PRIMARY_BLOCK_H,
            "sensitivity_blocks_h": list(stats.SENSITIVITY_BLOCKS_H),
        },
        "seconds": round(time.monotonic() - started, 1),
    }


def metrics_from_result(result: Mapping[str, Any]) -> dict[str, float | None]:
    """Protocol section 11 metric names; undefined values are None (omitted by ``log_run``)."""
    metrics: dict[str, float | None] = {}
    for h in result["horizons"]:
        s = result["scores"][str(h)]
        if not s.get("n_pairs"):
            continue
        sm, pe = s["vs_smoother"], s["vs_persistence"]
        metrics.update(
            {
                f"mae_{h}h": s["mae"],
                f"rmse_{h}h": s["rmse"],
                f"skill_vs_persistence_{h}h": s["skill_vs_persistence"],
                f"diff_vs_smoother_{h}h": sm["diff"],
                f"diff_vs_smoother_ci_low_{h}h": sm["ci"][0] if "ci" in sm else None,
                f"diff_vs_smoother_ci_high_{h}h": sm["ci"][1] if "ci" in sm else None,
                f"diff_vs_persistence_ci_low_{h}h": pe["ci"][0] if "ci" in pe else None,
                f"diff_vs_persistence_ci_high_{h}h": pe["ci"][1] if "ci" in pe else None,
                f"dm_pvalue_vs_smoother_{h}h": (sm.get("dm", {}).get("167") or {}).get("pvalue"),
                f"spike_recall_{h}h": s["spike"]["recall"],
                f"spike_precision_{h}h": s["spike"]["precision"],
                f"n_pairs_{h}h": s["n_pairs"],
            }
        )
    return metrics


def _params_from_result(result: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    plan = result["plan"]
    return {
        "seed": stats.BOOTSTRAP_SEED,
        "ci_level": result["ci_level"],
        "H0": plan["H0"],
        "E": plan["E"],
        "fold_starts": ",".join(plan["fold_starts"]),
        "bootstrap_B": stats.BOOTSTRAP_B,
        "bootstrap_seed": stats.BOOTSTRAP_SEED,
        "block_h": stats.PRIMARY_BLOCK_H,
        "dm_bandwidth": stats.DM_PRIMARY_BANDWIDTH,
        "horizons": ",".join(map(str, result["horizons"])),
        "stations": ",".join(result["stations"]),
        "refit": "weekly, at fold start - max(horizons)",
        "clipped_negative_predictions": result["counters"]["clipped"],
        **extra,
    }


def ci_level_for(finalists: int) -> float:
    """Protocol section 8: ``1 - 0.05 / k`` for ``k`` registered finalists."""
    if finalists < 1:
        raise EvaluationError("--finalists must be >= 1")
    return 1 - 0.05 / finalists


def holdout_already_scored(prereg_id: str) -> bool:
    """True if a holdout run tagged with this pre-registration id is already in the store."""
    client = tracking._client()
    exps = client.search_experiments(filter_string=f"name LIKE '{tracking.EXPERIMENT_PREFIX}%'")
    ids = [e.experiment_id for e in exps]
    if not ids:
        return False
    runs = client.search_runs(
        ids, filter_string=f"tags.prereg = '{prereg_id}' and tags.split = 'holdout'", max_results=1
    )
    return len(runs) > 0


def check_holdout_allowed(prereg: Prereg | None, *, flag: bool, logging_on: bool) -> None:
    if not flag:
        raise EvaluationError("the holdout needs the explicit --holdout flag")
    if prereg is None or not prereg.final or prereg.split != "holdout":
        raise EvaluationError(
            "refusing holdout: the pre-registration must exist and be final: true, split: holdout"
        )
    if not logging_on:
        raise EvaluationError("a holdout run must be logged to the tracker (no --no-log)")
    if tracking.git_state()[1]:
        raise EvaluationError("refusing holdout: git working tree is dirty (protocol section 12)")
    if holdout_already_scored(prereg.id):
        raise EvaluationError(
            f"holdout for {prereg.id} was already scored once (protocol section 8)"
        )


def execute(
    *,
    snapshot: Path,
    model: str,
    prereg: Prereg | None = None,
    holdout: bool = False,
    finalists: int = 1,
    horizons: Sequence[int] | None = None,
    experiment: str | None = None,
    run_name: str | None = None,
    ticket: str | None = None,
    agent: str | None = None,
    log: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Guards, evaluation, then ``tracking.log_run``. Returns the result (with ``run_id``)."""
    split = "holdout" if holdout else "dev"
    if split == "holdout":
        check_holdout_allowed(prereg, flag=holdout, logging_on=log)
    fit, short = resolve_factory(model)
    observations, vintages, _ = load_snapshot(snapshot)
    model_name = prereg.model if prereg else short
    result = run_evaluation(
        observations,
        vintages,
        fit=fit,
        split=split,
        horizons=list(horizons or (prereg.horizons if prereg else HORIZONS_H)),
        ci_level=ci_level_for(finalists),
        params=prereg.params if prereg else None,
    )
    sha = tracking.snapshot_hash(Path(snapshot))
    result["snapshot_sha256"] = sha
    result["model"] = model
    tags = {
        "ticket": prereg.ticket if prereg else (ticket or DEFAULT_TICKET),
        "agent": prereg.author_agent if prereg else (agent or DEFAULT_AGENT),
        "model_name": model_name,
        "model_family": prereg.model_family if prereg else "baseline",
        "feature_set": prereg.feature_set if prereg else "none",
        "snapshot_sha256": sha,
        "split": split,
        "prereg": prereg.id if prereg else "exploratory",
    }
    if result["invalid_reasons"]:
        tags["invalid"] = "true"
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="evaluate-"))
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "evaluation.json"
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    result["artifact"] = str(artifact)
    if log:
        slug = (prereg.id if prereg else f"e04-{model_name}").replace("_", "-")
        result["run_id"] = tracking.log_run(
            experiment or f"kan78/{slug}",
            run_name or f"{model_name}-{split}",
            params=_params_from_result(
                result, {"model": model, "n_snapshot_rows": len(observations)}
            ),
            metrics=metrics_from_result(result),
            tags=tags,
            artifacts=[artifact],
        )
    return result


def _summary(result: Mapping[str, Any]) -> str:
    lines = [
        f"split={result['split']} folds={result['plan']['n_folds']} "
        f"stations={len(result['stations'])}"
    ]
    for h in result["horizons"]:
        s = result["scores"][str(h)]
        if not s.get("n_pairs"):
            lines.append(f"+{h}h: no pairs")
            continue
        sm, pe = s["vs_smoother"], s["vs_persistence"]
        ci = lambda c: f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]" if "ci" in c else "n/a"  # noqa: E731
        lines.append(
            f"+{h}h n={s['n_pairs']} MAE={s['mae']:.3f} RMSE={s['rmse']:.3f} "
            f"skill={s['skill_vs_persistence']:+.3f} vs_pers={pe['diff']:+.3f} {ci(pe)} "
            f"vs_smoother={sm['diff']:+.3f} {ci(sm)}"
        )
    if result["invalid_reasons"]:
        lines.append("INVALID: " + "; ".join(result["invalid_reasons"]))
    if "run_id" in result:
        lines.append(f"run_id={result['run_id']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pm25_forecast.experiments.evaluate", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--prereg", type=Path, default=None, help="pre-registration YAML/JSON")
    parser.add_argument("--snapshot", type=Path, required=True, help="E02 snapshot directory")
    parser.add_argument("--model", required=True, help="module:factory, fit(train) -> predictor")
    parser.add_argument("--holdout", action="store_true", help="score the locked holdout (final)")
    parser.add_argument(
        "--finalists", type=int, default=1, help="k registered finalists (CI level)"
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--experiment", default=None, help="kan78/<id> (default from the prereg)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--ticket", default=None, help="without a prereg (default KAN-82)")
    parser.add_argument("--agent", default=None, help="without a prereg")
    parser.add_argument("--no-log", action="store_true", help="dry run: do not write to MLflow")
    parser.add_argument("--out-dir", type=Path, default=None, help="where evaluation.json goes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        prereg = load_prereg(args.prereg) if args.prereg else None
        result = execute(
            snapshot=args.snapshot,
            model=args.model,
            prereg=prereg,
            holdout=args.holdout,
            finalists=args.finalists,
            horizons=args.horizons,
            experiment=args.experiment,
            run_name=args.run_name,
            ticket=args.ticket,
            agent=args.agent,
            log=not args.no_log,
            out_dir=args.out_dir,
        )
    except (EvaluationError, tracking.TrackingError) as exc:
        logger.error("refused: %s", exc)
        return 2
    print(_summary(result))
    print(f"details: {result['artifact']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
