"""Train the pooled ridge challenger (KAN-100) and write its registry artifact and manifest.

The model is the KAN-84 best variant ``ridge-pool-raw-chg``: one pooled ridge per horizon (station
one-hot), raw target, prediction = latest reading + predicted change, feature set ``fs_v2_core``.
The fit is ``experiments.linear_models.fit_models``, the SAME code the experiment ran (inner
time-series CV for lambda, training-mean standardisation); features come from
``features.library_v2`` only. Serving is ``models.serving.RidgeModel``; the bundle layout is
``models.ridge_bundle``. Direct strategy: a separate small ridge for each requested horizon.

Training data is every observation matured at ``--as-of`` (default: now, floored to the hour),
read with the experiment dataset path (``experiments.dataset``: read-only DB transaction, same
filters and availability rule as the experiments) or from a frozen snapshot (``--snapshot``)::

    DATABASE_URL=... uv run python -m pm25_forecast.models.ridge --out-dir out/kan100
    uv run python -m pm25_forecast.models.ridge --out-dir out/kan100 \
        --snapshot ~/pm25-experiments/snapshots/e02-v1
    uv run python -m pm25_forecast.registry_cli register \
        --artifact out/kan100/artifact.json --manifest out/kan100/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pm25_forecast.backtest import Observation
from pm25_forecast.collector import ConfigError
from pm25_forecast.experiments import dataset
from pm25_forecast.experiments import linear_models as lm
from pm25_forecast.experiments.evaluate import TrainSnapshot
from pm25_forecast.features.library_v2 import feature_set_names
from pm25_forecast.models.ridge_bundle import MAX_HORIZON_H, MISSING_POLICIES, parse_ridge_bundle
from pm25_forecast.settings import get_settings

logger = logging.getLogger(__name__)

VARIANT = "ridge-pool-raw-chg"
VARIANT_PARAMS: dict[str, Any] = {
    "variant": VARIANT,
    "model": "ridge",
    "scope": "pooled",
    "target": "raw",
    "output": "change",
    "feature_set": "fs_v2_core",
}
DEFAULT_HORIZONS: tuple[int, ...] = tuple(range(1, MAX_HORIZON_H + 1))
ASSUMED_LAG_H = 1
# KAN-84 (prereg e06-ridge-pool-raw-chg), dev split, +24 h MAE, ug/m3.
KAN84_DEV_24H = {"ridge": 3.757, "persistence": 4.311, "smoother": 3.867}


class TrainingError(ValueError):
    """Not enough data to fit any requested horizon."""


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def train_snapshot(
    observations: Sequence[Observation], refit_time: datetime, horizons: Sequence[int]
) -> TrainSnapshot:
    """Matured rows only: ``ts <= refit_time`` and ``available_at <= refit_time``."""
    matured = tuple(o for o in observations if o.ts <= refit_time and o.available_at <= refit_time)
    stations = tuple(sorted({o.station_id for o in matured}))
    return TrainSnapshot(refit_time, stations, tuple(sorted(horizons)), matured, ())


def fit_bundle(
    train: TrainSnapshot, missing_policy: str = "skip"
) -> tuple[dict[str, Any], dict[int, int]]:
    """Fit the variant on ``train``; return the ``ridge`` bundle section and rows per horizon."""
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"missing_policy must be one of {MISSING_POLICIES}")
    lm.clear_cache()
    spec = {**lm.DEFAULTS, **VARIANT_PARAMS}
    models, stations, log = lm.fit_models(train, spec)
    rows = {rec["h"]: rec["n"] for rec in log}
    horizons: dict[str, Any] = {}
    for h in train.horizons:
        fitted = next((models[s, h] for s in stations if (s, h) in models), None)
        if fitted is None:
            continue  # fewer than MIN_TRAIN_ROWS: no forecast for this horizon, never invented
        horizons[str(h)] = {
            "intercept": float(fitted.b),
            "coef": [float(v) for v in fitted.w],
            "mean": [float(v) for v in fitted.std.mean],
            "scale": [float(v) for v in fitted.std.scale],
            "lambda": float(fitted.strength),
            "n_train": int(rows[h]),
        }
    if not horizons:
        raise TrainingError(f"no horizon has {lm.MIN_TRAIN_ROWS} training rows")
    section = {
        "feature_set": spec["feature_set"],
        "features": list(lm.names_for(spec["feature_set"])),
        "stations": list(stations),
        "target": spec["target"],
        "output": spec["output"],
        "missing_policy": missing_policy,
        "horizons": horizons,
    }
    return section, rows


def ridge_artifact(section: Mapping[str, Any]) -> bytes:
    payload = {"format": 1, "ridge": section}
    artifact = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    parse_ridge_bundle(json.loads(artifact))  # never write a bundle the loader would refuse
    return artifact


def ridge_manifest(
    artifact: bytes,
    section: Mapping[str, Any],
    *,
    version: str,
    window: tuple[datetime, datetime],
    data_sha256: str,
    requested: Sequence[int],
    dev_mae: Mapping[str, Mapping[str, float]] | None = None,
    trained_at: datetime | None = None,
) -> dict[str, Any]:
    """Registry manifest. ``dev_mae`` (``{"6": {"ridge":, "persistence":, "smoother":}}``) is the
    dev-split evidence per horizon; hours where the ridge is worse than persistence are flagged."""
    served = sorted(int(h) for h in section["horizons"])
    dev = {str(h): dict(dev_mae[str(h)]) for h in served if dev_mae and str(h) in dev_mae}
    worse = [
        h for h in served if str(h) in dev and dev[str(h)]["ridge"] > dev[str(h)]["persistence"]
    ]
    return {
        "version": version,
        "model_type": "ridge-pooled-direct-by-horizon",
        "variant": VARIANT,
        "metrics": {"dev_mae_by_horizon": dev, "kan84_dev_24h": KAN84_DEV_24H},
        "training_window": {"start": _iso(window[0]), "end": _iso(window[1])},
        "feature_list": list(section["features"]),
        "feature_set": section["feature_set"],
        "data_sha256": data_sha256,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "assumed_reporting_lag_hours": ASSUMED_LAG_H,
        "horizons": served,
        "horizons_requested": sorted(requested),
        "ridge_lambda": {h: v["lambda"] for h, v in section["horizons"].items()},
        "n_train": {h: v["n_train"] for h, v in section["horizons"].items()},
        "stations": list(section["stations"]),
        "scaling": "training-mean imputation then z-score per column (mean and scale in bundle)",
        "missing_policy": section["missing_policy"],
        "horizons_worse_than_persistence_dev": worse,
        "note": (
            "Dev-split MAE per horizon is in metrics.dev_mae_by_horizon (no holdout used). "
            "horizons_worse_than_persistence_dev lists hours where the ridge did not beat "
            "persistence on dev; score and show them separately."
        ),
        "trained_at": _iso(trained_at or datetime.now(UTC)),
    }


def load_observations(
    snapshot: Path | None, *, kan74_repaired: bool
) -> tuple[list[Observation], str]:
    """(observations, data fingerprint) from a frozen snapshot or the read-only database."""
    if snapshot is not None:
        observations, _vintages, manifest = dataset.load_snapshot(snapshot)
        return observations, str(manifest.get("snapshot_sha256", manifest.get("content_sha256")))
    settings = get_settings()
    if settings.database_url is None:
        raise ConfigError("DATABASE_URL is not set (or pass --snapshot)")
    raw_obs, raw_vin = dataset.fetch_rows(settings.database_url.get_secret_value())
    obs, vin = dataset.normalise_and_filter(raw_obs, raw_vin, kan74_repaired=kan74_repaired)
    rows = [Observation(s, ts, v, av) for s, ts, v, _src, av, _ing in obs]
    return rows, dataset.content_sha256(obs, vin)


def _dev_table(path: Path | None) -> dict[str, dict[str, float]] | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=None, help="frozen snapshot, not the DB")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="horizons in hours to fit, each 1..24 (default: every hour 1..24)",
    )
    parser.add_argument("--as-of", default=None, help="ISO time with offset (default: now)")
    parser.add_argument("--version", default=None)
    parser.add_argument("--missing-policy", choices=MISSING_POLICIES, default="skip")
    parser.add_argument("--kan74-repaired", action="store_true")
    parser.add_argument(
        "--dev-mae", type=Path, default=None, help="JSON {h: {ridge, persistence, smoother}}"
    )
    try:
        args = parser.parse_args(argv)
        if any(not 1 <= h <= MAX_HORIZON_H for h in args.horizons):
            parser.error(f"--horizons must be within 1..{MAX_HORIZON_H}")
        now = datetime.now(UTC) if args.as_of is None else datetime.fromisoformat(args.as_of)
        if now.tzinfo is None:
            parser.error("--as-of needs a UTC offset")
        refit_time = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        version = args.version or f"{VARIANT}-{refit_time:%Y%m%dT%H}Z"
        observations, data_sha = load_observations(
            args.snapshot.expanduser() if args.snapshot else None,
            kan74_repaired=args.kan74_repaired,
        )
        train = train_snapshot(observations, refit_time, args.horizons)
        if not train.observations:
            raise TrainingError("no matured observations")
        names = feature_set_names(VARIANT_PARAMS["feature_set"])
        logger.info("fitting %d horizons on %d stations", len(train.horizons), len(train.stations))
        section, _rows = fit_bundle(train, args.missing_policy)
        artifact = ridge_artifact(section)
        first = min(o.ts for o in train.observations)
        manifest = ridge_manifest(
            artifact,
            section,
            version=version,
            window=(first, refit_time),
            data_sha256=data_sha,
            requested=args.horizons,
            dev_mae=_dev_table(args.dev_mae),
        )
        assert manifest["feature_list"] == list(names)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "artifact.json").write_bytes(artifact)
        (args.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sys.stdout.write(f"wrote {version} (horizons {manifest['horizons']}) to {args.out_dir}\n")
        return 0
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except TrainingError as exc:
        logger.error("training failed: %s", exc)
        return 1
    except Exception as exc:  # driver errors can echo DATABASE_URL; log the type only
        logger.error("ridge training failed (%s)", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
