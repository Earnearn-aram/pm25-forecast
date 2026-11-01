"""Shared local MLflow tracking for experiments (KAN-81; protocol section 11).

All worktrees write to one store outside git: ``~/pm25-experiments/mlflow`` (SQLite
``mlflow.db`` plus an ``artifacts/`` directory). Override with ``MLFLOW_TRACKING_URI``.

The tag and metric names are enforced here, so a run cannot be logged with a missing or
misspelled name. ``mlflow`` lives in the optional ``experiments`` dependency group and is
imported lazily; the validation helpers work without it.
"""

import hashlib
import math
import os
import re
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

STORE_ENV = "MLFLOW_TRACKING_URI"
DEFAULT_STORE_DIR = Path("~/pm25-experiments/mlflow")
DEFAULT_EXPERIMENT = "pm25-forecast"

REQUIRED_TAGS = (
    "ticket",
    "agent",
    "model_name",
    "model_family",
    "feature_set",
    "git_sha",
    "git_dirty",
    "snapshot_sha256",
    "split",
    "prereg",
)
SPLITS = ("dev", "holdout")
HORIZONS = (6, 12, 24)

# Base names of protocol section 11; each is logged for +24 h, and with _6h/_12h variants.
_METRIC_BASES = (
    "mae",
    "rmse",
    "skill_vs_persistence",
    "diff_vs_smoother",
    "diff_vs_smoother_ci_low",
    "diff_vs_smoother_ci_high",
    "diff_vs_persistence_ci_low",
    "diff_vs_persistence_ci_high",
    "dm_pvalue_vs_smoother",
    "spike_recall",
    "spike_precision",
    "n_pairs",
)
ALLOWED_METRICS = frozenset(f"{base}_{h}h" for base in _METRIC_BASES for h in HORIZONS)
# Required for a valid run (section 12); the rest may be undefined and are then omitted.
REQUIRED_METRICS = ("mae_24h", "rmse_24h", "n_pairs_24h")

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class TrackingError(ValueError):
    """The run does not meet the logging contract."""


def store_dir() -> Path:
    return DEFAULT_STORE_DIR.expanduser()


def tracking_uri() -> str:
    """``MLFLOW_TRACKING_URI`` if set and non-blank, else the shared SQLite store."""
    return os.environ.get(STORE_ENV, "").strip() or f"sqlite:///{store_dir() / 'mlflow.db'}"


def artifact_root() -> str:
    return (store_dir() / "artifacts").as_uri()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_state() -> tuple[str, bool]:
    """(HEAD sha, dirty flag) of the code that produced the run."""
    sha = _git("rev-parse", "HEAD")
    return sha, bool(_git("status", "--porcelain"))


def validate_tags(tags: Mapping[str, Any]) -> dict[str, str]:
    missing = [t for t in REQUIRED_TAGS if tags.get(t) in (None, "")]
    if missing:
        raise TrackingError(f"missing required tags: {', '.join(missing)}")
    unknown = sorted(set(tags) - set(REQUIRED_TAGS) - {"invalid"})
    if unknown:
        raise TrackingError(f"unknown tags: {', '.join(unknown)}")
    out = {k: str(v) for k, v in tags.items()}
    if out["split"] not in SPLITS:
        raise TrackingError(f"split must be one of {SPLITS}, got {out['split']!r}")
    if not re.fullmatch(r"KAN-\d+", out["ticket"]):
        raise TrackingError(f"ticket must look like KAN-<n>, got {out['ticket']!r}")
    if not _SHA_RE.fullmatch(out["snapshot_sha256"]):
        raise TrackingError("snapshot_sha256 must be a 64-char lowercase hex digest")
    out["git_dirty"] = out["git_dirty"].lower()
    if out["git_dirty"] not in ("true", "false"):
        raise TrackingError("git_dirty must be true or false")
    if out["split"] == "holdout" and out["git_dirty"] == "true":
        raise TrackingError("holdout runs need git_dirty=false (protocol section 12)")
    return out


def validate_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Check names; drop undefined (None/NaN/inf) values rather than logging zero."""
    unknown = sorted(set(metrics) - ALLOWED_METRICS)
    if unknown:
        raise TrackingError(f"unknown metrics: {', '.join(unknown)}")
    out = {
        k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))
    }
    missing = [m for m in REQUIRED_METRICS if m not in out]
    if missing:
        raise TrackingError(f"missing required metrics: {', '.join(missing)}")
    return out


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "mlflow is not installed; run `uv sync --group experiments`"
        ) from exc
    return mlflow


def ensure_experiment(name: str = DEFAULT_EXPERIMENT) -> str:
    """Point MLflow at the shared store and return the experiment id (created on demand)."""
    mlflow = _mlflow()
    uri = tracking_uri()
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    root = (
        artifact_root()
        if not os.environ.get(STORE_ENV, "").strip()
        else None  # custom store: let the server pick its default artifact location
    )
    if root:
        (store_dir() / "artifacts").mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=root)


@contextmanager
def tracked_run(
    run_name: str,
    *,
    tags: Mapping[str, Any],
    params: Mapping[str, Any],
    experiment: str = DEFAULT_EXPERIMENT,
) -> Iterator[dict[str, float]]:
    """Open a run with validated tags and params; yields a dict to fill with metrics.

    Metrics are validated and logged on clean exit. If the body raises, nothing is logged
    as a result and the run is tagged ``invalid``.
    """
    checked = validate_tags(tags)
    mlflow = _mlflow()
    exp_id = ensure_experiment(experiment)
    metrics: dict[str, float] = {}
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
        mlflow.set_tags(checked)
        mlflow.log_params({k: str(v) for k, v in params.items()})
        try:
            yield metrics
            mlflow.log_metrics(validate_metrics(metrics))
        except BaseException:
            mlflow.set_tag("invalid", "true")
            raise
