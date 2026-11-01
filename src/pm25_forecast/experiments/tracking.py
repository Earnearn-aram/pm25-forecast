"""Shared local MLflow tracking for experiments (KAN-81; protocol section 11).

All worktrees write to one store outside git: ``~/pm25-experiments/mlflow`` (SQLite
``mlflow.db`` plus an ``artifacts/`` directory). Override with ``MLFLOW_TRACKING_URI``.

``log_run`` is the public entry point: it refuses a run with a missing or misspelled tag or
metric, fills ``git_sha`` / ``git_dirty`` itself and takes the E02 snapshot hash. ``mlflow`` lives
in the optional ``experiments`` dependency group and is imported lazily; validation works
without it.

CLI: ``python -m pm25_forecast.experiments.tracking {leaderboard,render-report,demo}``.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

STORE_ENV = "MLFLOW_TRACKING_URI"
DEFAULT_STORE_DIR = Path("~/pm25-experiments/mlflow")
EXPERIMENT_PREFIX = "kan78/"
_EXPERIMENT_RE = re.compile(r"^kan78/[a-z0-9][a-z0-9_-]{1,63}$")
BUSY_TIMEOUT_S = 30
LOCK_RETRIES = 6
LEADERBOARD_PATH = Path(__file__).resolve().parents[3] / "docs" / "experiments" / "leaderboard.md"

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
# Section 11 does not mark any metric required, so only the two headline errors are; the rest
# (including n_pairs_24h) may be undefined and are then omitted, never zero.
REQUIRED_METRICS = ("mae_24h", "rmse_24h")
AUTO_TAGS = ("git_sha", "git_dirty")

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class TrackingError(ValueError):
    """The run does not meet the logging contract."""


def store_dir() -> Path:
    return DEFAULT_STORE_DIR.expanduser()


def tracking_uri() -> str:
    """``MLFLOW_TRACKING_URI`` if set and non-blank, else the shared SQLite store."""
    # ``timeout`` is sqlite's busy timeout: a second writer waits instead of failing at once.
    default = f"sqlite:///{store_dir() / 'mlflow.db'}?timeout={BUSY_TIMEOUT_S}"
    return os.environ.get(STORE_ENV, "").strip() or default


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
    return sha, bool(_git("status", "--porcelain", "--untracked-files=no"))


def snapshot_hash(value: Path | str) -> str:
    """A 64-hex digest as is, or the ``snapshot_sha256`` of an E02 snapshot dir / manifest."""
    text = str(value)
    if _SHA_RE.fullmatch(text):
        return text
    path = Path(text).expanduser()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise TrackingError(f"snapshot_sha256 is neither a digest nor a manifest: {text}")
    manifest = json.loads(path.read_text())
    return str(manifest.get("snapshot_sha256", manifest.get("content_sha256", "")))


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
    out = {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}
    missing = [m for m in REQUIRED_METRICS if m not in out]
    if missing:
        raise TrackingError(f"missing required metrics: {', '.join(missing)}")
    return out


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("mlflow is not installed; run `uv sync --group experiments`") from exc
    return mlflow


def _client() -> Any:
    mlflow = _mlflow()
    uri = tracking_uri()
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///").split("?")[0]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return mlflow.MlflowClient()


def _ensure_experiment(client: Any, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    root = None
    if not os.environ.get(STORE_ENV, "").strip():
        (store_dir() / "artifacts").mkdir(parents=True, exist_ok=True)
        root = artifact_root()
    try:
        return client.create_experiment(name, artifact_location=root)
    except Exception:  # another agent created it first
        exp = client.get_experiment_by_name(name)
        if exp is None:
            raise
        return exp.experiment_id


def _is_locked(exc: BaseException) -> bool:
    return "locked" in str(exc).lower() or "busy" in str(exc).lower()


def _with_retry(fn: Any) -> Any:
    """Retry on SQLite lock errors with jittered backoff (the busy timeout covers short waits)."""
    for attempt in range(LOCK_RETRIES):
        try:
            return fn()
        except TrackingError:
            raise
        except Exception as exc:
            if not _is_locked(exc) or attempt == LOCK_RETRIES - 1:
                raise
            time.sleep(0.25 * 2**attempt + random.uniform(0, 0.25))


def log_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float | None],
    tags: Mapping[str, Any],
    artifacts: Sequence[Path | str] = (),
) -> str:
    """Validate, then write one finished run to the shared store; returns the run id.

    ``tags`` holds the eight caller tags; ``git_sha`` / ``git_dirty`` are filled here and may
    not be passed. ``tags["snapshot_sha256"]`` may be a digest or an E02 snapshot path.
    Nothing is written if validation fails.
    """
    if not _EXPERIMENT_RE.fullmatch(experiment):
        raise TrackingError(f"experiment must match {_EXPERIMENT_RE.pattern}, got {experiment!r}")
    if not run_name.strip():
        raise TrackingError("run_name is empty")
    given = sorted(set(tags) & set(AUTO_TAGS))
    if given:
        raise TrackingError(f"{', '.join(given)} are filled automatically; do not pass them")
    full = dict(tags)
    if full.get("snapshot_sha256") not in (None, ""):
        full["snapshot_sha256"] = snapshot_hash(full["snapshot_sha256"])
    sha, dirty = git_state()
    full["git_sha"], full["git_dirty"] = sha, str(dirty).lower()
    checked_tags = validate_tags(full)
    checked_metrics = validate_metrics(metrics)
    files = [Path(a).expanduser() for a in artifacts]
    absent = [str(f) for f in files if not f.is_file()]
    if absent:
        raise TrackingError(f"artifacts not found: {', '.join(absent)}")

    _mlflow()
    from mlflow.entities import Metric, Param

    client = _client()

    def write() -> str:
        run = client.create_run(
            _ensure_experiment(client, experiment), run_name=run_name, tags=checked_tags
        )
        run_id = run.info.run_id
        now = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[Metric(k, v, now, 0) for k, v in checked_metrics.items()],
            params=[Param(k, str(v)) for k, v in params.items()],
        )
        for f in files:
            client.log_artifact(run_id, str(f))
        client.set_terminated(run_id, "FINISHED")
        return run_id

    return _with_retry(write)


# ---- leaderboard ---------------------------------------------------------------------------

LEADERBOARD_COLUMNS = (
    "experiment",
    "run",
    "mae_24h",
    "rmse_24h",
    "skill vs persistence",
    "n_pairs",
    "git_sha",
    "snapshot hash",
    "dirty",
)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def collect_leaderboard(*, split: str = "dev") -> list[dict[str, Any]]:
    """Best valid run (lowest ``mae_24h``) of each ``kan78/*`` experiment for one split."""
    client = _client()
    rows = []
    for exp in client.search_experiments(filter_string=f"name LIKE '{EXPERIMENT_PREFIX}%'"):
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.split = '{split}' and metrics.mae_24h >= 0",
            order_by=["metrics.mae_24h ASC"],
            max_results=500,
        )
        for run in runs:
            if run.data.tags.get("invalid") == "true" or run.info.lifecycle_stage != "active":
                continue
            m, t = run.data.metrics, run.data.tags
            rows.append(
                {
                    "experiment": exp.name,
                    "run": f"{run.info.run_name} ({run.info.run_id[:8]})",
                    "mae": m.get("mae_24h"),
                    "rmse": m.get("rmse_24h"),
                    "skill": m.get("skill_vs_persistence_24h"),
                    "n_pairs": m.get("n_pairs_24h"),
                    "git_sha": t.get("git_sha", "")[:10],
                    "snapshot": t.get("snapshot_sha256", "")[:12],
                    "dirty": t.get("git_dirty") == "true",
                }
            )
            break  # runs are sorted, so the first valid one is the best
    return sorted(rows, key=lambda r: r["mae"])


def render_leaderboard(rows: Sequence[Mapping[str, Any]], fmt: str, *, split: str = "dev") -> str:
    cells = [
        [
            str(r["experiment"]),
            str(r["run"]),
            _fmt(r["mae"]),
            _fmt(r["rmse"]),
            _fmt(r["skill"]),
            "" if r["n_pairs"] is None else str(int(r["n_pairs"])),
            str(r["git_sha"]),
            str(r["snapshot"]),
            "DIRTY" if r["dirty"] else "",
        ]
        for r in rows
    ]
    cells = [[c.replace("|", "/") for c in row] for row in cells]
    title = f"Leaderboard ({split} split, best run per experiment, by mae_24h)"
    if fmt == "confluence":
        lines = ["||" + "||".join(LEADERBOARD_COLUMNS) + "||"]
        lines += ["|" + "|".join(c or " " for c in row) + "|" for row in cells]
        return "\n".join(lines) + "\n"
    if fmt != "md":
        raise TrackingError(f"unknown format {fmt!r}")
    lines = [
        f"# {title}",
        "",
        "Generated by `python -m pm25_forecast.experiments.tracking leaderboard --format md`.",
        "Invalid runs are excluded; DIRTY marks runs made from uncommitted code.",
        "",
        "| " + " | ".join(LEADERBOARD_COLUMNS) + " |",
        "|" + "|".join("---" for _ in LEADERBOARD_COLUMNS) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines) + "\n"


# ---- report rendering ----------------------------------------------------------------------


def render_report(
    notebook: Path | str,
    *,
    run_id: str | None = None,
    out_dir: Path | str = "report",
    execute: bool = False,
) -> Path:
    """Render a notebook to ``<out_dir>/report.html`` and attach it to the run as an artifact."""
    out = Path(out_dir)
    cmd = [sys.executable, "-m", "nbconvert", "--to", "html", "--output", "report"]
    cmd += ["--output-dir", str(out), str(notebook)]
    if execute:
        cmd.insert(3, "--execute")
    subprocess.run(cmd, check=True)
    html = out / "report.html"
    if run_id:
        _with_retry(lambda: _client().log_artifact(run_id, str(html), artifact_path="report"))
    return html


# ---- CLI -----------------------------------------------------------------------------------


def _demo() -> str:
    from pm25_forecast.experiments.demo import load_series, persistence_scores

    fixture = (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "tracking_demo_series.json"
    )
    scores = persistence_scores(load_series(fixture))
    return log_run(
        "kan78/e00-demo",
        "persistence-fixture",
        params={"horizon_h": 24, "fixture": fixture.name},
        metrics=scores,
        tags={
            "ticket": "KAN-81",
            "agent": "claude-engineer",
            "model_name": "persistence",
            "model_family": "baseline",
            "feature_set": "none",
            "snapshot_sha256": file_sha256(fixture),
            "split": "dev",
            "prereg": "none-demo",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pm25_forecast.experiments.tracking")
    sub = parser.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("leaderboard", help="best run per kan78/* experiment")
    lb.add_argument("--format", choices=("md", "confluence"), default="md")
    lb.add_argument("--holdout", action="store_true", help="holdout split instead of dev")
    lb.add_argument("--out", type=Path, default=None, help="md output file")
    rr = sub.add_parser("render-report", help="notebook -> report/report.html (+ artifact)")
    rr.add_argument("notebook", type=Path)
    rr.add_argument("--run-id", default=None)
    rr.add_argument("--out-dir", default="report")
    rr.add_argument("--execute", action="store_true")
    sub.add_parser("demo", help="log the persistence demo run on the recorded fixture")
    args = parser.parse_args(argv)

    if args.cmd == "leaderboard":
        split = "holdout" if args.holdout else "dev"
        text = render_leaderboard(collect_leaderboard(split=split), args.format, split=split)
        if args.format == "md":
            out = args.out or LEADERBOARD_PATH
            out.write_text(text)
            print(f"wrote {out}")
        else:
            print(text, end="")
    elif args.cmd == "render-report":
        html = render_report(
            args.notebook, run_id=args.run_id, out_dir=args.out_dir, execute=args.execute
        )
        print(html)
    else:
        print(_demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
