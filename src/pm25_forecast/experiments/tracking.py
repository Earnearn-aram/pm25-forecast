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
import contextlib
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
from urllib.parse import unquote, urlparse

STORE_ENV = "MLFLOW_TRACKING_URI"
ARTIFACT_ENV = "MLFLOW_ARTIFACT_ROOT"
DEFAULT_STORE_DIR = Path("~/pm25-experiments/mlflow")
EXPERIMENT_PREFIX = "kan78/"
_EXPERIMENT_RE = re.compile(r"^kan78/[a-z0-9][a-z0-9_-]{1,63}$")
BUSY_TIMEOUT_S = 30
LOCK_RETRIES = 6
REPO_ROOT = Path(__file__).resolve().parents[3]
LEADERBOARD_PATH = REPO_ROOT / "docs" / "experiments" / "leaderboard.md"
# Output directories that do not count as uncommitted code when computing ``git_dirty``.
DIRTY_IGNORE = ("report", ".claude", ".codex", "out")

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


def _env(name: str) -> str:
    """Process environment first, then ``.env`` in the working directory or the repo root."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    from dotenv import dotenv_values

    for base in (Path.cwd(), REPO_ROOT):
        value = (dotenv_values(base / ".env").get(name) or "").strip()
        if value:
            return value
    return ""


def _with_timeout(uri: str) -> str:
    """Add the SQLite busy timeout to a sqlite URI that has none."""
    if uri.startswith("sqlite:") and "timeout=" not in uri:
        return f"{uri}{'&' if '?' in uri else '?'}timeout={BUSY_TIMEOUT_S}"
    return uri


def tracking_uri() -> str:
    """``MLFLOW_TRACKING_URI`` (env or .env) if set, else the shared SQLite store."""
    # ``timeout`` is sqlite's busy timeout: a second writer waits instead of failing at once.
    return _with_timeout(_env(STORE_ENV) or f"sqlite:///{store_dir() / 'mlflow.db'}")


def _sqlite_path(uri: str) -> Path | None:
    match = re.match(r"^sqlite:///(.*?)(\?.*)?$", uri)
    return Path(match.group(1)).expanduser().resolve() if match else None


def artifact_root() -> str:
    """``MLFLOW_ARTIFACT_ROOT`` if set, else ``artifacts/`` next to the SQLite database.

    Always explicit, so artifacts never fall back to a per-worktree ``./mlruns``.
    """
    explicit = _env(ARTIFACT_ENV)
    if explicit:
        return explicit if "://" in explicit else Path(explicit).expanduser().resolve().as_uri()
    db = _sqlite_path(tracking_uri())
    if db is None:
        raise TrackingError(f"{ARTIFACT_ENV} is required when {STORE_ENV} is not a sqlite URI")
    return (db.parent / "artifacts").as_uri()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd or Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_state(repo: Path | None = None) -> tuple[str, bool]:
    """(HEAD sha, dirty flag) of the code that produced the run.

    Dirty means any tracked change or untracked file, except the output directories in
    ``DIRTY_IGNORE``.
    """
    pathspec = [":(top)", *(f":(top,exclude){d}" for d in DIRTY_IGNORE)]
    sha = _git("rev-parse", "HEAD", cwd=repo)
    return sha, bool(_git("status", "--porcelain", "--", *pathspec, cwd=repo))


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
    digest = str(manifest.get("snapshot_sha256", manifest.get("content_sha256", "")))
    if not _SHA_RE.fullmatch(digest):
        raise TrackingError(f"no valid snapshot_sha256 in manifest {path}")
    return digest


def validate_tags(tags: Mapping[str, Any]) -> dict[str, str]:
    missing = [t for t in REQUIRED_TAGS if tags.get(t) in (None, "")]
    if missing:
        raise TrackingError(f"missing required tags: {', '.join(missing)}")
    unknown = sorted(set(tags) - set(REQUIRED_TAGS))
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
    db = _sqlite_path(uri)
    if db is not None:
        db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    return mlflow.MlflowClient()


def _ensure_experiment(client: Any, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    root = artifact_root()
    if root.startswith("file://"):
        Path(unquote(urlparse(root).path)).mkdir(parents=True, exist_ok=True)
    from mlflow.exceptions import MlflowException

    try:
        return client.create_experiment(name, artifact_location=root)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        return client.get_experiment_by_name(name).experiment_id  # another agent got there first


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
    Nothing is written if validation fails. A run that fails while writing is marked FAILED.
    Using the MLflow client directly bypasses all of these checks.
    """
    if not _EXPERIMENT_RE.fullmatch(experiment):
        raise TrackingError(f"experiment must match {_EXPERIMENT_RE.pattern}, got {experiment!r}")
    if not run_name.strip():
        raise TrackingError("run_name is empty")
    given = sorted(set(tags) & set(AUTO_TAGS))
    if given:
        raise TrackingError(f"{', '.join(given)} are filled automatically; do not pass them")
    if "invalid" in tags:
        raise TrackingError("the invalid tag is set by tooling, not by callers")
    full = dict(tags)
    if full.get("snapshot_sha256") in (None, ""):
        raise TrackingError(
            "snapshot_sha256 not given: pass the E02 snapshot digest, or the snapshot dir or "
            "manifest.json to read it from"
        )
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

    state: dict[str, Any] = {"run_id": None, "batch": False, "files": 0}

    def write() -> str:
        # Every step records its completion, so a retry after a lock error resumes where it
        # stopped instead of creating a second run.
        if state["run_id"] is None:
            run = client.create_run(
                _ensure_experiment(client, experiment), run_name=run_name, tags=checked_tags
            )
            state["run_id"] = run.info.run_id
        run_id = state["run_id"]
        if not state["batch"]:
            now = int(time.time() * 1000)
            client.log_batch(
                run_id,
                metrics=[Metric(k, v, now, 0) for k, v in checked_metrics.items()],
                params=[Param(k, str(v)) for k, v in params.items()],
            )
            state["batch"] = True
        while state["files"] < len(files):
            client.log_artifact(run_id, str(files[state["files"]]))
            state["files"] += 1
        client.set_terminated(run_id, "FINISHED")
        return run_id

    try:
        return _with_retry(write)
    except BaseException:
        if state["run_id"]:
            with contextlib.suppress(Exception):  # best effort; the original error matters
                client.set_terminated(state["run_id"], "FAILED")
        raise


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
    "prereg",
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
            if (
                run.data.tags.get("invalid") == "true"
                or run.info.lifecycle_stage != "active"
                or run.info.status != "FINISHED"
            ):
                continue
            m, t = run.data.metrics, run.data.tags
            rows.append(
                {
                    "experiment": exp.name,
                    "ticket": t.get("ticket", ""),
                    "run": f"{run.info.run_name} ({run.info.run_id[:8]})",
                    "mae": m.get("mae_24h"),
                    "rmse": m.get("rmse_24h"),
                    "skill": m.get("skill_vs_persistence_24h"),
                    "n_pairs": m.get("n_pairs_24h"),
                    "git_sha": t.get("git_sha", "")[:10],
                    "snapshot": t.get("snapshot_sha256", "")[:12],
                    "dirty": t.get("git_dirty") == "true",
                    "prereg": t.get("prereg", ""),
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
            str(r.get("prereg", "")),
        ]
        for r in rows
    ]
    cells = [[c.replace("|", "/") for c in row] for row in cells]
    warnings = []
    for label, key in (("snapshot_sha256", "snapshot"), ("prereg", "prereg")):
        values = sorted({str(r.get(key, "")) for r in rows})
        if len(values) > 1:
            warnings.append(
                f"WARNING: ranked runs have different {label} values ({', '.join(values)}); "
                "they are not directly comparable."
            )
    title = f"Leaderboard ({split} split, best run per experiment, by mae_24h)"
    if fmt == "confluence":
        lines = [*warnings, "||" + "||".join(LEADERBOARD_COLUMNS) + "||"]
        lines += ["|" + "|".join(c or " " for c in row) + "|" for row in cells]
        return "\n".join(lines) + "\n"
    if fmt != "md":
        raise TrackingError(f"unknown format {fmt!r}")
    lines = [
        f"# {title}",
        "",
        "Generated by `python -m pm25_forecast.experiments.tracking leaderboard --format md`.",
        "Invalid and unfinished runs are excluded; DIRTY marks runs made from uncommitted code.",
        "",
        *[w for warning in warnings for w in (warning, "")],
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
        if split == "holdout":
            print("WARNING: holdout results; never commit them or tune on them.", file=sys.stderr)
        if args.format == "md":
            out = args.out or LEADERBOARD_PATH
            if split == "holdout" and (args.out is None or out.resolve().is_relative_to(REPO_ROOT)):
                raise TrackingError("--holdout --format md needs --out outside the repository")
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
