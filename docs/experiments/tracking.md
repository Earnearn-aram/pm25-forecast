# Experiment tracking with a shared local MLflow (KAN-81)

Tag and metric names are fixed by protocol section 11. `src/pm25_forecast/experiments/tracking.py`
enforces them: `log_run` refuses a run with a missing or unknown name and writes nothing.

## Store

One store for every worktree and agent (Claude and OpenAI alike), outside git:

    ~/pm25-experiments/mlflow/mlflow.db     SQLite backend
    ~/pm25-experiments/mlflow/artifacts/    artifact files

Exact tracking URI used by default (`$HOME` expanded; the helper builds it, paths with spaces are fine):

    MLFLOW_TRACKING_URI=sqlite:////Users/<you>/pm25-experiments/mlflow/mlflow.db?timeout=30

`MLFLOW_TRACKING_URI` in `.env` (placeholder in `.env.example`; blank means the default above)
overrides it. If you override it with a SQLite URI, keep `?timeout=30`, the SQLite busy timeout.
Concurrent writers: SQLite waits up to 30 s for the lock, and `log_run` also retries lock errors
six times with backoff, so two agents logging at once do not fail.

Cross-worktree write test (KAN-81 build): from the sandboxed agent session a file was written
to and removed from `~/pm25-experiments/mlflow/` while the shell was in two different directories
(the worktree and the main checkout, whose path contains a space). Not testable there: a second
agent session or a different OS user writing at the same time; the two-writer case is covered by
`tests/test_experiment_tracking.py::test_two_writers_at_once` on a temporary store.

## Naming

- Experiment: `kan78/<id>` with `<id>` lowercase letters, digits, `-`, `_` (for example `kan78/e05-lgbm-direct`).
- Run name: free text, unique per attempt (for example `lgbm-direct-seed1`).
- Tags (all required): `ticket`, `agent`, `model_name`, `model_family`, `feature_set`, `git_sha`,
  `git_dirty`, `snapshot_sha256`, `split` (`dev`|`holdout`), `prereg`. `git_sha` and `git_dirty` are
  filled by `log_run` (dirty = tracked files changed; untracked files are ignored) and cannot be
  passed. `git_dirty` must be false for holdout runs.
- Required metrics: `mae_24h`, `rmse_24h`. All others of protocol section 11 (including
  `n_pairs_24h`) are optional; undefined values are omitted, never logged as 0.

## Setup

The experiment dependencies are an optional group; production `uv sync --locked` does not install them.

    uv sync --locked --group experiments

## Log a run

    from pm25_forecast.experiments.tracking import log_run

    run_id = log_run(
        "kan78/e05-lgbm-direct", "lgbm-direct-seed1",
        params={"seed": 1}, metrics={"mae_24h": 5.1, "rmse_24h": 7.2, "n_pairs_24h": 900},
        tags={"ticket": "KAN-85", "agent": "claude-data-scientist", "model_name": "lgbm-direct",
              "model_family": "lightgbm", "feature_set": "base-v1",
              "snapshot_sha256": "<digest or path of the E02 snapshot dir / manifest.json>",
              "split": "dev", "prereg": "e05-lgbm-direct-v1"},
        artifacts=["figs/residuals.png"],
    )

Demo (persistence on a recorded fixture, experiment `kan78/e00-demo`):

    uv run --group experiments python -m pm25_forecast.experiments.tracking demo

## Report

`notebooks/_template_experiment_report.ipynb` has the eight sections of protocol section 13.
Copy it, fill it, then render it to `report/report.html` (git-ignored) and attach it to the run:

    uv run --group experiments python -m pm25_forecast.experiments.tracking render-report \
        notebooks/<file>.ipynb --run-id <RUN_ID> --execute

Strip notebook outputs before committing (a pre-commit hook for this would need a new dependency, so it is a manual step):

    uv run --group experiments jupyter nbconvert --clear-output --inplace notebooks/<file>.ipynb

## Leaderboard

Best valid run (lowest `mae_24h`, `invalid` runs excluded) of each `kan78/*` experiment, dev split;
`--holdout` shows the holdout split instead. Runs from uncommitted code are marked DIRTY.

    uv run --group experiments python -m pm25_forecast.experiments.tracking leaderboard --format md
    uv run --group experiments python -m pm25_forecast.experiments.tracking leaderboard --format confluence

`md` writes `docs/experiments/leaderboard.md`; `confluence` prints wiki-markup to paste.

## Browse the UI

    uv run --group experiments mlflow ui \
        --backend-store-uri "sqlite:///$HOME/pm25-experiments/mlflow/mlflow.db" \
        --default-artifact-root "file://$HOME/pm25-experiments/mlflow/artifacts"

## Back up

Stop writers, then copy the whole directory (the database and artifacts belong together):

    cp -R ~/pm25-experiments/mlflow ~/pm25-experiments/mlflow-backup-$(date -u +%Y%m%d)

or `sqlite3 ~/pm25-experiments/mlflow/mlflow.db ".backup 'mlflow-backup.db'"` for the database alone.
