# Experiment tracking with a shared local MLflow (KAN-81)

Tag and metric names are fixed by protocol section 11. `src/pm25_forecast/experiments/tracking.py`
enforces them: `log_run` refuses a run with a missing or unknown name and writes nothing.

## Store

One store for every worktree and agent (Claude and OpenAI alike), outside git:

    ~/pm25-experiments/mlflow/mlflow.db     SQLite backend
    ~/pm25-experiments/mlflow/artifacts/    artifact files

Exact tracking URI used by default (the helper builds it; paths with spaces are fine):

    MLFLOW_TRACKING_URI=sqlite:////Users/<you>/pm25-experiments/mlflow/mlflow.db?timeout=30

`MLFLOW_TRACKING_URI` can be set in the environment or in `.env` (cwd or repo root; the process
environment wins; placeholder in `.env.example`, blank means the default above). `?timeout=30`
(the SQLite busy timeout) is appended to any sqlite URI that lacks it.

Artifacts always go to an explicit root, never to a per-worktree `./mlruns`: `MLFLOW_ARTIFACT_ROOT`
(env or `.env`; a path or URI) if set, otherwise the `artifacts/` directory next to the SQLite file.
A non-sqlite tracking URI requires `MLFLOW_ARTIFACT_ROOT`. Experiments keep the artifact root they
were created with.

Concurrent writers: SQLite waits up to 30 s for the lock, and `log_run` retries lock errors up to
six times with backoff, resuming the same run (it never creates a second one). A run that still
fails is marked FAILED and the leaderboard ignores it.

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
  filled by `log_run` and cannot be passed. Dirty = any tracked change or untracked file outside
  `report/`, `.claude/`, `.codex/`, `out/`. `git_dirty` must be false for holdout runs.
- `invalid` is set by tooling only; `log_run` rejects it as a caller tag. Writing with the MLflow
  client or UI directly bypasses every check here; such runs are not valid evidence.
- Required metrics: `mae_24h`, `rmse_24h`. All others of protocol section 11 (including
  `n_pairs_24h`) are optional; undefined values are omitted, never logged as 0. The KAN-96 gate
  metrics (`station_worst_rel_diff_24h`, `station_worst_rel_diff_ci_high_24h`,
  `stations_worse_than_limit_24h`, `serving_lag_checks`, `serving_lag_mismatches`) are allowed
  without `_6h`/`_12h` variants.

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

`--execute` runs the notebook's code with your permissions; only render notebooks you trust.

Strip notebook outputs before committing (a pre-commit hook for this would need a new dependency, so it is a manual step):

    uv run --group experiments jupyter nbconvert --clear-output --inplace notebooks/<file>.ipynb

## Leaderboard

Best valid run (lowest `mae_24h`, `invalid` runs excluded) of each `kan78/*` experiment, dev split;
`--holdout` shows the holdout split instead; with `--format md` it requires `--out` outside the
repository (holdout numbers must not land in tracked files) and prints a warning. Only FINISHED
runs count. Runs from uncommitted code are marked DIRTY, and a warning is added when the ranked runs
have different `snapshot_sha256` or `prereg` values (they are not directly comparable).

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

## HTML reports per ticket (KAN-97)

One self-contained HTML file per experiment ticket, plus an index, written to `docs/experiments/reports/`
(committed; the written source stays `docs/experiments/e*.md`):

    uv run --group experiments python -m pm25_forecast.experiments.report --ticket KAN-83 \
        [--out docs/experiments/reports/] [--docs-dir docs/experiments] [--status merged]
    uv run --group experiments python -m pm25_forecast.experiments.report index

Open `docs/experiments/reports/index.html`: a table with one row per ticket (status, verdict,
best +24 h MAE, link) and the leaderboard with the DIRTY flag and the mixed-snapshot warning.

- Data: the best valid dev run of every `kan78/*` experiment tagged with the ticket, its
  `evaluation.json` artifact, the prereg YAMLs (`<docs-dir>/prereg/<prereg tag>.yaml`) and the
  `e*.md` whose title line names the ticket. Runs with a CI level other than 95% (for example a
  Bonferroni sensitivity run) are listed in the table but not plotted. Dev only: holdout runs are
  never read. Only aggregates are embedded (scores, CIs, per-fold, per-station, per-hour MAE).
- Plots (PNG embedded as base64, no network): 1 MAE per variant with persistence and smoother;
  2 paired difference vs the smoother with the 95% CI (72 h blocks) and fragile flags; 3 per-fold
  difference; 4 MAE by station and by hour for the best variant vs baselines; 5 feature ablation
  (only when `-abl-` experiments exist). Titles state the finding; a win is only claimed when the
  CI is entirely below zero, and it is always flagged as exploratory and unadjusted.
- `--status` is shown on the report and index (default `draft`; Jira is not read). The verdict is
  the `Verdict: **...**` line of the `e*.md`, else its first bold phrase near the top. For a ticket
  whose branch is not merged yet, point `--docs-dir` at that worktree's `docs/experiments`.
- Each report also writes `<ticket>.json` (title, status, verdict, best run); the index reads them.
- Needs `mlflow` and `matplotlib` (both come with the experiments group; matplotlib is a transitive
  dependency of mlflow, not declared in `pyproject.toml`). Output depends on the store contents, so
  regenerate after new runs.
