# Experiment tracking with a shared local MLflow (KAN-81)

Tag and metric names are fixed by protocol section 11; `src/pm25_forecast/experiments/tracking.py`
enforces them (missing or unknown names raise `TrackingError`).

## Store

One store for all worktrees, outside git: `~/pm25-experiments/mlflow/` with `mlflow.db` (SQLite)
and `artifacts/`. The default needs no setup; `MLFLOW_TRACKING_URI` (see `.env.example`) overrides
it, blank means the default. Paths may contain spaces (the repo path does); the helper builds the
URI itself, so do not hand-write one. SQLite handles a few concurrent writers; avoid running many
agents' long runs against it at once.

## Setup

Experiment dependencies are an optional group; production `uv sync --locked` does not install them.

    uv sync --locked --group experiments

## Logging a run

    from pm25_forecast.experiments import tracking
    sha, dirty = tracking.git_state()
    with tracking.tracked_run("name", tags={...all ten tags...}, params={...}) as metrics:
        metrics.update({"mae_24h": ..., "rmse_24h": ..., "n_pairs_24h": ...})

Rules enforced: all ten tags; `split` is `dev` or `holdout`; `snapshot_sha256` is a 64-hex digest
(take it from the snapshot manifest); `git_dirty` must be `false` for holdout. Required metrics are
`mae_24h`, `rmse_24h`, `n_pairs_24h`; undefined values (None, NaN, inf) are omitted, never logged
as zero. If the body raises or a required metric is missing, the run is tagged `invalid=true`.

## Notebook template

`notebooks/experiment_template.ipynb` loads the frozen snapshot and the committed prereg, then logs
through `tracked_run`. Execute with
`uv run --group experiments jupyter nbconvert --to notebook --execute --inplace <copy>`.
Keep notebook outputs cleared before committing. As shipped, the template logs placeholder
metrics, so it raises `TrackingError` and the run is tagged `invalid`; replace them with computed values.

## Viewing

    uv run --group experiments mlflow ui --backend-store-uri "sqlite:///$HOME/pm25-experiments/mlflow/mlflow.db"
