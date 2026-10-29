# pm25-forecast

Bangkok PM2.5 24-hour forecast, built as an MLOps portfolio project. The goal is to show an
end-to-end loop running unattended on free tiers: collect data, predict, store, score, monitor,
retrain. Private until the owner decides otherwise.

## Architecture (5 lines)

1. An hourly GitHub Actions job ingests Air4Thai readings and Open-Meteo weather plus the CAMS PM2.5 forecast (stored with its issue time).
2. Data-quality checks run, then observations are upserted into hosted storage (final choice in KAN-17).
3. A predict step uses the champion model (a persistence/CAMS baseline first, LightGBM as a shadow challenger later) and stores 24 h forecasts.
4. Matured forecasts are scored against actuals; drift (PSI) and data health are monitored, with failure alerts and a heartbeat.
5. A weekly retrain promotes a challenger only if it beats the champion; a static dashboard shows forecast vs actual vs baseline.

Diagrams: https://earnrw.atlassian.net/wiki/spaces/EARN/pages/66244/PM2.5+Architecture+Diagrams

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv if missing).

```bash
uv sync                 # install runtime + dev dependencies from uv.lock
uv run pytest           # tests
uv run ruff check       # lint
uv run ruff format --check
uv run pre-commit install   # optional: run hooks on every commit
cp .env.example .env        # optional: local settings, never commit .env
```

Collector (KAN-18): `uv run python -m pm25_forecast.collector` needs `DATABASE_URL` and
`STATION_LOCATIONS` (see `.env.example`). The workflow draft (every 20 minutes) is `docs/collector-workflow.yml.txt`.

Alerts and heartbeat (KAN-19): everything goes to `ALERT_WEBHOOK_URL` (Telegram or Discord; the
body carries both `text` and `content`). Without it, alerts are logged only, with a warning.
`python -m pm25_forecast.monitoring.notify failure` is the workflows' `if: failure()` step;
`python -m pm25_forecast.monitoring.heartbeat` posts a daily one-line summary
(`docs/heartbeat-workflow.yml.txt`). Each run logs one `RUN_SUMMARY {json}` line; log timestamps
are UTC.

Backfill (KAN-12 part 1, KAN-34): `uv run python -m pm25_forecast.backfill [--lookback-days 120]
[--stations 59t,61t]` upserts whatever hourly history Air4Thai still holds (a rolling window of
about 3 months) into `observations`. Same settings as the collector; idempotent, so it runs daily
as gap recovery. Workflow draft: `docs/backfill-workflow.yml.txt`.

Predict (KAN-28): `uv run python -m pm25_forecast.predict` reads stored observations and CAMS rows
and appends `persistence-24h` and `cams` forecasts for +6, +12 and +24 h per station to
`predictions` (idempotent per issue hour). Same settings as the collector; DB down means skip,
alert and exit non-zero. Baselines: `models/baselines.py`. Workflow draft: `docs/predict-workflow.yml.txt`.

Backtest (KAN-14): `uv run python -m pm25_forecast.backtest /private/path/input.json` runs
read-only rolling-origin evaluation with identical scoring pairs, availability gates, and a
28-day paired CAMS comparison rule. See [input contract and data limitations](docs/backtest.md).
No historical CAMS results are invented when stored vintages are missing.

First local model (KAN-32): on a fresh database, apply `src/pm25_forecast/storage/schema.sql`. On
an existing database, run `src/pm25_forecast/storage/migrations/001_model_manifest.sql` and
`src/pm25_forecast/storage/migrations/002_model_champion.sql` once. Then run this sequence from a
fresh clone:

```sh
uv sync --locked
uv run python -m pm25_forecast.export_history --output data/history.duckdb
uv run python -m pm25_forecast.train_model --export data/history.duckdb --push
```

The export reads observations and stored CAMS forecast vintages in a PostgreSQL read-only
transaction. Training reads only the local DuckDB file. The `--push` step writes the model binary
and manifest to the `models` table, reconnects, reads them back, checks that the artifact digest
and manifest match the submitted values, and deserializes the LightGBM bundle against the shared
ordered feature list. Model versions are immutable: retrying a version with identical content is
idempotent, while different content is rejected. The export and generated artifacts are local,
git-ignored files. `DATABASE_URL` comes from the environment or `.env` and is never included in
logs. The manifest records the UTC training window, held-out metrics, ordered feature list, export
hash, and artifact hash. Registry writes verify that manifest hashes match the exact binary.

Model versions remain immutable. The separate `model_champion` table stores one mutable
`production` pointer to a registered version; moving it never changes a model row. A freshly
trained version is stored without changing the pointer by default. To explicitly register and
promote a model in the same run, add `--promote` to the training command (`--promote` requires
`--push`). `storage.registry.read_champion` returns the pointer plus the content-verified bundle
for a consumer. Hourly prediction does not select this pointer yet; that integration remains in
KAN-29.

Promotions update the single production row. The chosen concurrent-write policy is last successful
upsert wins if PostgreSQL serializes competing writes on that row; there is no compare-and-swap or
model ranking guard. The registry test covers sequential replacement only; it does not verify
concurrent PostgreSQL behavior.

Training uses a strict as-of policy: an observation can be an input only when its exported
`available_at` is no later than the simulated issue time, and a target must be present in the
local snapshot. Hourly issue times are derived from target timestamps and do not require a reading
at the exact issue hour; only earlier readings that passed the availability gate become features.
If any horizon has fewer than 20 eligible examples, training stops with counts for all horizons
and counts of observations and forecast vintages available by their own timestamps. Do not relax
this gate by replacing `available_at` with `ts` or by assuming a historical download was available
when its timestamp occurred. The chronological validation split also requires 20 training rows
per horizon after its horizon-length embargo, so the required sample count can be higher. The live
observations table is upserted and refreshes `ingested_at`; it does not preserve original arrival
times or prior values. The export therefore
supports conservative eligibility checks for the current row, but it cannot reconstruct a fully
faithful historical feature/label snapshot. Collect immutable, timestamped captures over time
before claiming historical training or validation coverage. A current export with less than 24
hours of post-availability observations cannot supply +24h training targets under this policy.

KAN-32 trains and verifies a registered model. `pm25_forecast.predict` emits the KAN-28
persistence and CAMS baselines every hour.

Shadow challenger (KAN-29): set `CHALLENGER_MODEL_VERSION` (a GitHub Actions variable, not a
secret) to a registered model version and the hourly predict step also stores that model's
predictions in `predictions` under its own `model_version`, next to the baselines, for KAN-20 to
score. Nothing uses them for decisions. The model is read once per run from the registry
(content-verified) and loaded by `models.serving`, which accepts both bundle shapes registered as
`format: 1` (KAN-32 multi-horizon `horizons`, KAN-15 single-horizon `horizon_hours` + `booster`)
and requires every booster's feature names to equal the manifest `feature_list` exactly and to be
produced by the shared feature builder. The job builds features as of the floored issue hour, so
live inputs are always at least 1 h old; a manifest declaring `assumed_reporting_lag_hours` below 1
is refused as train/serve skew. This currently refuses the KAN-15 `kan15-retrospective-24h` model
(trained at 0 h lag); it must be retrained at a 1 h lag before it can be the challenger. Challenger
rows are all-or-nothing per run. A missing model, a bad bundle or a feature mismatch fails
the run loudly (alert, exit 1) after the baselines are written. Unset = no challenger.

Owner commands (`DATABASE_URL` from the environment or `.env`, never printed):

```sh
uv run python -m pm25_forecast.registry_cli register --artifact model.json --manifest manifest.json
uv run python -m pm25_forecast.registry_cli check kan15-retrospective-24h
uv run python -m pm25_forecast.registry_cli champion
uv run python -m pm25_forecast.registry_cli promote kan15-retrospective-24h --confirm
```

`register` publishes and reads back without moving the champion. `check` loads a version exactly
as the predict step would. `promote` is the manual champion flip; it needs `--confirm` and refuses
a version the serving loader rejects. The hourly baselines do not read the champion pointer yet;
automatic champion/challenger selection is KAN-23.

Layout: `src/pm25_forecast/{ingest,features,models,storage,monitoring}` and `settings.py`
(environment-driven configuration).

## Branch and PR rules

- One Jira ticket = one branch (`kan-<n>/<short-name>`) = one pull request.
- The PR description lists the acceptance criteria met.
- Nobody pushes to or merges into `main` except the owner, who reviews and merges.
- Commit messages: `KAN-<n>: <summary>`.
- No secrets and no data dumps in git.

## Time

All timestamps are UTC, in code, storage and logs.

## Data sources and attribution

- Weather and CAMS air-quality forecasts: [Open-Meteo](https://open-meteo.com/), licensed CC BY 4.0 (free tier is non-commercial use).
  Air-quality data contains modified Copernicus Atmosphere Monitoring Service (CAMS) information.
- PM2.5 station observations: Air4Thai, Pollution Control Department (PCD), Thailand.
- Historical backfill and coverage metrics: [OpenAQ](https://openaq.org/).

## Air4Thai TLS (KAN-35)

If Air4Thai fails with `CERTIFICATE_VERIFY_FAILED`, re-run the `api-probe` workflow (Actions tab,
"Run workflow") and read the "Air4Thai TLS diagnosis" section of the job summary or the
`probe-report` artifact. See `certs/README.md` for what to do with the result and
`AIR4THAI_CA_BUNDLE` in `.env.example`.

OpenAQ 730-day history for the five project stations (KAN-49):
`uv run python -m pm25_forecast.openaq_backfill`. See the
[operator guide](docs/openaq-backfill.md) for configuration, restart behavior, source precedence
and post-merge owner actions.
