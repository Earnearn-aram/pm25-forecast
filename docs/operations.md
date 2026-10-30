# Operations and component details

Moved from the README Quickstart. Commands and behaviour, component by component.


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
`STATION_LOCATIONS` (see `.env.example`). The workflow is `.github/workflows/collector.yml` (started by the external timer, see Architecture).

Alerts and heartbeat (KAN-19): everything goes to `ALERT_WEBHOOK_URL` (Discord; the
body carries both `text` and `content`). Without it, alerts are logged only, with a warning.
`python -m pm25_forecast.monitoring.notify failure` is the workflows' `if: failure()` step;
`python -m pm25_forecast.monitoring.heartbeat` posts a daily one-line summary
(`.github/workflows/heartbeat.yml`). Each run logs one `RUN_SUMMARY {json}` line; log timestamps
are UTC.

Backfill (KAN-12 part 1, KAN-34): `uv run python -m pm25_forecast.backfill [--lookback-days 120]
[--stations 59t,61t]` upserts whatever hourly history Air4Thai still holds (a rolling window of
about 3 months) into `observations`. Same settings as the collector; idempotent, so it runs daily
as gap recovery. Workflow: `.github/workflows/backfill.yml`.

Collector stores hourly PM2.5, not the 24 h average (KAN-74). Until KAN-74 the collector stored
`AQILast.PM25.value` from `getNewAQI_JSON` as `pm25`. On 2026-10-30 15:00 Bangkok that value equalled
the mean of the last 24 hourly history values for stations 02t, 59t and 61t (while the latest hours
were 15.5, 4.3 and 12.1), and PCD documents the headline as a 24 h average. The live `AQILast` block
has no separate 1 h field (only `color_id`, `aqi` and `value` per pollutant), so the collector now
reads the history endpoint (`getHistoryData`, `type=hr`) through the existing client (TLS bundle,
retries): the last 6 hours per station each run, one request per station, 2 s apart. Hours with no
value stay missing (no imputation); missing hours after the newest stored hour are logged and
alerted as gaps. A history failure for any station is a run failure (exit 1, alert) with only the
exception type in the message (no URL or secrets); CAMS is still collected. The 24 h headline is
not stored anywhere (it would need a new column or table, which this change does not add).

Why it matters: `UPSERT_OBSERVATION` lets the later same-source write win, so rows written by the
old collector are 24 h averages until the hourly backfill overwrites them, and every
`scores.pm25_obs` computed from them is skewed.

Fix-forward notes in the section above: a collector run that cannot reach Air4Thai stops at the
first transport failure (2 tries per request, then the remaining stations are skipped), writes the
stations it already has and fails the run; the deadline is checked before every station. Raw
history bodies are archived under `<RAW_ARCHIVE_DIR>/air4thai/air4thai-history-<station>-<UTC>.json.gz`
(git-ignored, public data only). Gap messages now say the next runs re-read the last 6 h and the
daily backfill covers older hours.

One-off repair (owner; needs `DATABASE_URL`, `STATION_LOCATIONS`, `AIR4THAI_CA_BUNDLE`). Order
matters: an upsert sets `ingested_at` to now, which is how the old rows are told apart.

1. Before anything: run the two queries in "Identify" below and keep the output.
2. Merge the KAN-74 PR and note the time `T` (UTC) from which the new collector is live (the
   first `RUN_SUMMARY` of the new code, or the merge time; the later one is safe).
3. Overwrite recent hours with hourly values by running the existing backfill over the whole
   retention window: GitHub, Actions, "Backfill", "Run workflow", `lookback_days` = `120`
   (draft: `docs/backfill-workflow.yml.txt`), or locally
   `AIR4THAI_CA_BUNDLE=certs/air4thai-ca.pem uv run python -m pm25_forecast.backfill --lookback-days 120`.
   Read the per-station log lines (rows upserted, empty or failed chunks).
4. Remove what the backfill could not overwrite ("Delete" below; only if step 3 had 0 failed chunks).
5. Re-score from 2026-10-27 (exact command below), then re-run drift and retrain only after 1-4.

Identify (read-only; run before the repair). Rows written by the old collector have `ingested_at`
within a few minutes (at most about 3 h with a late cron and PCD lag) of `ts`; the backfill writes
them hours or days later:

```sql
-- observations written at collection time since 2026-10-27 (likely 24 h averages)
SELECT station_id, count(*) AS rows, min(ts) AS first_ts, max(ts) AS last_ts
FROM observations
WHERE source = 'air4thai' AND ts >= '2026-10-27T00:00:00+07:00'
  AND ingested_at - ts < interval '3 hours'
GROUP BY station_id ORDER BY station_id;

-- scores computed in that period (re-score after the repair)
SELECT model_version, count(*) AS scores, min(valid_time) AS first_valid, max(valid_time) AS last_valid
FROM scores
WHERE valid_time >= '2026-10-27T00:00:00+07:00'
GROUP BY model_version ORDER BY model_version;
```

Delete (owner runs after step 3; replace `<T>` with the time from step 2, in UTC, e.g.
`2026-11-01T03:00:00+00:00`). HARD GATE: run it only if every station's step-3 backfill log line
shows 0 failed chunks; otherwise re-run step 3 first, because good hourly rows that a failed chunk
did not overwrite would match the filter and be deleted. Every row the backfill rewrote has
`ingested_at` after `<T>`, so what is still older than `<T>` and was written within 3 h of its
`ts` is an old collector row the backfill had no hourly value for.

1. Run the SELECT and compare its total with the "Identify" output.
2. Run the transaction below. Compare the observations `DELETE n` count with the SELECT total.
   `COMMIT` only if equal; otherwise `ROLLBACK` and investigate.

```sql
SELECT station_id, count(*), min(ts), max(ts) FROM observations
WHERE source = 'air4thai' AND ts >= '2026-10-27T00:00:00+07:00'
  AND ingested_at < '<T>' AND ingested_at - ts < interval '3 hours'
GROUP BY station_id;

BEGIN;
DELETE FROM observations
WHERE source = 'air4thai' AND ts >= '2026-10-27T00:00:00+07:00'
  AND ingested_at < '<T>' AND ingested_at - ts < interval '3 hours';
-- compare "DELETE n" with the SELECT total above, then decide:
-- scores whose actual no longer exists (the scorer never deletes a score on its own)
DELETE FROM scores s
WHERE s.valid_time >= '2026-10-27T00:00:00+07:00'
  AND NOT EXISTS (SELECT 1 FROM observations o
                  WHERE o.station_id = s.station_id AND o.ts = s.valid_time
                    AND o.pm25 IS NOT NULL);
-- equal counts: COMMIT;   otherwise: ROLLBACK;
```

Why delete and not impute or keep: a 24 h average is a different quantity (smoothed, lagged),
so a row holding it is wrong, not merely noisy; features, the persistence baseline, scores and
drift would all treat it as an hourly truth. A missing hour is handled everywhere (gaps are
reported, missing actuals are "waiting", features tolerate gaps) and imputing would invent data.
Deleted hours are not refetched because Air4Thai has none for them. A source-`openaq` row for the
same hour, if any, is untouched.

Re-score (after steps 3-4; reads and writes the database only, idempotent):

```sh
uv run python -m pm25_forecast.score --since 2026-10-27T00:00:00Z
```

`--since` takes an ISO datetime with a UTC offset and replaces the default 72 h rescore start for
that run (the metrics window is unchanged). The hourly predict workflow's own score step keeps the
default; run this once by hand.

Downstream effects while the 24 h values were in `observations` (2026-10-27 until the repair):
- Retrain: the first scheduled retrain is Mon 2026-11-02 03:17 UTC and has not run yet. Finish
  steps 1-4 before then (or disable it for the weekend) so it does not train on 24 h values.
- Drift: it runs daily on the last 7 days of observations, so since 2026-10-27 it has read some
  24 h-average rows; its results for those days may be false alarms or hide real shifts. Re-run it
  after the repair; treat `drift_checks` rows from 2026-10-27 until then as suspect.
- Scores: `persistence-24h` (and challenger) scores for valid_time from 2026-10-27 until the repair
  are suspect. The Identify query lists them; the re-score above recomputes the ones whose
  actual still exists, and the DELETE above removes the ones whose actual was deleted.

Predict (KAN-28): `uv run python -m pm25_forecast.predict` reads stored observations and CAMS rows
and appends `persistence-24h` forecasts (plus a `cams` pass-through of stored CAMS rows, not scored) for +6, +12 and +24 h per station to
`predictions` (idempotent per issue hour). Same settings as the collector; DB down means skip,
alert and exit non-zero. Baselines: `models/baselines.py`. Workflow: `.github/workflows/predict.yml`.

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
persistence baseline every hour (plus a CAMS pass-through that is stored but not scored).

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
