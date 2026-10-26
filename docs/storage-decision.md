# KAN-17: Storage decision and tables

## Options

Free-tier figures below are unverified: the agent could not confirm they were fetched from the official pages. Owner must re-check them before signing up.

| | Hosted Postgres: Neon | Hosted Postgres: Supabase | Object storage: Cloudflare R2 |
|---|---|---|---|
| Free allowance | 1 GB storage per project, 100 CU-hours per project, no card, permanent plan | 500 MB database, 2 active projects | 10 GB-month, 1M Class A and 10M Class B ops per month, free egress |
| Idle behaviour | Compute suspends after 5 min idle and wakes on connect; no data deleted at limits | Project paused after 1 week of inactivity | None |
| Upsert on natural keys | Native (`INSERT ... ON CONFLICT`) | Native | None: read-modify-write of Parquet objects, races between hourly jobs |
| Fit for hourly CI job | Good (cold-start of about a second is acceptable) | Pause risk if the pipeline stalls for a week | Needs object layout and compaction logic |

Sources: https://neon.com/pricing, https://supabase.com/pricing, https://developers.cloudflare.com/r2/pricing/

## Recommendation

Neon free-tier Postgres. Idempotent upserts on natural keys are a single SQL statement, runners stay stateless, and nothing is committed to git. Supabase's one-week pause is a real risk for a portfolio project that may sit idle. R2 would force us to hand-build upsert and dedup on top of files.

Estimated volume is corrected in Open decisions below: the earlier 'well under 100 MB per year' figure was wrong once forecast vintages are counted.

## Model binary

Stored as a `BYTEA` row in the `models` table (keyed by `model_version`, with a sha256). Models are small (LightGBM/sklearn, a few MB), the hourly job already has a DB connection, and this avoids a second provider and credentials. Move to R2 only if a model exceeds about 50 MB. The load path must verify the stored `sha256` against the artifact before use, and the writer must keep only the last N model versions (N to be set in the model-registry ticket) so artifacts do not eat the free quota.

## Keys and rules

- All timestamps are `timestamptz` and written as timezone-aware UTC. Naive values are rejected by `sql.write_rows`, which calls `require_utc` before executing (KAN-18+ writers must go through it).
- `observations`: natural key `(station_id, ts)`. Re-ingest and backfill overlaps upsert (last write wins), except that a NULL `pm25` never overwrites an existing value; `ingested_at` is refreshed on update.
- `features`: key `(station_id, ts, feature_version)`.
- `predictions`: key `(station_id, issue_time, valid_time, model_version)`. Append-only forecast vintage: a new `issue_time` is a new row, never an overwrite of an older forecast. Re-running the same issue is idempotent (`DO NOTHING`, so the first-written vintage is kept; `write_rows` logs inserted vs ignored counts). Rows carry `run_id`.
- `scores`: key `(station_id, issue_time, valid_time, model_version)`; upserted, since scoring is recomputed when late observations arrive.
- DDL: `src/pm25_forecast/storage/schema.sql`. Upsert statements: `src/pm25_forecast/storage/sql.py`.

## DB-down behaviour

The predict step runs inside `run_predict_step` (`storage/guard.py`). If the database is unreachable there is no stale-data fallback; the run is skipped, an alert is sent through the configured webhook (`ALERT_WEBHOOK_URL`), and `RunSkipped` is returned. `run_entrypoint` wraps the step, maps driver-level connection errors (configurable tuple, default `OSError`/`ConnectionError`) to `DatabaseUnavailableError`, logs and still skips if `alert()` itself fails, and returns an exit code. The caller must `sys.exit` with a non-zero code on a skipped run so CI shows red. It never reads cached or older data to produce a forecast.

## Owner actions

Create the Neon project and set `DATABASE_URL` (and `ALERT_WEBHOOK_URL`) as secrets; apply `schema.sql` once.

## Open decisions (Boss)

(a) DB driver and placeholder style: psycopg (`%(name)s`) vs SQLAlchemy `text` (`:name`). The statements in `sql.py` currently use `:name`; no driver is chosen or added. `run_entrypoint`'s error tuple should be set to the chosen driver's connection errors.

(b) Storage estimate, corrected: predictions are about 10 stations x 8,760 issues x 24 horizons = about 2.1M rows per year, roughly 0.3-0.4 GB per year, plus scores and the features jsonb. That would hit Neon's 1 GB free limit in about 12-18 months. Options: retention/rollup (raw vintages kept 90 days, scores aggregated), slimmer rows (a `runs` table holding run-level fields so prediction rows shrink), or a paid tier.
