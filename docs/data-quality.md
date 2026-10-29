# Data quality checks (KAN-21)

`python -m pm25_forecast.quality` runs after Score in the predict workflow. It reads the database
only and writes one row per check to `data_quality_checks`. All thresholds are named constants at
the top of `src/pm25_forecast/quality.py`; this page mirrors them. Change both together.

Timestamps are UTC. "Last 24 h" means the 24 whole hours before the current hour.

## Checks

| Check | Source (`source` column) | Meaning | Warn | Fail |
|---|---|---|---|---|
| `freshness` | `air4thai` | Age of the newest usable observation, per station | > 3 h | > 6 h, or none |
| `freshness` | `openaq` | Same; only stations that have OpenAQ data. OpenAQ is a gap filler, so stale is normal | > 24 h | never |
| `freshness` | `cams_global` | Age of the newest CAMS vintage (`issue_time`), per expected station | > 24 h | > 48 h, or none |
| `freshness` | `persistence-24h` and the challenger (`CHALLENGER_MODEL_VERSION`) | Age of the newest forecast `issue_time`, per expected station | > 3 h | > 6 h, or none |
| `missing_hours` | `air4thai` | Hours in the last 24 without a usable reading, per station | >= 3 | never |
| `out_of_range` | any observation source | Readings in the last 24 h below 0, above 1000, or non-finite | never | any (> 0) |
| `expected_station` | `air4thai` | A station in `STATION_LOCATIONS` with fewer than 1 row in 24 h | < 1 row | never |
| `null_rate` | any observation source | Share of rows in the last 24 h with NULL pm25 | > 0.2 | never |

"Usable" = not NULL, finite, and between 0 and 1000 inclusive.

## Warn vs fail, and alerting

* `ok`: within limits. `warn`: recorded only. `fail`: recorded and alerted.
* Only `freshness` and `out_of_range` can fail. Shape changes (`expected_station`, `null_rate`)
  and `missing_hours` are warn-only.
* `missing_hours` is never alerted here: the Collector already alerts on Air4Thai gaps
  (`find_gaps`), so a second alert would repeat it.
* One alert per run lists the `fail` items only (at most 15, then "+N more"). It contains source,
  station id, check name and numbers; no URLs or secrets.
* A run that finds failing data exits 0 (the alert is the signal; a red step would also trigger
  the workflow's failure alert). Exit 1 = the run itself failed or the DB is down (skipped and
  alerted); exit 2 = configuration error (`DATABASE_URL` or `STATION_LOCATIONS` missing or
  malformed).
* Predict also runs after every Collector run, so up to 3 quality runs per hour can repeat the same
  alert while a check keeps failing. Rows are upserted, so the table is not duplicated.

## Table `data_quality_checks`

| Column | Notes |
|---|---|
| `checked_at` | timestamptz, the run time floored to the hour |
| `source` | `air4thai`, `openaq`, `cams_global`, `persistence-24h` or a challenger model version |
| `station_id` | nullable (NULL = source-wide check) |
| `check_name` | `freshness`, `missing_hours`, `out_of_range`, `expected_station`, `null_rate` |
| `status` | `ok`, `warn` or `fail` |
| `value` | freshness: age in hours; `missing_hours`: hours; `out_of_range`: readings; `expected_station`: rows; `null_rate`: fraction. NULL when nothing was found |
| `threshold` | the limit that decided the status (fail limit for fail, warn limit otherwise) |
| `detail` | short human-readable text |

Unique on (`checked_at`, `source`, `COALESCE(station_id, '')`, `check_name`); a re-run in the same
hour updates the row. History for a dashboard:

```sql
SELECT checked_at, station_id, status, value
FROM data_quality_checks
WHERE source = 'air4thai' AND check_name = 'freshness'
ORDER BY checked_at;
```

Existing databases: apply `src/pm25_forecast/storage/migrations/003_data_quality.sql` once.
