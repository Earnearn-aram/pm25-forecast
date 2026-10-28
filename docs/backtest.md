# Rolling-origin backtest (KAN-14)

Run `uv run python -m pm25_forecast.backtest /private/path/input.json` to print an aggregate JSON
report. Keep input outside git: never commit production observations or forecast exports.
The runner is offline and read-only. Exit 2 means there are no complete scoring pairs; exit 0
means descriptive metrics exist, **not** that a challenger has beaten CAMS. Inspect each
`beats_cams` decision separately.

## Input contract

The JSON object has `start`, `end`, `as_of` (ISO 8601 timestamps with timezones), `stations`
(a fixed cohort of station IDs), `aqi_breakpoints` (strictly increasing concentration boundaries
in µg/m³), `observations`, `vintages`, and optional `challenger_models` (archived model names).
Each observation has `station_id`, `ts`, `value`, `available_at`; each vintage has `station_id`,
`issue_time`, `valid_time`, `value`, `available_at`, and `model` (default `cams`).

- `start` and `end` are UTC hour boundaries, with hourly prediction origins in `[start,end)`.
  `as_of` is the evaluation cutoff: targets and their observations must have matured by then.
- `available_at` is when the exact value was actually available locally, **not** its target time,
  a reconstructed model cycle, or the date a historical download describes. For stored CAMS
  `cams_global` rows map `model` to `cams`; conservatively use `created_at` as availability and
  preserve collector `issue_time`. Archived challenger predictions retain their model names.
- Observations used for fitting/features must have both `ts <= prediction_time` and
  `available_at <= prediction_time`. Forecast features must have both `issue_time` and
  `available_at <= prediction_time`. Labels are available to scoring only at `as_of`.
- All datetimes must be timezone-aware; origins and target grouping are normalized to UTC.
  Values must be finite and nonnegative. Missing values should be omitted, never filled in.
- A duplicate observation key is rejected: this input contract accepts one immutable value per
  station/hour. A duplicate forecast natural key (station/issue/target/model) is rejected too.
  Preserve first writes and append each new issue. Do not collapse vintage history to the newest
  forecast or overwrite a vintage with a revised forecast. Existing `INSERT_PREDICTION` already
  uses `ON CONFLICT DO NOTHING`; this harness never changes that table.

For programmatic training, call `evaluate(..., challengers={"model-version": callback})`.
The callback receives a `Snapshot` and horizon, and must return a finite nonnegative prediction
or `None` for missing inputs. It is invoked separately at each origin/target, allowing an
expanding-window fit with mature labels. Use the shared `features/` module when implemented
(KAN-31); do not duplicate feature transformations here. Do not close over future data or a
model fitted on the evaluation period. The harness supplies only eligible inputs; arbitrary
callback code remains responsible for respecting that interface. No ML dependency is added.

## Scores and comparison rule

Horizons default to +6, +12, +24 hours. Persistence-24h repeats the newest observation no more
than 3 hours old, using the existing live baseline implementation. Seasonal-naive repeats the
previous day's exact target hour (24-hour cycling for longer horizons); gaps stay gaps. CAMS
uses the latest available vintage for the exact target hour. `stored_predictor` replays a stored
challenger with the same availability gate.

Every reported model uses the common intersection of station, prediction origin, target hour,
and horizon across **all requested models** and all three baselines. Adding a sparse challenger
can reduce every model's sample count. `eligible_actual_pairs`, `paired_rows`, and per-model
`missing` counts expose attrition (missing counters can overlap). There is no unpaired
comparison, no CAMS reconstruction, and no claim that missing hours perform like observed hours.

Reports contain MAE, RMSE and category accuracy per horizon, plus MAE by station, target UTC month,
and horizon. Category accuracy uses caller-supplied concentration bands: values equal to a boundary
remain in the lower band. Supply and record the chosen AQI standard's boundaries. This hourly
concentration-category proxy does not implement official daily AQI averaging or interpolation.

A challenger **beats CAMS** only when all these prespecified conditions hold:

1. The origin window is exactly 28 days starting at UTC midnight, with all +24h targets mature.
2. At +24h, every station in the declared cohort has at least 22 shared hourly pairs on each
   of the 28 origin days (at least 91.7% daily coverage). Missing a station/day blocks the claim.
3. Paired absolute-error differences (`challenger - CAMS`) have a negative overall mean and a
   strictly negative upper one-sided 95% bootstrap confidence bound. Ties do not win.

The seeded bootstrap resamples whole UTC origin days, preserving station/hour dependence inside
each daily block; 2,000 replicates, seed 14. It reports mean MAE delta, upper bound, days and pairs.
This is a prespecified practical uncertainty rule, not protection against all multi-day serial
dependence, model-selection bias, multiple challenger testing, or seasonal distribution shift.
Choose the challenger before evaluating a fresh window; avoid repeated tuning on this window.
`insufficient_data` and `beats_cams: null` are distinct from a valid negative result (`false`).

## Current historical-data limitation

Inspection of `predict.py` records that stored CAMS collection began on 2026-10-27. Live vintages
cannot establish CAMS performance at earlier origins. The repository does not include a historical
previous-runs CAMS archive; a newly downloaded current forecast is not a substitute. Collect a
full 28-day window of real vintages and wait for the +24h labels to mature before claiming a win.
The harness intentionally publishes no production scores or fabricated historical CAMS results.

The live `observations` table upserts values and resets `ingested_at`. It does not preserve original
observation revisions or their availability. Exporting the current table alone therefore cannot
prove a historical training snapshot was available at the time. Use immutable, timestamped input
captures for rigorous replay; treat current-table `ingested_at` conservatively (which may exclude
historical inputs), and disclose that retrospective original revisions cannot be recovered.
No production database was queried as part of this implementation; these are inspected code/schema
constraints, not a claim about measured live coverage. Expanding ingest/archive coverage is separate
work; missing archives remain visible as an insufficient-data decision here.
