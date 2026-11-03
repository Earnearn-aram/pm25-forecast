# Training data readiness (KAN-118)

Use this read-only report before starting a retrain to see whether the current trainer has enough
time-safe examples. It does not train, score a model or evaluate the locked experiment holdout.
Choose the exclusive cutoff in advance. The operator must establish the exclusive boundary before
any locked holdout in the input and supply it as `--safe-before`; this command refuses a cutoff
after that boundary before opening the file. This is an operator assertion, not automatic provenance
verification. Do not use an export if its provenance or holdout boundary is unknown. Both timestamps
must include a timezone; all report timestamps are UTC.

```bash
uv run python -m pm25_forecast.training_readiness \
  --export /path/to/local-history.duckdb --cutoff '<UTC_CUTOFF>' \
  --safe-before '<VERIFIED_NON_HOLDOUT_BOUNDARY>'
```

The input is an existing local DuckDB history export, not a database URL. The command makes a
temporary training-only copy using the same exclusive timestamp and availability filter as retrain;
it never changes the input and removes the copy on exit. The JSON output contains counts only:

- `usable_mature_observations_before_cutoff`: usable labels in that filtered export;
- `usable_examples` per horizon: examples constructed by the production sample builder, using
  only features available at each simulated issue time;
- `training_after_embargo` and `validation_examples`: the trainer's chronological split. They
  are null when there are too few usable examples to attempt a split, rather than fake zeros.
- `status`: `count_ready` only if all three horizons meet the trainer's minimum both before and
  after embargo. It is a data-count result, not proof of safe availability or model quality.

The live observations table is mutable. If a value is corrected, its earlier value and arrival
time are not retained; current `ingested_at` is not proof of what was known earlier, nor of exact
transaction commit visibility. A late backfill is not made available retroactively. See KAN-115
for the capture/replay design and its migration and retention decisions. Do not relax the
availability cutoff or invent history to make the report ready.

The report requires no schema or production rollout. Existing training and alert behavior is
unchanged. Store any local exports outside version control; do not upload real data dumps. The
summary output does not include values or credentials. A ready count still requires the separately
documented evaluation and owner promotion policy.
