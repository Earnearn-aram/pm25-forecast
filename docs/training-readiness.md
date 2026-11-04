# Training data readiness (KAN-118)

The readiness command is a diagnostic report. It never trains, promotes, or returns a ready state
until a trusted, versioned provenance source binds an export to a verified non-holdout boundary and
immutable feature-availability history.

## Report a cutoff

The report requires an exclusive, timezone-aware UTC cutoff:

```bash
uv run python -m pm25_forecast.training_readiness \
  --cutoff '2026-01-03T00:00:00Z'
```

Without an export, the command prints `blocked_unverified_export_boundary`, confirms
`export_opened: false`, and exits with code 1. A cutoff supplied by an operator does not prove the
holdout boundary.

## Inspect an export locally

For reproducible diagnostics, an operator may supply a local DuckDB export:

```bash
uv run python -m pm25_forecast.training_readiness \
  --cutoff '2026-01-03T00:00:00Z' \
  --export data/history.duckdb
```

The export is opened read-only. The JSON includes its SHA-256, per-horizon export-level usable
examples, the minimum example count, and overall outcome counts that appear mature or late relative
to the exclusive cutoff. Maturity counts are reported once because outcome availability does not
depend on forecast horizon. These values follow the export's `available_at` fields and the shared
training sample builder. They are diagnostic counts only: because legacy `available_at` can derive from a
mutable current-state `ingested_at`, the report still returns `ready: false`,
`availability_history_verified: false`, and exit code 1 even when all thresholds are met. Do not
use these counts as evidence of historical feature visibility, model quality, or promotion
eligibility.

The report explicitly says correction history is unavailable. A current-state export cannot
reconstruct overwritten values or their earlier arrival times. Late and corrected synthetic
fixtures verify that outcomes arriving after the cutoff are excluded from mature counts; this does
not recreate a real correction timeline.

## Retention and rollout

Preserve immutable timestamped source captures and issue feature snapshots for at least the full
training lookback, outcome-maturity window, and an approved audit margin. Before production use,
prove the database visibility-snapshot contract and replay behavior, measure storage/WAL/backup
growth, and get an owner-approved retention budget and operational owner. Roll out schema/writes in
shadow mode first, verify counts and checksums while writers are controlled, then enable a reader
only after acceptance checks pass. Rollback disables history reads and retains the append-only
records; it does not fabricate or delete history.

See [observation availability history](observation-availability-design.md) for the detailed
semantics, acceptance checks, and migration proposal. Existing training and alert behavior is
unchanged. This report does not authorize retraining or promotion.
