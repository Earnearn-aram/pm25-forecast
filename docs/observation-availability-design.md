# Observation availability history: audit and design

**Status: proposed design; not implemented.** This document records the read-only audit evidence available on 2026-11-02 and a fail-closed design for a separately reviewed implementation. It does not establish operational readiness, create historical captures, authorize training, or replace the retraining guard.

## Audit findings

The read-only audit recorded 66,902 non-null rows in `observations`; the earliest current `ingested_at` was `2026-10-28T08:18:21Z`. For the failed retraining cutoff `2026-10-26T04:00Z`, zero rows had both `ts` and `ingested_at` before the cutoff. The export currently exposes `ingested_at` as `available_at`.

The serving table is keyed by `(station_id, ts)`. Re-ingest/backfill uses an upsert: a NULL PM2.5 value does not overwrite a non-NULL value; source precedence is applied by the current writer; `ingested_at` is retained for an identical value/source and refreshed when the value or source changes. The table therefore holds current serving state, not an immutable sequence of versions. A changed value/source overwrites the prior state and its prior `ingested_at`. Existing rows cannot prove earlier availability, and a backfill retrieval time cannot be projected into the past.

These are point-in-time audit observations, not claims about the current live database. No production query or write is part of this design PR.

## Semantics that must stay distinct

For each source event, preserve separate fields for:

1. `observed_at`: the source's event/measurement time.
2. `received_at`: when the collector received and validated the payload (useful operational provenance, not proof of DB visibility).
3. `committed_at`: when the event transaction committed (not `now()`/`CURRENT_TIMESTAMP`, which PostgreSQL defines as transaction start time).
4. `available_at`: the conservative time at which the training/serving reader is allowed to use the event. This must be no earlier than verified commit visibility and must fail closed when proof is missing.

Never substitute observation time, HTTP receipt, mutable `ingested_at`, a user-supplied cutoff, or a later backfill time for a verified availability boundary.

## Proposed event and serving model

Keep `observations` as the current-value serving table. Add a separately reviewed append-only per-source event history. Capture each valid source event, including a lower-precedence OpenAQ event that is not allowed to replace Air4Thai serving state. Record a stable event ID, natural key `(station_id, observed_at)`, source, value including explicit NULL, source payload/run provenance, `received_at`, database transaction identity, and the verified availability fields. Store schema/version metadata and enforce append-only access for the writer role; readers get read-only access.

The serving table continues to apply the existing precedence and NULL rules. Per natural key, writers serialize event comparison and serving-state update. Suppress only a retry that repeats the immediately previous state for that same source and key. Do not use a global unique `(key, value)` rule: A → B → A is a real correction and must retain all three transitions. Idempotency keys distinguish a retry from a repeated value in a later collection. Event capture and serving update must commit atomically; a rolled-back transaction leaves neither visible.

### Availability protocol (implementation must prove this)

An event is not eligible merely because its insert row contains a timestamp. The initial implementation should use a two-step, conservative publication protocol:

1. In one short transaction, append the event and update the serving row under the per-key serialization lock. Commit. A database-generated transaction-start timestamp may be retained for diagnostics only.
2. Only after the writer receives successful commit acknowledgement, obtain a trusted UTC time from the database in a new short operation. Add the configured maximum clock uncertainty, then round `available_at` **up** to the next UTC hour boundary (the forecasting issue grid). This deliberately delays eligibility; it cannot move an event earlier.
3. Persist the publication marker in a separate transaction keyed to the event ID. Readers ignore every event without a valid marker. If the process crashes after event commit and before marker commit, the event remains ineligible. A reconciler may mark it using recovery time plus the same uncertainty/rounding rule, never the earlier insert or receive time.
4. If clocks are unsynchronized, the uncertainty bound is unknown, the marker is missing/duplicated, or event and serving state disagree, reject that event for training and report a readiness error. Do not infer a boundary from operator input.

The marker is evidence of post-commit acknowledgement, not a claim to know the exact physical commit instant. Rounding and the uncertainty allowance make the boundary conservative for hourly issue times. This protocol and its database/Neon behavior must be validated before use; if post-commit acknowledgement time cannot be trusted and bounded, the system remains blocked and the design must use a separately proven watermark mechanism. PostgreSQL `track_commit_timestamp` is not the default solution: it is optional and commit timestamp records can be removed during vacuum.

## As-of replay and leakage rules

At a simulated feature issue time `T`, select only source events whose verified `available_at <= T`; then apply the same documented source precedence and correction rules that serving used. Do not first collapse to the latest current row. Labels/outcomes must separately satisfy `label_available_at <= training/evaluation cutoff`. Any missing provenance, marker, version, or trustworthy time excludes the row and increments an explicit blocked/unusable count. Training remains blocked unless every required horizon has enough usable examples under these rules.

Historical versions before capture rollout are unknown. Current rows are not a historical event stream and must not be assigned fabricated availability times. The first trustworthy replay window begins only after the verified cutover boundary and enough outcomes mature.

## Migration, overlap, and rollback plan

1. Implement behind a feature flag in a separate ticket/PR; deploy schema and indexes without changing readers.
2. Pause or fence all observation writers. Record a coordinated cutover boundary. While writers remain fenced, seed one baseline event per current row, with `history_complete=false` / `baseline_only=true`, no claim about prior versions, and availability no earlier than the conservative post-commit boundary. Verify counts and a deterministic checksum.
3. Enable the dual-write transaction and post-commit marker protocol; verify all writers are upgraded before unfencing. Keep a durable run ID and resume cursor so restart cannot silently skip a range. Any event lacking its marker is excluded and reconciled conservatively.
4. Shadow-read history alongside current serving state. Compare values, NULL handling, and precedence; monitor unmarked events, lag, duplicates, and per-source counts. Do not use shadow results for training until the acceptance checks pass.
5. Rollback by disabling history reads and returning to the existing serving table; retain append-only events and markers for diagnosis. Do not delete or rewrite history to roll back. Keep retraining fail-closed until a reviewed reader/export demonstrates verified as-of semantics.

If writers cannot be fenced for seeding, do not run an uncoordinated snapshot. First implement a tested overlap protocol (for example, change capture with a durable cursor) that proves no write is lost or double-applied; otherwise abort migration.

## Retention and operations proposal

Retain immutable source events, publication markers, and provenance for at least the maximum training lookback plus the full evaluation/maturity window and an agreed audit margin. Before rollout, measure rows/day, indexes, WAL and backup growth, and storage cost; set an owner-approved byte budget and alert threshold. Never prune events inside any model's required lookback or while a model/run references them. If budget is exceeded, stop new training/readiness claims first; do not silently delete required history. Keep the serving table's existing retention separate. Record access-control, backup/restore, schema version, producer version, collection run ID, and migration run ID.

The exact retention duration, cost budget, clock uncertainty bound, and operational owner are rollout decisions and must be approved before production implementation. Until then, the safe default is no history pruning and no training use.

## Acceptance tests required before implementation rollout

Use synthetic fixtures for deterministic tests and PostgreSQL integration tests for transaction/locking behavior. CI must not call live APIs or production databases.

- Identical retry is idempotent; changed value, changed source, and A → B → A preserve the correct event sequence.
- Air4Thai precedence, rejected lower-priority OpenAQ capture, explicit NULL, and NULL-not-overwrite behavior match current serving semantics.
- Concurrent writers on the same key serialize; retries cannot erase or reorder a correction.
- Event plus serving update roll back atomically; marker failure/crash leaves event unusable; recovery marks only at recovery time or later.
- A transaction spanning an issue boundary cannot appear available before its post-commit marker; exact-boundary ties use `<=` consistently after UTC hour-ceiling.
- Backfill is available only after its actual commit and publication marker; observation time never changes eligibility.
- Migration seed/writer overlap cannot lose a change; restart/resume is idempotent; rollback disables reads without deleting captures.
- As-of replay returns the version visible at each origin, not the current value; labels after the cutoff and rows with missing provenance are rejected.
- Integration checks verify schema constraints, grants, indexes, lock behavior, and restore/replay on the supported PostgreSQL service before rollout.

## Separate implementation scope and owner decisions

This audit/design does not implement schema, collector writes, export/training readers, retention jobs, production migration, or readiness-unblock controls. Track those as a distinct implementation ticket with one branch/PR, tech-lead review, tested rollback, and explicit owner approval for rollout. The owner must decide the retention budget/duration and operational owner; implementation must validate the post-commit marker clock/uncertainty protocol against the hosted PostgreSQL service. Until those conditions are met, KAN-115 is a design deliverable only and KAN-118/KAN-126 must remain fail-closed.
