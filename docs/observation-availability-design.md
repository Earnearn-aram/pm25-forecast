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
3. `committed_at`: may be stored only when verified from durable database evidence; `now()`/`CURRENT_TIMESTAMP` is transaction-start time, not commit time.
4. `available_at`: derived per feature origin from a trusted visibility snapshot, not assigned as a timeless event timestamp. Missing or ambiguous snapshot proof fails closed.

Never substitute observation time, HTTP receipt, mutable `ingested_at`, a user-supplied cutoff, or a later backfill time for a verified availability boundary.

## Proposed event and serving model

Keep `observations` as the current-value serving table. Add a separately reviewed append-only per-source event history. Capture each valid source event, including a lower-precedence OpenAQ event that is not allowed to replace Air4Thai serving state. Record a stable event ID, natural key `(station_id, observed_at)`, source, value including explicit NULL, source payload/run provenance, `received_at`, and database transaction identity. Store schema/version metadata and enforce append-only access for the writer role; readers get read-only access. Do not put an unverified availability timestamp on an event.

The serving table continues to apply the existing precedence and NULL rules. Per natural key, writers serialize event comparison and serving-state update. Suppress only a retry that repeats the immediately previous state for that same source and key. Do not use a global unique `(key, value)` rule: A → B → A is a real correction and must retain all three transitions. Idempotency keys distinguish a retry from a repeated value in a later collection. Event capture and serving update must commit atomically; a rolled-back transaction leaves neither visible.

### Visibility proof (implementation blocker; no rollout protocol approved)

Commit time alone does not prove that a particular feature reader could see a row at its issue origin. A reader may have opened its database snapshot before the event committed, and a proof record may be written later. Using only the event commit time would let retrospective replay include data that a proof-gated reader could not use at that origin. Do not resolve this by timestamping the later proof marker or by rounding timestamps.

The required contract is a durable **feature-read visibility snapshot**: for each forecast issue, establish a database snapshot (for example, PostgreSQL's `pg_current_snapshot()`) in a short read-only repeatable-read transaction before feature reads; perform all feature queries under that same transaction snapshot. Preserve that exact snapshot token and transaction/feature provenance with the forecast record. Event records carry their creating transaction ID. A replay may use an event only if the stored issue snapshot proves that event transaction was visible; it then applies the recorded source precedence and correction order. The later persistence time of the forecast record does not replace or backdate the captured snapshot. A missing snapshot, missing event provenance, or ambiguous visibility blocks the affected station/horizon. If a newer effective-state change is unproven, replay must not fall back to an older value and pretend that was still current.

### Nominal issue hour versus actual feature time

Current prediction code floors run time to a UTC `issue_time` hour, while database reads happen later at the actual workflow time. A snapshot taken at 10:07 cannot establish which corrected value was visible at the nominal 10:00 issue boundary; it may include a 10:05 commit. Treating that later snapshot as an as-of-10:00 snapshot would leak future-to-origin state. A future implementation must choose and test one explicit contract before any training reader consumes the snapshots:

- Keep the nominal hour as the origin and capture a trusted visibility boundary at or before that hour; if it is missed, mark that origin unusable. Feature versions committed after the nominal boundary must be excluded.
- Or make the actual feature snapshot/issuance time the canonical forecast origin, persist it separately from the nominal hourly bucket, and align target lead times, maturity cutoffs, and scoring with that actual origin. This is a forecast/scoring contract change requiring its own reviewed scope.

Until one contract is selected and implemented consistently by prediction, training, and scoring, a later snapshot cannot be used as proof for the floored nominal issue time. The existing hourly issue key alone is insufficient provenance; KAN-118/KAN-126 stay blocked.

For simulated historical origins that have no saved feature snapshot, availability cannot be reconstructed exactly from mutable rows or from receipt timestamps. Start producing immutable origin snapshots prospectively at the actual forecast/training-example origin; no pre-capture backfill may fabricate them. Only examples with retained, verifiable snapshots are eligible. Before implementation, prove that the supported PostgreSQL service can export/capture, persist, retain, and later evaluate snapshot visibility safely across transaction-ID wraparound, vacuum, failover, and restore. PostgreSQL commit-timestamp tracking may help diagnostics but is optional and its records are routinely removed during vacuum; it is not a substitute for the issue snapshot. Until the snapshot contract and retention behavior pass integration tests, the design remains blocked and KAN-118/KAN-126 stay fail-closed.

## As-of replay and leakage rules

At an issue origin `T`, replay only source events proven visible in that issue's recorded feature snapshot; then apply the same documented source precedence and correction rules that serving used. Do not first collapse to the latest current row. Labels/outcomes must separately have verified availability by the training/evaluation cutoff. Any missing snapshot, provenance, version, or trustworthy boundary excludes the affected key/example and increments an explicit blocked/unusable count. An unproven newer effective-state event invalidates the key at/after its possible arrival; do not fall back to a stale prior value. Training remains blocked unless every required horizon has enough usable examples under these rules.

Historical versions before capture rollout are unknown. Current rows are not a historical event stream and must not be assigned fabricated availability times. The first trustworthy replay window begins only after the verified cutover boundary and enough outcomes mature.

## Migration, overlap, and rollback plan

1. Implement behind a feature flag in a separate ticket/PR; deploy schema and indexes without changing readers.
2. Pause or fence all observation writers. Record a coordinated cutover boundary. While writers remain fenced, seed one baseline event per current row, with `history_complete=false` / `baseline_only=true`, no claim about prior versions or earlier arrival time. Baselines can be used only by feature snapshots captured after the verified cutover. Verify counts and a deterministic checksum.
3. Enable the dual-write transaction and issue-snapshot capture; verify all writers and snapshot recording are healthy before unfencing. Keep a durable run ID and resume cursor so restart cannot silently skip a range. A delayed/crashed snapshot write is not backdated: the affected origin remains unusable unless its exact transaction snapshot was durably preserved.
4. Shadow-read history alongside current serving state. Compare values, NULL handling, and precedence; monitor missing snapshots, unresolved keys, lag, duplicates, and per-source counts. Do not use shadow results for training until the acceptance checks pass.
5. Rollback by disabling history reads and returning to the existing serving table; retain append-only events and snapshots for diagnosis. Do not delete or rewrite history to roll back. Keep retraining fail-closed until a reviewed reader/export demonstrates verified as-of semantics.

If writers cannot be fenced for seeding, do not run an uncoordinated snapshot. First implement a tested overlap protocol (for example, change capture with a durable cursor) that proves no write is lost or double-applied; otherwise abort migration.

## Retention and operations proposal

Retain immutable source events, issue snapshots, and provenance for at least the maximum training lookback plus the full evaluation/maturity window and an agreed audit margin. Before rollout, measure rows/day, indexes, WAL and backup growth, and storage cost; set an owner-approved byte budget and alert threshold. Never prune events or snapshots inside any model's required lookback or while a model/run references them. If budget is exceeded, stop new training/readiness claims first; do not silently delete required history. Keep the serving table's existing retention separate. Record access-control, backup/restore, schema version, producer version, collection run ID, and migration run ID.

The exact retention duration, cost budget, snapshot retention horizon, database-service support, and operational owner are rollout decisions and must be approved before production implementation. Until then, the safe default is no history pruning and no training use.

## Acceptance tests required before implementation rollout

Use synthetic fixtures for deterministic tests and PostgreSQL integration tests for transaction/locking behavior. CI must not call live APIs or production databases.

- Identical retry is idempotent; changed value, changed source, and A → B → A preserve the correct event sequence.
- Air4Thai precedence, rejected lower-priority OpenAQ capture, explicit NULL, and NULL-not-overwrite behavior match current serving semantics.
- Concurrent writers on the same key serialize; retries cannot erase or reorder a correction.
- Event plus serving update roll back atomically; delayed/crashed snapshot persistence leaves the affected origin unusable, with no retrospective replacement snapshot.
- Feature reads use one recorded repeatable-read snapshot; an event committed after that snapshot is excluded, even if it commits before forecast output is written.
- An event committed after a nominal hour but before a later feature snapshot is never treated as available at the earlier nominal origin; tests cover both contract options and the +6/+12/+24h target and maturity implications.
- Persisting a previously captured snapshot after an issue boundary preserves that exact snapshot; it must never be replaced by a later snapshot. If the process crashes before that token is durably associated with the forecast, the origin stays unusable. Missing/ambiguous transaction visibility and newer unproven corrections fail closed without stale fallback.
- Backfill is eligible only when the exact issue snapshot proves visibility; observation and receipt times never change eligibility.
- Migration seed/writer overlap cannot lose a change; restart/resume is idempotent; rollback disables reads without deleting captures.
- As-of replay returns the version visible at each origin, not the current value; labels after the cutoff and rows with missing provenance are rejected.
- Integration checks verify schema constraints, grants, indexes, lock behavior, and restore/replay on the supported PostgreSQL service before rollout.

## Separate implementation scope and owner decisions

This audit/design does not implement schema, collector writes, export/training readers, retention jobs, production migration, or readiness-unblock controls. Track those as a distinct implementation ticket with one branch/PR, tech-lead review, tested rollback, and explicit owner approval for rollout. The owner must decide the retention budget/duration and operational owner; implementation must validate durable issue-snapshot persistence and later transaction-visibility evaluation against the hosted PostgreSQL service. Until those conditions are met, KAN-115 is a design deliverable only and KAN-118/KAN-126 must remain fail-closed.
