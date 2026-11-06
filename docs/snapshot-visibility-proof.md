# KAN-130: PostgreSQL snapshot visibility proof

The optional test `tests/test_snapshot_visibility_postgres.py` uses a dedicated local PostgreSQL
database and synthetic rows. It captures a top-level transaction ID (`xid8`) for each writer and a
`pg_current_snapshot()` token in one repeatable-read, read-only reader. It demonstrates that a
committed earlier write is visible, while a write still in progress when the snapshot was taken
and a later committed write are not visible to that reader, including after the reader finishes.
An aborted insert is absent from the committed table. `pg_visible_in_snapshot` is a snapshot
membership check; it is not proof that a transaction committed.

The test uses the same local-only DSN guard as the serving rehearsal: a `/tmp` Unix socket, a
database named `pm25_test_*`, and verification of the connected server and role before creating a
unique temporary test schema. To run, create a disposable local database as described in
`docs/lifecycle-rehearsal.md`, then set `PM25_TEST_POSTGRES_DSN` for that database and run:

```sh
uv run pytest -q tests/test_snapshot_visibility_postgres.py
```

The owner-created Neon `staging` branch reported PostgreSQL 18.6 and the availability of
`pg_visible_in_snapshot(xid8, pg_snapshot)` in a read-only capability check. That establishes only
function availability. This test has not been run against hosted staging and cannot be pointed at
it: the local target guard deliberately rejects remote DSNs. Hosted testing needs a separately
reviewed procedure that independently verifies the staging branch identity before writes and uses
dedicated staging credentials.

This proof does not establish that an actual prediction and all its inputs share a durable
snapshot, that source event transactions committed, or that snapshots remain interpretable after
restore, failover, transaction-ID reuse/wraparound or retention expiry. The owner has selected the
actual feature-read snapshot time as the future canonical forecast origin; this test does not
implement that contract. Persisted run-to-prediction lineage, all feature and label sources,
cluster identity/generation and restore behavior still need reviewed implementation and tests.
The KAN-126 retrain guard stays in place.
