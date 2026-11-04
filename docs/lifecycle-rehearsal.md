# KAN-122 local lifecycle rehearsal

This rehearsal connects the registry, real Predict runner, immutable serving ledger, Score runner,
and dashboard chart using synthetic models and observations. It is an integration check of the code
path, not evidence that the proposed promotion policy is approved or that a production champion is
healthy.

## Run it

The optional integration tests require a dedicated local PostgreSQL database whose name starts
with `pm25_test_`. Use a Unix-domain socket under `/tmp`; TCP, `hostaddr`, libpq service files,
multi-host targets, and ambient `PG*` routing variables are rejected or cleared. Before any test
schema is created, the connection verifies the database and role, confirms PostgreSQL reports a
Unix socket (no network server address), and checks that the server's socket directory matches the
requested path. A database reached through a TCP tunnel is therefore not accepted.

```sh
createdb -h /tmp -p 5432 -U "$(id -un)" pm25_test_kan122
PM25_TEST_POSTGRES_DSN="host=/tmp port=5432 dbname=pm25_test_kan122 user=$(id -un)" \
  uv run pytest -q tests/test_serving_postgres.py::test_local_lifecycle_rehearsal_register_promote_predict_score_and_rollback
dropdb -h /tmp -p 5432 -U "$(id -un)" pm25_test_kan122
```

Every integration test validates the same local-only connection before creating anything. The
rehearsal creates a uniquely named schema, applies the repository's PostgreSQL schema there, and
drops that schema in `finally`. It uses no configured `DATABASE_URL`, Neon connection, live API,
experiment dataset, or locked holdout. The only registered models and observations are generated
inside the test. Removing the dedicated database after the test is part of cleanup.

## What the rehearsal verifies

- Register two immutable synthetic bundles and verify each through the same loader used by Predict.
- Refuse a rollback when there is no previous champion. Once A is champion, reject an invalid
  bundle without changing the pointer or adding a promotion event.
- Run Predict with A, promote B during that already-selected issue hour, and retry. The frozen
  issue remains attributed to A; a new issue selects B.
- Roll back B to A and verify a later issue selects A. Every served row joins to the exact persisted
  prediction and run ID. Unsupported model horizons are recorded as `baseline_fallback`.
- Run Score against synthetic mature outcomes. Served-version metrics are compared with persistence
  on the same issue, station, and valid-time keys; both model versions remain measurable. One
  missing +24 h outcome stays in the waiting count and is excluded from metrics.
- Query the latest issue through the dashboard's real chart reader and renderer; the chart keeps the
  served forecast identifiable by model and reason.

The test fixes simulated issue clocks while PostgreSQL supplies actual creation timestamps. It does
not prove timestamp visibility against a hosted service, nor does it simulate a real trained
candidate, an owner-reviewed decision, or end-to-end deployment. Passing this optional test is not
evidence of a production rollout; it only verifies the local fixture path described above.

## Remaining KAN-122 work

Before claiming the full promotion rehearsal complete, KAN-120's owner-approved eligibility policy,
verified training/evaluation provenance, and an approved candidate must exist. A controlled staging
environment and owner review are still required for a real promotion and rollback. Do not set
`AUTO_PROMOTE=true` or run this against production as part of this fixture rehearsal.
