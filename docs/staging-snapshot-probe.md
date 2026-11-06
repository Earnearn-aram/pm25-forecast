# Staging snapshot proof

This manual workflow is a synthetic-only rehearsal of PostgreSQL snapshot visibility. It is not
the production retraining workflow and does not authorize or unblock retraining.

## Current status

The reviewed target is deliberately disabled in `staging_snapshot_probe.py`. There is no committed
`staging-target-evidence.json`, no pinned Neon host or database role, and no hosted connection has
been attempted. The branch name, endpoint ID, empty-table query, and UI details supplied during
setup are useful leads, but they are not independent control-plane evidence of branch ancestry,
database identity, role isolation, or absence of protected holdout data.

Before enabling a target, the owner and reviewer must verify the Neon control plane and record a
reviewable, non-secret evidence reference that binds the project, staging branch, endpoint, exact
direct hostname, database, restricted role, parent branch, schema-only creation mode, and explicit
confirmation that no production or locked-holdout data is present. The evidence JSON must match the
code's fixed target and its reviewed SHA-256 pin. A JSON claim or digest alone does not establish
truth; approval of the source evidence is required. Do not include connection strings, passwords,
or data rows in the evidence file or pull request.

This change does not add or enable a GitHub workflow, GitHub environment, or staging secret. The
repository's experiment-work instructions prohibit changes to `.github/workflows/`, Neon, and
secrets. A future separately authorized change must configure the `staging` GitHub environment
with required reviewers and branch restrictions that permit only protected `main`; a workflow
branch condition and environment name do not enforce those settings by themselves. Any future
`PM25_STAGING_DATABASE_URL` must be a dedicated restricted-role DSN for the pinned direct host,
with `sslmode=verify-full`. Never use a production credential, owner role, or a user-provided DSN
to select the target.

## Probe behavior

The runner rejects an absent or changed target/evidence before connecting. It then validates the
exact DSN, rejects ambient `PG*` settings, verifies TLS, server database and role, and checks that
the role is not privileged, owns the database, inherits memberships, or has access to existing
application schemas, relations, sequences, or routines. Only then does it create a uniquely named
schema, run a synthetic multi-session transaction-visibility exercise, close sessions, and drop
that schema. An interrupted runner can leave an orphaned `probe_*` schema; inspect and clean up
only the exact synthetic schema after verifying it contains no user data.

Local tests exercise the catalog checks and transaction probe against a temporary PostgreSQL
instance. Local TLS and hostname metadata are simulated by the test wrapper; that is not evidence
of hosted Neon connectivity, lineage, or GitHub environment protection. A hosted run is allowed
only after the independent evidence review and protected-environment setup above.
