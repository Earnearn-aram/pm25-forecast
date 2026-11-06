"""Manual, fail-closed synthetic staging snapshot probe; no production target is configured.

Before use, a reviewed change must pin an independently verified Neon project, branch, endpoint,
direct hostname, database and dedicated role in APPROVED_TARGET. The workflow must be run only
from protected main and with a staging-only environment secret. This probe does not unblock retrain.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql


class ProbeRefused(ValueError):
    """A target or permission check failed before, or during, the synthetic proof."""


@dataclass(frozen=True)
class ApprovedTarget:
    project_id: str
    branch_id: str
    endpoint_id: str
    host: str
    database: str
    role: str
    parent_branch_id: str
    evidence_sha256: str


# Intentionally disabled. Never derive this value from environment, command arguments or a DSN.
# Pin a real target only after the owner independently verifies the control-plane association.
APPROVED_TARGET: ApprovedTarget | None = None
EVIDENCE_PATH = Path(__file__).resolve().parents[2] / "docs" / "staging-target-evidence.json"
_ALLOWED_PARAMS = {"host", "dbname", "user", "password", "sslmode", "connect_timeout"}


def verify_reviewed_evidence(target: ApprovedTarget, data: bytes) -> None:
    """Check fixed-path evidence against a protected, independently reviewed source pin.

    The owner/reviewer must verify the Neon control-plane evidence itself before approving a pin;
    this check detects changes but cannot independently establish the truth of its claims.
    """
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), target.evidence_sha256):
        raise ProbeRefused("staging evidence does not match reviewed digest")
    try:
        evidence = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProbeRefused("invalid staging evidence") from exc
    expected = {
        "schema_version": 1,
        "project_id": target.project_id,
        "branch_id": target.branch_id,
        "endpoint_id": target.endpoint_id,
        "host": target.host,
        "database": target.database,
        "role": target.role,
        "parent_branch_id": target.parent_branch_id,
        "creation_mode": "schema_only",
        "no_production_or_holdout_data": True,
    }
    if not isinstance(evidence, dict) or any(evidence.get(k) != v for k, v in expected.items()):
        raise ProbeRefused("staging evidence does not match target or isolation requirements")
    if not isinstance(evidence.get("verification_ref"), str) or not evidence["verification_ref"]:
        raise ProbeRefused("staging evidence has no reviewed control-plane verification reference")


def validate_dsn(dsn: str, target: ApprovedTarget | None) -> dict[str, str]:
    """Reject unknown/unreviewed endpoints and alternate libpq routing before any connection."""
    if target is None:
        raise ProbeRefused("no reviewed staging target")
    if (
        not target.project_id
        or not target.branch_id.startswith("br-")
        or not target.endpoint_id.startswith("ep-")
        or not target.host.startswith(target.endpoint_id + ".")
        or not target.host.endswith(".aws.neon.tech")
        or not target.database
        or not target.role
        or not target.parent_branch_id.startswith("br-")
        or target.parent_branch_id == target.branch_id
        or len(target.evidence_sha256) != 64
        or any(c not in "0123456789abcdef" for c in target.evidence_sha256)
    ):
        raise ProbeRefused("invalid target pin")
    try:
        params = psycopg.conninfo.conninfo_to_dict(dsn)
    except psycopg.Error as exc:
        raise ProbeRefused("invalid staging DSN") from exc
    if (
        set(params) - _ALLOWED_PARAMS
        or params.get("host") != target.host
        or params.get("dbname") != target.database
        or params.get("user") != target.role
        or not params.get("password")
        or params.get("sslmode") != "verify-full"
    ):
        raise ProbeRefused("staging DSN does not match reviewed target")
    return params


def check_connection(con: psycopg.Connection, target: ApprovedTarget) -> str:
    """Verify the connected role and reject privileges on application tables before any DDL."""
    if con.info.host != target.host or not con.pgconn.ssl_in_use:
        raise ProbeRefused("connected endpoint or TLS mismatch")
    row = con.execute(
        "SELECT current_database(), current_user, current_setting('server_version'), "
        "r.rolsuper, r.rolcreatedb, r.rolcreaterole, r.rolreplication, r.rolbypassrls, "
        "has_database_privilege(current_user, current_database(), 'CREATE'), "
        "has_schema_privilege(current_user, 'public', 'CREATE'), "
        "EXISTS (SELECT 1 FROM pg_auth_members m WHERE m.member = r.oid), "
        "(SELECT pg_get_userbyid(datdba) = current_user FROM pg_database "
        "WHERE datname = current_database()) "
        "FROM pg_roles r WHERE r.rolname = current_user"
    ).fetchone()
    if (
        row is None
        or row[0] != target.database
        or row[1] != target.role
        or any(row[3:8])
        or not row[8]
        or row[9]
        or row[10]
        or row[11]
    ):
        raise ProbeRefused("staging database or role privileges do not match")
    schema_access = con.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_namespace n WHERE "
        "n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema' "
        "AND has_schema_privilege(current_user, n.oid, 'CREATE'))"
    ).fetchone()
    relation_access = con.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema' AND ("
        "(c.relkind IN ('r','p','v','m','f') AND ("
        "has_table_privilege(current_user, c.oid, "
        "'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') OR "
        "has_any_column_privilege(current_user, c.oid, 'SELECT,INSERT,UPDATE,REFERENCES'))) OR "
        "(c.relkind = 'S' AND has_sequence_privilege(current_user, c.oid, 'USAGE,SELECT,UPDATE'))"
        "))"
    ).fetchone()
    routine_access = con.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema' "
        "AND has_function_privilege(current_user, p.oid, 'EXECUTE'))"
    ).fetchone()
    if (
        schema_access is None
        or schema_access[0]
        or relation_access is None
        or relation_access[0]
        or routine_access is None
        or routine_access[0]
    ):
        raise ProbeRefused("staging role has existing schema, relation or routine privileges")
    return str(row[2])


def run_probe(dsn: str, target: ApprovedTarget | None = APPROVED_TARGET) -> str:
    """Perform only synthetic writes after all endpoint, connection and role checks pass."""
    params = validate_dsn(dsn, target)
    assert target is not None
    try:
        evidence = EVIDENCE_PATH.read_bytes()
    except OSError as exc:
        raise ProbeRefused("reviewed staging evidence is unavailable") from exc
    verify_reviewed_evidence(target, evidence)
    # Reject ambient libpq configuration; it can change TLS and routing behind the DSN.
    if any(key.startswith("PG") for key in os.environ):
        raise ProbeRefused("ambient libpq variables are not allowed")

    def connect(*, autocommit: bool = False) -> psycopg.Connection:
        con = psycopg.connect(**params, autocommit=autocommit)
        try:
            check_connection(con, target)
            if not autocommit:
                con.rollback()
        except Exception:
            con.close()
            raise
        return con

    schema = "probe_" + uuid4().hex
    table = sql.Identifier(schema, "events")
    with connect(autocommit=True) as setup:
        version = check_connection(setup, target)
        created = False
        try:
            # Every connection is verified before the first DDL, including writers and reader.
            with ExitStack() as stack:
                before, inflight, reader, later = (stack.enter_context(connect()) for _ in range(4))
                existing = setup.execute(
                    "SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,)
                ).fetchone()
                if existing is not None:
                    raise ProbeRefused("generated probe schema already exists")
                setup.execute(sql.SQL("CREATE SCHEMA {} ").format(sql.Identifier(schema)))
                # Do not clean up unless CREATE succeeded. A racing name collision must never
                # turn cleanup into a CASCADE drop of a schema created by someone else.
                created = True
                setup.execute(sql.SQL("CREATE TABLE {} (label text PRIMARY KEY)").format(table))
                before_id = before.execute("SELECT pg_current_xact_id()::text").fetchone()[0]
                before.execute(sql.SQL("INSERT INTO {} VALUES ('before')").format(table))
                before.commit()
                inflight_id = inflight.execute("SELECT pg_current_xact_id()::text").fetchone()[0]
                inflight.execute(sql.SQL("INSERT INTO {} VALUES ('inflight')").format(table))
                reader.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
                snapshot = reader.execute("SELECT pg_current_snapshot()::text").fetchone()[0]
                if reader.execute(sql.SQL("SELECT label FROM {}").format(table)).fetchall() != [
                    ("before",)
                ]:
                    raise ProbeRefused("initial snapshot did not match")
                inflight.commit()
                after_id = later.execute("SELECT pg_current_xact_id()::text").fetchone()[0]
                later.execute(sql.SQL("INSERT INTO {} VALUES ('after')").format(table))
                later.commit()
                later.execute("SELECT pg_current_xact_id()")
                later.execute(sql.SQL("INSERT INTO {} VALUES ('aborted')").format(table))
                later.rollback()
                if reader.execute(sql.SQL("SELECT label FROM {}").format(table)).fetchall() != [
                    ("before",)
                ]:
                    raise ProbeRefused("repeatable-read snapshot changed")
                reader.commit()
                visible = setup.execute(
                    "SELECT pg_visible_in_snapshot(%s::xid8, %s::pg_snapshot), "
                    "pg_visible_in_snapshot(%s::xid8, %s::pg_snapshot), "
                    "pg_visible_in_snapshot(%s::xid8, %s::pg_snapshot)",
                    (before_id, snapshot, inflight_id, snapshot, after_id, snapshot),
                ).fetchone()
                rows = setup.execute(
                    sql.SQL("SELECT label FROM {} ORDER BY label").format(table)
                ).fetchall()
                if visible != (True, False, False) or rows != [
                    ("after",),
                    ("before",),
                    ("inflight",),
                ]:
                    raise ProbeRefused("visibility or rollback result did not match")
        finally:
            # ExitStack closes or rolls back all sessions first, releasing locks before cleanup.
            if created:
                setup.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )
    return version


def main() -> int:
    try:
        version = run_probe(os.environ.get("PM25_STAGING_DATABASE_URL", ""))
    except Exception as exc:
        # Driver errors may contain credentials; expose only the exception class.
        print(f"staging probe refused or failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"synthetic staging visibility proof passed; PostgreSQL {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
