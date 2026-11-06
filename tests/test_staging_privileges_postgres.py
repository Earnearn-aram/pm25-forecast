"""Optional local SQL privilege proof; never connects to a hosted database."""

import hashlib
import json
import os
from dataclasses import replace
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from pm25_forecast import staging_snapshot_probe as probe
from test_serving_postgres import _connect_local_test_db, _validated_test_dsn


@pytest.mark.skipif(
    not os.environ.get("PM25_TEST_POSTGRES_DSN"), reason="local PostgreSQL not configured"
)
def test_restricted_role_passes_real_catalog_checks_and_table_grant_is_refused(
    monkeypatch, tmp_path
):
    dsn = _validated_test_dsn()
    role = "probe_" + uuid4().hex
    table = "probe_" + uuid4().hex
    params = psycopg.conninfo.conninfo_to_dict(dsn)
    role_dsn = psycopg.conninfo.make_conninfo(dsn, user=role)
    for key in tuple(os.environ):
        if key.startswith("PG"):
            monkeypatch.delenv(key)
    target = probe.ApprovedTarget(
        "synthetic-project",
        "br-synthetic",
        "ep-synthetic",
        "ep-synthetic.ap-southeast-1.aws.neon.tech",
        params["dbname"],
        role,
        "br-synthetic-parent",
        "0" * 64,
    )
    evidence = json.dumps(
        {
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
            "verification_ref": "synthetic-local-test",
        },
        sort_keys=True,
    ).encode()
    target = replace(target, evidence_sha256=hashlib.sha256(evidence).hexdigest())
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_bytes(evidence)
    monkeypatch.setattr(probe, "EVIDENCE_PATH", evidence_path)

    class LocalCatalogConnection:
        """Test only: real SQL, with simulated TLS/host metadata for local Unix socket."""

        def __init__(self, connection):
            self.connection = connection
            self.info = type("Info", (), {"host": target.host})()
            self.pgconn = type("Pgconn", (), {"ssl_in_use": True})()

        def execute(self, *args):
            return self.connection.execute(*args)

    with _connect_local_test_db(dsn, autocommit=True) as admin:
        public_create = admin.execute(
            "SELECT has_schema_privilege('public', 'public', 'CREATE')"
        ).fetchone()[0]
        if public_create:
            admin.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        role_created = False
        try:
            admin.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(role)))
            role_created = True
            admin.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(params["dbname"]), sql.Identifier(role)
                )
            )
            with _connect_local_test_db(
                role_dsn, autocommit=True, verify_socket_directory=False
            ) as connection:
                assert connection.info.host == params["host"]
                assert connection.execute(
                    "SELECT current_database(), current_user, inet_server_addr()"
                ).fetchone() == (params["dbname"], role, None)
                wrapped = LocalCatalogConnection(connection)
                assert probe.check_connection(wrapped, target)
                admin.execute(
                    sql.SQL("CREATE TABLE public.{} (value int)").format(sql.Identifier(table))
                )
                admin.execute(
                    sql.SQL("GRANT SELECT ON public.{} TO {}").format(
                        sql.Identifier(table), sql.Identifier(role)
                    )
                )
                with pytest.raises(probe.ProbeRefused, match="schema, relation or routine"):
                    probe.check_connection(wrapped, target)
                admin.execute(sql.SQL("DROP TABLE public.{}").format(sql.Identifier(table)))

                # Exercise the entire synthetic proof against the restricted local role. TLS and
                # the remote hostname are simulated; no hosted endpoint or credentials are used.
                real_connect = psycopg.connect
                failure = {"inject": False}
                collision = {"inject": False, "schema": None}

                class LocalProbeConnection(LocalCatalogConnection):
                    def __enter__(self):
                        self.connection.__enter__()
                        return self

                    def __exit__(self, *args):
                        return self.connection.__exit__(*args)

                    def close(self):
                        self.connection.close()

                    def rollback(self):
                        self.connection.rollback()

                    def commit(self):
                        self.connection.commit()

                    def execute(self, statement, *args):
                        text = (
                            statement.as_string(self.connection)
                            if isinstance(statement, sql.Composable)
                            else statement
                        )
                        if failure["inject"] and "SELECT label FROM" in text:
                            failure["inject"] = False
                            raise RuntimeError("synthetic failure")
                        if collision["inject"] and text.startswith("CREATE SCHEMA "):
                            collision["inject"] = False
                            collision["schema"] = text.split('"')[1]
                            admin.execute(
                                sql.SQL("CREATE SCHEMA {}").format(
                                    sql.Identifier(collision["schema"])
                                )
                            )
                            raise psycopg.errors.DuplicateSchema("simulated concurrent creator")
                        return self.connection.execute(statement, *args)

                def local_connect(**kwargs):
                    local = real_connect(role_dsn, autocommit=kwargs.get("autocommit", False))
                    return LocalProbeConnection(local)

                monkeypatch.setattr(psycopg, "connect", local_connect)
                synthetic_dsn = psycopg.conninfo.make_conninfo(
                    host=target.host,
                    dbname=target.database,
                    user=role,
                    password="synthetic-password",  # pragma: allowlist secret
                    sslmode="verify-full",
                )
                assert probe.run_probe(synthetic_dsn, target)
                failure["inject"] = True
                with pytest.raises(RuntimeError, match="synthetic failure"):
                    probe.run_probe(synthetic_dsn, target)
                assert admin.execute(
                    "SELECT count(*) FROM pg_namespace WHERE nspname LIKE 'probe_%'"
                ).fetchone() == (0,)
                collision["inject"] = True
                with pytest.raises(psycopg.errors.DuplicateSchema):
                    probe.run_probe(synthetic_dsn, target)
                assert admin.execute(
                    "SELECT count(*) FROM pg_namespace WHERE nspname = %s",
                    (collision["schema"],),
                ).fetchone() == (1,)
                admin.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(collision["schema"]))
                )
        finally:
            admin.execute(sql.SQL("DROP TABLE IF EXISTS public.{}").format(sql.Identifier(table)))
            if role_created:
                admin.execute(
                    sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(
                        sql.Identifier(params["dbname"]), sql.Identifier(role)
                    )
                )
                admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
            if public_create:
                admin.execute("GRANT CREATE ON SCHEMA public TO PUBLIC")
