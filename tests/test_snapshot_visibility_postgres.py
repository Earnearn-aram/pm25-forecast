"""Synthetic local PostgreSQL proof; no hosted, production or experiment data access."""

import os
from uuid import uuid4

import pytest
from psycopg import sql

from test_serving_postgres import _connect_local_test_db, _validated_test_dsn


@pytest.mark.skipif(
    not os.environ.get("PM25_TEST_POSTGRES_DSN"), reason="local PostgreSQL not configured"
)
def test_repeatable_read_snapshot_visibility_after_reader_finishes():
    dsn = _validated_test_dsn()
    schema = "test_" + uuid4().hex
    table = sql.Identifier(schema, "events")
    with _connect_local_test_db(dsn, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {} ").format(sql.Identifier(schema)))
        try:
            setup.execute(sql.SQL("CREATE TABLE {} (label text PRIMARY KEY)").format(table))
            with (
                _connect_local_test_db(dsn) as before,
                _connect_local_test_db(dsn) as inflight,
                _connect_local_test_db(dsn) as reader,
                _connect_local_test_db(dsn) as later,
            ):
                before_id = before.execute("SELECT pg_current_xact_id()::text").fetchone()[0]
                before.execute(sql.SQL("INSERT INTO {} VALUES ('before')").format(table))
                before.commit()

                inflight_id = inflight.execute("SELECT pg_current_xact_id()::text").fetchone()[0]
                inflight.execute(sql.SQL("INSERT INTO {} VALUES ('inflight')").format(table))

                # The connection safety probe starts a transaction; reset before choosing isolation.
                reader.rollback()
                reader.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
                snapshot = reader.execute("SELECT pg_current_snapshot()::text").fetchone()[0]
                assert reader.execute(sql.SQL("SELECT label FROM {}").format(table)).fetchall() == [
                    ("before",)
                ]

                inflight.commit()
                after_id = later.execute("SELECT pg_current_xact_id()::text").fetchone()[0]
                later.execute(sql.SQL("INSERT INTO {} VALUES ('after')").format(table))
                later.commit()
                later.execute("SELECT pg_current_xact_id()")
                later.execute(sql.SQL("INSERT INTO {} VALUES ('aborted')").format(table))
                later.rollback()

                assert reader.execute(sql.SQL("SELECT label FROM {}").format(table)).fetchall() == [
                    ("before",)
                ]
                reader.commit()

            # pg_visible_in_snapshot is a snapshot membership check, not proof of commit. The
            # aborted insert is checked through row absence, not by treating its xid as committed.
            assert setup.execute(
                "SELECT pg_visible_in_snapshot(%s::xid8, %s::pg_snapshot), "
                "pg_visible_in_snapshot(%s::xid8, %s::pg_snapshot), "
                "pg_visible_in_snapshot(%s::xid8, %s::pg_snapshot)",
                (before_id, snapshot, inflight_id, snapshot, after_id, snapshot),
            ).fetchone() == (True, False, False)
            assert setup.execute(
                sql.SQL("SELECT label FROM {} ORDER BY label").format(table)
            ).fetchall() == [("after",), ("before",), ("inflight",)]
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
