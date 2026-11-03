"""Optional local PostgreSQL concurrency test; never connects to production."""

import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from pm25_forecast import predict
from pm25_forecast.settings import Settings
from pm25_forecast.storage import sql


@pytest.mark.skipif(
    not os.environ.get("PM25_TEST_POSTGRES_DSN"), reason="local PostgreSQL not configured"
)
def test_decision_concurrent_winner_and_rollback():
    dsn = os.environ["PM25_TEST_POSTGRES_DSN"]
    schema = "test_" + uuid4().hex
    issue = datetime(2026, 11, 2, 12, tzinfo=UTC)
    migration = Path(sql.__file__).parent / "migrations" / "006_serving_decisions.sql"
    with psycopg.connect(dsn, autocommit=True) as setup:
        setup.execute("SET client_encoding TO 'UTF8'")
        setup.execute(f'CREATE SCHEMA "{schema}"')
        try:
            setup.execute(f'SET search_path TO "{schema}"')
            setup.execute("CREATE TABLE model_champion (slot text PRIMARY KEY, model_version text)")
            setup.execute(
                "CREATE TABLE predictions (station_id text, issue_time timestamptz, "
                "valid_time timestamptz, model_version text, "
                "PRIMARY KEY (station_id, issue_time, valid_time, model_version))"
            )
            setup.execute(migration.read_text())
            setup.execute("INSERT INTO model_champion VALUES ('production', 'v1')")
            with psycopg.connect(dsn) as first, psycopg.connect(dsn) as second:
                for con in (first, second):
                    con.execute(f'SET search_path TO "{schema}"')
                first.execute(sql.INSERT_SERVING_DECISION, {"issue_time": issue, "run_id": "first"})
                first.rollback()  # A failed attempt must not freeze the hour.
                first.execute(f'SET search_path TO "{schema}"')
                first.execute(
                    sql.INSERT_SERVING_DECISION, {"issue_time": issue, "run_id": "winner"}
                )
                setup.execute("UPDATE model_champion SET model_version = 'v2'")
                started = threading.Event()
                done = threading.Event()

                def contender():
                    started.set()
                    second.execute(
                        sql.INSERT_SERVING_DECISION, {"issue_time": issue, "run_id": "loser"}
                    )
                    second.commit()
                    done.set()

                thread = threading.Thread(target=contender)
                thread.start()
                assert started.wait(3)
                assert not done.wait(0.2)  # Unique conflict waits for the first transaction.
                first.commit()
                thread.join(5)
                assert done.is_set()
                assert setup.execute(
                    "SELECT champion_version, selection_run_id FROM serving_issue_decisions"
                ).fetchone() == ("v1", "winner")
                second.execute(
                    sql.INSERT_SERVING_DECISION,
                    {"issue_time": issue + timedelta(hours=1), "run_id": "next-hour"},
                )
                second.commit()
                assert setup.execute(
                    "SELECT champion_version FROM serving_issue_decisions "
                    "WHERE selection_run_id = 'next-hour'"
                ).fetchone() == ("v2",)
        finally:
            setup.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.mark.skipif(
    not os.environ.get("PM25_TEST_POSTGRES_DSN"), reason="local PostgreSQL not configured"
)
def test_real_predict_retry_preserves_first_persisted_fallback(monkeypatch):
    dsn = os.environ["PM25_TEST_POSTGRES_DSN"]
    schema = "test_" + uuid4().hex
    issue = datetime(2026, 11, 2, 12, tzinfo=UTC)
    with psycopg.connect(dsn, autocommit=True) as setup:
        setup.execute("SET client_encoding TO 'UTF8'")
        setup.execute(f'CREATE SCHEMA "{schema}"')
        try:
            setup.execute(f'SET search_path TO "{schema}"')
            setup.execute((Path(sql.__file__).parent / "schema.sql").read_text())
            setup.execute(
                "INSERT INTO observations(station_id, ts, pm25, source, ingested_at) "
                "VALUES ('station', %s, 12, 'air4thai', %s)",
                (issue - timedelta(hours=1), issue - timedelta(hours=1)),
            )
            setup.execute(
                "INSERT INTO models(model_version, sha256, artifact, manifest) "
                "VALUES ('v1', 'hash', ''::bytea, '{}'::jsonb)"
            )
            setup.execute(
                "INSERT INTO model_champion(slot, model_version) VALUES ('production', 'v1')"
            )

            def connect(_url):
                con = psycopg.connect(
                    dsn, options=f"-c search_path={schema} -c client_encoding=UTF8"
                )
                return con

            class Model:
                model_version = "v1"
                horizons = (24,)

                def predict(self, horizon, features):
                    return 15.0

            def broken(*_args):
                raise ValueError("invalid artifact")

            settings = Settings(
                _env_file=None,
                database_url="postgresql://unused/unused",
                station_locations="station=13,100",
            )
            monkeypatch.setattr(predict, "load_registered_model", broken)
            result = predict.run(
                settings, connect=connect, clock=lambda: issue + timedelta(minutes=10)
            )
            assert result.failures
            monkeypatch.setattr(predict, "load_registered_model", lambda *_: Model())
            predict.run(settings, connect=connect, clock=lambda: issue + timedelta(minutes=20))
            assert setup.execute(
                "SELECT DISTINCT model_version, reason FROM serving_forecasts"
            ).fetchall() == [("persistence-24h", "baseline_fallback")]
            assert setup.execute(
                "SELECT count(*) FROM serving_forecasts s JOIN predictions p "
                "USING (station_id, issue_time, valid_time, model_version, run_id)"
            ).fetchone() == (3,)
            setup.execute("DELETE FROM predictions WHERE model_version = 'persistence-24h'")
            assert setup.execute("SELECT count(*) FROM serving_forecasts").fetchone() == (0,)
        finally:
            setup.execute(f'DROP SCHEMA "{schema}" CASCADE')
