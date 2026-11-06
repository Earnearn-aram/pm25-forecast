"""Optional local PostgreSQL concurrency test; never connects to production."""

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from pm25_forecast import dashboard, predict, registry_cli, score
from pm25_forecast.settings import Settings
from pm25_forecast.storage import registry, sql
from test_serving import KAN15_FEATURES, booster_text, manifest


def _validated_test_dsn() -> str:
    dsn = os.environ.get("PM25_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("local PostgreSQL not configured")
    params = psycopg.conninfo.conninfo_to_dict(dsn)
    host = params.get("host")
    if host not in {"/tmp", "/private/tmp"} or "," in host:
        raise ValueError("PM25_TEST_POSTGRES_DSN must use one explicit Unix socket under /tmp")
    if any(params.get(key) for key in ("hostaddr", "service", "servicefile")):
        raise ValueError("PM25_TEST_POSTGRES_DSN cannot use hostaddr or libpq service files")
    if not params.get("dbname", "").startswith("pm25_test_"):
        raise ValueError("PM25_TEST_POSTGRES_DSN must select a dedicated pm25_test_* database")
    if not params.get("user"):
        raise ValueError("PM25_TEST_POSTGRES_DSN must name the local test role explicitly")
    return dsn


def _connect_local_test_db(dsn: str, *, options: str | None = None, autocommit: bool = False):
    """Connect only through a local socket, ignoring ambient libpq route settings."""
    params = psycopg.conninfo.conninfo_to_dict(dsn)
    saved = {key: value for key, value in os.environ.items() if key.startswith("PG")}
    for key in saved:
        os.environ.pop(key, None)
    try:
        con = psycopg.connect(dsn, options=options, autocommit=autocommit)
    finally:
        os.environ.update(saved)
    try:
        actual = con.execute(
            "SELECT current_database(), current_user, inet_server_addr(), "
            "current_setting('unix_socket_directories')"
        ).fetchone()
        socket_dirs = {str(Path(item).resolve()) for item in actual[3].split(",")}
        requested_socket = str(Path(params["host"]).resolve())
        if (
            actual[:3] != (params["dbname"], params["user"], None)
            or requested_socket not in socket_dirs
            or con.info.host != params["host"]
        ):
            raise ValueError("connected PostgreSQL server does not match the local test target")
    except Exception:
        con.close()
        raise
    return con


@pytest.mark.skipif(
    not os.environ.get("PM25_TEST_POSTGRES_DSN"), reason="local PostgreSQL not configured"
)
def test_decision_concurrent_winner_and_rollback():
    dsn = _validated_test_dsn()
    schema = "test_" + uuid4().hex
    issue = datetime(2026, 11, 2, 12, tzinfo=UTC)
    migration = Path(sql.__file__).parent / "migrations" / "006_serving_decisions.sql"
    with _connect_local_test_db(dsn, autocommit=True) as setup:
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
            with _connect_local_test_db(dsn) as first, _connect_local_test_db(dsn) as second:
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
    dsn = _validated_test_dsn()
    schema = "test_" + uuid4().hex
    issue = datetime(2026, 11, 2, 12, tzinfo=UTC)
    with _connect_local_test_db(dsn, autocommit=True) as setup:
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
                con = _connect_local_test_db(
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


@pytest.mark.skipif(
    not os.environ.get("PM25_TEST_POSTGRES_DSN"), reason="local PostgreSQL not configured"
)
def test_local_lifecycle_rehearsal_register_promote_predict_score_and_rollback(tmp_path):
    """Exercise the real lifecycle code against an explicitly local, disposable PostgreSQL DB."""
    dsn = _validated_test_dsn()

    schema = "test_" + uuid4().hex
    issue0 = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    issues = [issue0 + timedelta(hours=offset) for offset in range(3)]
    versions = ("rehearsal-a", "rehearsal-b")

    def connect(_url):
        return _connect_local_test_db(
            dsn, options=f"-c search_path={schema} -c client_encoding=UTF8"
        )

    def cli(*args):
        output = []
        code = registry_cli.run(
            list(args),
            database_url="postgresql://local-test/unused",
            connect=connect,
            out=output.append,
        )
        return code, "\n".join(output)

    def fixture_recommendation(con, version, previous_version):
        # Synthetic evidence for exercising the promotion interface; it is not a real evaluation.
        registry.record_event(
            con,
            "recommend",
            version,
            previous_version=previous_version,
            actor="retrain",
            metrics={
                "station_policy": {
                    "passed": True,
                    "locked_holdout_boundary_verified": True,
                    "candidate_training_window_verified": True,
                    "incumbent_training_window_verified": True,
                    "evaluation_start": "2026-01-01T00:00:00+00:00",
                    "evaluation_end": "2026-01-29T00:00:00+00:00",
                }
            },
        )

    with _connect_local_test_db(dsn, autocommit=True) as setup:
        setup.execute("SET client_encoding TO 'UTF8'")
        setup.execute(f'CREATE SCHEMA "{schema}"')
        try:
            setup.execute(f'SET search_path TO "{schema}"')
            setup.execute((Path(sql.__file__).parent / "schema.sql").read_text())

            # No previous champion is a safe refusal, not a pointer or audit mutation.
            code, output = cli("rollback", "--confirm")
            assert code == registry_cli.EXIT_FAILED and "nothing to roll back" in output
            assert setup.execute("SELECT count(*) FROM model_events").fetchone() == (0,)
            assert setup.execute("SELECT count(*) FROM model_champion").fetchone() == (0,)

            for offset, version in enumerate(versions):
                artifact = json.dumps(
                    {
                        "format": 1,
                        "horizon_hours": 24,
                        "booster": booster_text(KAN15_FEATURES, offset=float(offset * 3)),
                    },
                    separators=(",", ":"),
                ).encode()
                model_manifest = manifest(version, artifact, KAN15_FEATURES)
                artifact_path = tmp_path / f"{version}.json"
                manifest_path = tmp_path / f"{version}.manifest.json"
                artifact_path.write_bytes(artifact)
                manifest_path.write_text(json.dumps(model_manifest), encoding="utf-8")
                code, output = cli(
                    "register", "--artifact", str(artifact_path), "--manifest", str(manifest_path)
                )
                assert code == 0 and "champion unchanged" in output
                assert cli("check", version)[0] == 0

            assert cli("promote", versions[0], "--confirm")[0] == registry_cli.EXIT_FAILED
            assert setup.execute("SELECT count(*) FROM model_champion").fetchone() == (0,)
            assert setup.execute("SELECT count(*) FROM model_events").fetchone() == (0,)
            fixture_recommendation(setup, versions[0], None)
            assert cli("promote", versions[0], "--confirm")[0] == 0
            # Invalid registered bytes cannot replace the current champion or append an event.
            bad_artifact = b'{"format":1,"horizon_hours":24,"booster":"not-a-booster"}'
            bad_manifest = manifest("rehearsal-invalid", bad_artifact, KAN15_FEATURES)
            setup.execute(
                "INSERT INTO models(model_version, sha256, artifact, manifest) "
                "VALUES (%s, %s, %s, %s::jsonb)",
                (
                    "rehearsal-invalid",
                    registry.artifact_sha256(bad_artifact),
                    bad_artifact,
                    json.dumps(bad_manifest),
                ),
            )
            fixture_recommendation(setup, "rehearsal-invalid", versions[0])
            code, output = cli("promote", "rehearsal-invalid", "--confirm")
            assert code == registry_cli.EXIT_FAILED and "failed" in output
            assert setup.execute(
                "SELECT model_version FROM model_champion WHERE slot = 'production'"
            ).fetchone() == (versions[0],)
            assert setup.execute("SELECT count(*) FROM model_events").fetchone() == (3,)

            for index, issue in enumerate(issues):
                # Provide only historical input at the issue boundary; all later outcome rows
                # are timestamped as ingested after their valid time and cannot leak into input.
                input_ts = issue - timedelta(hours=1)
                setup.execute(
                    "INSERT INTO observations(station_id, ts, pm25, source, ingested_at) "
                    "VALUES ('59t', %s, %s, 'air4thai', %s) ON CONFLICT DO NOTHING",
                    (input_ts, 18.0 + index, input_ts + timedelta(minutes=1)),
                )
                for horizon in (6, 12, 24):
                    if index == 2 and horizon == 24:
                        continue  # A mature ledger row without an actual remains waiting.
                    valid = issue + timedelta(hours=horizon)
                    setup.execute(
                        "INSERT INTO observations(station_id, ts, pm25, source, ingested_at) "
                        "VALUES ('59t', %s, %s, 'air4thai', %s) ON CONFLICT DO NOTHING",
                        (valid, 20.0 + index, valid + timedelta(minutes=1)),
                    )
            # First issue freezes A. Promoting B in that issue hour cannot rewrite that decision.
            first = predict.run(
                Settings(
                    _env_file=None,
                    database_url="postgresql://local-test/unused",
                    station_locations="59t=13.783185,100.540489",
                ),
                connect=connect,
                clock=lambda: issues[0] + timedelta(minutes=10),
            )
            assert not first.failures
            fixture_recommendation(setup, versions[1], versions[0])
            assert cli("promote", versions[1], "--confirm")[0] == 0
            same_hour_retry = predict.run(
                Settings(
                    _env_file=None,
                    database_url="postgresql://local-test/unused",
                    station_locations="59t=13.783185,100.540489",
                ),
                connect=connect,
                clock=lambda: issues[0] + timedelta(minutes=20),
            )
            assert not same_hour_retry.failures
            assert setup.execute(
                "SELECT champion_version FROM serving_issue_decisions WHERE issue_time = %s",
                (issues[0],),
            ).fetchone() == (versions[0],)

            # New hour sees B; rollback to A is audited, and the next hour serves A again.
            second = predict.run(
                Settings(
                    _env_file=None,
                    database_url="postgresql://local-test/unused",
                    station_locations="59t=13.783185,100.540489",
                ),
                connect=connect,
                clock=lambda: issues[1] + timedelta(minutes=10),
            )
            assert not second.failures
            assert cli("rollback", "--confirm")[0] == 0
            third = predict.run(
                Settings(
                    _env_file=None,
                    database_url="postgresql://local-test/unused",
                    station_locations="59t=13.783185,100.540489",
                ),
                connect=connect,
                clock=lambda: issues[2] + timedelta(minutes=10),
            )
            assert not third.failures

            choices = setup.execute(
                "SELECT issue_time, valid_time, model_version, reason, run_id "
                "FROM serving_forecasts ORDER BY issue_time, valid_time"
            ).fetchall()
            by_issue = {}
            for row in choices:
                by_issue.setdefault(row[0], []).append(row[2:4])
            assert by_issue[issues[0]] == [
                ("persistence-24h", "baseline_fallback"),
                ("persistence-24h", "baseline_fallback"),
                (versions[0], "champion"),
            ]
            assert by_issue[issues[1]][-1] == (versions[1], "champion")
            assert by_issue[issues[2]][-1] == (versions[0], "champion")
            assert setup.execute(
                "SELECT count(*) FROM serving_forecasts sf JOIN predictions p "
                "USING (station_id, issue_time, valid_time, model_version, run_id)"
            ).fetchone() == (9,)
            assert setup.execute(
                "SELECT action, model_version, previous_version FROM model_events ORDER BY id"
            ).fetchall() == [
                ("recommend", versions[0], None),
                ("promote", versions[0], None),
                ("recommend", "rehearsal-invalid", versions[0]),
                ("recommend", versions[1], versions[0]),
                ("promote", versions[1], versions[0]),
                ("rollback", versions[0], versions[1]),
            ]

            # Score against synthetic matured outcomes via the actual score runner.
            score_now = issues[2] + timedelta(hours=25)
            scored = score.run(
                Settings(_env_file=None, database_url="postgresql://local-test/unused"),
                connect=connect,
                clock=lambda: score_now,
            )
            assert scored.served_status == "ok"
            assert scored.served_metrics[versions[0]][24].n == 1
            assert scored.served_metrics[versions[1]][24].n == 1
            assert scored.served_metrics[versions[0]][24].n_matched == 1
            assert scored.served_waiting == 1
            errors = dict(
                setup.execute(
                    "SELECT model_version, abs_error FROM scores "
                    "WHERE station_id = '59t' AND issue_time = %s AND valid_time = %s "
                    "AND model_version IN (%s, %s)",
                    (issues[0], issues[0] + timedelta(hours=24), versions[0], "persistence-24h"),
                ).fetchall()
            )
            served_metric = scored.served_metrics[versions[0]][24]
            assert served_metric.mae == pytest.approx(errors[versions[0]])
            assert served_metric.skill_vs_persistence == pytest.approx(
                1 - errors[versions[0]] / errors["persistence-24h"]
            )

            # Render the actual latest served issue using the dashboard's real read/query/render
            # path while outcomes are still future relative to this chart snapshot.
            chart_now = max(
                issues[2] + timedelta(hours=2), datetime.now(UTC) + timedelta(minutes=5)
            )
            with connect("unused") as chart_con:

                class Reader:
                    def query(self, statement, params):
                        return chart_con.execute(statement, params).fetchall()

                reader = Reader()
                observations = dashboard._observations(reader, chart_now)
                chart = dashboard._chart(reader, observations, None, chart_now)
                station = next(row for row in chart["stations"] if row["id"] == "59t")
                assert chart["latest_serving_issue"] == issues[2].isoformat()
                assert (24, versions[0]) in {
                    (row["horizon"], row["model_version"]) for row in station["future_served"]
                }
                block = dashboard._chart_block(station, chart, True, chart_now)
                assert "rehearsal-a" in block and 'data-en="Served"' in block
        finally:
            setup.execute(f'DROP SCHEMA "{schema}" CASCADE')
