"""KAN-25: static dashboard build. Offline: SQLite built from schema.sql (FakeDb)."""

import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest

from pm25_forecast import dashboard
from pm25_forecast.dashboard import headline
from pm25_forecast.models.baselines import PERSISTENCE_MODEL_VERSION
from pm25_forecast.settings import Settings
from test_predict import DB_PASSWORD, DB_URL, FakeDb

CHALLENGER = "kan15-retrospective-24h-lag1h"
NOW = datetime(2026, 10, 30, 12, 20, tzinfo=UTC)
H = timedelta(hours=1)
P = PERSISTENCE_MODEL_VERSION
TOP_KEYS = {
    "generated_at",
    "generated_at_bangkok",
    "status",
    "sections",
    "challenger_version",
    "headline",
    "kpi",
    "scores",
    "freshness",
    "chart",
    "quality",
    "drift",
    "models",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def settings(**kw):
    values = {"database_url": DB_URL, "challenger_model_version": CHALLENGER}
    return Settings(_env_file=None, **{**values, **kw})


def build(fake, out: Path, **kw) -> int:
    return dashboard.run(settings(**kw), out, connect=fake, clock=lambda: NOW)


def load(out: Path) -> dict:
    return json.loads((out / "data.json").read_text())


def seed(fake: FakeDb, *, challenger_scores: bool = True) -> None:
    """Two stations, 3 days of hourly data, +24 h forecasts and scores for both models."""
    con = fake.con
    start = NOW.replace(minute=0, second=0, microsecond=0) - timedelta(days=3)
    for station, base in (("59t", 40.0), ("61t", 55.0)):
        for i in range(0, 73):
            ts = start + i * H
            con.execute(
                "INSERT INTO observations (station_id, ts, pm25, source)"
                " VALUES (?, ?, ?, 'air4thai')",
                (station, ts.isoformat(), base + i % 7),
            )
            if i < 48:
                issue, valid = ts, ts + 24 * H
                models = [(P, 10.0)] + ([(CHALLENGER, 5.0)] if challenger_scores else [])
                for model, err in models:
                    con.execute(
                        "INSERT INTO predictions (station_id, issue_time, valid_time,"
                        " model_version, run_id, pm25_pred) VALUES (?, ?, ?, ?, 'r', ?)",
                        (station, issue.isoformat(), valid.isoformat(), model, base + err),
                    )
                    con.execute(
                        "INSERT INTO scores (station_id, issue_time, valid_time, model_version,"
                        " pm25_obs, abs_error) VALUES (?, ?, ?, ?, ?, ?)",
                        (station, issue.isoformat(), valid.isoformat(), model, base, err),
                    )
    # Hourly quality history for the last 12 hours, with one warning and a gap.
    for k in range(1, 13):
        if k == 5:
            continue
        at = (NOW.replace(minute=0, second=0, microsecond=0) - k * H).isoformat()
        for source, station, check, status, value in (
            ("air4thai", "59t", "freshness", "warn" if k == 1 else "ok", 0.4),
            ("air4thai", "59t", "missing_hours", "ok", 1.0),
            ("air4thai", "61t", "freshness", "ok", 0.3),
            (P, "59t", "freshness", "ok", 0.5),
        ):
            con.execute(
                "INSERT INTO data_quality_checks (checked_at, source, station_id, check_name,"
                " status, value) VALUES (?, ?, ?, ?, ?, ?)",
                (at, source, station, check, status, value),
            )
    day = NOW.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    ref = (NOW - timedelta(days=60)).isoformat()
    for feature, psi, status in (("pm25_latest", 0.05, "ok"), ("cams_pm25", 0.31, "alert")):
        con.execute(
            "INSERT INTO drift_checks (checked_at, model_version, feature, psi, status, ref_start,"
            " ref_end, recent_start, recent_end, n_ref, n_recent) VALUES (?, ?, ?, ?, ?, ?, ?, ?,"
            " ?, 100, 50)",
            (day, CHALLENGER, feature, psi, status, ref, ref, ref, ref),
        )
    con.execute(
        "INSERT INTO models (model_version, sha256, artifact, manifest, created_at)"
        " VALUES (?, 'x', x'00', '{}', ?)",
        ("retrain-20261026", (NOW - timedelta(days=4)).isoformat()),
    )
    con.commit()


def test_scored_build_json_shape_and_page(tmp_path):
    fake = FakeDb()
    seed(fake)
    assert build(fake, tmp_path) == 0
    data = load(tmp_path)
    assert data.keys() >= TOP_KEYS
    assert set(data["sections"]) == set(dashboard.SECTIONS)
    assert all(v == "ok" for v in data["sections"].values())
    assert data["headline"] == {"state": "yes", "text": "Yes, by 50.0%", "skill_pct": 50.0}
    assert data["kpi"]["mae_challenger_24h"] == 5.0
    assert data["kpi"]["mae_persistence_24h"] == 10.0
    assert data["kpi"]["stations_total"] == 2
    assert data["kpi"]["worst_psi"] == 0.31
    assert data["kpi"]["worst_psi_feature"] == "cams_pm25"
    assert data["status"] == "degraded"  # a warn in the latest checks, stations not reporting
    assert data["chart"]["has_challenger"] is True
    assert [s["id"] for s in data["chart"]["stations"]] == ["59t", "61t"]
    assert data["models"]["last_retrain"]["version"] == "retrain-20261026"
    pipeline = {p["name"]: p for p in data["quality"]["pipeline"]}
    assert set(pipeline) == {"Collect", "Predict", "Quality"}
    assert all(len(p["cells"]) == 24 for p in pipeline.values())
    assert "missed" in pipeline["Collect"]["cells"]
    page = (tmp_path / "index.html").read_text()
    for needle in (
        "Is it beating the naive guess",
        "Yes, by 50.0%",
        "Beats the baseline by",
        "Typical error (MAE, µg/m³)",
        "Data freshness",
        "Input drift (PSI)",
        "Pause animation",
        'aria-pressed="false"',
        "<caption>",
        'scope="col"',
        'tabindex="0"',
        "Alert",
        "cell miss",
    ):
        assert needle in page
    assert "Challenger (not scored yet)" not in page
    assert (tmp_path / "dashboard.css").exists() and (tmp_path / "dashboard.js").exists()


def test_empty_db_renders_waiting(tmp_path):
    assert build(FakeDb(), tmp_path) == 0
    data = load(tmp_path)
    assert data.keys() >= TOP_KEYS
    assert all(v == "ok" for v in data["sections"].values())
    assert data["status"] == "waiting"
    assert data["headline"]["state"] == "waiting"
    assert data["kpi"]["skill_24h"] is None and data["kpi"]["freshness_min"] is None
    assert data["chart"]["stations"] == []
    page = (tmp_path / "index.html").read_text()
    assert "Not scored yet" in page
    assert "Challenger (not scored yet)" in page
    assert "first +24 h forecasts are scored" in page
    assert 'class="val waiting"' in page


def test_waiting_note_uses_first_forecast_date(tmp_path):
    fake = FakeDb()
    seed(fake, challenger_scores=False)
    fake.con.execute(
        "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id,"
        " pm25_pred) VALUES ('59t', ?, ?, ?, 'r', 1.0)",
        (NOW.isoformat(), (NOW + 24 * H).isoformat(), CHALLENGER),
    )
    fake.con.commit()
    build(fake, tmp_path)
    data = load(tmp_path)
    assert data["headline"]["state"] == "waiting"
    assert data["scores"]["first_scores_from"] == (NOW + 24 * H).isoformat()
    assert "31 Oct, 12:20 UTC" in (tmp_path / "index.html").read_text()


def test_no_challenger_configured(tmp_path):
    fake = FakeDb()
    seed(fake)
    build(fake, tmp_path, challenger_model_version=None)
    page = (tmp_path / "index.html").read_text()
    assert load(tmp_path)["headline"]["state"] == "waiting"
    assert "No challenger model is configured" in page


def test_missing_table_degrades_one_section(tmp_path):
    fake = FakeDb()
    seed(fake)
    fake.con.execute("DROP TABLE drift_checks")
    assert build(fake, tmp_path) == 0
    data = load(tmp_path)
    assert data["sections"]["drift"] == "unavailable"
    assert data["drift"] is None and data["kpi"]["worst_psi"] is None
    assert data["sections"]["scores"] == "ok"
    assert data["status"] == "degraded"
    assert "Drift unavailable right now" in (tmp_path / "index.html").read_text()


def test_every_section_failing_writes_nothing(tmp_path):
    def down(_url):
        raise psycopg.OperationalError(f"connection refused {DB_URL}")

    out = tmp_path / "site"
    assert dashboard.run(settings(), out, connect=down, clock=lambda: NOW) == 1
    assert not out.exists()


def test_missing_database_url_is_config_error(tmp_path):
    with pytest.raises(dashboard.ConfigError):
        dashboard.run(settings(database_url=None), tmp_path, connect=FakeDb())


def test_no_secrets_in_output(tmp_path):
    fake = FakeDb()
    seed(fake)
    fake.con.execute("DROP TABLE drift_checks")  # also exercise the failure path
    build(fake, tmp_path)
    for path in tmp_path.iterdir():
        text = path.read_text()
        assert DB_URL not in text and DB_PASSWORD not in text
        assert "postgresql://" not in text


@pytest.mark.parametrize(
    ("skill", "state", "text"),
    [
        (None, "waiting", "Not scored yet"),
        (float("nan"), "waiting", "Not scored yet"),
        (0.123, "yes", "Yes, by 12.3%"),
        (-0.081, "not_yet", "Not yet (\u22128.1%)"),
        (0.0, "not_yet", "Not yet (0.0%)"),
    ],
)
def test_headline_logic(skill, state, text):
    result = headline(skill)
    assert (result["state"], result["text"]) == (state, text)


def test_negative_skill_page_uses_amber_not_yet(tmp_path):
    fake = FakeDb()
    seed(fake)
    fake.con.execute("UPDATE scores SET abs_error = 20 WHERE model_version = ?", (CHALLENGER,))
    fake.con.commit()
    build(fake, tmp_path)
    data = load(tmp_path)
    assert data["headline"]["state"] == "not_yet"
    assert data["headline"]["text"].startswith("Not yet (\u2212")
    assert 'class="answer notyet"' in (tmp_path / "index.html").read_text()


def test_html_escapes_station_ids(tmp_path):
    fake = FakeDb()
    fake.con.execute(
        "INSERT INTO observations (station_id, ts, pm25, source) VALUES (?, ?, 20, 'air4thai')",
        ("<script>x</script>", (NOW - H).isoformat()),
    )
    fake.con.commit()
    build(fake, tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page
