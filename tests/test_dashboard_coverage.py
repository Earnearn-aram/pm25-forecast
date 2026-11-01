"""KAN-102: measured source coverage and the drift 'not comparable' text on the dashboard."""

import json
from datetime import timedelta

from pm25_forecast import dashboard
from pm25_forecast.drift import DATA_SOURCE_BREAK
from test_dashboard import NOW, Fake, H, build, load, seed

STATIONS = list(dashboard.STATION_MAP)
END = NOW.replace(minute=0, second=0, microsecond=0) - H  # 11:00, the last hour already due


def put_obs(fake, station, ts, source, pm25=20.0):
    fake.con.execute(
        "INSERT INTO observations (station_id, ts, pm25, source) VALUES (?, ?, ?, ?)",
        (station, ts.isoformat(), pm25, source),
    )


def put_cams(fake, issue):
    fake.con.execute(
        "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id,"
        " pm25_pred) VALUES ('59t', ?, ?, 'cams_global', 'collector', 10)",
        (issue.isoformat(), (issue + H).isoformat()),
    )


def test_air4thai_coverage_counts_station_hours_and_never_rounds_up(tmp_path):
    fake = Fake()
    for sid in STATIONS:  # 5 stations x last 24 h = full, except one missing hour
        for i in range(24):
            if sid == "59t" and i == 3:
                continue
            put_obs(fake, sid, END - i * H, "air4thai")
    put_obs(fake, "other", END, "air4thai")  # not a pilot station: ignored
    fake.con.commit()
    assert build(fake, tmp_path) == 0
    a = load(tmp_path)["coverage"]["air4thai"]
    assert a["available"] and a["stations"] == 5
    assert a["pct_24h"] == 99.1  # 119/120 = 99.17, floored: never shown as 100
    assert a["pct_168h"] == 14.1  # 119/840
    assert a["newest_age_min"] == 80
    page = (tmp_path / "index.html").read_text()
    assert "99.1% last 24 h" in page and "14.1% last 7 d" in page
    assert "5 stations x 1 reading per UTC hour" in page
    assert "Coverage not measured" not in page


def test_openaq_coverage_is_history_only_with_range(tmp_path):
    fake = Fake()
    first = DATA_SOURCE_BREAK - timedelta(hours=10)
    for sid in STATIONS:
        for i in range(10):
            if sid == "61t" and i < 5:
                continue
            put_obs(fake, sid, first + i * H, "openaq")
    put_obs(fake, "59t", DATA_SOURCE_BREAK, "openaq")  # inside the Air4Thai era: not counted
    fake.con.commit()
    build(fake, tmp_path)
    o = load(tmp_path)["coverage"]["openaq"]
    assert o["pct"] == 90.0  # 45 of 5 x 10
    assert o["first"].startswith("2026-07-31T07:00") and o["last"].startswith("2026-07-31T16:00")
    page = (tmp_path / "index.html").read_text()
    assert "OpenAQ is history only, not live" in page
    assert "90.0% of history" in page
    assert "OpenAQ is history only" in dashboard._coverage_texts("openaq", o)[1][0]


def test_cams_coverage_counts_hours_with_a_vintage(tmp_path):
    fake = Fake()
    for i in range(12):  # twelve of the last 24 hours, three vintages per hour
        for minute in (0, 20, 40):
            put_cams(fake, NOW.replace(minute=0) - i * H + timedelta(minutes=minute))
    fake.con.commit()
    build(fake, tmp_path)
    c = load(tmp_path)["coverage"]["cams"]
    assert c["pct_24h"] == 50.0 and c["pct_168h"] == 7.1  # 12/168 = 7.14
    assert c["newest_age_min"] is not None


def test_no_rows_is_not_available_never_100(tmp_path):
    assert build(Fake(), tmp_path) == 0
    data = load(tmp_path)
    for src in data["coverage"].values():
        assert src == {"available": False, "reason": "no_rows"}
    assert data["sections"]["coverage"] == "ok"
    page = (tmp_path / "index.html").read_text()
    assert "not available (no rows stored yet)" in page
    assert "ไม่พร้อมใช้งาน (ยังไม่มีข้อมูลในฐานข้อมูล)" in page
    assert "100" not in json.dumps(data["coverage"])


def test_failing_query_degrades_only_that_source(tmp_path):
    fake = Fake()
    seed(fake)
    fake.con.execute("DROP TABLE data_quality_checks")  # unrelated section fails too
    real = fake.execute

    def execute(statement, params=()):
        if "model_version = %(m)s AND issue_time >=" in statement:
            raise RuntimeError("boom secret-host")
        return real(statement, params)

    fake.execute = execute
    assert build(fake, tmp_path) == 0
    data = load(tmp_path)
    assert data["coverage"]["cams"] == {"available": False, "reason": "query_failed"}
    assert data["coverage"]["air4thai"]["available"]
    assert data["sections"]["coverage"] == "degraded"
    page = (tmp_path / "index.html").read_text()
    assert "the database query failed" in page and "secret-host" not in page


def test_drift_insufficient_shows_reason_in_both_languages(tmp_path):
    fake = Fake()
    seed(fake)
    for feature in ("cams_pm25", "pm25_latest"):
        fake.con.execute(
            "UPDATE drift_checks SET psi = NULL, status = 'insufficient' WHERE feature = ?",
            (feature,),
        )
    fake.con.commit()
    build(fake, tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert ">n/a<" not in page.split("Drift")[1].split("PSI measures")[0]
    assert "this model was trained on data from before the current data source" in page
    assert "จึงไม่มีข้อมูลอ้างอิงที่ตรงกัน" in page
    assert "CAMS history is still too short" in page
