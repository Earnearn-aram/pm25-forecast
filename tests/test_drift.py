"""KAN-22: drift detection (PSI). Offline: SQLite built from schema.sql."""

import json
import math
import socket
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from tenacity import wait_none

from pm25_forecast import drift
from pm25_forecast.drift import (
    ALERT,
    INSUFFICIENT,
    OK,
    TARGET_FEATURE,
    WARN,
    DriftResult,
    classify,
    latest_drift_alert,
    psi,
    run,
)
from pm25_forecast.features import FEATURE_NAMES
from pm25_forecast.settings import Settings
from pm25_forecast.storage import db
from pm25_forecast.storage.guard import RunSkipped, run_entrypoint
from test_predict import DB_PASSWORD, DB_URL, FakeDb

CHALLENGER = "kan15-retrospective-24h-lag1h"
CHAMPION = "champ-v1"
NOW = datetime(2026, 10, 30, 12, 20, tzinfo=UTC)
H = timedelta(hours=1)
REF_START = datetime(2026, 9, 1, tzinfo=UTC)
REF_END = datetime(2026, 9, 21, tzinfo=UTC)
# Hour-aligned recent origins: CAMS valid times must fall on the hour to match build_features.
RECENT_FROM = datetime(2026, 10, 22, 12, tzinfo=UTC)
RECENT_TO = datetime(2026, 10, 30, 12, tzinfo=UTC)
SHIFT_FROM =datetime(2026, 10, 23, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_a, **_k):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(db, "RETRY_WAIT", wait_none())


def settings(**kw):
    values = {"database_url": DB_URL, "challenger_model_version": CHALLENGER}
    return Settings(_env_file=None, **{**values, **kw})


def do_run(fake, now=NOW, **kw) -> DriftResult:
    return run(settings(**kw), connect=fake, clock=lambda: now)


@pytest.fixture
def fake():
    return FakeDb()


def pm25(ts: datetime, shift: float = 0.0) -> float:
    return 20.0 + (int(ts.timestamp() // 3600) * 37) % 23 + shift  # stationary, deterministic


def add_model(fake, version, lag=0, start=REF_START, end=REF_END, horizons=(24,)):
    manifest = {
        "version": version,
        "training_window": {"start": start.isoformat(), "end": end.isoformat()},
        "assumed_reporting_lag_hours": lag,
        "metrics": {str(h): {"mae": 1.0} for h in horizons},
    }
    fake.con.execute(
        "INSERT INTO models (model_version, sha256, artifact, manifest) VALUES (?, 'x', x'00', ?)",
        (version, json.dumps(manifest)),
    )
    fake.con.commit()


def add_series(fake, start=REF_START - 2 * H, end=NOW, shift=0.0, shift_from=SHIFT_FROM, fn=None):
    ts, rows = start, []
    while ts <= end:
        value = fn(ts) if fn else pm25(ts, shift if ts >= shift_from else 0.0)
        rows.append(("59t", ts.isoformat(), value))
        ts += H
    fake.con.executemany(
        "INSERT INTO observations (station_id, ts, pm25, source) VALUES (?, ?, ?, 'test') "
        "ON CONFLICT (station_id, ts) DO UPDATE SET pm25 = excluded.pm25",
        rows,
    )
    fake.con.commit()


def shift_recent(fake, delta):
    fake.con.execute(
        "UPDATE observations SET pm25 = pm25 + ? WHERE ts >= ?", (delta, SHIFT_FROM.isoformat())
    )
    fake.con.commit()


def add_cams(fake, start, end, every=1):
    """CAMS vintages (issued 1 h before an origin, valid 24 h after it) for every n-th origin."""
    rows, origin, i = [], start, 0
    while origin <= end:
        if i % every == 0:
            valid = origin + 24 * H
            rows.append(("59t", (origin - H).isoformat(), valid.isoformat(), pm25(valid)))
        origin += H
        i += 1
    fake.con.executemany(
        "INSERT INTO predictions (station_id, issue_time, valid_time, model_version, run_id,"
        f" pm25_pred) VALUES (?, ?, ?, '{drift.CAMS_MODEL_VERSION}', 'collector', ?)",
        rows,
    )
    fake.con.commit()


def two_alert_days(fake):
    """Shifted recent week seen on two consecutive daily runs; returns the second run's time."""
    add_model(fake, CHALLENGER)
    add_series(fake, shift=60.0)
    do_run(fake)
    later = NOW + timedelta(days=1)
    add_series(fake, start=NOW, end=later, shift=60.0)
    do_run(fake, now=later)
    return later


def checks(fake):
    return fake.con.execute(
        "SELECT checked_at, model_version, feature, psi, status, n_ref, n_recent "
        "FROM drift_checks ORDER BY 1, 2, 3"
    ).fetchall()


def by_feature(result, model=CHALLENGER):
    return {c.feature: c for c in result.checks if c.model_version == model}


# --- psi ---------------------------------------------------------------------------------------


def test_identical_distributions_give_zero():
    data = [float(i % 17) for i in range(500)]
    assert psi(data, list(data)) == pytest.approx(0.0, abs=1e-12)


def test_known_shift_matches_hand_computed_value():
    # bins=2: the edge is the reference median 2.5; shares ref 0.5/0.5, recent 0.75/0.25.
    # PSI = (0.75-0.5)ln(0.75/0.5) + (0.25-0.5)ln(0.25/0.5) = 0.25 ln 3
    assert psi([1, 2, 3, 4], [1, 1, 1, 4], bins=2) == pytest.approx(0.25 * math.log(3))


def test_empty_bin_uses_epsilon():
    eps = drift.PSI_EPSILON
    expected = (eps - 0.5) * math.log(eps / 0.5) + (1 - 0.5) * math.log(1 / 0.5)
    assert psi([1, 2, 3, 4], [5, 6, 7, 8], bins=2) == pytest.approx(expected)


def test_non_finite_values_are_dropped():
    ref, rec = [1, 2, 3, 4], [1, 1, 1, 4]
    dirty_ref = [*ref, float("nan"), None, float("inf")]
    dirty_rec = [*rec, float("-inf"), float("nan")]
    assert psi(dirty_ref, dirty_rec, bins=2) == psi(ref, rec, bins=2)


def test_psi_rejects_empty_input_and_tiny_bins():
    with pytest.raises(ValueError):
        psi([float("nan")], [1.0])
    with pytest.raises(ValueError):
        psi([1.0], [])
    with pytest.raises(ValueError):
        psi([1.0], [1.0], bins=1)


def test_constant_reference_still_detects_a_shift():
    assert psi([5.0] * 50, [5.0] * 50) == pytest.approx(0.0)
    assert psi([5.0] * 50, [9.0] * 50) > 0.2


@pytest.mark.parametrize(
    ("value", "status"),
    [(0.0, OK), (0.0999, OK), (0.1, WARN), (0.2, WARN), (0.2001, ALERT), (3.0, ALERT)],
)
def test_status_thresholds(value, status):
    assert classify(value, 500, 200) == status


def test_too_few_points_is_insufficient_whatever_the_psi():
    assert classify(5.0, drift.MIN_REFERENCE - 1, 500) == INSUFFICIENT
    assert classify(5.0, 500, drift.MIN_RECENT - 1) == INSUFFICIENT
    assert classify(5.0, drift.MIN_REFERENCE, drift.MIN_RECENT) == ALERT
    assert classify(None, 500, 500) == INSUFFICIENT


# --- run ---------------------------------------------------------------------------------------


def test_stable_data_is_ok_for_features_and_target(fake):
    add_model(fake, CHALLENGER)
    add_series(fake)
    result = do_run(fake)
    got = by_feature(result)
    assert set(got) == {*FEATURE_NAMES, TARGET_FEATURE}
    for name in ("pm25_latest", "pm25_same_hour_24h", TARGET_FEATURE):
        assert got[name].status == OK and got[name].psi < 0.1
    # No CAMS vintages stored: too few points, and that is never an alert.
    assert got["cams_pm25"].status == INSUFFICIENT and got["cams_pm25"].psi is None
    assert not result.alerts and not result.failures
    assert not latest_drift_alert(fake)


def test_first_day_above_threshold_is_only_a_warning(fake):
    add_model(fake, CHALLENGER)
    add_series(fake, shift=60.0)
    result = do_run(fake)
    got = by_feature(result)
    assert got["pm25_latest"].status == WARN and got[TARGET_FEATURE].status == WARN
    assert got[TARGET_FEATURE].psi > 0.2  # the PSI is stored; only the status waits a day
    assert not result.alerts and not latest_drift_alert(fake, now=NOW)
    window = got[TARGET_FEATURE]
    assert window.ref.start == REF_START
    assert window.recent.end == datetime(2026, 10, 30, 12, tzinfo=UTC)


def test_second_consecutive_day_above_threshold_alerts(fake):
    later = two_alert_days(fake)
    rows = {r[2]: r[4] for r in checks(fake) if r[0].startswith("2026-10-31")}
    assert rows["pm25_latest"] == ALERT and rows[TARGET_FEATURE] == ALERT
    assert latest_drift_alert(fake, now=later)


def test_alert_needs_the_previous_run_to_be_recent_and_above_threshold(fake):
    add_model(fake, CHALLENGER)
    add_series(fake, shift=60.0)
    do_run(fake)
    add_series(fake, start=NOW, end=NOW + timedelta(days=4), shift=60.0)
    result = do_run(fake, now=NOW + timedelta(days=3))  # the run before is 3 days old
    assert not result.alerts
    fake.con.execute("UPDATE drift_checks SET psi = 0.15")  # previous run only moderate
    fake.con.commit()
    result = do_run(fake, now=NOW + timedelta(days=4))
    assert not result.alerts


def test_manifest_lag_without_extra_horizons_has_no_same_hour_feature(fake):
    # A 24 h model with a reporting lag never sees the origin's own reading, in training either.
    add_model(fake, CHALLENGER, lag=1)
    add_series(fake)
    got = by_feature(do_run(fake))
    assert got["pm25_same_hour_24h"].status == INSUFFICIENT
    assert got["pm25_same_hour_24h"].n_ref == 0
    assert got["pm25_latest"].status == OK


def test_same_hour_feature_has_data_with_lag_for_shorter_horizons(fake):
    add_model(fake, CHALLENGER, lag=1, horizons=(6, 12, 24))
    add_series(fake)
    got = by_feature(do_run(fake))["pm25_same_hour_24h"]
    assert got.n_ref > 0 and got.n_recent > 0
    assert got.status != INSUFFICIENT


def test_too_little_recent_data_is_insufficient_and_no_alert(fake):
    add_model(fake, CHALLENGER)
    add_series(fake, end=NOW - timedelta(days=6, hours=12), shift=60.0)  # ~12 recent hours
    result = do_run(fake)
    got = by_feature(result)
    assert all(got[n].status == INSUFFICIENT for n in (*FEATURE_NAMES, TARGET_FEATURE))
    assert got[TARGET_FEATURE].n_recent < drift.MIN_RECENT
    assert not result.alerts and not latest_drift_alert(fake)


def test_rerun_same_day_upserts_and_next_day_adds_rows(fake):
    add_model(fake, CHALLENGER)
    add_series(fake)
    do_run(fake)
    first = checks(fake)
    assert len(first) == len(FEATURE_NAMES) + 1
    do_run(fake, now=NOW + timedelta(minutes=30))  # same UTC day and hour, same data
    assert checks(fake) == first
    shift_recent(fake, 80.0)
    do_run(fake, now=NOW + timedelta(hours=4))
    updated = checks(fake)
    assert len(updated) == len(first) and updated != first  # same keys, new values
    assert {r[4] for r in updated if r[2] == TARGET_FEATURE} == {WARN}  # first day above 0.2
    do_run(fake, now=NOW + timedelta(days=1))
    assert len(checks(fake)) == 2 * len(first)


def test_latest_drift_alert_filters_by_model_and_ignores_stale_runs(fake):
    later = two_alert_days(fake)
    assert latest_drift_alert(fake, now=later)
    assert latest_drift_alert(fake, CHALLENGER, now=later)
    assert not latest_drift_alert(fake, "some-other-model", now=later)
    assert latest_drift_alert(fake, now=later + timedelta(hours=35))
    assert not latest_drift_alert(fake, now=later + timedelta(hours=37))
    assert not latest_drift_alert(fake, CHALLENGER, now=later + timedelta(days=5))


def test_latest_drift_alert_is_false_on_an_empty_table(fake):
    assert latest_drift_alert(fake, now=NOW) is False
    assert latest_drift_alert(fake, CHALLENGER, now=NOW) is False


def test_reference_is_cached_per_model_and_not_rebuilt_daily(fake):
    add_model(fake, CHALLENGER)
    add_series(fake)
    do_run(fake)
    cached = fake.con.execute("SELECT feature FROM drift_reference ORDER BY 1").fetchall()
    assert len(cached) == len(FEATURE_NAMES) + 1
    first = checks(fake)
    # The training-window observations are gone; only the cached reference can explain the result.
    fake.con.execute("DELETE FROM observations WHERE ts < '2026-10-01'")
    fake.con.commit()
    do_run(fake, now=NOW + timedelta(minutes=30))
    assert checks(fake) == first


def test_cams_needs_half_of_the_reference_origins_covered(fake):
    add_model(fake, CHALLENGER)
    add_series(fake)
    add_cams(fake, REF_START, REF_END, every=5)  # 20 % of the reference origins
    add_cams(fake, RECENT_FROM, RECENT_TO)
    assert by_feature(do_run(fake))["cams_pm25"].status == INSUFFICIENT


def test_cams_with_full_coverage_is_compared(fake):
    add_model(fake, CHALLENGER)
    add_series(fake)
    add_cams(fake, REF_START, REF_END)
    add_cams(fake, RECENT_FROM, RECENT_TO)
    got =by_feature(do_run(fake))["cams_pm25"]
    assert got.n_ref >= drift.MIN_REFERENCE and got.status in {OK, WARN}


def test_integer_heavy_reference_with_fewer_than_three_bins_is_insufficient(fake):
    def mostly_ten(ts):
        return 11.0 if ts.timestamp() // 3600 % 20 == 0 else 10.0  # 95 % identical values

    add_model(fake, CHALLENGER)
    add_series(fake, fn=mostly_ten)
    result = do_run(fake)
    assert {c.status for c in result.checks} == {INSUFFICIENT}
    assert all(c.psi is None for c in result.checks)


def test_make_reference_rules():
    spread = [float(i % 40) for i in range(400)]
    assert drift.make_reference("pm25_latest", spread, 400).usable
    assert not drift.make_reference("pm25_latest", [10.0] * 380 + [11.0] * 20, 400).usable
    assert not drift.make_reference("pm25_latest", spread[:100], 100).usable  # too few points
    assert not drift.make_reference("cams_pm25", spread, 1000).usable  # 40 % coverage
    assert drift.make_reference("cams_pm25", spread, 700).usable  # 57 % coverage


def test_psi_with_stored_profile_equals_direct_psi():
    ref = [float(i % 40) for i in range(400)]
    recent = [float(i % 25) for i in range(200)]
    edges, shares = drift.reference_profile(ref, 5)
    assert drift.psi_vs_profile(edges, shares, recent) == pytest.approx(psi(ref, recent, 5))


def test_champion_is_checked_too_and_not_twice_when_same(fake):
    add_model(fake, CHALLENGER)
    add_model(fake, CHAMPION)
    add_series(fake)
    fake.con.execute(
        "INSERT INTO model_champion (slot, model_version) VALUES ('production', ?)", (CHAMPION,)
    )
    fake.con.commit()
    do_run(fake)
    assert {r[1] for r in checks(fake)} == {CHALLENGER, CHAMPION}
    fake.con.execute("UPDATE model_champion SET model_version = ?", (CHALLENGER,))
    fake.con.execute("DELETE FROM drift_checks")
    fake.con.commit()
    do_run(fake)
    assert {r[1] for r in checks(fake)} == {CHALLENGER}


def test_unknown_model_is_a_failure_and_nothing_is_written(fake):
    add_series(fake)
    result = do_run(fake)
    assert result.failures and "not found" in result.failures[0]
    assert checks(fake) == []


def test_bad_manifest_window_is_a_failure(fake):
    fake.con.execute(
        "INSERT INTO models (model_version, sha256, artifact, manifest) VALUES (?, 'x', x'00', ?)",
        (CHALLENGER, json.dumps({"training_window": {"start": "nope"}})),
    )
    fake.con.commit()
    assert "training_window" in do_run(fake).failures[0]


def test_nothing_to_check_is_a_failure(fake):
    assert do_run(fake, challenger_model_version=None).failures


def test_missing_database_url_is_a_config_error(fake):
    with pytest.raises(drift.ConfigError):
        run(Settings(_env_file=None), connect=fake, clock=lambda: NOW)


def test_naive_clock_is_rejected(fake):
    with pytest.raises(ValueError):
        do_run(fake, now=NOW.replace(tzinfo=None))


def test_db_down_skips_alerts_and_exits_nonzero_without_leaking_url(caplog):
    def refuse(_url):
        raise psycopg.OperationalError(f"connection refused {DB_URL}")

    alerts: list[str] = []
    result, code = run_entrypoint(
        lambda: run(settings(), connect=refuse, clock=lambda: NOW),
        alerts.append,
        connection_errors=db.CONNECTION_ERRORS,
    )
    assert isinstance(result, RunSkipped) and code == 1
    assert len(alerts) == 1 and DB_PASSWORD not in alerts[0] and DB_PASSWORD not in caplog.text


def test_main_exit_codes_and_a_single_alert_per_run(monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(drift, "get_settings", lambda: settings())
    monkeypatch.setattr(drift, "make_alert", lambda *_a, **_k: sent.append)

    def use(fn):
        monkeypatch.setattr(drift, "run", fn)

    day = datetime(2026, 10, 30, tzinfo=UTC)
    window = drift.Window(start=REF_START, end=REF_END)

    def check(feature, status, value):
        return drift.Check(CHALLENGER, feature, value, status, window, window, 500, 168)

    use(lambda *_a, **_k: DriftResult(run_id="r", checked_at=day))
    assert drift.main() == 0 and sent == []

    use(
        lambda *_a, **_k: DriftResult(
            run_id="r",
            checked_at=day,
            checks=[
                check("pm25_latest", ALERT, 0.31),
                check(TARGET_FEATURE, ALERT, 0.44),
                check("cams_pm25", INSUFFICIENT, None),
                check("pm25_same_hour_24h", WARN, 0.15),
            ],
        )
    )
    assert drift.main() == 0  # drift is reported, not a failure
    assert len(sent) == 1
    assert "pm25_latest" in sent[0] and TARGET_FEATURE in sent[0]
    assert "cams_pm25" not in sent[0] and "pm25_same_hour_24h" not in sent[0]

    sent.clear()
    use(lambda *_a, **_k: DriftResult(run_id="r", checked_at=day, failures=["x"]))
    assert drift.main() == 1 and len(sent) == 1

    def down(*_a, **_k):
        raise psycopg.OperationalError("connection refused")

    use(down)
    assert drift.main() == 1

    def bad_config(*_a, **_k):
        raise drift.ConfigError("nope")

    use(bad_config)
    assert drift.main() == 2

    def crash(*_a, **_k):
        raise RuntimeError("boom")

    use(crash)
    assert drift.main() == 1
