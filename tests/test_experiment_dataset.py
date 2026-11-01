import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pm25_forecast.backtest import Observation, Vintage
from pm25_forecast.experiments import dataset
from pm25_forecast.experiments.dataset import (
    DATA_SOURCE_BREAK,
    KAN74_CUTOFF,
    export_snapshot,
    load_snapshot,
    normalise_and_filter,
)

FIXTURE = Path(__file__).parent / "fixtures" / "experiment_dataset_rows.json"
CREATED = datetime(2026, 11, 1, tzinfo=UTC)


def _rows():
    data = json.loads(FIXTURE.read_text())
    parse = datetime.fromisoformat
    obs = [(s, parse(t), v, src, parse(a)) for s, t, v, src, a in data["observations"]]
    vin = [(s, parse(i), parse(v), val, parse(a), m) for s, i, v, val, a, m in data["vintages"]]
    return obs, vin


class FakeConn:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement):
        self.statements.append(statement)
        return self

    def fetchall(self):
        obs, vin = _rows()
        return obs if "FROM observations" in self.statements[-1] else vin


def test_constants_are_aware_utc():
    assert datetime(2026, 7, 31, 17, tzinfo=UTC) == DATA_SOURCE_BREAK
    assert datetime(2026, 10, 26, tzinfo=UTC) == KAN74_CUTOFF
    assert DATA_SOURCE_BREAK.utcoffset() is not None and KAN74_CUTOFF.utcoffset() is not None


def test_filters_default_and_repaired():
    obs, vin = _rows()
    o, v = normalise_and_filter(obs, vin)
    assert [(r[0], r[1].isoformat()) for r in o] == [
        ("02t", "2026-07-31T18:00:00+00:00"),
        ("02t", "2026-07-31T20:00:00+00:00"),
        ("59t", "2026-07-31T17:00:00+00:00"),
        ("59t", "2026-07-31T19:00:00+00:00"),
        ("59t", "2026-10-25T23:00:00+00:00"),
    ]
    assert [r[2].isoformat() for r in v] == [
        "2026-07-31T18:00:00+00:00",
        "2026-10-25T23:00:00+00:00",
    ]
    assert {r[5] for r in v} == {"cams"}
    o2, v2 = normalise_and_filter(obs, vin, kan74_repaired=True)
    assert len(o2) == len(o) + 1 and len(v2) == len(v) + 1


def test_air4thai_available_at_is_ts_plus_one_hour_and_audit_column_kept():
    obs, vin = normalise_and_filter(*_rows())
    row = next(r for r in obs if r[0] == "59t" and r[1].hour == 17)
    assert row[4] == datetime(2026, 7, 31, 18, tzinfo=UTC)
    assert row[5] == datetime(2026, 7, 31, 17, 5, tzinfo=UTC)
    assert vin[0][4] == datetime(2026, 7, 31, 12, 30, tzinfo=UTC)


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        normalise_and_filter([("a", datetime(2026, 8, 1), 1.0, "x", CREATED)], [])  # noqa: DTZ001


def test_export_is_read_only_and_hash_deterministic(tmp_path):
    conn = FakeConn()
    m1 = export_snapshot(tmp_path / "a", "postgresql://unused", connect=lambda _url: conn)
    assert conn.statements[0] == "SET TRANSACTION READ ONLY"
    assert not any(
        s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "DROP", "CREATE"))
        for s in conn.statements
    )
    m2 = export_snapshot(tmp_path / "b", "postgresql://unused", connect=lambda _url: FakeConn())
    assert m1["snapshot_sha256"] == m2["snapshot_sha256"]
    m3 = export_snapshot(
        tmp_path / "c", "postgresql://unused", kan74_repaired=True, connect=lambda _url: FakeConn()
    )
    assert m3["snapshot_sha256"] != m1["snapshot_sha256"]


def test_hash_ignores_input_order():
    obs, vin = _rows()
    a = dataset.content_sha256(*normalise_and_filter(obs, vin))
    b = dataset.content_sha256(*normalise_and_filter(obs[::-1], vin[::-1]))
    assert a == b


def test_manifest_fields_and_round_trip(tmp_path):
    obs, vin = normalise_and_filter(*_rows())
    manifest = dataset.write_snapshot(
        tmp_path / "s", obs, vin, kan74_repaired=False, created_at=CREATED, git_sha="abc123"
    )
    on_disk = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["snapshot_sha256"] == manifest["content_sha256"]
    assert "0 vintages" in manifest["vintages_note"]
    assert manifest["created_at"] == "2026-11-01T00:00:00Z"
    assert manifest["git_sha"] == "abc123"
    assert manifest["stations"] == ["02t", "59t"]
    assert manifest["time_range"]["first_usable_target"] == "2026-07-31T17:00:00Z"
    assert manifest["time_range"]["last_usable_target"] == "2026-10-25T23:00:00Z"
    rc = manifest["row_counts"]
    assert rc["observations_total"] == 5 and rc["vintages_total"] == 2
    assert rc["observations_by_station_source"]["59t"] == {"air4thai": 2, "openaq": 1}
    assert rc["vintages_by_station"] == {"59t": 2}
    span = datetime(2026, 10, 25, 23, tzinfo=UTC) - DATA_SOURCE_BREAK
    expected_hours = int(span.total_seconds() / 3600) + 1
    assert manifest["missing_hour_rate_by_station"]["59t"] == round(1 - 3 / expected_hours, 6)
    assert manifest["filters"]["kan74_repaired"] is False
    assert manifest["filters"]["data_source_break_utc"] == "2026-07-31T17:00:00Z"

    loaded_obs, loaded_vin, loaded_manifest = load_snapshot(tmp_path / "s")
    assert loaded_manifest == manifest
    assert all(isinstance(o, Observation) for o in loaded_obs)
    assert all(isinstance(v, Vintage) for v in loaded_vin)
    # air4thai: available_at = ts + 1 h, not the real ingested_at (18:05)
    assert loaded_obs[0] == Observation(
        "02t",
        datetime(2026, 7, 31, 18, tzinfo=UTC),
        15.25,
        datetime(2026, 7, 31, 19, tzinfo=UTC),
    )
    # openaq keeps its ingested_at
    openaq = next(o for o in loaded_obs if o.station_id == "59t" and o.ts.hour == 19)
    assert openaq.available_at == datetime(2026, 7, 31, 19, 10, tzinfo=UTC)
    assert "ts + 1 h" in manifest["filters"]["available_at"]
    assert loaded_vin[0].model == "cams" and loaded_vin[0].valid_time.tzinfo is not None
    assert len(loaded_obs) == 5 and len(loaded_vin) == 2


def test_load_detects_tampering(tmp_path):
    obs, vin = normalise_and_filter(*_rows())
    dataset.write_snapshot(tmp_path / "s", obs, vin, kan74_repaired=False, git_sha="x")
    path = tmp_path / "s" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["snapshot_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(tmp_path / "s")


def test_refuses_non_empty_directory(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_text("1")
    with pytest.raises(FileExistsError):
        dataset.write_snapshot(tmp_path / "x", [], [], kan74_repaired=False)


def test_cli_rejects_out_inside_repo():
    root = Path(dataset.__file__).resolve().parents[3]
    assert dataset.main(["export", "--out", str(root / "snap")]) == 2


def test_cli_fails_closed_without_repo_root(monkeypatch, tmp_path):
    def boom(*_a):
        raise OSError("no git")

    monkeypatch.setattr(dataset, "_git", boom)
    assert dataset.main(["export", "--out", str(tmp_path / "s")]) == 2


def test_off_hour_observation_rejected():
    row = ("a", datetime(2026, 8, 1, 0, 30, tzinfo=UTC), 1.0, "air4thai", CREATED, CREATED)
    with pytest.raises(ValueError, match="UTC hour"):
        dataset.build_manifest([row], [], kan74_repaired=False, git_sha="x")


def test_cutoff_is_the_settings_constant():
    from pm25_forecast.settings import KAN74_BUG_CUTOFF

    assert dataset.KAN74_CUTOFF is KAN74_BUG_CUTOFF
