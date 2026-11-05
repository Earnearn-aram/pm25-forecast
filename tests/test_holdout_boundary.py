"""Synthetic boundary proofs only; never load the locked experiment snapshot."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pm25_forecast.experiments import folds
from pm25_forecast.holdout_boundary import ReviewedAnchor, fixture_row_allowed, verify_manifest


def digest(data):
    return hashlib.sha256(data).hexdigest()


def metadata(*, altered_start=False):
    last = datetime(2026, 10, 25, 23, tzinfo=UTC)
    split = folds.derive_folds(last)
    snapshot_sha = "a" * 64
    snapshot = json.dumps(
        {"snapshot_sha256": snapshot_sha, "time_range": {"last_usable_target": last.isoformat()}}
    ).encode()
    protocol = b"synthetic reviewed protocol"
    code_digest = digest(Path(folds.__file__).read_bytes())
    start = split.holdout_start + timedelta(hours=1) if altered_start else split.holdout_start
    data = json.dumps(
        {
            "schema_version": 1,
            "boundary_id": "synthetic-boundary",
            "snapshot_sha256": snapshot_sha,
            "snapshot_manifest_sha256": digest(snapshot),
            "protocol_sha256": digest(protocol),
            "derivation_code_sha256": code_digest,
            "lock_ref": "refs/tags/synthetic-lock",
            "s0": folds.S0.isoformat(),
            "last_usable_target": last.isoformat(),
            "holdout_start": start.isoformat(),
            "holdout_end": split.end.isoformat(),
            "exclusion_columns": {
                "observations": ["ts"],
                "forecast_vintages": ["issue_time", "valid_time"],
            },
        }
    ).encode()
    anchor = ReviewedAnchor(
        manifest_sha256=digest(data),
        snapshot_sha256=snapshot_sha,
        snapshot_manifest_sha256=digest(snapshot),
        protocol_sha256=digest(protocol),
        derivation_code_sha256=code_digest,
        lock_ref="refs/tags/synthetic-lock",
    )
    return data, anchor, snapshot, protocol


def verify(data, anchor, snapshot, protocol):
    return verify_manifest(data, anchor, snapshot_manifest=snapshot, protocol=protocol)


def test_reviewed_metadata_reproduces_boundary_without_reading_snapshot_rows():
    data, anchor, snapshot, protocol = metadata()
    result = verify(data, anchor, snapshot, protocol)
    assert result.holdout_start == datetime(2026, 10, 5, tzinfo=UTC)
    assert result.holdout_end == datetime(2026, 10, 26, tzinfo=UTC)


def test_rejects_changed_manifest_and_independently_pinned_inputs():
    data, anchor, snapshot, protocol = metadata()
    with pytest.raises(ValueError, match="reviewed digest"):
        verify(data + b" ", anchor, snapshot, protocol)
    with pytest.raises(ValueError, match="snapshot manifest"):
        verify(data, anchor, snapshot + b" ", protocol)
    with pytest.raises(ValueError, match="protocol"):
        verify(data, anchor, snapshot, protocol + b" ")
    with pytest.raises(ValueError, match="derivation_code_sha256"):
        verify(data, replace(anchor, derivation_code_sha256="b" * 64), snapshot, protocol)


def test_rejects_boundary_inconsistent_with_derived_fold_even_if_hash_is_pinned():
    data, anchor, snapshot, protocol = metadata(altered_start=True)
    with pytest.raises(ValueError, match="protocol-derived"):
        verify(data, anchor, snapshot, protocol)


def test_rejects_pinned_snapshot_manifest_with_a_different_target_range():
    data, anchor, snapshot, protocol = metadata()
    snapshot_data = json.loads(snapshot)
    snapshot_data["time_range"]["last_usable_target"] = "2026-10-25T22:00:00+00:00"
    changed_snapshot = json.dumps(snapshot_data).encode()
    boundary_data = json.loads(data)
    boundary_data["snapshot_manifest_sha256"] = digest(changed_snapshot)
    changed_boundary = json.dumps(boundary_data).encode()
    anchor = replace(
        anchor,
        manifest_sha256=digest(changed_boundary),
        snapshot_manifest_sha256=digest(changed_snapshot),
    )
    with pytest.raises(ValueError, match="target range"):
        verify(changed_boundary, anchor, changed_snapshot, protocol)


def test_short_snapshot_cannot_forge_holdout_using_an_earlier_start():
    data, anchor, snapshot, protocol = metadata()
    last = datetime(2026, 9, 30, 23, tzinfo=UTC)
    assert folds.derive_folds(last).holdout_start is None
    fake_s0 = datetime(2026, 5, 1, tzinfo=UTC)
    forged_split = folds.derive_folds(last, fake_s0)
    assert forged_split.holdout_start is not None
    snapshot_data = json.loads(snapshot)
    snapshot_data["time_range"]["last_usable_target"] = last.isoformat()
    changed_snapshot = json.dumps(snapshot_data).encode()
    boundary_data = json.loads(data)
    boundary_data.update(
        s0=fake_s0.isoformat(),
        last_usable_target=last.isoformat(),
        holdout_start=forged_split.holdout_start.isoformat(),
        holdout_end=forged_split.end.isoformat(),
        snapshot_manifest_sha256=digest(changed_snapshot),
    )
    changed_boundary = json.dumps(boundary_data).encode()
    anchor = replace(
        anchor,
        manifest_sha256=digest(changed_boundary),
        snapshot_manifest_sha256=digest(changed_snapshot),
    )
    with pytest.raises(ValueError, match="fixed S0"):
        verify(changed_boundary, anchor, changed_snapshot, protocol)


def test_target_at_protocol_cutoff_is_rejected_even_when_metadata_is_pinned():
    data, anchor, snapshot, protocol = metadata()
    snapshot_data = json.loads(snapshot)
    snapshot_data["time_range"]["last_usable_target"] = folds.TARGET_CUT.isoformat()
    changed_snapshot = json.dumps(snapshot_data).encode()
    boundary_data = json.loads(data)
    boundary_data.update(
        last_usable_target=folds.TARGET_CUT.isoformat(),
        snapshot_manifest_sha256=digest(changed_snapshot),
    )
    changed_boundary = json.dumps(boundary_data).encode()
    anchor = replace(
        anchor,
        manifest_sha256=digest(changed_boundary),
        snapshot_manifest_sha256=digest(changed_snapshot),
    )
    with pytest.raises(ValueError, match="protocol cutoff"):
        verify(changed_boundary, anchor, changed_snapshot, protocol)


def test_synthetic_exclusions_cover_both_boundaries_for_all_source_times():
    boundary = verify(*metadata())
    before = boundary.holdout_start - timedelta(hours=1)
    inside = boundary.holdout_start + timedelta(hours=1)
    assert fixture_row_allowed("observations", {"ts": before}, boundary)
    assert not fixture_row_allowed("observations", {"ts": boundary.holdout_start}, boundary)
    assert not fixture_row_allowed("observations", {"ts": inside}, boundary)
    assert fixture_row_allowed("observations", {"ts": boundary.holdout_end}, boundary)
    for column in ("issue_time", "valid_time"):
        row = {"issue_time": before, "valid_time": before}
        for timestamp in (boundary.holdout_start, inside):
            row[column] = timestamp
            assert not fixture_row_allowed("forecast_vintages", row, boundary)
        row[column] = boundary.holdout_end
        assert fixture_row_allowed("forecast_vintages", row, boundary)
    with pytest.raises(ValueError, match="unknown source"):
        fixture_row_allowed("observations", {}, boundary)
