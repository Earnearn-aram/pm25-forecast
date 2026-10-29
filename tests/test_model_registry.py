import json
import threading
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pytest

from pm25_forecast.storage.registry import (
    artifact_sha256,
    promote_model,
    publish_model,
    read_champion,
    read_model,
    write_model,
)


class InMemoryRegistry:
    def __init__(self):
        self.rows = {}
        self.champion = None
        self.lock = threading.Lock()

    @property
    def row(self):
        return self.rows.get("test-v1")

    @row.setter
    def row(self, value):
        self.rows["test-v1"] = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement, params):
        if statement.lstrip().startswith("INSERT INTO models"):
            with self.lock:
                self.rows.setdefault(
                    params["model_version"],
                    (params["sha256"], params["artifact"], params["manifest"]),
                )
        elif statement.lstrip().startswith("INSERT INTO model_champion"):
            with self.lock:
                self.champion = params["model_version"]
        elif "FROM model_champion" in statement:
            with self.lock:
                self._selected = (self.champion,) if self.champion else None
        elif statement.lstrip().startswith("SELECT"):
            with self.lock:
                self._selected = self.rows.get(params["model_version"])
        return self

    def fetchone(self):
        with self.lock:
            return self._selected


FEATURE_LIST = ["pm25_latest", "cams_pm25"]


def model_artifact(offset=0):
    features = np.asarray([[1.0 + offset, 2.0], [2.0 + offset, 1.0], [3.0 + offset, 0.0]])
    dataset = lgb.Dataset(features, label=np.asarray([1.0, 2.0, 3.0]), feature_name=FEATURE_LIST)
    booster_text = lgb.train(
        {"objective": "regression", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 1},
        dataset,
        num_boost_round=2,
    ).model_to_string()
    return json.dumps(
        {
            "format": 1,
            "training_id": offset,
            "horizons": {str(horizon): booster_text for horizon in (6, 12, 24)},
        },
        separators=(",", ":"),
    ).encode()


ARTIFACT = model_artifact()

MANIFEST = {
    "version": "test-v1",
    "metrics": {"24": {"mae": 1.2}},
    "training_window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-02-01T00:00:00+00:00"},
    "feature_list": FEATURE_LIST,
    "data_sha256": "1" * 64,
    "artifact_sha256": artifact_sha256(ARTIFACT),
}


def manifest_for(artifact):
    return {**MANIFEST, "artifact_sha256": artifact_sha256(artifact)}


def test_registry_write_reads_back_and_verifies_artifact():
    con = InMemoryRegistry()
    digest = write_model(con, "test-v1", ARTIFACT, MANIFEST)

    artifact, manifest = read_model(con, "test-v1")
    assert artifact == ARTIFACT
    assert manifest == MANIFEST
    assert con.row[0] == digest


def test_registry_rejects_manifest_artifact_hash_mismatch():
    with pytest.raises(ValueError, match="artifact_sha256 does not match"):
        write_model(InMemoryRegistry(), "test-v1", model_artifact(offset=1), MANIFEST)


def test_registry_rejects_manifest_version_that_differs_from_registry_key():
    with pytest.raises(ValueError, match="manifest version does not match model_version"):
        write_model(InMemoryRegistry(), "test-v2", ARTIFACT, MANIFEST)


def test_registry_rejects_feature_list_that_differs_from_model_bundle():
    incompatible = {**MANIFEST, "feature_list": ["pm25_latest"]}
    with pytest.raises(ValueError, match="feature_list does not match model artifact"):
        write_model(InMemoryRegistry(), "test-v1", ARTIFACT, incompatible)


def test_registry_read_rejects_manifest_version_tampering():
    con = InMemoryRegistry()
    write_model(con, "test-v1", ARTIFACT, MANIFEST)
    con.row = (con.row[0], con.row[1], {**MANIFEST, "version": "other-version"})

    with pytest.raises(ValueError, match="manifest version does not match model_version"):
        read_model(con, "test-v1")


def test_registry_identical_version_retry_is_idempotent():
    con = InMemoryRegistry()
    digest = write_model(con, "test-v1", ARTIFACT, MANIFEST)

    assert write_model(con, "test-v1", ARTIFACT, MANIFEST) == digest
    assert read_model(con, "test-v1") == (ARTIFACT, MANIFEST)


def test_registry_rejects_reusing_version_for_different_artifact_or_manifest():
    con = InMemoryRegistry()
    write_model(con, "test-v1", ARTIFACT, MANIFEST)

    replacement = model_artifact(offset=2)
    with pytest.raises(ValueError, match="different artifact"):
        write_model(con, "test-v1", replacement, manifest_for(replacement))
    changed_manifest = {**MANIFEST, "metrics": {"24": {"mae": 0.1}}}
    with pytest.raises(ValueError, match="different artifact or manifest"):
        write_model(con, "test-v1", ARTIFACT, changed_manifest)
    assert read_model(con, "test-v1") == (ARTIFACT, MANIFEST)


def test_concurrent_version_collision_cannot_overwrite_first_writer():
    con = InMemoryRegistry()
    artifacts = (model_artifact(offset=3), model_artifact(offset=4))

    def attempt(artifact):
        try:
            return write_model(con, "test-v1", artifact, manifest_for(artifact))
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, artifacts))

    assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
    stored_artifact, stored_manifest = read_model(con, "test-v1")
    assert stored_artifact in artifacts
    assert stored_manifest == manifest_for(stored_artifact)
    successful_digests = [outcome for outcome in outcomes if isinstance(outcome, str)]
    assert successful_digests == [artifact_sha256(stored_artifact)]


def test_registry_read_rejects_corrupted_artifact():
    con = InMemoryRegistry()
    write_model(con, "test-v1", ARTIFACT, MANIFEST)
    con.row = (con.row[0], b"corrupted", con.row[2])

    with pytest.raises(ValueError, match="checksum"):
        read_model(con, "test-v1")


def test_registry_requires_manifest_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        write_model(InMemoryRegistry(), "test-v1", b"model", {"version": "test-v1"})


def test_publish_reconnects_to_read_back_committed_model():
    store = InMemoryRegistry()
    connections = []

    def connect(_url):
        connections.append(store)
        return store

    digest = publish_model("unused", "test-v1", ARTIFACT, MANIFEST, connect=connect)

    assert digest == store.row[0]
    assert len(connections) == 2


@pytest.mark.parametrize("change", ["artifact", "manifest"])
def test_publish_rejects_second_connection_readback_mismatch(change):
    store = InMemoryRegistry()
    connections = []

    def connect(_url):
        connections.append(store)
        if len(connections) == 2:
            if change == "artifact":
                changed = model_artifact(offset=5)
                store.row = (
                    artifact_sha256(changed),
                    changed,
                    manifest_for(changed),
                )
            else:
                store.row = (store.row[0], store.row[1], {**MANIFEST, "metrics": {}})
        return store

    with pytest.raises(ValueError, match="does not match submitted"):
        publish_model("unused", "test-v1", ARTIFACT, MANIFEST, connect=connect)


def test_sequential_champion_promotions_replace_pointer_and_read_verified_bundle():
    con = InMemoryRegistry()
    first_artifact = model_artifact(offset=6)
    second_artifact = model_artifact(offset=7)
    first_manifest = {
        **MANIFEST,
        "version": "test-v1",
        "artifact_sha256": artifact_sha256(first_artifact),
    }
    write_model(con, "test-v1", first_artifact, first_manifest)
    second_manifest = {
        **MANIFEST,
        "version": "test-v2",
        "artifact_sha256": artifact_sha256(second_artifact),
    }
    write_model(con, "test-v2", second_artifact, second_manifest)

    promote_model(con, "test-v1")
    assert read_champion(con) == ("test-v1", first_artifact, first_manifest)
    promote_model(con, "test-v2")
    assert read_champion(con) == ("test-v2", second_artifact, second_manifest)


def test_champion_pointer_requires_registered_model_and_is_empty_initially():
    con = InMemoryRegistry()
    with pytest.raises(LookupError, match="champion model is not set"):
        read_champion(con)
    with pytest.raises(LookupError, match="model version not found"):
        promote_model(con, "missing")
    assert con.champion is None
