import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from pm25_forecast.storage.registry import artifact_sha256, publish_model, read_model, write_model


class InMemoryRegistry:
    def __init__(self):
        self.row = None
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement, params):
        if statement.lstrip().startswith("INSERT"):
            with self.lock:
                if self.row is None:
                    self.row = (params["sha256"], params["artifact"], params["manifest"])
        elif statement.lstrip().startswith("SELECT"):
            with self.lock:
                self._selected = self.row
        return self

    def fetchone(self):
        with self.lock:
            return self._selected


MANIFEST = {
    "version": "test-v1",
    "metrics": {"24": {"mae": 1.2}},
    "training_window": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-02-01T00:00:00+00:00"},
    "feature_list": ["pm25_latest", "cams_pm25"],
}


def test_registry_write_reads_back_and_verifies_artifact():
    con = InMemoryRegistry()
    digest = write_model(con, "test-v1", b"serialized-model", MANIFEST)

    artifact, manifest = read_model(con, "test-v1")
    assert artifact == b"serialized-model"
    assert manifest == MANIFEST
    assert con.row[0] == digest


def test_registry_identical_version_retry_is_idempotent():
    con = InMemoryRegistry()
    digest = write_model(con, "test-v1", b"serialized-model", MANIFEST)

    assert write_model(con, "test-v1", b"serialized-model", MANIFEST) == digest
    assert read_model(con, "test-v1") == (b"serialized-model", MANIFEST)


def test_registry_rejects_reusing_version_for_different_artifact_or_manifest():
    con = InMemoryRegistry()
    write_model(con, "test-v1", b"serialized-model", MANIFEST)

    with pytest.raises(ValueError, match="different artifact"):
        write_model(con, "test-v1", b"replacement-model", MANIFEST)
    changed_manifest = {**MANIFEST, "metrics": {"24": {"mae": 0.1}}}
    with pytest.raises(ValueError, match="different artifact or manifest"):
        write_model(con, "test-v1", b"serialized-model", changed_manifest)
    assert read_model(con, "test-v1") == (b"serialized-model", MANIFEST)


def test_concurrent_version_collision_cannot_overwrite_first_writer():
    con = InMemoryRegistry()
    artifacts = (b"model-a", b"model-b")

    def attempt(artifact):
        try:
            return write_model(con, "test-v1", artifact, MANIFEST)
        except ValueError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, artifacts))

    assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
    stored_artifact, stored_manifest = read_model(con, "test-v1")
    assert stored_artifact in artifacts
    assert stored_manifest == MANIFEST
    successful_digests = [outcome for outcome in outcomes if isinstance(outcome, str)]
    assert successful_digests == [artifact_sha256(stored_artifact)]


def test_registry_read_rejects_corrupted_artifact():
    con = InMemoryRegistry()
    write_model(con, "test-v1", b"serialized-model", MANIFEST)
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

    digest = publish_model("unused", "test-v1", b"serialized-model", MANIFEST, connect=connect)

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
                changed = b"concurrent replacement"
                store.row = (artifact_sha256(changed), changed, store.row[2])
            else:
                store.row = (store.row[0], store.row[1], {**MANIFEST, "metrics": {}})
        return store

    with pytest.raises(ValueError, match="does not match submitted"):
        publish_model("unused", "test-v1", b"serialized-model", MANIFEST, connect=connect)
