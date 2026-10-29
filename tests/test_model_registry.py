import pytest

from pm25_forecast.storage.registry import publish_model, read_model, write_model


class InMemoryRegistry:
    def __init__(self):
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement, params):
        if statement.lstrip().startswith("INSERT"):
            self.row = (params["sha256"], params["artifact"], params["manifest"])
        return self

    def fetchone(self):
        return self.row


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
