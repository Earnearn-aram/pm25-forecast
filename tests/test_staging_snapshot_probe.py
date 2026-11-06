"""Offline guard checks. No hosted or live database access."""

import hashlib
import json
from dataclasses import replace

import psycopg
import pytest

from pm25_forecast import staging_snapshot_probe as probe

TARGET = probe.ApprovedTarget(
    project_id="synthetic-project",
    branch_id="br-synthetic",
    endpoint_id="ep-synthetic",
    host="ep-synthetic.ap-southeast-1.aws.neon.tech",
    database="synthetic_db",
    role="synthetic_probe",
    parent_branch_id="br-synthetic-parent",
    evidence_sha256="0" * 64,
)


def synthetic_evidence(target=TARGET):
    return json.dumps(
        {
            "schema_version": 1,
            "project_id": target.project_id,
            "branch_id": target.branch_id,
            "endpoint_id": target.endpoint_id,
            "host": target.host,
            "database": target.database,
            "role": target.role,
            "parent_branch_id": target.parent_branch_id,
            "creation_mode": "schema_only",
            "no_production_or_holdout_data": True,
            "verification_ref": "synthetic-review-record",
        },
        sort_keys=True,
    ).encode()


def target_with_evidence(target=TARGET):
    data = synthetic_evidence(target)
    return replace(target, evidence_sha256=hashlib.sha256(data).hexdigest()), data


def conninfo(**overrides):
    params = {
        "host": TARGET.host,
        "dbname": TARGET.database,
        "user": TARGET.role,
        "password": "synthetic-password",  # pragma: allowlist secret
        "sslmode": "verify-full",
    }
    params.update(overrides)
    return psycopg.conninfo.make_conninfo(**params)


def test_no_reviewed_target_refuses_before_connect(monkeypatch):
    monkeypatch.setattr(psycopg, "connect", lambda **_: pytest.fail("must not connect"))
    with pytest.raises(probe.ProbeRefused, match="no reviewed"):
        probe.run_probe(conninfo())


def test_exact_staging_dsn_is_accepted():
    result = probe.validate_dsn(conninfo(), TARGET)
    assert result["host"] == TARGET.host
    assert result["sslmode"] == "verify-full"


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "ep-production.ap-southeast-1.aws.neon.tech"},
        {"host": TARGET.host + ",other"},
        {"dbname": "production"},
        {"user": "production_owner"},
        {"password": ""},
        {"sslmode": "require"},
        {"hostaddr": "127.0.0.1"},
        {"service": "production"},
        {"options": "-c search_path=public"},
        {"sslrootcert": "/tmp/ca.pem"},
    ],
)
def test_rejects_alternate_routes_or_privileged_target(overrides):
    with pytest.raises(probe.ProbeRefused):
        probe.validate_dsn(conninfo(**overrides), TARGET)


def test_endpoint_must_match_the_reviewed_direct_host():
    with pytest.raises(probe.ProbeRefused, match="invalid target pin"):
        probe.validate_dsn(conninfo(), replace(TARGET, endpoint_id="ep-other"))


def test_ambient_libpq_variables_refuse_before_connect(monkeypatch, tmp_path):
    target, evidence = target_with_evidence()
    path = tmp_path / "evidence.json"
    path.write_bytes(evidence)
    monkeypatch.setattr(probe, "EVIDENCE_PATH", path)
    monkeypatch.setenv("PGHOSTADDR", "127.0.0.1")
    monkeypatch.setattr(psycopg, "connect", lambda **_: pytest.fail("must not connect"))
    with pytest.raises(probe.ProbeRefused, match="ambient"):
        probe.run_probe(conninfo(), target)


def test_reviewed_evidence_must_match_digest_and_target():
    target, evidence = target_with_evidence()
    probe.verify_reviewed_evidence(target, evidence)
    with pytest.raises(probe.ProbeRefused, match="digest"):
        probe.verify_reviewed_evidence(target, evidence + b" ")
    changed = json.loads(evidence)
    changed["parent_branch_id"] = "br-production"
    changed_bytes = json.dumps(changed, sort_keys=True).encode()
    target, _ = target_with_evidence(target)
    with pytest.raises(probe.ProbeRefused, match="isolation requirements"):
        probe.verify_reviewed_evidence(
            replace(target, evidence_sha256=hashlib.sha256(changed_bytes).hexdigest()),
            changed_bytes,
        )


def test_target_requires_distinct_pinned_parent_and_evidence_digest():
    with pytest.raises(probe.ProbeRefused, match="invalid target pin"):
        probe.validate_dsn(conninfo(), replace(TARGET, parent_branch_id=TARGET.branch_id))


class FakeCursor:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, *, attributes=None, app_access=False, ssl=True):
        self.info = type("Info", (), {"host": TARGET.host})()
        self.pgconn = type("Pgconn", (), {"ssl_in_use": ssl})()
        self.attributes = attributes or (
            TARGET.database,
            TARGET.role,
            "18.6",
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        )
        self.app_access = app_access

    def execute(self, statement, _params=None):
        row = (self.app_access,) if "FROM pg_namespace n" in statement else self.attributes
        if "FROM pg_class c" in statement:
            row = (self.app_access,)
        if "FROM pg_proc p" in statement:
            row = (self.app_access,)
        return FakeCursor(row)


def test_connected_role_must_have_limited_privileges_and_tls():
    assert probe.check_connection(FakeConnection(), TARGET) == "18.6"
    with pytest.raises(probe.ProbeRefused, match="TLS"):
        probe.check_connection(FakeConnection(ssl=False), TARGET)
    with pytest.raises(probe.ProbeRefused, match="schema, relation or routine"):
        probe.check_connection(FakeConnection(app_access=True), TARGET)
    for index, value in ((3, True), (7, True), (8, False), (9, True), (10, True), (11, True)):
        row = list(FakeConnection().attributes)
        row[index] = value
        with pytest.raises(probe.ProbeRefused, match="role privileges"):
            probe.check_connection(FakeConnection(attributes=tuple(row)), TARGET)


def test_cli_never_prints_driver_exception_details(monkeypatch, capsys):
    def fail(_dsn):
        raise ValueError("synthetic-password-should-not-appear")

    monkeypatch.setattr(probe, "run_probe", fail)
    assert probe.main() == 1
    assert "synthetic-password" not in capsys.readouterr().err
