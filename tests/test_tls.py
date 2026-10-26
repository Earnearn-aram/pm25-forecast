"""Offline tests for the Air4Thai TLS helpers (KAN-35).

Certificates are generated at test time with openssl and served by a local 127.0.0.1 TLS
server, so no live host is contacted and nothing certificate-like is committed.
"""

import socket
import ssl
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from conftest import Pki
from pm25_forecast.ingest import tls
from pm25_forecast.ingest.air4thai import Air4ThaiClient
from pm25_forecast.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "air4thai_getNewAQI_synthetic.json"
NOW = datetime(2026, 10, 26, tzinfo=UTC)


# ---------------------------------------------------------------- DER / PEM parsing


def test_parse_cert_leaf_intermediate_root(pki: Pki) -> None:
    leaf = tls.parse_cert(pki.leaf_der)
    inter = tls.parse_cert(pki.inter_der)
    root = tls.parse_cert(pki.root_der)
    assert leaf.subject == "CN=localhost"
    assert leaf.issuer == inter.subject == "C=TH, O=Test Org, CN=Test Intermediate"
    assert leaf.ca_issuers == ["http://ca.test/inter.der"]
    assert inter.ca_issuers == ["http://ca.test/root.der"]
    assert root.ca_issuers == []
    assert root.self_signed and not leaf.self_signed
    assert leaf.not_after.tzinfo is not None and leaf.not_after > leaf.not_before
    assert len(leaf.sha256) == 64


def test_parse_cert_rejects_garbage(pki: Pki) -> None:
    for bad in (b"", b"\x00", b"not a cert", pki.leaf_der[:40], b"\x30\x84\xff\xff\xff\xff"):
        with pytest.raises(tls.CertParseError):
            tls.parse_cert(bad)


def test_pem_roundtrip_and_download_formats(pki: Pki) -> None:
    pem = tls.der_to_pem(pki.inter_der)
    assert pem.startswith("-----BEGIN CERTIFICATE-----\n")
    assert tls.pem_to_ders(pem) == [pki.inter_der]
    assert tls.pem_to_ders(pki.chain_pem.read_text()) == [pki.leaf_der, pki.inter_der]
    assert tls.pem_to_ders("-----BEGIN CERTIFICATE-----\n!!!\n-----END CERTIFICATE-----") == []
    assert tls.der_from_download(pki.inter_der) == pki.inter_der
    assert tls.der_from_download(pem.encode()) == pki.inter_der
    with pytest.raises(tls.CertParseError):
        tls.der_from_download(b"-----BEGIN CERTIFICATE-----\nxx\n-----END CERTIFICATE-----")


# ---------------------------------------------------------------- redirects and targets


def test_redirect_info_records_status_and_absolute_location() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"Location": "https://air4thai.pcd.go.th/x?a=1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    info = tls.redirect_info(client, "http://air4thai.pcd.go.th/x")
    assert info["status"] == 301
    assert info["location"] == "https://air4thai.pcd.go.th/x?a=1"
    assert info["error"] is None


def test_redirect_info_relative_location_and_tls_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(302, headers={"Location": "/y"}))
    )
    assert tls.redirect_info(client, "http://h.test/x")["location"] == "http://h.test/y"

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] bad", request=request)

    failed = tls.redirect_info(httpx.Client(transport=httpx.MockTransport(boom)), "https://h.test")
    assert failed["status"] is None and failed["tls_verify_failed"] is True
    assert "CERTIFICATE_VERIFY_FAILED" in failed["error"]


def test_https_targets_dedupes_and_follows_redirects() -> None:
    urls = ["http://a.test/x", "http://b.test/y"]
    redirects = [
        {"location": "https://a.test/x"},
        {"location": "https://c.test:8443/z"},
        {"location": None},
    ]
    assert tls.https_targets(urls, redirects) == [
        ("a.test", 443),
        ("b.test", 443),
        ("c.test", 8443),
    ]


# ---------------------------------------------------------------- issuer download


def _issuer_client(pki: Pki, overrides: dict[str, httpx.Response] | None = None) -> httpx.Client:
    routes = {
        "http://ca.test/inter.der": httpx.Response(200, content=pki.inter_der),
        "http://ca.test/root.der": httpx.Response(200, content=pki.root_der),
        **(overrides or {}),
    }
    return httpx.Client(
        transport=httpx.MockTransport(lambda r: routes.get(str(r.url), httpx.Response(404)))
    )


def test_download_issuers_offers_intermediate_but_never_a_root(pki: Pki) -> None:
    leaf = tls.parse_cert(pki.leaf_der)
    records = tls.download_issuers(_issuer_client(pki), leaf, NOW)
    assert [r["url"] for r in records] == ["http://ca.test/inter.der", "http://ca.test/root.der"]
    assert records[0]["ok"] and records[0]["matches_issuer"]
    assert records[0]["pem"] == tls.der_to_pem(pki.inter_der)
    assert records[1]["self_signed"] is True
    assert records[1]["pem"] is None  # a root fetched over http is not offered as an anchor


def test_download_issuers_accepts_pem_and_rejects_wrong_subject(pki: Pki) -> None:
    leaf = tls.parse_cert(pki.leaf_der)
    as_pem = _issuer_client(
        pki, {"http://ca.test/inter.der": httpx.Response(200, text=tls.der_to_pem(pki.inter_der))}
    )
    assert tls.download_issuers(as_pem, leaf, NOW)[0]["pem"] is not None
    wrong = _issuer_client(
        pki, {"http://ca.test/inter.der": httpx.Response(200, content=pki.root_der)}
    )
    rec = tls.download_issuers(wrong, leaf, NOW)
    assert len(rec) == 1 and rec[0]["matches_issuer"] is False and rec[0]["pem"] is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404),
        httpx.Response(200, content=b"garbage"),
        httpx.Response(200, content=b"x" * 200_000),
    ],
)
def test_download_issuers_failures_are_recorded_not_raised(
    pki: Pki, response: httpx.Response
) -> None:
    leaf = tls.parse_cert(pki.leaf_der)
    rec = tls.download_issuers(_issuer_client(pki, {"http://ca.test/inter.der": response}), leaf)
    assert rec[0]["ok"] is False and rec[0]["error"] and rec[0]["pem"] is None


def test_diagnose_host_with_injected_network(pki: Pki) -> None:
    report = tls.diagnose_host(
        _issuer_client(pki),
        "h.test",
        443,
        NOW,
        peek=lambda host, port: (pki.leaf_der, {"tls_version": "TLSv1.3", "cipher": "X"}),
        verify_error=lambda host, port: {
            "verify_code": 20,
            "message": "unable to get local issuer",
        },
        chain=lambda host, port: [tls.der_to_pem(pki.leaf_der)],  # server omitted the intermediate
    )
    assert report["presented_count"] == 1
    assert report["verify_error"]["verify_code"] == 20
    assert [c["source"] for c in report["chain"]] == ["leaf"]
    assert report["chain"][0]["ca_issuers"] == ["http://ca.test/inter.der"]
    bundle = tls.bundle_from_report([report])
    assert bundle.endswith(tls.der_to_pem(pki.inter_der))
    header = bundle.splitlines()[0]
    assert header.startswith("# UNVERIFIED: fetched over plain http from http://ca.test/inter.der")
    assert "2026-10-26T00:00:00+00:00" in header and "SHA-256 (DER): " in header
    assert header.endswith("verify before committing")
    assert tls.pem_to_ders(bundle) == [pki.inter_der]  # header does not disturb PEM readers


def test_diagnose_host_records_unreachable_host() -> None:
    def refuse(host: str, port: int) -> tuple[bytes, dict]:
        raise ConnectionRefusedError("nope")

    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    report = tls.diagnose_host(
        client,
        "h.test",
        443,
        NOW,
        peek=refuse,
        verify_error=lambda h, p: None,
        chain=lambda h, p: [],
    )
    assert "ConnectionRefusedError" in report["error"]
    assert tls.bundle_from_report([report]) == ""


def test_diagnose_air4thai_combines_redirects_and_hosts(pki: Pki, monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ca.test":
            return httpx.Response(200, content=pki.inter_der)
        return httpx.Response(
            301, headers={"Location": str(request.url).replace("http:", "https:")}
        )

    monkeypatch.setattr(tls, "peek_leaf_certificate", lambda h, p=443, t=0: (pki.leaf_der, {}))
    monkeypatch.setattr(tls, "verified_handshake_error", lambda h, p=443, t=0: {"message": "x"})
    monkeypatch.setattr(tls, "presented_chain_pems", lambda h, p=443, t=0: [])
    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = tls.diagnose_air4thai(
        client,
        [("station_list", "http://a.test/s", None), ("history", "http://b.test/h", {"q": "1"})],
        NOW,
    )
    assert [r["status"] for r in out["redirects"]] == [301, 301]
    assert [h["host"] for h in out["hosts"]] == ["a.test", "b.test"]
    pem = tls.der_to_pem(pki.inter_der)
    assert out["bundle_pem"].count(pem) == 1  # deduplicated across hosts
    assert out["bundle_pem"].startswith(
        "# UNVERIFIED: fetched over plain http from http://ca.test/"
    )


# ---------------------------------------------------------------- CA bundle option


def test_build_verify_defaults_and_errors(tmp_path: Path) -> None:
    assert tls.build_verify(None) is True
    assert tls.build_verify("") is True
    with pytest.raises(FileNotFoundError):
        tls.build_verify(tmp_path / "missing.pem")


def test_build_verify_with_bundle_keeps_verification_on(pki: Pki) -> None:
    ctx = tls.build_verify(pki.inter_pem)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True
    # A bundled intermediate must never be a trust anchor on its own.
    assert not ctx.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN


def test_settings_ca_bundle_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIR4THAI_CA_BUNDLE", raising=False)
    assert Settings(_env_file=None).air4thai_ca_bundle is None
    monkeypatch.setenv("AIR4THAI_CA_BUNDLE", "")
    assert Settings(_env_file=None).air4thai_ca_bundle is None
    monkeypatch.setenv("AIR4THAI_CA_BUNDLE", "certs/x.pem")
    assert Settings(_env_file=None).air4thai_ca_bundle == Path("certs/x.pem")


def test_client_rejects_bundle_with_injected_client(pki: Pki) -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ValueError, match="ca_bundle"):
        Air4ThaiClient(client=http, ca_bundle=pki.inter_pem)


# ---------------------------------------------------------------- real TLS against localhost


@pytest.fixture
def tls_server(pki: Pki) -> Iterator[int]:
    """A local HTTPS server that sends leaf + intermediate (not the root)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # The server sends the leaf only, i.e. it omits the intermediate (the KAN-35 failure mode).
    context.load_cert_chain(str(pki.leaf_pem), str(pki.leaf_key))
    body = FIXTURE.read_bytes()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            try:
                conn.settimeout(2)
                with context.wrap_socket(conn, server_side=True) as secure:
                    data = b""
                    while b"\r\n\r\n" not in data:
                        chunk = secure.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    if data:
                        head = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
                        secure.sendall(head.encode() + b"Connection: close\r\n\r\n" + body)
            except (OSError, ssl.SSLError):
                pass
            finally:
                conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield listener.getsockname()[1]
    stop.set()
    thread.join(timeout=2)
    listener.close()


@pytest.fixture
def root_trusted(pki: Pki, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the throwaway root the only default trust anchor (httpx honours SSL_CERT_FILE)."""
    monkeypatch.setenv("SSL_CERT_FILE", str(pki.root_pem))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


@pytest.mark.usefixtures("root_trusted")
def test_missing_intermediate_fails_verified_and_bundle_fixes_it(pki: Pki, tls_server: int) -> None:
    url = f"https://127.0.0.1:{tls_server}/"
    with pytest.raises(httpx.ConnectError) as err, httpx.Client(verify=True) as plain:
        plain.get(url)
    assert tls.is_tls_verify_error(err.value)
    # The intermediate chains to the trusted root, so the bundle makes verification succeed.
    with httpx.Client(verify=tls.build_verify(pki.inter_pem)) as fixed:
        assert fixed.get(url).status_code == 200


@pytest.mark.usefixtures("root_trusted")
def test_bundle_with_headers_from_report_still_loads_and_verifies(
    pki: Pki, tls_server: int, tmp_path: Path
) -> None:
    host = {
        "issuer_downloads": [
            {
                "url": "http://ca.test/inter.der",
                "pem": tls.der_to_pem(pki.inter_der),
                "sha256": "ab" * 32,
                "fetched_at": NOW.isoformat(),
            }
        ]
    }
    bundle = tmp_path / "bundle.pem"
    bundle.write_text(tls.bundle_from_report([host]))
    with httpx.Client(verify=tls.build_verify(bundle)) as client:
        assert client.get(f"https://127.0.0.1:{tls_server}/").status_code == 200


@pytest.mark.usefixtures("root_trusted")
def test_forged_intermediate_in_bundle_is_rejected(pki: Pki, tls_server: int) -> None:
    """Same subject DN, different key, self-signed: must not make the leaf verify."""
    url = f"https://127.0.0.1:{tls_server}/"
    with (
        pytest.raises(httpx.ConnectError) as err,
        httpx.Client(verify=tls.build_verify(pki.forged_inter_pem)) as client,
    ):
        client.get(url)
    assert tls.is_tls_verify_error(err.value)


def test_intermediate_alone_is_not_a_trust_anchor(pki: Pki, tls_server: int) -> None:
    """Without the root in the default store, the downloaded intermediate must not be enough."""
    url = f"https://127.0.0.1:{tls_server}/"
    with (
        pytest.raises(httpx.ConnectError) as err,
        httpx.Client(verify=tls.build_verify(pki.inter_pem)) as client,
    ):
        client.get(url)
    assert tls.is_tls_verify_error(err.value)


@pytest.mark.usefixtures("root_trusted")
def test_air4thai_client_uses_ca_bundle(pki: Pki, tls_server: int) -> None:
    url = f"https://127.0.0.1:{tls_server}/"
    client = Air4ThaiClient(url=url, ca_bundle=pki.inter_pem, max_attempts=1)
    try:
        assert client.fetch()
    finally:
        client.close()
    bare = Air4ThaiClient(url=url, max_attempts=1)
    try:
        with pytest.raises(httpx.ConnectError):
            bare.fetch()
    finally:
        bare.close()


def test_diagnostic_handshake_reads_certificate_only(pki: Pki, tls_server: int) -> None:
    der, details = tls.peek_leaf_certificate("127.0.0.1", tls_server)
    assert der == pki.leaf_der
    assert details["tls_version"]
    failure = tls.verified_handshake_error("127.0.0.1", tls_server)
    assert failure is not None and failure["verify_code"] in (20, 21)
    chain = tls.presented_chain_pems("127.0.0.1", tls_server)
    assert chain is not None  # openssl is required by the pki fixture
    assert [tls.parse_cert(d).subject for p in chain for d in tls.pem_to_ders(p)] == [
        "CN=localhost"  # the server omitted the intermediate
    ]


# ---------------------------------------------------------------- never raises / limits


def test_redirect_info_survives_malformed_location() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(302, headers={"Location": "http://[::1/x"})
        )
    )
    info = tls.redirect_info(client, "http://h.test/x")
    assert info["location"] is None and info["error"]  # recorded, not raised


def test_malformed_aia_url_is_recorded_not_raised(pki: Pki) -> None:
    leaf = tls.parse_cert(pki.leaf_der).model_copy(update={"ca_issuers": ["http://[bad/x"]})
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    records = tls.download_issuers(client, leaf, NOW)
    assert records[0]["ok"] is False and records[0]["error"]
    assert tls.https_targets(["http://[bad/x", "http://h.test:notaport/"], []) == []


def test_download_stops_reading_after_the_cap(pki: Pki) -> None:
    leaf = tls.parse_cert(pki.leaf_der)
    sent = 0

    def endless():
        nonlocal sent
        while True:
            sent += 1
            yield b"x" * 10_000

    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=endless()))
    )
    records = tls.download_issuers(client, leaf, NOW)
    assert "too large" in records[0]["error"]
    assert sent <= tls.MAX_CERT_BYTES // 10_000 + 2


def test_parse_cert_rejects_oversized_input() -> None:
    with pytest.raises(tls.CertParseError, match="too large"):
        tls.parse_cert(b"\x30" * (tls.MAX_CERT_BYTES + 1))


def test_openssl_unavailable_vs_connection_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def missing(cmd, **kw):
        commands.append(cmd)
        raise FileNotFoundError("openssl")

    monkeypatch.setattr(tls.subprocess, "run", missing)
    assert tls.presented_chain_pems("::1", 8443) is None
    assert commands[0][-1] == "[::1]:8443"  # IPv6 literal is bracketed

    class Done:
        stdout = "CONNECTED(00000003)\nconnect:errno=111\n"

    monkeypatch.setattr(tls.subprocess, "run", lambda cmd, **kw: Done())
    assert tls.presented_chain_pems("h.test") == []


def test_presented_count_distinguishes_none_from_zero(pki: Pki) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

    def run(chain_result: list[str] | None) -> dict:
        return tls.diagnose_host(
            client,
            "h.test",
            443,
            NOW,
            peek=lambda h, p: (pki.leaf_der, {}),
            verify_error=lambda h, p: None,
            chain=lambda h, p: chain_result,
        )

    assert run(None)["presented_count"] is None
    assert run([])["presented_count"] == 0
