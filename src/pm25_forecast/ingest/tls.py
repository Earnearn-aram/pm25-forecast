"""TLS helpers for Air4Thai: a CA-bundle option for real clients and diagnostics for the probe.

Two separate concerns live here, and they never mix:

* ``build_verify`` builds the ``verify=`` argument for httpx. Certificate verification is always
  on; an optional extra CA bundle (PEM) can be added on top of the default trust store.
* The ``diagnose_*`` helpers inspect the certificate chain a server presents so a human can see
  why verification fails. They perform a handshake with verification switched off, read the
  certificates, and close the socket. They never send or read application data, and nothing they
  return is ever used as an API payload.

Certificates are parsed with a tiny stdlib-only DER reader (no new dependency). Anything
malformed raises ``CertParseError``; callers in the probe catch it and record the failure.
"""

from __future__ import annotations

import base64
import hashlib
import re
import socket
import ssl
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

HANDSHAKE_TIMEOUT = 10.0
MAX_AIA_HOPS = 3
MAX_CERT_BYTES = 100_000
# Host used when no station list is available to pick a station for the history endpoint probe.
FALLBACK_HISTORY_STATION = "02t"

_PEM_RE = re.compile(r"-----BEGIN CERTIFICATE-----(.+?)-----END CERTIFICATE-----", re.DOTALL)
_OID_NAMES = {
    "2.5.4.3": "CN",
    "2.5.4.5": "serialNumber",
    "2.5.4.6": "C",
    "2.5.4.7": "L",
    "2.5.4.8": "ST",
    "2.5.4.10": "O",
    "2.5.4.11": "OU",
    "1.2.840.113549.1.9.1": "emailAddress",
}
_OID_AIA = "1.3.6.1.5.5.7.1.1"
_OID_CA_ISSUERS = "1.3.6.1.5.5.7.48.2"


class CertParseError(ValueError):
    """The bytes are not a certificate this reader understands."""


# ---------------------------------------------------------------- verify= for real clients


def build_verify(ca_bundle: str | Path | None = None) -> ssl.SSLContext | bool:
    """Return the httpx ``verify=`` value.

    With no bundle this is ``True``: httpx's default trust store, verification on. With a bundle
    (PEM file) it is an ``SSLContext`` holding the default trust store *plus* the bundle, so a
    missing intermediate can be supplied without dropping the normal roots. Chain and hostname
    verification are never disabled, and the partial-chain flag is deliberately NOT set: a
    certificate in the bundle (typically an intermediate fetched over plain http) is never a
    trust anchor by itself and must still chain to a root in the default trust store.
    """
    if ca_bundle is None or str(ca_bundle).strip() == "":
        return True
    path = Path(ca_bundle)
    if not path.is_file():
        raise FileNotFoundError(f"CA bundle not found: {path}")
    context = httpx.create_ssl_context(verify=True)
    context.load_verify_locations(cafile=str(path))
    return context


def is_tls_verify_error(exc: BaseException) -> bool:
    """True if an exception (or its cause chain) is a certificate verification failure."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


# ---------------------------------------------------------------- minimal DER reader


class CertInfo(BaseModel):
    """The facts about one certificate that matter for diagnosing a broken chain."""

    subject: str
    issuer: str
    not_before: datetime
    not_after: datetime
    ca_issuers: list[str] = Field(default_factory=list)
    sha256: str
    self_signed: bool


def _tlv(buf: bytes, pos: int) -> tuple[int, int, int]:
    """Read one tag-length-value header at ``pos``: returns (tag, value_start, value_end)."""
    if pos + 2 > len(buf):
        raise CertParseError("truncated DER")
    tag = buf[pos]
    if tag & 0x1F == 0x1F:
        raise CertParseError("high tag numbers are not supported")
    first = buf[pos + 1]
    start = pos + 2
    if first < 0x80:
        length = first
    else:
        count = first & 0x7F
        if count == 0 or count > 4 or start + count > len(buf):
            raise CertParseError("bad DER length")
        length = int.from_bytes(buf[start : start + count], "big")
        start += count
    end = start + length
    if end > len(buf):
        raise CertParseError("DER length exceeds data")
    return tag, start, end


def _children(buf: bytes, start: int, end: int) -> list[tuple[int, int, int]]:
    out = []
    pos = start
    while pos < end:
        tag, s, e = _tlv(buf, pos)
        out.append((tag, s, e))
        pos = e
    if pos != end:
        raise CertParseError("misaligned DER")
    return out


def _oid(buf: bytes, start: int, end: int) -> str:
    if start >= end:
        raise CertParseError("empty OID")
    first = buf[start]
    parts = [first // 40, first % 40]
    value = 0
    for byte in buf[start + 1 : end]:
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            parts.append(value)
            value = 0
    return ".".join(str(p) for p in parts)


def _text(tag: int, raw: bytes) -> str:
    if tag == 0x1E:  # BMPString
        return raw.decode("utf-16-be", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _name(buf: bytes, start: int, end: int) -> str:
    """Render an X.500 Name as ``C=TH, O=Example, CN=Example CA`` (certificate order)."""
    pieces = []
    for _, rs, re_ in _children(buf, start, end):  # each RDN (SET)
        for _, as_, ae in _children(buf, rs, re_):  # each AttributeTypeAndValue (SEQUENCE)
            items = _children(buf, as_, ae)
            if len(items) != 2 or items[0][0] != 0x06:
                raise CertParseError("bad name attribute")
            oid = _oid(buf, items[0][1], items[0][2])
            vtag, vs, ve = items[1]
            pieces.append(f"{_OID_NAMES.get(oid, oid)}={_text(vtag, buf[vs:ve])}")
    return ", ".join(pieces)


def _time(buf: bytes, tag: int, start: int, end: int) -> datetime:
    raw = buf[start:end].decode("ascii", errors="replace")
    try:
        if tag == 0x17 and re.fullmatch(r"\d{12}Z", raw):
            year = int(raw[:2])
            return datetime.strptime(
                f"{1900 + year if year >= 50 else 2000 + year}{raw[2:]}", "%Y%m%d%H%M%SZ"
            ).replace(tzinfo=UTC)
        if tag == 0x18 and re.fullmatch(r"\d{14}Z", raw):
            return datetime.strptime(raw, "%Y%m%d%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        pass
    raise CertParseError(f"bad certificate time {raw!r}")


def _ca_issuers(buf: bytes, start: int, end: int) -> list[str]:
    """Extract caIssuers URLs from an Authority Information Access extension value."""
    urls = []
    seq_tag, ss, se = _tlv(buf, start)
    if seq_tag != 0x30 or se > end:
        raise CertParseError("bad AIA extension")
    for _, ds, de in _children(buf, ss, se):
        items = _children(buf, ds, de)
        if (
            len(items) == 2
            and items[0][0] == 0x06
            and items[1][0] == 0x86
            and _oid(buf, items[0][1], items[0][2]) == _OID_CA_ISSUERS
        ):
            urls.append(buf[items[1][1] : items[1][2]].decode("ascii", errors="replace"))
    return urls


def parse_cert(der: bytes) -> CertInfo:
    """Parse a DER certificate into ``CertInfo``; raises ``CertParseError`` on bad input."""
    if len(der) > MAX_CERT_BYTES:
        raise CertParseError("certificate too large")
    try:
        tag, s, _ = _tlv(der, 0)
        if tag != 0x30:
            raise CertParseError("certificate is not a SEQUENCE")
        tbs_tag, ts, te = _tlv(der, s)
        if tbs_tag != 0x30:
            raise CertParseError("TBSCertificate is not a SEQUENCE")
        fields = _children(der, ts, te)
        idx = 1 if fields and fields[0][0] == 0xA0 else 0  # optional explicit version
        # serial, signature algorithm, issuer, validity, subject, subject public key info
        serial, sigalg, issuer, validity, subject = fields[idx : idx + 5]
        del serial, sigalg
        v_items = _children(der, validity[1], validity[2])
        if len(v_items) != 2:
            raise CertParseError("bad validity")
        not_before = _time(der, *v_items[0])
        not_after = _time(der, *v_items[1])
        urls: list[str] = []
        for ftag, fs, _ in fields[idx + 6 :]:
            if ftag != 0xA3:  # extensions are [3] EXPLICIT
                continue
            for _, xs, xe in _children(der, *_tlv(der, fs)[1:]):
                parts = _children(der, xs, xe)
                if (
                    parts
                    and parts[0][0] == 0x06
                    and _oid(der, parts[0][1], parts[0][2]) == _OID_AIA
                ):
                    octets = parts[-1]
                    if octets[0] == 0x04:
                        urls += _ca_issuers(der, octets[1], octets[2])
        subject_text = _name(der, subject[1], subject[2])
        issuer_text = _name(der, issuer[1], issuer[2])
    except (ValueError, IndexError) as exc:
        if isinstance(exc, CertParseError):
            raise
        raise CertParseError(f"malformed certificate: {exc}") from exc
    return CertInfo(
        subject=subject_text,
        issuer=issuer_text,
        not_before=not_before,
        not_after=not_after,
        ca_issuers=urls,
        sha256=hashlib.sha256(der).hexdigest(),
        self_signed=subject_text == issuer_text,
    )


def der_to_pem(der: bytes) -> str:
    body = base64.encodebytes(der).decode("ascii")
    return f"-----BEGIN CERTIFICATE-----\n{body}-----END CERTIFICATE-----\n"


def pem_to_ders(text: str) -> list[bytes]:
    """Every certificate in a PEM text, as DER. Invalid base64 blocks are skipped."""
    out = []
    for block in _PEM_RE.findall(text):
        try:
            out.append(base64.b64decode("".join(block.split()), validate=True))
        except ValueError:
            continue
    return out


def der_from_download(body: bytes) -> bytes:
    """A downloaded caIssuers body is DER (usual) or PEM; return DER or raise."""
    if b"-----BEGIN CERTIFICATE-----" in body:
        ders = pem_to_ders(body.decode("ascii", errors="replace"))
        if not ders:
            raise CertParseError("no certificate in PEM download")
        return ders[0]
    return body


# ---------------------------------------------------------------- diagnostics (network)


def redirect_info(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> dict:
    """GET without following redirects; record status and Location. Never raises."""
    record: dict[str, Any] = {"url": url, "status": None, "location": None, "error": None}
    try:
        resp = client.get(url, params=params, follow_redirects=False)
        record["status"] = resp.status_code
        location = resp.headers.get("location")
        if location:
            record["location"] = str(httpx.URL(url).join(location))
    except httpx.HTTPError as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        record["tls_verify_failed"] = is_tls_verify_error(exc)
    except (httpx.InvalidURL, ValueError) as exc:  # e.g. a malformed Location header
        record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return record


def peek_leaf_certificate(host: str, port: int = 443, timeout: float = HANDSHAKE_TIMEOUT):
    """DIAGNOSTIC ONLY: handshake without verification and return (leaf DER, tls details).

    Only the certificate is read. No request is sent and no application data is received, so
    nothing from an unverified connection can reach the data pipeline.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with (
        socket.create_connection((host, port), timeout=timeout) as sock,
        context.wrap_socket(sock, server_hostname=host) as tls,
    ):
        der = tls.getpeercert(binary_form=True)
        cipher = tls.cipher()
        details = {"tls_version": tls.version(), "cipher": cipher[0] if cipher else None}
    if not der:
        raise CertParseError("server presented no certificate")
    return der, details


def verified_handshake_error(host: str, port: int = 443, timeout: float = HANDSHAKE_TIMEOUT):
    """Try a normal verified handshake (default trust store); return the failure, or None."""
    context = httpx.create_ssl_context(verify=True)
    try:
        with (
            socket.create_connection((host, port), timeout=timeout) as sock,
            context.wrap_socket(sock, server_hostname=host),
        ):
            return None
    except ssl.SSLCertVerificationError as exc:
        return {"verify_code": exc.verify_code, "message": exc.verify_message}
    except (OSError, ssl.SSLError) as exc:
        return {"verify_code": None, "message": f"{type(exc).__name__}: {str(exc)[:200]}"}


def presented_chain_pems(
    host: str, port: int = 443, timeout: float = HANDSHAKE_TIMEOUT
) -> list[str] | None:
    """DIAGNOSTIC ONLY: the PEM chain the server sends, via ``openssl s_client`` if installed.

    stdin is closed immediately, so no application data is sent or read. Returns ``None`` when
    openssl is unavailable (cannot run), and ``[]`` when it ran but no certificate was obtained.
    """
    target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    cmd = ["openssl", "s_client", "-showcerts", "-servername", host, "-connect", target]
    try:
        done = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=False,
        )
    except FileNotFoundError:
        return None
    except (OSError, subprocess.SubprocessError):
        return []
    return [der_to_pem(d) for d in pem_to_ders(done.stdout) if len(d) <= MAX_CERT_BYTES]


def _describe(info: CertInfo, now: datetime, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "subject": info.subject,
        "issuer": info.issuer,
        "not_after": info.not_after.isoformat(),
        "expired": info.not_after < now,
        "self_signed": info.self_signed,
        "ca_issuers": info.ca_issuers,
        "sha256": info.sha256,
    }


def download_issuers(
    client: httpx.Client, leaf: CertInfo, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Follow ``caIssuers`` links upward from ``leaf`` (at most ``MAX_AIA_HOPS``). Never raises.

    Each record says whether the download parsed and whether its subject matches the child's
    issuer. ``pem`` is only set for a matching, non-self-signed certificate: a root fetched over
    plain http is never offered as a trust anchor.
    """
    records: list[dict[str, Any]] = []
    child = leaf
    seen: set[str] = set()
    for _ in range(MAX_AIA_HOPS):
        url = next((u for u in child.ca_issuers if u not in seen), None)
        if url is None:
            break
        try:
            scheme = urlsplit(url).scheme
        except ValueError:
            records.append({"url": url, "ok": False, "error": "malformed URL", "pem": None})
            break
        if scheme not in ("http", "https"):
            break
        seen.add(url)
        record: dict[str, Any] = {"url": url, "ok": False, "error": None, "pem": None}
        records.append(record)
        try:
            with client.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > MAX_CERT_BYTES:  # stop reading: never buffer more
                        raise CertParseError("download too large")
            der = der_from_download(body)
            info = parse_cert(der)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            break
        matches = info.subject == child.issuer
        record.update(
            ok=True,
            matches_issuer=matches,
            fetched_at=(now or datetime.now(UTC)).isoformat(),
            **_describe(info, now or datetime.now(UTC), "aia"),
        )
        if matches and not info.self_signed:
            record["pem"] = der_to_pem(der)
        if not matches or info.self_signed:
            break
        child = info
    return records


def diagnose_host(
    client: httpx.Client,
    host: str,
    port: int = 443,
    now: datetime | None = None,
    *,
    peek: Callable[..., Any] | None = None,
    verify_error: Callable[..., Any] | None = None,
    chain: Callable[..., list[str]] | None = None,
) -> dict[str, Any]:
    """Describe the chain a host presents and try to fetch its missing intermediates.

    The three network callables default to the module functions (looked up at call time) and
    exist so tests can run offline.
    """
    peek = peek or peek_leaf_certificate
    verify_error = verify_error or verified_handshake_error
    chain = chain or presented_chain_pems
    now = now or datetime.now(UTC)
    result: dict[str, Any] = {"host": host, "port": port, "error": None}
    result["verify_error"] = verify_error(host, port)
    try:
        der, details = peek(host, port)
        leaf = parse_cert(der)
    except (OSError, ssl.SSLError, ValueError) as exc:  # CertParseError is a ValueError
        result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return result
    result.update(details)
    presented_raw = chain(host, port)
    presented = presented_raw or []
    # None: openssl unavailable; 0: openssl ran but obtained no certificate (connection failed)
    result["presented_count"] = None if presented_raw is None else len(presented)
    certs = [_describe(leaf, now, "leaf")]
    for pem in presented[1:]:
        for extra in pem_to_ders(pem):
            try:
                certs.append(_describe(parse_cert(extra), now, "presented"))
            except CertParseError as exc:
                certs.append({"source": "presented", "error": str(exc)})
    result["chain"] = certs
    result["presented_pem"] = presented[1:]
    result["issuer_downloads"] = download_issuers(client, leaf, now)
    return result


def https_targets(urls: list[str], redirects: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Unique (host, port) pairs to inspect: each URL's https twin plus https redirect targets."""
    targets: list[tuple[str, int]] = []
    candidates = list(urls) + [r["location"] for r in redirects if r.get("location")]
    for url in candidates:
        try:
            parts = urlsplit(url)
            host, port = parts.hostname, parts.port
        except ValueError:  # malformed URL or port: skip it
            continue
        if host and parts.scheme in ("http", "https"):
            pair = (host, (port or 443) if parts.scheme == "https" else 443)
            if pair not in targets:
                targets.append(pair)
    return targets


def _unverified_header(record: dict[str, Any]) -> str:
    url = re.sub(r"[^\x21-\x7e]", "?", str(record.get("url", "")))
    how = "plain http" if url.startswith("http:") else "https"
    return (
        f"# UNVERIFIED: fetched over {how} from {url} on {record.get('fetched_at')}; "
        f"SHA-256 (DER): {record.get('sha256')}; verify before committing\n"
    )


def bundle_from_report(hosts: list[dict[str, Any]]) -> str:
    """Concatenate the matching, non-self-signed issuer PEMs found for all hosts (deduplicated).

    Each certificate is preceded by an ``# UNVERIFIED`` comment (source URL, UTC time, SHA-256);
    PEM readers ignore text outside the BEGIN/END lines.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for host in hosts:
        for rec in host.get("issuer_downloads", []):
            pem = rec.get("pem")
            if pem and pem not in seen:
                seen.add(pem)
                parts.append(_unverified_header(rec) + pem)
    return "".join(parts)


def diagnose_air4thai(
    client: httpx.Client,
    probe_requests: list[tuple[str, str, dict[str, Any] | None]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Redirects + presented chains for the Air4Thai endpoints; the report section for KAN-35.

    ``probe_requests`` is ``[(name, url, params), ...]``. Never raises.
    """
    now = now or datetime.now(UTC)
    redirects = []
    for name, url, params in probe_requests:
        redirects.append({"name": name, **redirect_info(client, url, params)})
    hosts = [
        diagnose_host(client, host, port, now)
        for host, port in https_targets([u for _, u, _ in probe_requests], redirects)
    ]
    return {"redirects": redirects, "hosts": hosts, "bundle_pem": bundle_from_report(hosts)}
