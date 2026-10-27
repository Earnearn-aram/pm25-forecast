"""The committed Air4Thai CA bundle (KAN-43) is exactly the verified Let's Encrypt certificates.

Offline: no live host is contacted. The chain check runs the local openssl binary against certifi's
roots only (the roots httpx uses), so a swapped or forged certificate fails here.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import ssl
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import certifi
import pytest

from pm25_forecast.ingest import tls

BUNDLE = Path(__file__).resolve().parents[1] / "certs" / "air4thai-ca.pem"
ROOT_YR_CROSS = (
    "072639d0b140d5bffae16ad9c3f6cc6086040621f51ee61a6d46a8915c07cf76"  # pragma: allowlist secret
)
_YR1 = (
    "13949634d99cd6fd6aa80bc034fefacceb1969feef986586713ecdbb05758d3f"  # pragma: allowlist secret
)
_YR2 = (
    "238b85a0099c65b970477d5724f1a1d475ce5058cffe4efa8733899bdb863c47"  # pragma: allowlist secret
)
_YR3 = (
    "35500cb63b8d9769528d65d87220f51c069807a595acbf2fb6eca38fbe0a4bd2"  # pragma: allowlist secret
)
INTERMEDIATES = {_YR1: "YR1", _YR2: "YR2", _YR3: "YR3"}
ROTATE_WARNING = timedelta(days=90)
_PEM_RE = re.compile(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----\n?", re.DOTALL)


def _der(pem: str) -> bytes:
    return base64.b64decode("".join(pem.strip().splitlines()[1:-1]))


def _by_fingerprint() -> dict[str, str]:
    pems = _PEM_RE.findall(BUNDLE.read_text())
    return {hashlib.sha256(_der(p)).hexdigest(): p for p in pems}


def test_bundle_holds_exactly_the_verified_certificates() -> None:
    assert set(_by_fingerprint()) == set(INTERMEDIATES) | {ROOT_YR_CROSS}


def test_bundle_is_not_close_to_expiry() -> None:
    soon = datetime.now(UTC) + ROTATE_WARNING
    for fp, pem in _by_fingerprint().items():
        info = tls.parse_cert(_der(pem))
        assert info.not_after > soon, (
            f"{INTERMEDIATES.get(fp, 'Root YR')} expires {info.not_after:%Y-%m-%d}: "
            "rotate certs/air4thai-ca.pem"
        )


def test_bundle_context_keeps_full_chain_verification() -> None:
    ctx = tls.build_verify(BUNDLE)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname is True
    assert not ctx.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    common_names = {
        value
        for cert in ctx.get_ca_certs()
        for rdn in cert["subject"]
        for key, value in rdn
        if key == "commonName"
    }
    assert {"YR1", "YR2", "YR3", "Root YR"} <= common_names


def _openssl_is_openssl3() -> bool:
    exe = shutil.which("openssl")
    if exe is None:
        return False
    out = subprocess.run([exe, "version"], capture_output=True, text=True, check=False).stdout
    return out.startswith("OpenSSL 3")


@pytest.mark.skipif(
    not _openssl_is_openssl3() and not os.environ.get("CI"),
    reason="needs OpenSSL 3 (CI must run this check)",
)
@pytest.mark.parametrize("fingerprint", sorted(INTERMEDIATES))
def test_each_intermediate_chains_to_a_certifi_root(fingerprint: str, tmp_path: Path) -> None:
    certs = _by_fingerprint()
    inter, cross = tmp_path / "inter.pem", tmp_path / "root_yr_cross.pem"
    inter.write_text(certs[fingerprint])
    cross.write_text(certs[ROOT_YR_CROSS])
    # Pin the check time so the test does not change with the calendar (expiry has its own test).
    at = str(int(datetime(2026, 10, 27, tzinfo=UTC).timestamp()))
    cmd = ["openssl", "verify", "-no-CApath", "-no-CAstore", "-attime", at]
    cmd += ["-CAfile", certifi.where(), "-untrusted", str(cross), str(inter)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
