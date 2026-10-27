"""The committed Air4Thai CA bundle (KAN-43) is exactly the verified intermediates.

Offline: no live host is contacted. The chain check uses the local openssl binary and certifi's
root store (the same roots httpx uses), so a swapped or forged certificate fails here.
"""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import certifi
import pytest

from pm25_forecast.ingest.tls import build_verify

BUNDLE = Path(__file__).resolve().parents[1] / "certs" / "air4thai-ca.pem"
EXPECTED_SHA256 = {
    # Let's Encrypt YR1
    "13949634d99cd6fd6aa80bc034fefacceb1969feef986586713ecdbb05758d3f",  # pragma: allowlist secret
    # ISRG Root YR cross-signed by ISRG Root X1
    "072639d0b140d5bffae16ad9c3f6cc6086040621f51ee61a6d46a8915c07cf76",  # pragma: allowlist secret
}
_PEM_RE = re.compile(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----\n?", re.DOTALL)


def _pems() -> list[str]:
    return _PEM_RE.findall(BUNDLE.read_text())


def test_bundle_holds_exactly_the_verified_certificates() -> None:
    fingerprints = set()
    for pem in _pems():
        body = "".join(pem.strip().splitlines()[1:-1])
        fingerprints.add(hashlib.sha256(base64.b64decode(body)).hexdigest())
    assert fingerprints == EXPECTED_SHA256


def test_bundle_loads_on_top_of_default_store() -> None:
    assert build_verify(BUNDLE) is not True


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")
def test_bundle_chains_to_a_default_root(tmp_path: Path) -> None:
    yr1, root_yr = _pems()
    (tmp_path / "yr1.pem").write_text(yr1)
    (tmp_path / "root_yr.pem").write_text(root_yr)
    result = subprocess.run(
        [
            "openssl",
            "verify",
            "-CAfile",
            certifi.where(),
            "-untrusted",
            str(tmp_path / "root_yr.pem"),
            str(tmp_path / "yr1.pem"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
