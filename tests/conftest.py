"""Shared test helpers: throwaway certificates generated at test time (never committed)."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class Pki:
    """A throwaway root -> intermediate -> leaf chain for 127.0.0.1 / localhost."""

    root_pem: Path
    inter_pem: Path
    leaf_pem: Path
    leaf_key: Path
    forged_inter_pem: Path  # same subject DN as the intermediate, different key
    chain_pem: Path  # leaf + intermediate
    root_der: bytes
    inter_der: bytes
    leaf_der: bytes


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _der(pem: Path, cwd: Path) -> bytes:
    out = subprocess.run(
        ["openssl", "x509", "-in", str(pem), "-outform", "DER"],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return out.stdout


@pytest.fixture(scope="session")
def pki(tmp_path_factory: pytest.TempPathFactory) -> Pki:
    if shutil.which("openssl") is None:
        if os.environ.get("CI"):
            pytest.fail("openssl is required in CI for the TLS tests")
        pytest.skip("openssl not available")
    d = tmp_path_factory.mktemp("pki")
    ec = ["-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"]
    _run(
        "openssl", "req", "-x509", *ec, "-keyout", "root.key", "-out", "root.pem",
        "-subj", "/C=TH/O=Test Org/CN=Test Root", "-days", "2",
        "-addext", "basicConstraints=critical,CA:TRUE",
        cwd=d,
    )  # fmt: skip
    (d / "inter.ext").write_text(
        "basicConstraints=critical,CA:TRUE\n"
        "authorityInfoAccess=caIssuers;URI:http://ca.test/root.der\n"
    )
    (d / "leaf.ext").write_text(
        "basicConstraints=CA:FALSE\n"
        "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "authorityInfoAccess=caIssuers;URI:http://ca.test/inter.der\n"
    )
    for name, ca, ext, subj in (
        ("inter", "root", "inter.ext", "/C=TH/O=Test Org/CN=Test Intermediate"),
        ("leaf", "inter", "leaf.ext", "/CN=localhost"),
    ):
        _run("openssl", "req", *ec, "-keyout", f"{name}.key", "-out", f"{name}.csr",
             "-subj", subj, cwd=d)  # fmt: skip
        _run("openssl", "x509", "-req", "-in", f"{name}.csr", "-CA", f"{ca}.pem",
             "-CAkey", f"{ca}.key", "-CAcreateserial", "-out", f"{name}.pem", "-days", "2",
             "-extfile", ext, cwd=d)  # fmt: skip
    _run(
        "openssl", "req", "-x509", *ec, "-keyout", "forged.key", "-out", "forged.pem",
        "-subj", "/C=TH/O=Test Org/CN=Test Intermediate", "-days", "2",
        "-addext", "basicConstraints=critical,CA:TRUE",
        cwd=d,
    )  # fmt: skip
    (d / "chain.pem").write_text((d / "leaf.pem").read_text() + (d / "inter.pem").read_text())
    return Pki(
        root_pem=d / "root.pem",
        inter_pem=d / "inter.pem",
        leaf_pem=d / "leaf.pem",
        leaf_key=d / "leaf.key",
        forged_inter_pem=d / "forged.pem",
        chain_pem=d / "chain.pem",
        root_der=_der(d / "root.pem", d),
        inter_der=_der(d / "inter.pem", d),
        leaf_der=_der(d / "leaf.pem", d),
    )
