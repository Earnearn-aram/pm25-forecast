# certs/

Extra CA bundle for Air4Thai (KAN-35). `air4thai-ca.pem` was verified and committed on 2026-10-27
(KAN-43); `tests/test_ca_bundle.py` pins its fingerprints. Nothing here is a secret (CA certificates
are public), but only add a PEM that was obtained and reviewed by the owner.

How it gets here:

1. Run the `api-probe` workflow (see `docs/probe-workflow.yml.txt`).
2. Open `probe-report.md`, section "Air4Thai TLS diagnosis". If the server omits its intermediate
   certificate, the report prints the intermediate (fetched from the certificate's AIA link over
   plain http) as PEM together with its SHA-256.
3. Verify each certificate (below), then save the PEM as `certs/air4thai-ca.pem` (the probe also
   uploads it as `probe-report.ca.pem`). Each certificate in the file starts with an
   `# UNVERIFIED: fetched over plain http ...` line; it is fetched without authentication, so
   treat it as a candidate only.
4. Point the client at it: `AIR4THAI_CA_BUNDLE=certs/air4thai-ca.pem`.

The bundle is added to the default trust store; certificate verification is never switched off.

## Owner checks before committing a PEM

The report prints one SHA-256 per intermediate; check each one against the value the CA publishes
(out of band, not from the Air4Thai server). Then, for each certificate saved as `f.pem`:

```bash
openssl x509 -in f.pem -noout -subject -issuer -enddate -fingerprint -sha256
openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt f.pem
```

The fingerprint must match the published one, the subject must be the issuer shown for the Air4Thai
certificate in the report, the certificate must not be expired, and `openssl verify` must print
`OK` (the intermediate chains to a root in the normal trust store). Do not commit a PEM that fails
any check. The client does not treat the bundle as a trust anchor: the chain must still end at a
root in the default trust store.

## How the committed bundle was verified (2026-10-27, KAN-43)

The committed `air4thai-ca.pem` came from the official files served over HTTPS at
`https://letsencrypt.org/certs/gen-y/` (int-yr1/2/3 and root-yr-by-x1). They were cross-checked
against the fingerprints in the probe report for YR1 and Root YR. YR1 has no self-signed root in
the default stores yet, so verify each intermediate through the cross-signed Root YR:

```bash
openssl verify -no-CApath -no-CAstore -CAfile "$(python -m certifi)" \
  -untrusted root-yr-by-x1.pem int-yr1.pem
```

YR2 and YR3 are included because Let's Encrypt may issue the next Air4Thai leaf from any of them.
Known gap: ECDSA intermediates (YE1-3, via Root YE / ISRG Root X2) are not included. Add them if
Air4Thai moves to an ECDSA key. Remove the bundle once Air4Thai serves a correct chain (the probe's
default-trust-store handshake passes without it). `tests/test_ca_bundle.py` pins the fingerprints,
re-checks the chains and fails 90 days before any certificate expires.
