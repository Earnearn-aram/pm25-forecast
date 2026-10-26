# certs/

Placeholder for an optional extra CA bundle for Air4Thai (KAN-35). **No certificate is committed
yet.** Nothing here is a secret (CA certificates are public), but only add a PEM that was obtained
and reviewed by the owner.

How it gets here:

1. Run the `api-probe` workflow (see `docs/probe-workflow.yml.txt`).
2. Open `probe-report.md`, section "Air4Thai TLS diagnosis". If the server omits its intermediate
   certificate, the report prints the intermediate (fetched from the certificate's AIA link over
   plain http) as PEM together with its SHA-256.
3. Verify that SHA-256 against the CA's published value, then save the PEM as
   `certs/air4thai-ca.pem` (the probe also uploads it as `probe-report.ca.pem`).
4. Point the client at it: `AIR4THAI_CA_BUNDLE=certs/air4thai-ca.pem`.

The bundle is added to the default trust store; certificate verification is never switched off.
