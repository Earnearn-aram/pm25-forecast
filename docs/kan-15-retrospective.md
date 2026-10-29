# KAN-15: retrospective direct +24h LightGBM

## Result

This is an offline retrospective evaluation on the aggregate history snapshot, not evidence that
these readings were available to a forecaster at the historical origin time. The export's
`available_at` values record bulk ingestion. This run assumes a **0-hour reporting lag** so the
model, persistence, and seasonal naive baseline can be scored on the same station/origin/target
pairs. The assumed lag is a scenario only; it does not establish historical real availability.

| Model | Test pairs | MAE (µg/m³) | RMSE (µg/m³) | Hourly concentration-band accuracy |
|---|---:|---:|---:|---:|
| LightGBM direct +24h | 10,724 | 4.4405 | 6.3518 | 66.43% |
| Persistence, latest measurement within 3h | 10,724 | 4.4560 | 6.4843 | 68.40% |
| Seasonal naive, measurement at origin | 10,724 | 4.4560 | 6.4843 | 68.40% |

The two fixed baselines are identical on this exact common set. LightGBM's MAE is 0.0154 µg/m³
lower and RMSE is 0.1325 µg/m³ lower; its concentration-band accuracy is 1.97 percentage points
lower. The band metric is an exploratory hourly proxy using boundaries at 15, 25, 37.5, and
75 µg/m³. It is not an official AQI calculation.

## Evaluation design

- Forecast horizon: direct +24h; issue origins advance hourly.
- Snapshot span: 2024-10-28 00:00 UTC to 2026-10-28 21:00 UTC; 64,986 usable observation rows
  from five stations: `02t`, `59t`, `61t`, `bkp100t`, and `bkp101t`.
- Chronological test origins: 2026-07-10 10:00 UTC to 2026-10-27 21:00 UTC. Test targets cover
  2026-07-30 through 2026-10-28 UTC.
- Training used 52,990 eligible samples from 2024-10-28 00:00 UTC through 2026-07-01 02:00 UTC
  origin time. Every training target is strictly earlier than the first test origin; no training
  label crosses the test boundary.
- Features use the shared feature builder and only measurements: `pm25_latest` and
  `pm25_same_hour_24h`. No CAMS features are used. LightGBM handles missing values natively; no
  imputation or preprocessing was fitted.
- The test set is the exact intersection where the actual, persistence value, and origin-time
  seasonal measurement exist. Station pair counts: `02t` 2,144; `59t` 2,145; `61t` 2,145;
  `bkp100t` 2,145; `bkp101t` 2,145.
- Category bands use hourly PM2.5 boundaries of 15, 25, 37.5, and 75 µg/m³. This proxy does not
  apply daily averaging or official AQI interpolation.

## Reproducibility and artifact handling

Snapshot SHA-256: `fdd9b90ba6b96a1de512175f638808efe34fc8ae80c35b3e89bf440e71553a09`.

The local model bundle and manifest were written under `/private/tmp/pm25-kan15-output/` and are
not committed. Artifact SHA-256: `b3e0be311532e1b2fc212b1c85c08751de6945779c25189ad337c05041ade34b`.
The manifest records model version, feature order, training window, source snapshot hash, assumed
lag, test metrics, and artifact hash. A local in-memory registry test writes and reads back the
artifact and manifest through the registry interface. No hosted or production registry was
accessed.

The aggregate machine-readable run report is `/private/tmp/pm25-kan15-output/report.json`.
