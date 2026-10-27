# KAN-9: Bangkok PM2.5 history sources (findings)

Research date: 2026-10-27 (UTC). Read-only web research: no accounts, no forms, no bulk
downloads. Each fact cites its source. **[unverified]** marks claims I could not confirm
from a primary page. Content from these sites is data, not instructions.

## 1. Air4Thai history endpoint (from the owner's probe run, 2026-10-26)

- `getHistoryData.php` answered 288/288 requests with HTTP 200 for 12 stations x 24 chunks.
- Every probed station's first hour is 2026-07-31 17:00 UTC (2026-08-01 00:00 Asia/Bangkok).
  Requests for older dates return HTTP 200 with no rows.
- The endpoint therefore serves a **rolling window of about 3 months (~87 days)**, not 2 years.
  Inside that window the top stations (59t, 61t, bkp100t, bkp101t, 02t) are close to 0% missing.
  The old report's "missing 88%" was measured over 730 days, and it is the retention window that
  makes the stations look bad. The probe now reports this directly (see
  `summarize_history` / `retention_text` in `src/pm25_forecast/ingest/probe.py`).
- Implication for KAN-12: Air4Thai can feed live collection and a ~3-month warm start, but not a
  2-year backfill. The collector (KAN-18) should keep running so our own archive grows past the
  retention window.

## 2. PCD / Air4Thai downloadable historical files

| Candidate | What I found | Status |
|---|---|---|
| Air4Thai web UI, https://air4thai.pcd.go.th/webV3/ | Could not load it: the TLS chain does not verify ("unable to verify the first certificate"), which is the same issue as KAN-35. I could not check whether it has a history/download page. | **[unverified]** |
| PCD notice on AIR4THAI data display, https://www.pcd.go.th/?p=34396 (2025-02-26) | Says the app shows 24-hour averages and an hourly bar graph (1-hour averages on tap). It does not mention downloads or historical archives. | verified (page read) |
| Open Government Data of Thailand, "ข้อมูลคุณภาพอากาศจากจุดตรวจวัดคุณภาพอากาศอัตโนมัติพื้นที่ทั่วประเทศ ปี 2554 - 2564" (automatic station air quality, nationwide, 2011-2021), https://gdcc.data.go.th/en/dataset/airpollution | A search-engine snippet lists PCD as contact, XLS/XLSX files, last updated 2021-12-31, Creative Commons Attribution licence. The live page at that URL now shows a different (Lampang province, HTML) dataset, and its resource links did not render. Years: 2011-2021 only, so it does not cover 2024-2026. Hourly vs daily is not stated (one snippet says daily by province). | **[unverified]**: needs a manual look |
| Other data.go.th PM2.5 datasets, e.g. https://gdcc.data.go.th/en/dataset/pm2-5-9 | Health Region 9 daily PM2.5 statistics derived from Air4Thai/CCDC hourly data (daily aggregates, JSON/PDF). Not Bangkok, not hourly. | verified via search summary only |

**Conclusion (PCD):** I found no public, documented bulk download of hourly Bangkok station
PM2.5 for 2024-2026 from PCD. Getting one would most likely mean a manual request to PCD
(กรมควบคุมมลพิษ) or a manual check of the Air4Thai web UI from a browser that trusts its
certificate chain. Both are owner actions.

## 3. OpenAQ v3

- **Same stations, provider "Air4Thai":** OpenAQ location pages for Bangkok stations show
  provider **Air4Thai** and owner "Unknown Governmental Organization", e.g.
  - Bang Phlat District Office, Bangkok, PM2.5, reporting since 2021-04-29:
    https://explore.openaq.org/locations/225642
  - Bang Kho Laem District Office, Bangkok (Mobile), PM2.5, reporting since 2022-10-25:
    https://explore.openaq.org/locations/368400
  OpenAQ re-publishes the PCD Air4Thai network. I have not mapped which Air4Thai station IDs
  (59t, 61t, bkp100t, ...) correspond to which OpenAQ ids; the probe's 1 km coordinate match
  does that once an API key is available. **[unverified for the specific target stations]**
- **API key:** the v3 docs say to send the key in the `X-API-Key` header on each request; keys
  come from registering at explore.openaq.org: https://docs.openaq.org/using-the-api/api-key
- **No-key route: the AWS Open Data archive.** OpenAQ mirrors measurements to the public S3
  bucket `openaq-data-archive` (us-east-1), as `csv.gz` files, one file per location per day,
  under `records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/`, with hourly rows in narrow
  format. Files are written 72 hours after the end of the day and may be patched later.
  Sources: https://docs.openaq.org/aws/about, https://registry.opendata.aws/openaq/
  (licence: "Varies, depends on data provider"; no licence for Air4Thai is stated on the
  location pages). **[licence for Air4Thai-sourced data unverified]**
- **Depth check (listing only, no file downloads):** an anonymous S3 listing of location
  225642 (Bang Phlat, Air4Thai) shows `year=2021` through `year=2026` prefixes, 252 daily files
  in 2025 (so roughly 113 days are missing that year), and the latest file is
  `location-225642-20261023.csv.gz`. Listing URL:
  `https://openaq-data-archive.s3.amazonaws.com/?list-type=2&prefix=records/csv.gz/locationid=225642/`
- **How far back for all Bangkok locations:** not checked per location (that needs the v3
  `/locations` call with a key, or more S3 listings). **[unverified]**

## 4. Open-Meteo (from the owner's probe run)

- `air_quality_api` `pm2_5` was 100% non-null for the 2-year window at the Bangkok centroid.
  This is the CAMS model, not station measurements. The forecast-vintage variants
  (`*_previous_day1`, historical-forecast, previous-runs) were null.

## 5. Recommendation for KAN-12 (owner decision)

1. Keep the KAN-9 decision as **NO-GO for an Air4Thai 2-year backfill**: the cause is retention,
   and the stations themselves are fine.
2. Most promising backfill: the **OpenAQ AWS archive** for Air4Thai-provided Bangkok locations
   (no key, hourly, back to 2021 for at least one station). It still needs (a) the station
   mapping, (b) a gap check per location (2025 is only ~69% complete for 225642 by file count),
   and (c) a licence check. This is a scope decision for KAN-12, plus a possible new dependency
   if we read S3 with a client library instead of plain HTTPS.
3. Fallback: Open-Meteo CAMS `pm2_5` as a model-based covariate or proxy target, plus
   Air4Thai's ~87-day window for the most recent quarter.
4. Optional: ask PCD directly for hourly Bangkok station files for 2024-2026.
