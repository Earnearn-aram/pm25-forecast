# OpenAQ 730-day backfill (KAN-49)

Run after merging KAN-49, from an environment with access to OpenAQ and the existing
PostgreSQL database. No schema migration or new dependency is needed. This is an explicit
one-off command; the daily Air4Thai job continues to use its existing entry point.

## Owner action after merge

1. Check out the merged main and run `uv sync --locked`.
2. Set `DATABASE_URL` and `STATION_LOCATIONS` to the existing deployment values. The latter
   must include the verified public coordinates for all five target stations:
   `59t`, `61t`, `bkp100t`, `bkp101t`, `02t`, using `ID=lat,lon;ID=lat,lon` syntax.
   Coordinates are required so ingestion cannot silently guess a station mapping.
3. Inject `OPENAQ_API_KEY` into the process environment from your secret manager. The command
   reads that environment variable only, never a CLI argument or the Settings/.env key.
   Do not paste credentials into shell history or logs.
4. Run:

   ```sh
   uv run python -m pm25_forecast.openaq_backfill
   ```

   By default this requests `[current full UTC hour - 730 days, current full UTC hour)`.
   For repeatable retries, pin the same explicit window on every attempt, for example:

   ```sh
   uv run python -m pm25_forecast.openaq_backfill \
     --start 2024-10-28T00:00:00Z --end 2026-10-28T00:00:00Z
   ```

5. Check the exit code and per-station chunk logs, then inspect the stored date span/counts:

   ```sql
   SELECT station_id, source, count(*), min(ts), max(ts)
   FROM observations
   WHERE station_id IN ('59t', '61t', 'bkp100t', 'bkp101t', '02t')
   GROUP BY station_id, source ORDER BY station_id, source;
   ```

   Available history can have gaps; requesting 730 days does not imply 730 days are present.
   Compare the combined Air4Thai/OpenAQ span rather than requiring an OpenAQ row where
   Air4Thai already owns that hour. No live backfill was executed during implementation.

Exit codes: **0** = all selected stations supplied usable rows with no fetch failures,
**1** = a failed fetch/chunk, database/runtime failure, or a station with no usable rows,
**2** = invalid configuration. Rows blocked by source precedence still count as usable,
so a fully overlapping successful rerun is not incorrectly reported as a failure.

## Matching, validation and precedence

The default station set is exactly the five targets. `--stations 59t,61t` can narrow a repair
run; other station ids are rejected. Reuse the KAN-46 parser and candidate ranking on main:
nearest OpenAQ location within 1,000 m, ties by location id; PM2.5 sensor ids embedded in
that location take priority over the compatibility `/locations/{id}/sensors` lookup.
All unique PM2.5 sensors are fetched. Their ids are sorted, and the **highest sensor id's
usable value wins** an overlapping hour. This is a deterministic tie-break, not a claim
about sensor quality. Null/invalid values never displace usable readings.

Payload envelopes and hourly readings are pydantic-validated. Ignore malformed rows,
naive timestamps, non-hour timestamps, non-finite/negative values and PM2.5 above 1,000
µg/m³; these remain gaps. Normalize aware timestamps to UTC; filter to `[start, end)`.
Location and sensor/hour pagination are bounded to 32 pages each; hitting the cap is a
failure, not silent truncation. Both explicit window bounds must be whole aware hours.

The existing natural key remains `(station_id, ts)`:

- A usable **Air4Thai** reading always wins over OpenAQ, independent of arrival order.
- OpenAQ fills missing rows or rows whose PM2.5 is null. It cannot overwrite an existing
  usable Air4Thai reading (or another source's usable reading).
- A null Air4Thai reading cannot take ownership of an existing OpenAQ value. A later usable
  Air4Thai reading replaces it and updates provenance.
- Same-source updates remain allowed, so reruns correct values without creating duplicates.

This rule lives in the shared atomic SQL conflict clause, protecting both the collector and
both backfill commands, including concurrent writes. No delete/reinsert race or schema
change is required. OpenAQ provenance is `source='openaq'`; retain attribution when sharing
exported data, per [OpenAQ terms](https://docs.openaq.org/about/terms).

## Restart and failure behavior

Each station is processed in at most 31-day chunks, and each completed chunk is committed
in its own transaction. Memory and transaction size are bounded to one chunk across that
station's sensors. Request timeout is 30 seconds; transport errors, 429 and 5xx get at most
four attempts with exponential fallback and Retry-After support (numeric or HTTP date),
with each retry wait capped at 60 seconds. Every request is paced by one second.
Redirects are disabled and failures log only fixed messages/status/exception type, never
response bodies, request headers or credential-bearing exception text.

If any sensor/page fails, the entire station/chunk is left untouched; other chunks and
stations continue and the final exit is nonzero. This deliberately avoids storing a
lower-priority sensor during a partial replay over previously stored higher-priority data.
An interrupted database transaction rolls back only its chunk. Already committed chunks
survive. Rerun the pinned window, or repair the logged interval with `--stations`, `--start`
and `--end`. There is no max-timestamp shortcut that could skip interior gaps, and no local
checkpoint file to drift from committed database state. Replaying completed chunks costs
requests but does not duplicate natural keys. Empty/malformed readings remain visible as
coverage gaps; they are not fabricated or interpolated.

## Duration and cost estimate

With one PM2.5 sensor per target, 730 days means 24 chunks per station: roughly **125 GETs**
(5 location lookups + 120 hourly pages), up to five more fallback sensor lookups. A full
window contains at most **87,600 unique station/hour rows** before gaps and precedence.
Allow **5–20 minutes** for API latency and database writes under normal conditions; this
is a planning estimate, not a measured production runtime. Multiple sensors, quota waits,
retries or slow database connections increase it. Each request has at most three retry
waits (up to 180 seconds total), and failures remain bounded per request.

OpenAQ [general API use is free](https://docs.openaq.org/using-the-api/rate-limits), with
60 requests/minute and 2,000/hour per key as checked on 2026-10-28. One-second pacing fits
the minute quota for this process; shared key traffic can still cause 429s. The usual
one-sensor run is well below the hourly quota. There is no new paid service or scheduled
runner; existing database compute/storage and any chosen runner may incur their normal
charges. No paid quota upgrade is required for the estimated request volume.

## Verification

All tests use synthetic HTTP responses and SQLite against the real shared SQL; no live
OpenAQ, Air4Thai or production database call is needed. Focused tests cover all five target
mappings, embedded/fallback sensors, deterministic multiple-sensor overlap, partial sensor
failure, pagination/caps, UTC/invalid-row validation, committed-chunk restart, idempotent
reruns, both Air4Thai/OpenAQ arrival orders, missing key and bounded retry behavior.

Run the repository-prescribed checks:

```sh
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run pytest
uv run pre-commit run --all-files
```
