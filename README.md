# pm25-forecast

Bangkok PM2.5 24-hour forecast, built as an MLOps portfolio project. The goal is to show an
end-to-end loop running unattended on free tiers: collect data, predict, store, score, monitor,
retrain. Private until the owner decides otherwise.

## Architecture (5 lines)

1. An hourly GitHub Actions job ingests Air4Thai readings and Open-Meteo weather plus the CAMS PM2.5 forecast (stored with its issue time).
2. Data-quality checks run, then observations are upserted into hosted storage (final choice in KAN-17).
3. A predict step uses the champion model (a persistence/CAMS baseline first, LightGBM as a shadow challenger later) and stores 24 h forecasts.
4. Matured forecasts are scored against actuals; drift (PSI) and data health are monitored, with failure alerts and a heartbeat.
5. A weekly retrain promotes a challenger only if it beats the champion; a static dashboard shows forecast vs actual vs baseline.

Diagrams: https://earnrw.atlassian.net/wiki/spaces/EARN/pages/66244/PM2.5+Architecture+Diagrams

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv if missing).

```bash
uv sync                 # install runtime + dev dependencies from uv.lock
uv run pytest           # tests
uv run ruff check       # lint
uv run ruff format --check
uv run pre-commit install   # optional: run hooks on every commit
cp .env.example .env        # optional: local settings, never commit .env
```

Collector (KAN-18): `uv run python -m pm25_forecast.collector` needs `DATABASE_URL` and
`STATION_LOCATIONS` (see `.env.example`). The workflow draft (every 20 minutes) is `docs/collector-workflow.yml.txt`.

Layout: `src/pm25_forecast/{ingest,features,models,storage,monitoring}` and `settings.py`
(environment-driven configuration).

## Branch and PR rules

- One Jira ticket = one branch (`kan-<n>/<short-name>`) = one pull request.
- The PR description lists the acceptance criteria met.
- Nobody pushes to or merges into `main` except the owner, who reviews and merges.
- Commit messages: `KAN-<n>: <summary>`.
- No secrets and no data dumps in git.

## Time

All timestamps are UTC, in code, storage and logs.

## Data sources and attribution

- Weather and CAMS air-quality forecasts: [Open-Meteo](https://open-meteo.com/), licensed CC BY 4.0 (free tier is non-commercial use).
  Air-quality data contains modified Copernicus Atmosphere Monitoring Service (CAMS) information.
- PM2.5 station observations: Air4Thai, Pollution Control Department (PCD), Thailand.
- Historical backfill and coverage metrics: [OpenAQ](https://openaq.org/).
