-- PM2.5 storage schema (PostgreSQL). All timestamps are timestamptz, stored as UTC.

CREATE TABLE IF NOT EXISTS observations (
    station_id   text             NOT NULL,
    ts           timestamptz      NOT NULL,
    pm25         double precision,
    source       text             NOT NULL,
    ingested_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (station_id, ts)
);

CREATE TABLE IF NOT EXISTS features (
    station_id       text        NOT NULL,
    ts               timestamptz NOT NULL,
    feature_version  text        NOT NULL,
    values_json      jsonb       NOT NULL,
    PRIMARY KEY (station_id, ts, feature_version)
);

CREATE TABLE IF NOT EXISTS predictions (
    station_id     text             NOT NULL,
    issue_time     timestamptz      NOT NULL,
    valid_time     timestamptz      NOT NULL,
    model_version  text             NOT NULL,
    run_id         text             NOT NULL,
    pm25_pred      double precision NOT NULL,
    created_at     timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (station_id, issue_time, valid_time, model_version)
);

CREATE TABLE IF NOT EXISTS scores (
    station_id     text             NOT NULL,
    issue_time     timestamptz      NOT NULL,
    valid_time     timestamptz      NOT NULL,
    model_version  text             NOT NULL,
    pm25_obs       double precision NOT NULL,
    abs_error      double precision NOT NULL,
    scored_at      timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (station_id, issue_time, valid_time, model_version)
);

CREATE TABLE IF NOT EXISTS models (
    model_version  text        PRIMARY KEY,
    sha256         text        NOT NULL,
    artifact       bytea       NOT NULL,
    manifest       jsonb       NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- Keep the mutable production alias separate from immutable model versions.
CREATE TABLE IF NOT EXISTS model_champion (
    slot           text        PRIMARY KEY CHECK (slot = 'production'),
    model_version  text        NOT NULL REFERENCES models(model_version),
    promoted_at    timestamptz NOT NULL DEFAULT now()
);

-- Data quality check history (KAN-21). One row per (hour, source, station, check); a re-run in the
-- same hour overwrites. `source` is an observation source (air4thai, openaq) or a model_version
-- (persistence-24h, a challenger, cams_global). station_id is '' for a source-wide check.
-- `check` is a reserved word, so the column is check_name. `threshold` is the limit that decided
-- the status (the fail limit for fail, the warn limit otherwise).
CREATE TABLE IF NOT EXISTS data_quality_checks (
    checked_at   timestamptz      NOT NULL,
    source       text             NOT NULL,
    station_id   text             NOT NULL DEFAULT '',
    check_name   text             NOT NULL,
    status       text             NOT NULL CHECK (status IN ('ok', 'warn', 'fail')),
    value        double precision,
    threshold    double precision,
    detail       text             NOT NULL DEFAULT '',
    recorded_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (checked_at, source, station_id, check_name)
);

-- Dashboard history for one series: WHERE source = ? AND check_name = ? ORDER BY checked_at.
CREATE INDEX IF NOT EXISTS data_quality_checks_series
    ON data_quality_checks (source, check_name, checked_at);
