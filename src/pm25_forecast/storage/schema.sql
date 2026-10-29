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
