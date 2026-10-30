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

-- Drift checks: one row per model and feature (or `target_pm25`) per UTC day.
-- `checked_at` is the UTC day (midnight) the check belongs to, so a re-run on the same day updates
-- the row; `computed_at` is when it was last computed. psi is NULL when status = 'insufficient'.
CREATE TABLE IF NOT EXISTS drift_checks (
    checked_at     timestamptz      NOT NULL,
    model_version  text             NOT NULL,
    feature        text             NOT NULL,
    psi            double precision,
    status         text             NOT NULL CHECK (status IN ('ok', 'warn', 'alert', 'insufficient')),
    ref_start      timestamptz      NOT NULL,
    ref_end        timestamptz      NOT NULL,
    recent_start   timestamptz      NOT NULL,
    recent_end     timestamptz      NOT NULL,
    n_ref          integer          NOT NULL,
    n_recent       integer          NOT NULL,
    computed_at    timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (checked_at, model_version, feature)
);

-- Reference distribution per model and feature, computed once from the model's training window
-- (immutable per model_version) so the daily run only reads the recent days. `edges` and `shares`
-- are JSON arrays; `usable` is false when the reference is too small, has too few distinct bins,
-- or (cams_pm25) covers too few of the training origins.
CREATE TABLE IF NOT EXISTS drift_reference (
    model_version  text        NOT NULL,
    feature        text        NOT NULL,
    edges          text        NOT NULL,
    shares         text        NOT NULL,
    n_ref          integer     NOT NULL,
    n_origins      integer     NOT NULL,
    usable         boolean     NOT NULL,
    computed_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model_version, feature)
);
