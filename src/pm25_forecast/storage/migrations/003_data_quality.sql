-- Run once on existing databases to add the data quality check history (KAN-21).
-- station_id is '' for a source-wide check.
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

CREATE INDEX IF NOT EXISTS data_quality_checks_series
    ON data_quality_checks (source, check_name, checked_at);
