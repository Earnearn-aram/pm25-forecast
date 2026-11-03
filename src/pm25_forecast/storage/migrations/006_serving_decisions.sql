-- Apply before enabling KAN-119. Existing forecasts are not retroactively assigned a role.
CREATE TABLE IF NOT EXISTS serving_issue_decisions (
    issue_time timestamptz PRIMARY KEY,
    champion_version text,
    selected_at timestamptz NOT NULL,
    selection_run_id text NOT NULL
);

CREATE TABLE IF NOT EXISTS serving_forecasts (
    station_id text NOT NULL,
    issue_time timestamptz NOT NULL,
    valid_time timestamptz NOT NULL,
    model_version text NOT NULL,
    reason text NOT NULL,
    run_id text NOT NULL,
    PRIMARY KEY (station_id, issue_time, valid_time),
    FOREIGN KEY (station_id, issue_time, valid_time, model_version)
        REFERENCES predictions (station_id, issue_time, valid_time, model_version) ON DELETE CASCADE
);
