-- Run once on existing databases to add the production model pointer.
-- Model rows remain immutable; this singleton alias can be moved between versions.
CREATE TABLE IF NOT EXISTS model_champion (
    slot           text        PRIMARY KEY CHECK (slot = 'production'),
    model_version  text        NOT NULL REFERENCES models(model_version),
    promoted_at    timestamptz NOT NULL DEFAULT now()
);
