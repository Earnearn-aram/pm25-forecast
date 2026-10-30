-- Run once on existing databases (KAN-24).
-- Promotion/rollback log: one row per champion change or retrain decision. A champion change
-- (promote, rollback) is written in the same transaction as the model_champion pointer change.
-- `recommend` and `keep` are retrain decisions that leave production alone. `metrics` is a JSON
-- object (text) with the comparison numbers. `id` gives the order of events; `at` is the database
-- clock. No foreign key: pruning a model must not erase history.
CREATE TABLE IF NOT EXISTS model_events (
    id bigserial PRIMARY KEY,
    at                timestamptz NOT NULL DEFAULT now(),
    action            text        NOT NULL CHECK (action IN ('promote', 'rollback', 'recommend', 'keep')),
    model_version     text        NOT NULL,
    previous_version  text,
    actor             text        NOT NULL CHECK (actor IN ('cli', 'retrain', 'auto')),
    metrics           text,
    reason            text
);

-- Drift reference hardening: the training window the cached reference was built from. A reference
-- whose window differs from the manifest's training_window is rebuilt. NULL (rows written before
-- this migration) also triggers one rebuild.
ALTER TABLE drift_reference ADD COLUMN IF NOT EXISTS ref_start timestamptz;
ALTER TABLE drift_reference ADD COLUMN IF NOT EXISTS ref_end timestamptz;

-- Same-season drift reference: the season the cached reference was built for, e.g.
-- 'doy±30@2026-W44'. A different key (or NULL, rows written before this) triggers a rebuild.
ALTER TABLE drift_reference ADD COLUMN IF NOT EXISTS season_key text;
