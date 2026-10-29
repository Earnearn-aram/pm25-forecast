-- Run once against an existing database before publishing the first KAN-32 model.
ALTER TABLE models ADD COLUMN IF NOT EXISTS manifest jsonb;
UPDATE models SET manifest = '{}'::jsonb WHERE manifest IS NULL;
ALTER TABLE models ALTER COLUMN manifest SET NOT NULL;
