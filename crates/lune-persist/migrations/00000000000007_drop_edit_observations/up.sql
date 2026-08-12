-- The confidence-calibration history this table was built for was never wired
-- up: nothing in the application ever wrote or read a row. Drop it rather than
-- keep accumulating per-dictation data no code consumes.
DROP INDEX IF EXISTS idx_edit_observations_created_at;
DROP TABLE IF EXISTS edit_observations;
