-- Maintained word count per session, so `session_stats` can SUM server-side
-- instead of pulling every transcript into memory to call split_whitespace().
-- NULL means "not computed yet"; `Database::backfill_word_counts` fills the
-- existing rows once, right after migrations run.
ALTER TABLE sessions ADD COLUMN word_count INTEGER;
