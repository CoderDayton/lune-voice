CREATE TABLE edit_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    observed_correct INTEGER NOT NULL,
    margin REAL,
    flip_fixes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_edit_observations_created_at ON edit_observations (created_at DESC);
