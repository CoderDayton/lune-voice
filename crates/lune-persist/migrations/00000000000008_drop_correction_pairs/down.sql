CREATE TABLE correction_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    raw TEXT NOT NULL,
    final_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_correction_pairs_created_at ON correction_pairs (created_at DESC);
