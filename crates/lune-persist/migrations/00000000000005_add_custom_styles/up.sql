CREATE TABLE custom_styles (
    id TEXT NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    style_instructions TEXT NOT NULL,
    formatting TEXT NOT NULL,
    few_shot_pairs TEXT NOT NULL DEFAULT '[]',
    few_shot_dirty INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
