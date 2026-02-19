CREATE TABLE dictionary (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    phrase TEXT NOT NULL,
    replacement TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE snippets (
    id TEXT NOT NULL PRIMARY KEY,
    trigger_phrase TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE sessions (
    id TEXT NOT NULL PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    style TEXT NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    transcript TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
