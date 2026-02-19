use std::path::Path;

use diesel::prelude::*;
use diesel::sqlite::SqliteConnection;
use diesel_migrations::{EmbeddedMigrations, MigrationHarness, embed_migrations};

use crate::error::{PersistError, PersistResult};
use crate::models::{NewDictionaryEntry, NewSession, NewSnippet};
use crate::schema::{dictionary, sessions, snippets};

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

/// SQLite database for dictionary, snippets, session history.
pub struct Database {
    conn: SqliteConnection,
}

impl Database {
    /// Open (or create) the database at the given path and run migrations.
    pub fn open(path: &Path) -> PersistResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let url = path.to_string_lossy();
        let mut conn = SqliteConnection::establish(&url)?;
        diesel::sql_query("PRAGMA journal_mode=WAL")
            .execute(&mut conn)?;
        diesel::sql_query("PRAGMA foreign_keys=ON")
            .execute(&mut conn)?;
        let mut db = Self { conn };
        db.run_migrations()?;
        Ok(db)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> PersistResult<Self> {
        let mut conn = SqliteConnection::establish(":memory:")?;
        diesel::sql_query("PRAGMA foreign_keys=ON")
            .execute(&mut conn)?;
        let mut db = Self { conn };
        db.run_migrations()?;
        Ok(db)
    }

    fn run_migrations(&mut self) -> PersistResult<()> {
        self.conn
            .run_pending_migrations(MIGRATIONS)
            .map_err(|e| PersistError::Migration(e.to_string()))?;
        tracing::info!("database migrations applied");
        Ok(())
    }

    // ── Dictionary CRUD ──────────────────────────────────────────────────

    pub fn insert_dictionary_entry(
        &mut self,
        phrase: &str,
        replacement: &str,
        priority: u8,
    ) -> PersistResult<i32> {
        let new = NewDictionaryEntry {
            phrase,
            replacement,
            priority: i32::from(priority),
        };
        let id = diesel::insert_into(dictionary::table)
            .values(&new)
            .returning(dictionary::id)
            .get_result::<i32>(&mut self.conn)?;
        Ok(id)
    }

    pub fn list_dictionary_entries(
        &mut self,
    ) -> PersistResult<Vec<lune_core::dict::DictionaryEntry>> {
        use crate::schema::dictionary::dsl::*;

        let rows = dictionary
            .select((phrase, replacement, priority))
            .order(priority.desc())
            .load::<(String, String, i32)>(&mut self.conn)?;

        Ok(rows
            .into_iter()
            .map(|(p, r, pr)| lune_core::dict::DictionaryEntry {
                phrase: p,
                replacement: r,
                priority: pr as u8,
            })
            .collect())
    }

    pub fn delete_dictionary_entry(&mut self, entry_id: i32) -> PersistResult<bool> {
        let count = diesel::delete(dictionary::table.filter(dictionary::id.eq(entry_id)))
            .execute(&mut self.conn)?;
        Ok(count > 0)
    }

    // ── Snippet CRUD ─────────────────────────────────────────────────────

    pub fn insert_snippet(
        &mut self,
        snippet_id: &str,
        trigger_phrase: &str,
        body: &str,
    ) -> PersistResult<()> {
        let new = NewSnippet {
            id: snippet_id,
            trigger_phrase,
            body,
        };
        diesel::insert_into(snippets::table)
            .values(&new)
            .execute(&mut self.conn)?;
        Ok(())
    }

    pub fn list_snippets(&mut self) -> PersistResult<Vec<lune_core::dict::Snippet>> {
        use crate::schema::snippets::dsl::*;

        let rows = snippets
            .select((id, trigger_phrase, body))
            .order(trigger_phrase.asc())
            .load::<(String, String, String)>(&mut self.conn)?;

        Ok(rows
            .into_iter()
            .map(|(i, tp, b)| lune_core::dict::Snippet {
                id: i,
                trigger_phrase: tp,
                body: b,
            })
            .collect())
    }

    pub fn get_snippet_by_trigger(
        &mut self,
        trigger: &str,
    ) -> PersistResult<Option<lune_core::dict::Snippet>> {
        use crate::schema::snippets::dsl::*;

        let row = snippets
            .select((id, trigger_phrase, body))
            .filter(trigger_phrase.eq(trigger))
            .first::<(String, String, String)>(&mut self.conn)
            .optional()?;

        Ok(row.map(|(i, tp, b)| lune_core::dict::Snippet {
            id: i,
            trigger_phrase: tp,
            body: b,
        }))
    }

    pub fn delete_snippet(&mut self, snippet_id: &str) -> PersistResult<bool> {
        let count = diesel::delete(snippets::table.filter(snippets::id.eq(snippet_id)))
            .execute(&mut self.conn)?;
        Ok(count > 0)
    }

    // ── Session history ──────────────────────────────────────────────────

    pub fn insert_session(
        &mut self,
        session: &lune_core::session::DictationSession,
    ) -> PersistResult<()> {
        let status_str = serde_json::to_string(&session.status)
            .map_err(|e| PersistError::Migration(e.to_string()))?;
        let backend_str = serde_json::to_string(&session.backend)
            .map_err(|e| PersistError::Migration(e.to_string()))?;
        let started = session.started_at.to_rfc3339();
        let ended = session.ended_at.map(|t| t.to_rfc3339());

        let new = NewSession {
            id: &session.id.to_string(),
            started_at: &started,
            ended_at: ended.as_deref(),
            style: &session.style,
            backend: &backend_str,
            status: &status_str,
            transcript: session.transcript.as_deref(),
        };

        diesel::insert_into(sessions::table)
            .values(&new)
            .execute(&mut self.conn)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use lune_core::asr::AsrBackendKind;
    use lune_core::session::{DictationSession, SessionStatus};
    use uuid::Uuid;

    #[test]
    fn migration_from_empty() {
        let _db = Database::open_in_memory().unwrap();
    }

    #[test]
    fn migration_idempotent() {
        let mut db = Database::open_in_memory().unwrap();
        // Running migrations again should be a no-op.
        db.run_migrations().unwrap();
    }

    #[test]
    fn dictionary_crud() {
        let mut db = Database::open_in_memory().unwrap();

        let id = db.insert_dictionary_entry("rust lang", "Rust", 10).unwrap();
        assert!(id > 0);

        db.insert_dictionary_entry("github", "GitHub", 5).unwrap();

        let entries = db.list_dictionary_entries().unwrap();
        assert_eq!(entries.len(), 2);
        // Ordered by priority DESC
        assert_eq!(entries[0].phrase, "rust lang");
        assert_eq!(entries[1].phrase, "github");

        let deleted = db.delete_dictionary_entry(id).unwrap();
        assert!(deleted);

        let entries = db.list_dictionary_entries().unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn snippet_crud() {
        let mut db = Database::open_in_memory().unwrap();

        db.insert_snippet("greet", "insert greeting", "Hello {name}!")
            .unwrap();

        let snippets = db.list_snippets().unwrap();
        assert_eq!(snippets.len(), 1);
        assert_eq!(snippets[0].id, "greet");

        let found = db.get_snippet_by_trigger("insert greeting").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().body, "Hello {name}!");

        let not_found = db.get_snippet_by_trigger("nonexistent").unwrap();
        assert!(not_found.is_none());

        let deleted = db.delete_snippet("greet").unwrap();
        assert!(deleted);
        assert_eq!(db.list_snippets().unwrap().len(), 0);
    }

    #[test]
    fn session_insert() {
        let mut db = Database::open_in_memory().unwrap();

        let session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
            style: "general".into(),
            backend: AsrBackendKind::VoxtralLocal,
            status: SessionStatus::Completed,
            transcript: Some("Hello world.".into()),
        };

        db.insert_session(&session).unwrap();
    }

    #[test]
    fn file_based_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");

        {
            let mut db = Database::open(&path).unwrap();
            db.insert_dictionary_entry("test", "Test", 1).unwrap();
        }

        // Reopen
        let mut db2 = Database::open(&path).unwrap();
        let entries = db2.list_dictionary_entries().unwrap();
        assert_eq!(entries.len(), 1);
    }
}
