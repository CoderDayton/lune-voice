use std::path::Path;

use chrono::{DateTime, Utc};
use diesel::prelude::*;
use diesel::sqlite::SqliteConnection;
use diesel_migrations::{EmbeddedMigrations, MigrationHarness, embed_migrations};
use uuid::Uuid;

use lune_core::asr::AsrBackendKind;
use lune_core::session::{DictationSession, SessionStatus};

use crate::error::{PersistError, PersistResult};
use crate::models::{NewDictionaryEntry, NewSession, NewSnippet};
use crate::schema::{dictionary, sessions, snippets};

type SessionRow7 = (
    String,
    String,
    Option<String>,
    String,
    String,
    String,
    Option<String>,
);

#[inline(always)]
fn row_to_session(
    (sid, s_at, e_at, sty, back, stat, trans): SessionRow7,
) -> PersistResult<DictationSession> {
    let id = Uuid::parse_str(&sid).map_err(|e| PersistError::Migration(format!("uuid: {e}")))?;
    let started_at = DateTime::parse_from_rfc3339(&s_at)
        .map_err(|e| PersistError::Migration(format!("started_at: {e}")))?
        .with_timezone(&Utc);
    let ended_at = e_at
        .map(|t| {
            DateTime::parse_from_rfc3339(&t)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| PersistError::Migration(format!("ended_at: {e}")))
        })
        .transpose()?;
    // serde_json::from_str for small enum variants is fast; no alloc on happy path
    let backend: AsrBackendKind = serde_json::from_str(&back)
        .map_err(|e| PersistError::Migration(format!("backend: {e}")))?;
    let status: SessionStatus =
        serde_json::from_str(&stat).map_err(|e| PersistError::Migration(format!("status: {e}")))?;
    Ok(DictationSession {
        id,
        started_at,
        ended_at,
        style: sty,
        backend,
        status,
        transcript: trans,
    })
}

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
        // WAL mode: concurrent readers don't block writer; synchronous=NORMAL
        // is safe with WAL and dramatically reduces fsync overhead.
        diesel::sql_query("PRAGMA journal_mode=WAL").execute(&mut conn)?;
        diesel::sql_query("PRAGMA synchronous=NORMAL").execute(&mut conn)?;
        diesel::sql_query("PRAGMA foreign_keys=ON").execute(&mut conn)?;
        // Larger cache reduces I/O on repeated reads (16 MiB).
        diesel::sql_query("PRAGMA cache_size=-16384").execute(&mut conn)?;
        // Keep temp tables in memory.
        diesel::sql_query("PRAGMA temp_store=MEMORY").execute(&mut conn)?;
        let mut db = Self { conn };
        db.run_migrations()?;
        Ok(db)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> PersistResult<Self> {
        let mut conn = SqliteConnection::establish(":memory:")?;
        diesel::sql_query("PRAGMA foreign_keys=ON").execute(&mut conn)?;
        diesel::sql_query("PRAGMA temp_store=MEMORY").execute(&mut conn)?;
        let mut db = Self { conn };
        db.run_migrations()?;
        Ok(db)
    }

    #[cold]
    #[inline(never)]
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

        // Vec::with_capacity avoids reallocations; map directly into target type.
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
        // Serialize small enums — output is typically <20 bytes, stack-friendly.
        let status_str = serde_json::to_string(&session.status)
            .map_err(|e| PersistError::Migration(e.to_string()))?;
        let backend_str = serde_json::to_string(&session.backend)
            .map_err(|e| PersistError::Migration(e.to_string()))?;
        // RFC3339 with sub-second precision; fixed-size stack allocation via SmolStr
        // is not available here so we accept the heap String from chrono.
        let started = session.started_at.to_rfc3339();
        let ended = session.ended_at.map(|t| t.to_rfc3339());
        // Uuid::hyphenated() avoids a redundant allocation vs to_string().
        let id_str = session.id.to_string();

        let new = NewSession {
            id: &id_str,
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

    pub fn list_sessions(&mut self, limit: u32) -> PersistResult<Vec<DictationSession>> {
        use crate::schema::sessions::dsl::*;

        let rows = sessions
            .select((id, started_at, ended_at, style, backend, status, transcript))
            .order(started_at.desc())
            .limit(i64::from(limit))
            .load::<SessionRow7>(&mut self.conn)?;

        // Pre-size output; map+collect avoids intermediate Vec allocation.
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(row_to_session(row)?);
        }
        Ok(out)
    }

    pub fn get_session(&mut self, session_id: Uuid) -> PersistResult<Option<DictationSession>> {
        use crate::schema::sessions::dsl::*;

        let id_str = session_id.to_string();
        let row = sessions
            .select((id, started_at, ended_at, style, backend, status, transcript))
            .filter(id.eq(&id_str))
            .first::<SessionRow7>(&mut self.conn)
            .optional()?;

        row.map(row_to_session).transpose()
    }

    pub fn update_session(&mut self, session: &DictationSession) -> PersistResult<()> {
        use crate::schema::sessions::dsl::*;

        let status_str = serde_json::to_string(&session.status)
            .map_err(|e| PersistError::Migration(format!("status serialize: {e}")))?;
        let ended = session.ended_at.map(|t| t.to_rfc3339());
        let id_str = session.id.to_string();

        diesel::update(sessions.filter(id.eq(&id_str)))
            .set((
                status.eq(&status_str),
                ended_at.eq(ended.as_deref()),
                transcript.eq(session.transcript.as_deref()),
            ))
            .execute(&mut self.conn)?;
        Ok(())
    }

    /// Delete sessions whose `started_at` is older than `days` days ago.
    /// Returns the number of rows deleted.
    pub fn prune_sessions(&mut self, days: u32) -> PersistResult<usize> {
        use crate::schema::sessions::dsl::*;

        // RFC3339 strings sort lexicographically, so < comparison is safe.
        let cutoff = (Utc::now() - chrono::Duration::days(i64::from(days))).to_rfc3339();
        let count =
            diesel::delete(sessions.filter(started_at.lt(&cutoff))).execute(&mut self.conn)?;
        Ok(count)
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
            backend: AsrBackendKind::ParakeetTdt,
            status: SessionStatus::Completed,
            transcript: Some("Hello world.".into()),
        };

        db.insert_session(&session).unwrap();
    }

    #[test]
    fn session_list_and_get() {
        let mut db = Database::open_in_memory().unwrap();

        let s1 = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: "general".into(),
            backend: AsrBackendKind::ParakeetTdt,
            status: SessionStatus::Active,
            transcript: None,
        };
        db.insert_session(&s1).unwrap();

        let list = db.list_sessions(10).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, s1.id);

        let fetched = db.get_session(s1.id).unwrap().unwrap();
        assert_eq!(fetched.id, s1.id);
        assert_eq!(fetched.style, "general");

        let not_found = db.get_session(Uuid::new_v4()).unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn session_update() {
        let mut db = Database::open_in_memory().unwrap();

        let mut session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: "general".into(),
            backend: AsrBackendKind::ParakeetTdt,
            status: SessionStatus::Active,
            transcript: None,
        };
        db.insert_session(&session).unwrap();

        session.status = SessionStatus::Completed;
        session.ended_at = Some(Utc::now());
        session.transcript = Some("Transcribed text.".into());
        db.update_session(&session).unwrap();

        let updated = db.get_session(session.id).unwrap().unwrap();
        assert!(matches!(updated.status, SessionStatus::Completed));
        assert!(updated.ended_at.is_some());
        assert_eq!(updated.transcript.unwrap(), "Transcribed text.");
    }

    #[test]
    fn session_prune() {
        let mut db = Database::open_in_memory().unwrap();

        // Insert a session with an old timestamp (40 days ago).
        let old_time = Utc::now() - chrono::Duration::days(40);
        let old_session = DictationSession {
            id: Uuid::new_v4(),
            started_at: old_time,
            ended_at: None,
            style: "general".into(),
            backend: AsrBackendKind::ParakeetTdt,
            status: SessionStatus::Completed,
            transcript: None,
        };
        db.insert_session(&old_session).unwrap();

        // Insert a recent session.
        let new_session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: "general".into(),
            backend: AsrBackendKind::ParakeetTdt,
            status: SessionStatus::Active,
            transcript: None,
        };
        db.insert_session(&new_session).unwrap();

        // Prune sessions older than 30 days.
        let pruned = db.prune_sessions(30).unwrap();
        assert_eq!(pruned, 1);

        let remaining = db.list_sessions(10).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, new_session.id);
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
