pub mod models;
pub mod schema;

use std::path::Path;

use chrono::{DateTime, Utc};
use diesel::prelude::*;
use diesel::sqlite::SqliteConnection;
use diesel_migrations::{EmbeddedMigrations, MigrationHarness, embed_migrations};

/// The history DB holds full dictated transcripts, so the file must never be
/// world-readable. `open` hardens the file (and its WAL/SHM sidecars, which
/// carry the same content in WAL mode) to owner-only, mirroring the 0600
/// `settings.rs` applies to the config file.
#[cfg(unix)]
fn restrict_to_owner(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Restrict the database file and its WAL/SHM sidecars to the owner. The
/// sidecars are created lazily by SQLite and can be absent at open time, so
/// only existing ones are touched; the process umask guard in `main` covers
/// files created later.
#[cfg(unix)]
fn restrict_db_files(path: &Path) {
    let mut candidates = vec![path.to_path_buf()];
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push("-wal");
    candidates.push(sidecar.into());
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push("-shm");
    candidates.push(sidecar.into());

    for candidate in candidates {
        if candidate.exists() {
            let _ = restrict_to_owner(&candidate);
        }
    }
}

#[cfg(not(unix))]
fn restrict_db_files(_path: &Path) {}

use uuid::Uuid;

use lune_core::asr::AsrBackendKind;
use lune_core::config::{FormatToggles, StylePreset};
use lune_core::session::{DictationSession, SessionStatus};

use self::models::{
    CustomStyleRow, NewCorrectionPair, NewCustomStyle, NewDictionaryEntry, NewSession, NewSnippet,
};
use self::schema::{custom_styles, dictionary, sessions, snippets};
use crate::error::{PersistError, PersistResult};
use diesel::result::Error as DieselError;

/// Which write [`Database::save_snippet_unique`] should perform.
#[derive(Debug, Clone, Copy)]
pub enum SnippetSaveMode<'a> {
    Insert { id: &'a str },
    Update { id: &'a str },
}

impl SnippetSaveMode<'_> {
    fn id(&self) -> &str {
        match self {
            Self::Insert { id } | Self::Update { id } => id,
        }
    }
}

/// Result of [`Database::save_snippet_unique`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnippetSaveOutcome {
    Saved,
    /// Another snippet already answers to this trigger phrase.
    TriggerTaken,
    /// Update mode, but no snippet has that id.
    NotFound,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SessionStats {
    pub sessions: u32,
    pub words: u32,
    pub seconds: u32,
    /// Same three totals, restricted to sessions at or after the `today_start`
    /// given to [`Database::session_stats`].
    ///
    /// Aggregated here rather than by walking a slice of recent sessions on the
    /// client: that slice is capped, so a busy day silently under-reports once
    /// it runs past the cap.
    pub sessions_today: u32,
    pub words_today: u32,
    pub seconds_today: u32,
}

type SessionRow7 = (
    String,
    String,
    Option<String>,
    String,
    String,
    String,
    Option<String>,
);

/// Outcome of decoding a stored backend kind from its JSON column value.
///
/// Non-parsing cases fall back to `WhisperCpp` so history still loads after a
/// backend is removed or renamed, but we distinguish an *expected* legacy
/// migration from a genuinely unrecognized value so only the latter warns.
enum BackendDecode {
    /// Parsed cleanly to a current variant.
    Parsed(AsrBackendKind),
    /// A known pre-migration backend (`Qwen3Asr` / its `OrtQwenAsr` alias),
    /// migrated to the current default. Expected, not an error.
    LegacyMigrated(AsrBackendKind),
    /// Unrecognized value; fell back to the default and worth surfacing.
    Unknown { kind: AsrBackendKind, error: String },
}

fn decode_backend(raw: &str) -> BackendDecode {
    match serde_json::from_str::<AsrBackendKind>(raw) {
        Ok(k) => BackendDecode::Parsed(k),
        // Known backends removed in the whisper.cpp migration — quietly migrated.
        Err(e) => match raw.trim().trim_matches('"') {
            "Qwen3Asr" | "OrtQwenAsr" => BackendDecode::LegacyMigrated(AsrBackendKind::WhisperCpp),
            _ => BackendDecode::Unknown {
                kind: AsrBackendKind::WhisperCpp,
                error: e.to_string(),
            },
        },
    }
}

fn row_to_session(
    (sid, s_at, e_at, sty, back, stat, trans): SessionRow7,
) -> PersistResult<DictationSession> {
    let id = Uuid::parse_str(&sid).map_err(|e| PersistError::RowDecode(format!("uuid: {e}")))?;
    let started_at = DateTime::parse_from_rfc3339(&s_at)
        .map_err(|e| PersistError::RowDecode(format!("started_at: {e}")))?
        .with_timezone(&Utc);
    let ended_at = e_at
        .map(|t| {
            DateTime::parse_from_rfc3339(&t)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| PersistError::RowDecode(format!("ended_at: {e}")))
        })
        .transpose()?;
    let backend = match decode_backend(&back) {
        BackendDecode::Parsed(kind) => kind,
        BackendDecode::LegacyMigrated(kind) => {
            tracing::debug!(
                session_id = %sid,
                stored_backend = %back,
                "migrated legacy session backend to the current default",
            );
            kind
        }
        BackendDecode::Unknown { kind, error } => {
            tracing::warn!(
                session_id = %sid,
                stored_backend = %back,
                error = %error,
                "session row stored an unknown backend kind; falling back to WhisperCpp",
            );
            kind
        }
    };
    let status: SessionStatus =
        serde_json::from_str(&stat).map_err(|e| PersistError::RowDecode(format!("status: {e}")))?;
    Ok(DictationSession {
        id,
        started_at,
        ended_at,
        // The `style` column is non-null TEXT; the empty string is the on-disk
        // representation of "no styling" (`None`).
        style: if sty.is_empty() { None } else { Some(sty) },
        backend,
        status,
        transcript: trans,
    })
}

fn row_to_snippet((i, tp, b): (String, String, String)) -> lune_core::dict::Snippet {
    lune_core::dict::Snippet {
        id: i,
        trigger_phrase: tp,
        body: b,
    }
}

/// Escape the LIKE metacharacters `%`, `_`, and `\` in a user search term so it
/// matches literally. Paired with `ESCAPE '\'` in the query.
fn escape_like(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for ch in term.chars() {
        if matches!(ch, '\\' | '%' | '_') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

/// Word count stored in `sessions.word_count`. The one definition of "a word"
/// for statistics — `split_whitespace`, exactly as the old in-memory count did,
/// so the maintained column can never disagree with what it replaced.
fn count_words(transcript: Option<&str>) -> i32 {
    transcript
        .map(|t| t.split_whitespace().count().min(i32::MAX as usize) as i32)
        .unwrap_or(0)
}

/// The single row predicate deciding whether a session "counts", shared by
/// `session_stats`, `count_sessions`, and `list_sessions_page` so the overview
/// total, the History total, and the paged rows are computed from the exact
/// same set and can never disagree. A session counts when it has a transcript
/// with non-whitespace content and a parseable start timestamp. The trim set
/// enumerates the ASCII whitespace bytes (space, tab, LF, VT, FF, CR) so a
/// whitespace-only transcript is dropped exactly as `str::split_whitespace`
/// treats it as wordless.
const COUNTABLE_WHERE: &str = "transcript IS NOT NULL \
     AND length(trim(transcript, char(32, 9, 10, 11, 12, 13))) > 0 \
     AND julianday(started_at) IS NOT NULL";

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

#[derive(Debug, Clone)]
pub struct StoredDictionaryEntry {
    pub id: i32,
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

/// A user-created styling preset loaded from the `custom_styles` table, paired
/// with its `few_shot_dirty` flag (the preset itself carries no such field).
#[derive(Debug, Clone)]
pub struct StoredCustomStyle {
    pub preset: StylePreset,
    pub few_shot_dirty: bool,
}

/// SQLite `datetime('now')`-style UTC timestamp for `updated_at` writes, so an
/// UPDATE stamps the same format the column's insert default uses.
fn sqlite_now() -> String {
    Utc::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Decode a `custom_styles` row, deserializing its JSON `formatting` and
/// `few_shot_pairs` columns back into the in-memory `StylePreset`.
fn row_to_custom_style(row: CustomStyleRow) -> PersistResult<StoredCustomStyle> {
    let formatting: FormatToggles = serde_json::from_str(&row.formatting)
        .map_err(|e| PersistError::RowDecode(format!("custom_style formatting: {e}")))?;
    let few_shot_pairs: Vec<(String, String)> = serde_json::from_str(&row.few_shot_pairs)
        .map_err(|e| PersistError::RowDecode(format!("custom_style few_shot_pairs: {e}")))?;
    Ok(StoredCustomStyle {
        preset: StylePreset {
            id: row.id,
            name: row.name,
            description: row.description,
            style_instructions: row.style_instructions,
            few_shot_pairs,
            formatting,
        },
        few_shot_dirty: row.few_shot_dirty != 0,
    })
}

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
        // First, before anything that can itself block: `journal_mode=WAL`
        // takes an exclusive lock, and at SQLite's default busy timeout of 0 a
        // second connection opening concurrently would fail outright instead of
        // waiting.
        diesel::sql_query("PRAGMA busy_timeout=5000").execute(&mut conn)?;
        diesel::sql_query("PRAGMA journal_mode=WAL").execute(&mut conn)?;
        diesel::sql_query("PRAGMA synchronous=NORMAL").execute(&mut conn)?;
        diesel::sql_query("PRAGMA cache_size=-16384").execute(&mut conn)?;
        diesel::sql_query("PRAGMA temp_store=MEMORY").execute(&mut conn)?;
        diesel::sql_query("PRAGMA mmap_size=268435456").execute(&mut conn)?;
        diesel::sql_query("PRAGMA wal_autocheckpoint=1000").execute(&mut conn)?;
        let mut db = Self { conn };
        db.run_migrations()?;
        // Migrations may have created the WAL/SHM sidecars; harden everything
        // now that the schema is in place.
        restrict_db_files(path);
        Ok(db)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> PersistResult<Self> {
        let mut conn = SqliteConnection::establish(":memory:")?;
        diesel::sql_query("PRAGMA temp_store=MEMORY").execute(&mut conn)?;
        diesel::sql_query("PRAGMA busy_timeout=5000").execute(&mut conn)?;
        let mut db = Self { conn };
        db.run_migrations()?;
        Ok(db)
    }

    #[cold]
    #[inline(never)]
    fn run_migrations(&mut self) -> PersistResult<()> {
        let applied = self
            .conn
            .run_pending_migrations(MIGRATIONS)
            .map_err(|e| PersistError::Migration(e.to_string()))?;
        if !applied.is_empty() {
            tracing::info!(count = applied.len(), "database migrations applied");
        }
        self.backfill_word_counts()?;
        Ok(())
    }

    /// Fill `sessions.word_count` for rows written before the column existed.
    ///
    /// The column is maintained on insert/update, so this runs once: the probe
    /// short-circuits on the first NULL and, after the backfill, finds none.
    /// This is the only place transcript text is read to count words — the
    /// point of the column is that `session_stats` never has to.
    ///
    /// Done in batches, never as one `load()`. A long-lived install with
    /// retention disabled can hold tens of MB of transcripts, and this runs
    /// inside `Database::open` — pulling all of it into one `Vec` would stall
    /// startup and spike RSS, which is the very thing the column exists to
    /// avoid. Each batch is its own transaction, so an interrupted run just
    /// leaves fewer rows for the next open to finish.
    #[cold]
    #[inline(never)]
    fn backfill_word_counts(&mut self) -> PersistResult<()> {
        use self::schema::sessions::dsl::*;

        /// Rows of transcript text held in memory at once.
        const BATCH: i64 = 500;

        let mut filled = 0usize;
        loop {
            // Always take the first NULL rows: each batch clears the ones it
            // read, so the next query starts where this one finished — no
            // OFFSET needed, and no row can be skipped.
            let rows: Vec<(String, Option<String>)> = sessions
                .filter(word_count.is_null())
                .select((id, transcript))
                .limit(BATCH)
                .load(&mut self.conn)?;
            if rows.is_empty() {
                break;
            }
            filled += rows.len();
            self.conn.transaction(|conn| {
                for (row_id, text) in rows {
                    diesel::update(sessions.filter(id.eq(&row_id)))
                        .set(word_count.eq(count_words(text.as_deref())))
                        .execute(conn)?;
                }
                Ok::<_, diesel::result::Error>(())
            })?;
        }
        if filled > 0 {
            tracing::info!(rows = filled, "backfilled session word counts");
        }
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

    /// Insert a dictionary entry unless one already answers to `new_phrase`.
    ///
    /// Returns `None` when a duplicate blocked the insert. Matching is
    /// case-insensitive because correction is: storing both `github → GitHub`
    /// and `GitHub → Git Hub` leaves which one applies down to row order, which
    /// the user has no way to see or control. The check shares the insert's
    /// transaction so two saves racing cannot both pass it.
    pub fn insert_dictionary_entry_unique(
        &mut self,
        new_phrase: &str,
        new_replacement: &str,
        new_priority: u8,
        phrases_collide: impl Fn(&str, &str) -> bool,
    ) -> PersistResult<Option<i32>> {
        use self::schema::dictionary::dsl::*;

        let inserted = self.conn.transaction::<_, DieselError, _>(|conn| {
            let existing = dictionary.select(phrase).load::<String>(conn)?;
            if existing.iter().any(|p| phrases_collide(p, new_phrase)) {
                return Ok(None);
            }
            let new_id = diesel::insert_into(dictionary)
                .values(&NewDictionaryEntry {
                    phrase: new_phrase,
                    replacement: new_replacement,
                    priority: i32::from(new_priority),
                })
                .returning(id)
                .get_result::<i32>(conn)?;
            Ok(Some(new_id))
        })?;
        Ok(inserted)
    }

    pub fn list_dictionary_entries(
        &mut self,
    ) -> PersistResult<Vec<lune_core::dict::DictionaryEntry>> {
        Ok(self
            .list_dictionary_entries_with_ids()?
            .into_iter()
            .map(|entry| lune_core::dict::DictionaryEntry {
                phrase: entry.phrase,
                replacement: entry.replacement,
                priority: entry.priority,
            })
            .collect())
    }

    pub fn list_dictionary_entries_with_ids(
        &mut self,
    ) -> PersistResult<Vec<StoredDictionaryEntry>> {
        use self::schema::dictionary::dsl::*;

        let rows = dictionary
            .select((id, phrase, replacement, priority))
            .order((priority.desc(), id.asc()))
            .load::<(i32, String, String, i32)>(&mut self.conn)?;

        Ok(rows
            .into_iter()
            .map(|(entry_id, p, r, pr)| StoredDictionaryEntry {
                id: entry_id,
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

    /// Update a dictionary entry unless another one already answers to
    /// `new_phrase`.
    ///
    /// The counterpart to [`Self::insert_dictionary_entry_unique`]: without it
    /// the uniqueness rule holds only for new rows, and editing an existing
    /// entry's phrase to collide with another re-creates exactly the
    /// row-order-decides-the-winner situation that rule exists to prevent.
    ///
    /// Returns `None` when a collision blocked the write, `Some(false)` when no
    /// entry has that id. The check shares the update's transaction, so two
    /// saves racing cannot both pass it.
    pub fn update_dictionary_entry_unique(
        &mut self,
        entry_id: i32,
        new_phrase: &str,
        new_replacement: &str,
        new_priority: u8,
        phrases_collide: impl Fn(&str, &str) -> bool,
    ) -> PersistResult<Option<bool>> {
        use self::schema::dictionary::dsl::*;

        let outcome = self.conn.transaction::<_, DieselError, _>(|conn| {
            let existing = dictionary
                .select((id, phrase))
                .load::<(i32, String)>(conn)?;
            if existing
                .iter()
                .any(|(row_id, p)| *row_id != entry_id && phrases_collide(p, new_phrase))
            {
                return Ok(None);
            }
            let count = diesel::update(dictionary.filter(id.eq(entry_id)))
                .set((
                    phrase.eq(new_phrase),
                    replacement.eq(new_replacement),
                    priority.eq(i32::from(new_priority)),
                    updated_at.eq(Utc::now().to_rfc3339()),
                ))
                .execute(conn)?;
            Ok(Some(count > 0))
        })?;
        Ok(outcome)
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

    /// Insert or update a snippet with its trigger-collision check inside the
    /// same transaction as the write.
    ///
    /// Checking for a colliding trigger with `list_snippets` and *then* writing
    /// leaves a window in which two concurrent saves both see no collision and
    /// both land, producing two snippets that the same utterance matches —
    /// spoken matching then picks whichever sorts first, not the one the user
    /// meant. `collides` is supplied by the caller because the normalization
    /// rule (case, punctuation, whitespace) lives with the matcher, not here.
    pub fn save_snippet_unique(
        &mut self,
        mode: SnippetSaveMode<'_>,
        new_trigger: &str,
        new_body: &str,
        collides: impl Fn(&str, &str) -> bool,
    ) -> PersistResult<SnippetSaveOutcome> {
        use self::schema::snippets::dsl::*;

        let outcome = self.conn.transaction::<_, DieselError, _>(|conn| {
            let existing = snippets
                .select((id, trigger_phrase))
                .load::<(String, String)>(conn)?;
            let own_id = mode.id();
            let taken = existing.iter().any(|(row_id, row_trigger)| {
                (!matches!(mode, SnippetSaveMode::Update { .. }) || row_id != own_id)
                    && collides(row_trigger, new_trigger)
            });
            if taken {
                return Ok(SnippetSaveOutcome::TriggerTaken);
            }
            match mode {
                SnippetSaveMode::Insert { id: new_id } => {
                    diesel::insert_into(snippets)
                        .values(&NewSnippet {
                            id: new_id,
                            trigger_phrase: new_trigger,
                            body: new_body,
                        })
                        .execute(conn)?;
                    Ok(SnippetSaveOutcome::Saved)
                }
                SnippetSaveMode::Update { id: target } => {
                    let count = diesel::update(snippets.filter(id.eq(target)))
                        .set((trigger_phrase.eq(new_trigger), body.eq(new_body)))
                        .execute(conn)?;
                    Ok(if count > 0 {
                        SnippetSaveOutcome::Saved
                    } else {
                        SnippetSaveOutcome::NotFound
                    })
                }
            }
        })?;
        Ok(outcome)
    }

    pub fn list_snippets(&mut self) -> PersistResult<Vec<lune_core::dict::Snippet>> {
        use self::schema::snippets::dsl::*;

        let rows = snippets
            .select((id, trigger_phrase, body))
            .order(trigger_phrase.asc())
            .load::<(String, String, String)>(&mut self.conn)?;

        Ok(rows.into_iter().map(row_to_snippet).collect())
    }

    pub fn get_snippet_by_trigger(
        &mut self,
        trigger: &str,
    ) -> PersistResult<Option<lune_core::dict::Snippet>> {
        use self::schema::snippets::dsl::*;

        let row = snippets
            .select((id, trigger_phrase, body))
            .filter(trigger_phrase.eq(trigger))
            .first::<(String, String, String)>(&mut self.conn)
            .optional()?;

        Ok(row.map(row_to_snippet))
    }

    /// Update an existing snippet's trigger and body. Returns `false` when no
    /// snippet has that id.
    pub fn update_snippet(
        &mut self,
        snippet_id: &str,
        new_trigger: &str,
        new_body: &str,
    ) -> PersistResult<bool> {
        use self::schema::snippets::dsl::*;
        let count = diesel::update(snippets.filter(id.eq(snippet_id)))
            .set((trigger_phrase.eq(new_trigger), body.eq(new_body)))
            .execute(&mut self.conn)?;
        Ok(count > 0)
    }

    pub fn delete_snippet(&mut self, snippet_id: &str) -> PersistResult<bool> {
        let count = diesel::delete(snippets::table.filter(snippets::id.eq(snippet_id)))
            .execute(&mut self.conn)?;
        Ok(count > 0)
    }

    // ── Custom style presets ─────────────────────────────────────────────

    /// All custom styles in stable display order (insertion order).
    pub fn list_custom_styles(&mut self) -> PersistResult<Vec<StoredCustomStyle>> {
        use self::schema::custom_styles::dsl::*;
        custom_styles
            .select(CustomStyleRow::as_select())
            // Insertion order (stable display order). `created_at` alone is not
            // enough: its `datetime('now')` default is whole-second, so styles
            // migrated in a tight loop share a timestamp and would then sort by
            // random UUID. The implicit rowid is strictly monotonic per insert,
            // and this app never VACUUMs, so rowids never renumber.
            .order(diesel::dsl::sql::<diesel::sql_types::BigInt>("rowid"))
            .load::<CustomStyleRow>(&mut self.conn)?
            .into_iter()
            .map(row_to_custom_style)
            .collect()
    }

    /// Look up a single custom style by id.
    pub fn get_custom_style(&mut self, style_id: &str) -> PersistResult<Option<StoredCustomStyle>> {
        use self::schema::custom_styles::dsl::*;
        custom_styles
            .select(CustomStyleRow::as_select())
            .filter(id.eq(style_id))
            .first::<CustomStyleRow>(&mut self.conn)
            .optional()?
            .map(row_to_custom_style)
            .transpose()
    }

    /// Current number of stored custom styles (for the per-user cap).
    pub fn count_custom_styles(&mut self) -> PersistResult<i64> {
        Ok(custom_styles::table.count().get_result(&mut self.conn)?)
    }

    /// Insert a new custom style. `dirty` marks its few-shot examples as stale.
    pub fn insert_custom_style(&mut self, preset: &StylePreset, dirty: bool) -> PersistResult<()> {
        let fmt = serde_json::to_string(&preset.formatting)
            .map_err(|e| PersistError::RowDecode(format!("serialize formatting: {e}")))?;
        let pairs = serde_json::to_string(&preset.few_shot_pairs)
            .map_err(|e| PersistError::RowDecode(format!("serialize few_shot_pairs: {e}")))?;
        let new = NewCustomStyle {
            id: &preset.id,
            name: &preset.name,
            description: &preset.description,
            style_instructions: &preset.style_instructions,
            formatting: &fmt,
            few_shot_pairs: &pairs,
            few_shot_dirty: dirty as i32,
        };
        diesel::insert_into(custom_styles::table)
            .values(&new)
            .execute(&mut self.conn)?;
        Ok(())
    }

    /// Insert legacy styles during the one-time `settings.toml` → DB migration.
    /// `INSERT OR IGNORE` skips any id already present (from a prior partial
    /// run), so the migration is convergent and safe to retry without erroring
    /// or duplicating. The whole batch runs in one transaction: any error rolls
    /// it back and leaves the DB untouched, so the source TOML stays
    /// authoritative until the migration fully succeeds. Returns the number of
    /// rows actually inserted.
    pub fn insert_custom_styles_migration(
        &mut self,
        styles: &[(StylePreset, bool)],
    ) -> PersistResult<usize> {
        self.conn.transaction(|conn| {
            let mut inserted = 0usize;
            for (preset, dirty) in styles {
                let fmt = serde_json::to_string(&preset.formatting)
                    .map_err(|e| PersistError::RowDecode(format!("serialize formatting: {e}")))?;
                let pairs = serde_json::to_string(&preset.few_shot_pairs).map_err(|e| {
                    PersistError::RowDecode(format!("serialize few_shot_pairs: {e}"))
                })?;
                let new = NewCustomStyle {
                    id: &preset.id,
                    name: &preset.name,
                    description: &preset.description,
                    style_instructions: &preset.style_instructions,
                    formatting: &fmt,
                    few_shot_pairs: &pairs,
                    few_shot_dirty: *dirty as i32,
                };
                inserted += diesel::insert_or_ignore_into(custom_styles::table)
                    .values(&new)
                    .execute(conn)?;
            }
            Ok(inserted)
        })
    }

    /// Update a custom style's editable fields (name/description/instructions/
    /// formatting) and its stale flag. Few-shot pairs are updated separately via
    /// [`set_custom_style_few_shot_pairs`]. Returns `false` if the id is unknown.
    pub fn update_custom_style_fields(
        &mut self,
        style_id: &str,
        name_val: &str,
        description_val: &str,
        instructions_val: &str,
        formatting_val: &FormatToggles,
        dirty: bool,
    ) -> PersistResult<bool> {
        use self::schema::custom_styles::dsl::*;
        let fmt = serde_json::to_string(formatting_val)
            .map_err(|e| PersistError::RowDecode(format!("serialize formatting: {e}")))?;
        let count = diesel::update(custom_styles.filter(id.eq(style_id)))
            .set((
                name.eq(name_val),
                description.eq(description_val),
                style_instructions.eq(instructions_val),
                formatting.eq(fmt),
                few_shot_dirty.eq(dirty as i32),
                updated_at.eq(sqlite_now()),
            ))
            .execute(&mut self.conn)?;
        Ok(count > 0)
    }

    /// Replace a custom style's few-shot pairs and set its stale flag. Returns
    /// `false` if the id is unknown.
    pub fn set_custom_style_few_shot_pairs(
        &mut self,
        style_id: &str,
        pairs_val: &[(String, String)],
        dirty: bool,
    ) -> PersistResult<bool> {
        use self::schema::custom_styles::dsl::*;
        let json = serde_json::to_string(pairs_val)
            .map_err(|e| PersistError::RowDecode(format!("serialize few_shot_pairs: {e}")))?;
        let count = diesel::update(custom_styles.filter(id.eq(style_id)))
            .set((
                few_shot_pairs.eq(json),
                few_shot_dirty.eq(dirty as i32),
                updated_at.eq(sqlite_now()),
            ))
            .execute(&mut self.conn)?;
        Ok(count > 0)
    }

    /// Delete a custom style. Returns `false` if the id was not present.
    pub fn delete_custom_style(&mut self, style_id: &str) -> PersistResult<bool> {
        let count = diesel::delete(custom_styles::table.filter(custom_styles::id.eq(style_id)))
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
            .map_err(|e| PersistError::RowEncode(format!("status: {e}")))?;
        let backend_str = serde_json::to_string(&session.backend)
            .map_err(|e| PersistError::RowEncode(format!("backend: {e}")))?;
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
            // Persist "no styling" (`None`) as the empty string (column is non-null).
            style: session.style.as_deref().unwrap_or(""),
            backend: &backend_str,
            status: &status_str,
            transcript: session.transcript.as_deref(),
            word_count: count_words(session.transcript.as_deref()),
        };

        diesel::insert_into(sessions::table)
            .values(&new)
            .execute(&mut self.conn)?;
        Ok(())
    }

    /// Aggregate stats across the entire sessions table (not capped by history limits).
    /// Counts only sessions with non-empty transcripts so it matches what the UI shows
    /// in the recent-sessions list. Rows with malformed timestamps are skipped entirely
    /// (no count, no words, no seconds) — `julianday()` returns NULL on parse failure,
    /// which propagates through the CASE and gives us a uniform skip path.
    /// Lifetime totals for the overview screen. Deliberately not capped by the
    /// history limits, which is why it aggregates entirely in SQL: loading
    /// every transcript to count words was an unbounded full-table scan
    /// pulling megabytes of text through the query layer for three integers.
    /// `word_count` is maintained on insert/update, so no transcript text
    /// crosses the query boundary here.
    /// Aggregate session totals, all-time and since `today_start`.
    ///
    /// `today_start` is an RFC3339 timestamp — the caller's local midnight,
    /// converted to UTC. Compared as a string, which is sound because every
    /// `started_at` is written by `Utc::now().to_rfc3339()` and so shares one
    /// offset; the same assumption `prune_sessions` runs on.
    pub fn session_stats(&mut self, today_start: &str) -> PersistResult<SessionStats> {
        use diesel::sql_types::{BigInt, Text};

        #[derive(diesel::QueryableByName)]
        struct Row {
            #[diesel(sql_type = BigInt)]
            sessions: i64,
            #[diesel(sql_type = BigInt)]
            words: i64,
            #[diesel(sql_type = BigInt)]
            secs: i64,
            #[diesel(sql_type = BigInt)]
            sessions_today: i64,
            #[diesel(sql_type = BigInt)]
            words_today: i64,
            #[diesel(sql_type = BigInt)]
            secs_today: i64,
        }

        // Duration of one row, in whole seconds. Repeated verbatim in the
        // today-scoped sum below, so keep the two in step.
        const SECS: &str = "CASE \
                 WHEN ended_at IS NULL OR julianday(ended_at) IS NULL THEN 0 \
                 ELSE CAST(MAX(1, (julianday(ended_at) - julianday(started_at)) * 86400) AS INTEGER) \
             END";

        // COUNTABLE_WHERE already excludes malformed `started_at` and
        // whitespace-only transcripts, so this walks the exact same rows as
        // `count_sessions` — the sessions total is identical by construction.
        // A session with no (or an unparseable) `ended_at` contributes 0
        // seconds, matching the old per-row `None` skip.
        let row: Row = diesel::sql_query(format!(
            "SELECT \
                 COUNT(*) AS sessions, \
                 COALESCE(SUM(word_count), 0) AS words, \
                 COALESCE(SUM({SECS}), 0) AS secs, \
                 COALESCE(SUM(CASE WHEN started_at >= ? THEN 1 ELSE 0 END), 0) AS sessions_today, \
                 COALESCE(SUM(CASE WHEN started_at >= ? THEN word_count ELSE 0 END), 0) AS words_today, \
                 COALESCE(SUM(CASE WHEN started_at >= ? THEN {SECS} ELSE 0 END), 0) AS secs_today \
             FROM sessions WHERE {COUNTABLE_WHERE}"
        ))
        .bind::<Text, _>(today_start)
        .bind::<Text, _>(today_start)
        .bind::<Text, _>(today_start)
        .get_result(&mut self.conn)?;

        let clamp_u32 = |v: i64| v.clamp(0, i64::from(u32::MAX)) as u32;
        Ok(SessionStats {
            sessions: clamp_u32(row.sessions),
            words: clamp_u32(row.words),
            seconds: clamp_u32(row.secs),
            sessions_today: clamp_u32(row.sessions_today),
            words_today: clamp_u32(row.words_today),
            seconds_today: clamp_u32(row.secs_today),
        })
    }

    pub fn list_sessions(&mut self, limit: u32) -> PersistResult<Vec<DictationSession>> {
        use self::schema::sessions::dsl::*;

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

    /// Count sessions with a non-empty transcript, optionally filtered by a
    /// case-insensitive transcript substring. The unfiltered count uses the
    /// same predicate as `session_stats`, so the History total matches the
    /// overview's lifetime session count.
    pub fn count_sessions(&mut self, query: Option<&str>) -> PersistResult<i64> {
        use diesel::sql_types::{BigInt, Text};

        #[derive(diesel::QueryableByName)]
        struct Count {
            #[diesel(sql_type = BigInt)]
            n: i64,
        }

        let row: Count = match query.map(str::trim).filter(|q| !q.is_empty()) {
            // `LIKE` is case-insensitive for ASCII only — SQLite has no Unicode
            // case folding without the ICU extension or FTS, which would be
            // disproportionate here; adequate for transcript substring search.
            Some(q) => diesel::sql_query(format!(
                "SELECT COUNT(*) AS n FROM sessions WHERE {COUNTABLE_WHERE} AND transcript LIKE ? ESCAPE '\\'"
            ))
            .bind::<Text, _>(format!("%{}%", escape_like(q)))
            .get_result(&mut self.conn)?,
            None => diesel::sql_query(format!(
                "SELECT COUNT(*) AS n FROM sessions WHERE {COUNTABLE_WHERE}"
            ))
            .get_result(&mut self.conn)?,
        };
        Ok(row.n)
    }

    /// A page of sessions (newest first) using the same non-empty-transcript
    /// predicate as `count_sessions`, so pages and totals stay consistent.
    /// `query` filters by a case-insensitive transcript substring.
    pub fn list_sessions_page(
        &mut self,
        limit: u32,
        offset: u32,
        query: Option<&str>,
    ) -> PersistResult<Vec<DictationSession>> {
        use diesel::sql_types::{BigInt, Nullable, Text};

        #[derive(diesel::QueryableByName)]
        struct Row {
            #[diesel(sql_type = Text)]
            id: String,
            #[diesel(sql_type = Text)]
            started_at: String,
            #[diesel(sql_type = Nullable<Text>)]
            ended_at: Option<String>,
            #[diesel(sql_type = Text)]
            style: String,
            #[diesel(sql_type = Text)]
            backend: String,
            #[diesel(sql_type = Text)]
            status: String,
            #[diesel(sql_type = Nullable<Text>)]
            transcript: Option<String>,
        }

        let cols = format!(
            "SELECT id, started_at, ended_at, style, backend, status, transcript \
             FROM sessions WHERE {COUNTABLE_WHERE}"
        );
        const TAIL: &str = " ORDER BY started_at DESC LIMIT ? OFFSET ?";

        let rows: Vec<Row> = match query.map(str::trim).filter(|q| !q.is_empty()) {
            // ASCII-only case folding, as in `count_sessions`.
            Some(q) => diesel::sql_query(format!("{cols} AND transcript LIKE ? ESCAPE '\\'{TAIL}"))
                .bind::<Text, _>(format!("%{}%", escape_like(q)))
                .bind::<BigInt, _>(i64::from(limit))
                .bind::<BigInt, _>(i64::from(offset))
                .load(&mut self.conn)?,
            None => diesel::sql_query(format!("{cols}{TAIL}"))
                .bind::<BigInt, _>(i64::from(limit))
                .bind::<BigInt, _>(i64::from(offset))
                .load(&mut self.conn)?,
        };

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(row_to_session((
                r.id,
                r.started_at,
                r.ended_at,
                r.style,
                r.backend,
                r.status,
                r.transcript,
            ))?);
        }
        Ok(out)
    }

    pub fn get_session(&mut self, session_id: Uuid) -> PersistResult<Option<DictationSession>> {
        use self::schema::sessions::dsl::*;

        let id_str = session_id.to_string();
        let row = sessions
            .select((id, started_at, ended_at, style, backend, status, transcript))
            .filter(id.eq(&id_str))
            .first::<SessionRow7>(&mut self.conn)
            .optional()?;

        row.map(row_to_session).transpose()
    }

    pub fn update_session(&mut self, session: &DictationSession) -> PersistResult<()> {
        use self::schema::sessions::dsl::*;

        let status_str = serde_json::to_string(&session.status)
            .map_err(|e| PersistError::RowEncode(format!("status: {e}")))?;
        let ended = session.ended_at.map(|t| t.to_rfc3339());
        let id_str = session.id.to_string();

        diesel::update(sessions.filter(id.eq(&id_str)))
            .set((
                status.eq(&status_str),
                ended_at.eq(ended.as_deref()),
                transcript.eq(session.transcript.as_deref()),
                // Kept in lockstep with `transcript` — the two must never
                // disagree, or lifetime word totals silently drift.
                word_count.eq(count_words(session.transcript.as_deref())),
            ))
            .execute(&mut self.conn)?;
        Ok(())
    }

    /// Record the raw ASR text and final delivered text for a session — the
    /// labelled `(raw → final)` history. Best-effort; the caller treats failure
    /// as non-fatal.
    pub fn insert_correction_pair(
        &mut self,
        session_id: Uuid,
        raw: &str,
        final_text: &str,
    ) -> PersistResult<()> {
        use self::schema::correction_pairs::dsl as cp;
        // Keep only the newest rows so the table can't grow without bound.
        const CAP: i64 = 5000;
        let sid = session_id.to_string();
        let now = Utc::now().to_rfc3339();
        // Insert + cap-trim atomically: a failed trim can't leave the table
        // over CAP.
        self.conn.transaction(|conn| {
            diesel::insert_into(cp::correction_pairs)
                .values(NewCorrectionPair {
                    session_id: &sid,
                    raw,
                    final_text,
                    created_at: &now,
                })
                .execute(conn)?;
            diesel::sql_query(
                "DELETE FROM correction_pairs \
                 WHERE id <= (SELECT MAX(id) FROM correction_pairs) - ?",
            )
            .bind::<diesel::sql_types::BigInt, _>(CAP)
            .execute(conn)?;
            Ok(())
        })
    }

    /// Most recent `(raw, final)` correction pairs, newest first.
    ///
    /// Nothing in the app reads these — they are written for retention and
    /// cascade-delete only. The reader exists so benchmarks and offline
    /// analysis can mine real `raw → final` deltas (what the corrector actually
    /// did to your dictation) without going around the crate with raw SQL.
    ///
    /// Contains verbatim transcript text: never log the result, and honour the
    /// same redaction rules as the rest of the pipeline.
    pub fn list_correction_pairs(&mut self, limit: u32) -> PersistResult<Vec<(String, String)>> {
        use self::schema::correction_pairs::dsl as cp;
        let rows = cp::correction_pairs
            .select((cp::raw, cp::final_text))
            .order((cp::created_at.desc(), cp::id.desc()))
            .limit(i64::from(limit))
            .load::<(String, String)>(&mut self.conn)?;
        Ok(rows)
    }

    pub fn delete_session(&mut self, session_id: &str) -> PersistResult<bool> {
        // Cascade: drop the session's calibration pairs so deleting a session
        // leaves no transcript text behind. There is no SQL foreign key and the
        // connection doesn't enable `PRAGMA foreign_keys`, so delete explicitly.
        // One transaction, so a crash mid-cascade can't leave the `sessions`
        // row — which carries the transcript — behind after its pairs are gone.
        self.conn.transaction(|conn| {
            {
                use self::schema::correction_pairs::dsl as cp;
                diesel::delete(cp::correction_pairs.filter(cp::session_id.eq(session_id)))
                    .execute(conn)?;
            }
            use self::schema::sessions::dsl::*;
            let count = diesel::delete(sessions.filter(id.eq(session_id))).execute(conn)?;
            Ok(count > 0)
        })
    }

    /// Delete several sessions at once, cascading to their correction pairs.
    ///
    /// One transaction for the whole batch, so the History tab's bulk delete
    /// either removes every selected row or none — a partial commit would leave
    /// the user re-selecting whatever survived. Returns how many `sessions`
    /// rows actually existed; ids that were already gone simply don't count.
    pub fn delete_sessions(&mut self, session_ids: &[String]) -> PersistResult<usize> {
        if session_ids.is_empty() {
            return Ok(0);
        }
        self.conn.transaction(|conn| {
            {
                use self::schema::correction_pairs::dsl as cp;
                diesel::delete(cp::correction_pairs.filter(cp::session_id.eq_any(session_ids)))
                    .execute(conn)?;
            }
            use self::schema::sessions::dsl::*;
            let count = diesel::delete(sessions.filter(id.eq_any(session_ids))).execute(conn)?;
            Ok(count)
        })
    }

    /// Close out sessions left open by a previous run.
    ///
    /// A row is inserted `Active` with no `ended_at` when a dictation starts,
    /// and the session's own teardown closes it. A hard kill — SIGKILL, an
    /// OOM-kill, a power cut — has no teardown, so the row stays `Active`
    /// forever and shows up as a permanently in-progress entry in the history.
    /// The in-process panic supervisor cannot reach these; only a sweep at
    /// startup can. Returns the number of rows closed.
    ///
    /// Scoped to rows that are still `Active`, not merely rows with a null
    /// `ended_at`: an init failure (mic busy, model missing) records its own
    /// `Failed(reason)` without an end time, and overwriting that would replace
    /// the real cause with a generic "interrupted".
    pub fn close_orphaned_sessions(&mut self) -> PersistResult<usize> {
        use self::schema::sessions::dsl::*;
        let now = Utc::now().to_rfc3339();
        let active = serde_json::to_string(&SessionStatus::Active)
            .map_err(|e| PersistError::RowEncode(format!("status serialize: {e}")))?;
        let failed = serde_json::to_string(&SessionStatus::Failed(
            "interrupted — Lune Voice stopped before this dictation finished".into(),
        ))
        .map_err(|e| PersistError::RowEncode(format!("status serialize: {e}")))?;
        let closed = diesel::update(sessions.filter(ended_at.is_null().and(status.eq(active))))
            .set((status.eq(failed), ended_at.eq(now)))
            .execute(&mut self.conn)?;
        Ok(closed)
    }

    /// Delete every session and every correction pair.
    ///
    /// This is what switching retention to "Don't save history" runs: turning
    /// the setting off has to remove what was already collected, not merely
    /// stop adding to it. Returns the number of sessions deleted.
    pub fn purge_history(&mut self) -> PersistResult<usize> {
        self.conn.transaction(|conn| {
            let count = {
                use self::schema::sessions::dsl::*;
                diesel::delete(sessions).execute(conn)?
            };
            use self::schema::correction_pairs::dsl as cp;
            diesel::delete(cp::correction_pairs).execute(conn)?;
            Ok(count)
        })
    }

    /// Delete sessions whose `started_at` is older than `days` days ago.
    /// Returns the number of rows deleted.
    pub fn prune_sessions(&mut self, days: u32) -> PersistResult<usize> {
        // RFC3339 strings sort lexicographically, so < comparison is safe.
        let cutoff = (Utc::now() - chrono::Duration::days(i64::from(days))).to_rfc3339();
        // Same atomicity discipline as `delete_session`: the retention sweep is
        // one cascade, so it commits or rolls back as one.
        self.conn.transaction(|conn| {
            let count = {
                use self::schema::sessions::dsl::*;
                diesel::delete(sessions.filter(started_at.lt(&cutoff))).execute(conn)?
            };
            // Prune calibration pairs in the same retention window.
            {
                use self::schema::correction_pairs::dsl as cp;
                diesel::delete(cp::correction_pairs.filter(cp::created_at.lt(&cutoff)))
                    .execute(conn)?;
            }
            Ok(count)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use lune_core::asr::AsrBackendKind;
    use lune_core::session::{DictationSession, SessionStatus};
    use uuid::Uuid;

    /// A `today_start` far enough in the past that every row counts as "today",
    /// so the all-time and today-scoped totals coincide and a test asserting on
    /// one is really asserting on both.
    const EPOCH: &str = "1970-01-01T00:00:00+00:00";

    /// Every migration ships a `down.sql`, but nothing exercised them — a
    /// `down` that does not actually reverse its `up` is only discovered during
    /// an emergency rollback, which is the worst possible time to find out.
    ///
    /// Reverting to empty and re-applying proves each `down` undoes its `up`
    /// cleanly and that the resulting schema still accepts real writes.
    #[test]
    fn migrations_revert_and_reapply_cleanly() {
        let mut db = Database::open_in_memory().expect("open");

        db.conn
            .revert_all_migrations(MIGRATIONS)
            .expect("every down.sql should reverse its up.sql");

        // Re-applying onto the reverted schema must work — a `down` that left
        // a table or index behind would collide here.
        db.run_migrations().expect("re-apply after revert");

        // And the round-tripped schema must still be usable, not merely present.
        let session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Completed,
            transcript: Some("Round trip.".into()),
        };
        db.insert_session(&session)
            .expect("schema is usable after a revert/re-apply cycle");
    }

    /// A corrupt database file must surface as a typed error the caller can
    /// act on, not a panic. The DB lives in the user's data dir where a power
    /// loss mid-write or a bad restore can leave garbage behind, and the app
    /// has to be able to report that instead of dying at startup.
    #[test]
    fn a_corrupt_database_file_errors_instead_of_panicking() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("lune.db");
        // Valid SQLite header magic followed by garbage: gets past the initial
        // format sniff, then fails once pages are actually read.
        let mut bytes = b"SQLite format 3\0".to_vec();
        bytes.extend(std::iter::repeat_n(0xAB, 4096));
        std::fs::write(&path, &bytes).expect("write corrupt db");

        let result = Database::open(&path);
        assert!(
            result.is_err(),
            "a corrupt database should be reported, not opened"
        );
    }

    /// An unreadable path (a directory where the file should be) must also be
    /// an error rather than a panic — this is what a botched install or a
    /// stale mount point looks like.
    #[test]
    fn an_unopenable_database_path_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("lune.db");
        std::fs::create_dir(&path).expect("create dir in the db's place");

        assert!(
            Database::open(&path).is_err(),
            "a directory in the database's place should be reported"
        );
    }

    /// Reverting one migration at a time must also work: a partial rollback is
    /// the realistic emergency, not an all-or-nothing one.
    #[test]
    fn the_most_recent_migration_reverts_on_its_own() {
        let mut db = Database::open_in_memory().expect("open");

        db.conn
            .revert_last_migration(MIGRATIONS)
            .expect("the latest down.sql should apply on its own");
        db.run_migrations()
            .expect("re-apply the reverted migration");
    }

    #[test]
    fn migration_from_empty() {
        let _db = Database::open_in_memory().unwrap();
    }

    #[test]
    fn custom_style_crud() {
        use lune_core::config::{FormatToggles, StylePreset};
        let mut db = Database::open_in_memory().unwrap();

        let preset = StylePreset {
            id: "custom-1".into(),
            name: "My Style".into(),
            description: "desc".into(),
            style_instructions: "Be terse.".into(),
            few_shot_pairs: vec![("in".into(), "out".into())],
            formatting: FormatToggles {
                auto_capitalization: false,
                ..FormatToggles::default()
            },
        };
        db.insert_custom_style(&preset, true).unwrap();

        // list/get round-trip, including the JSON formatting + few_shot columns.
        let all = db.list_custom_styles().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].preset.id, "custom-1");
        assert!(all[0].few_shot_dirty);
        assert_eq!(
            all[0].preset.few_shot_pairs,
            vec![("in".to_string(), "out".to_string())]
        );
        assert!(!all[0].preset.formatting.auto_capitalization);
        assert_eq!(db.count_custom_styles().unwrap(), 1);

        // Field update sets the stale flag and rewrites formatting.
        assert!(
            db.update_custom_style_fields(
                "custom-1",
                "Renamed",
                "desc2",
                "Be verbose.",
                &FormatToggles::default(),
                false,
            )
            .unwrap()
        );
        let got = db.get_custom_style("custom-1").unwrap().unwrap();
        assert_eq!(got.preset.name, "Renamed");
        assert_eq!(got.preset.style_instructions, "Be verbose.");
        assert!(!got.few_shot_dirty);
        assert!(got.preset.formatting.auto_capitalization);

        // Few-shot pairs replace independently.
        assert!(
            db.set_custom_style_few_shot_pairs("custom-1", &[("a".into(), "b".into())], false)
                .unwrap()
        );
        let got = db.get_custom_style("custom-1").unwrap().unwrap();
        assert_eq!(
            got.preset.few_shot_pairs,
            vec![("a".to_string(), "b".to_string())]
        );

        // Unknown ids report `false`, never error.
        assert!(
            !db.update_custom_style_fields("nope", "n", "d", "i", &FormatToggles::default(), false)
                .unwrap()
        );
        assert!(
            !db.set_custom_style_few_shot_pairs("nope", &[], false)
                .unwrap()
        );

        // Delete is idempotent.
        assert!(db.delete_custom_style("custom-1").unwrap());
        assert!(!db.delete_custom_style("custom-1").unwrap());
        assert_eq!(db.count_custom_styles().unwrap(), 0);
        assert!(db.get_custom_style("custom-1").unwrap().is_none());
    }

    /// Build a minimal `StylePreset` with the given id for ordering/migration
    /// tests (name mirrors the id; everything else is default).
    fn style_with_id(id: &str) -> StylePreset {
        StylePreset {
            id: id.into(),
            name: id.into(),
            description: String::new(),
            style_instructions: "x".into(),
            few_shot_pairs: Vec::new(),
            formatting: FormatToggles::default(),
        }
    }

    #[test]
    fn list_custom_styles_preserves_insertion_order_within_one_second() {
        let mut db = Database::open_in_memory().unwrap();
        // Insert with ids whose lexical order is the REVERSE of insertion order,
        // so a fallback to `ORDER BY id` (the old bug) would reshuffle them.
        // All land in the same whole second, defeating a `created_at` tiebreak.
        for id in ["zeta", "mike", "alpha"] {
            db.insert_custom_style(&style_with_id(id), false).unwrap();
        }
        let ids: Vec<_> = db
            .list_custom_styles()
            .unwrap()
            .into_iter()
            .map(|s| s.preset.id)
            .collect();
        assert_eq!(ids, vec!["zeta", "mike", "alpha"]);
    }

    #[test]
    fn migration_insert_is_atomic_and_convergent() {
        let mut db = Database::open_in_memory().unwrap();
        let batch = vec![
            (style_with_id("custom-a"), true),
            (style_with_id("custom-b"), false),
        ];
        // First run inserts both, preserving the dirty flag per row.
        assert_eq!(db.insert_custom_styles_migration(&batch).unwrap(), 2);
        assert_eq!(db.count_custom_styles().unwrap(), 2);
        assert!(
            db.get_custom_style("custom-a")
                .unwrap()
                .unwrap()
                .few_shot_dirty
        );

        // A re-run (e.g. after a prior strip failure) is a no-op: INSERT OR
        // IGNORE skips the present ids, so it neither errors nor duplicates.
        assert_eq!(db.insert_custom_styles_migration(&batch).unwrap(), 0);
        assert_eq!(db.count_custom_styles().unwrap(), 2);
    }

    #[test]
    fn decode_backend_passes_known_through() {
        assert!(matches!(
            decode_backend("\"OpenAiCompatible\""),
            BackendDecode::Parsed(AsrBackendKind::OpenAiCompatible)
        ));
    }

    #[test]
    fn decode_backend_migrates_legacy_qwen3_quietly() {
        // The Qwen3-ASR backend and its `OrtQwenAsr` alias were removed; legacy
        // session rows must still load by migrating to the current default —
        // and quietly, since it's an expected migration rather than an error.
        for raw in ["\"Qwen3Asr\"", "\"OrtQwenAsr\""] {
            assert!(
                matches!(
                    decode_backend(raw),
                    BackendDecode::LegacyMigrated(AsrBackendKind::WhisperCpp)
                ),
                "legacy backend value {raw} must migrate quietly to WhisperCpp",
            );
        }
    }

    #[test]
    fn decode_backend_falls_back_on_unknown() {
        assert!(matches!(
            decode_backend("\"SomeRemovedBackend\""),
            BackendDecode::Unknown {
                kind: AsrBackendKind::WhisperCpp,
                ..
            }
        ));
    }

    #[test]
    fn row_to_session_bad_uuid_returns_row_decode() {
        let row: SessionRow7 = (
            "not-a-uuid".into(),
            "2024-01-01T00:00:00Z".into(),
            None,
            "minimal".into(),
            "\"WhisperCpp\"".into(),
            "\"Completed\"".into(),
            None,
        );
        let err = row_to_session(row).unwrap_err();
        assert!(
            matches!(err, PersistError::RowDecode(_)),
            "bad uuid should yield RowDecode, got {err:?}",
        );
    }

    #[test]
    fn row_to_session_bad_started_at_returns_row_decode() {
        let row: SessionRow7 = (
            Uuid::new_v4().to_string(),
            "not-a-date".into(),
            None,
            "minimal".into(),
            "\"WhisperCpp\"".into(),
            "\"Completed\"".into(),
            None,
        );
        let err = row_to_session(row).unwrap_err();
        assert!(
            matches!(err, PersistError::RowDecode(_)),
            "bad started_at should yield RowDecode, got {err:?}",
        );
    }

    #[test]
    fn row_to_session_bad_status_returns_row_decode() {
        let row: SessionRow7 = (
            Uuid::new_v4().to_string(),
            "2024-01-01T00:00:00Z".into(),
            None,
            "minimal".into(),
            "\"WhisperCpp\"".into(),
            "\"NotAStatus\"".into(),
            None,
        );
        let err = row_to_session(row).unwrap_err();
        assert!(
            matches!(err, PersistError::RowDecode(_)),
            "bad status should yield RowDecode, got {err:?}",
        );
    }

    #[test]
    fn decode_backend_falls_back_on_malformed_json() {
        assert!(matches!(
            decode_backend("not-valid-json"),
            BackendDecode::Unknown {
                kind: AsrBackendKind::WhisperCpp,
                ..
            }
        ));
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
    fn dictionary_entries_with_ids() {
        let mut db = Database::open_in_memory().unwrap();

        let rust_id = db.insert_dictionary_entry("rust lang", "Rust", 10).unwrap();
        let git_id = db.insert_dictionary_entry("github", "GitHub", 5).unwrap();

        let entries = db.list_dictionary_entries_with_ids().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, rust_id);
        assert_eq!(entries[0].phrase, "rust lang");
        assert_eq!(entries[1].id, git_id);
        assert_eq!(entries[1].replacement, "GitHub");
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
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
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
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Active,
            transcript: None,
        };
        db.insert_session(&s1).unwrap();

        let list = db.list_sessions(10).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, s1.id);

        let fetched = db.get_session(s1.id).unwrap().unwrap();
        assert_eq!(fetched.id, s1.id);
        assert_eq!(fetched.style.as_deref(), Some("casual"));

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
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
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
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Completed,
            transcript: None,
        };
        db.insert_session(&old_session).unwrap();

        // Insert a recent session.
        let new_session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
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
    fn session_delete() {
        let mut db = Database::open_in_memory().unwrap();

        let s1 = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Completed,
            transcript: Some("First session.".into()),
        };
        let s2 = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
            style: Some("formal".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Completed,
            transcript: Some("Second session.".into()),
        };
        db.insert_session(&s1).unwrap();
        db.insert_session(&s2).unwrap();
        assert_eq!(db.list_sessions(10).unwrap().len(), 2);

        let deleted = db.delete_session(&s1.id.to_string()).unwrap();
        assert!(deleted);

        let remaining = db.list_sessions(10).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, s2.id);

        // Verify it's really gone
        assert!(db.get_session(s1.id).unwrap().is_none());
    }

    #[test]
    fn session_delete_many_removes_only_the_listed_rows_and_their_pairs() {
        let mut db = Database::open_in_memory().unwrap();
        let row = |transcript: &str| DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
            style: None,
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Completed,
            transcript: Some(transcript.into()),
        };
        let (a, b, keep) = (row("one"), row("two"), row("three"));
        for s in [&a, &b, &keep] {
            db.insert_session(s).unwrap();
            db.insert_correction_pair(s.id, "raw", "final").unwrap();
        }

        let ids = [a.id.to_string(), b.id.to_string()];
        assert_eq!(db.delete_sessions(&ids).unwrap(), 2);

        let remaining = db.list_sessions(10).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, keep.id);
        // The cascade is per-id, not a purge: the untouched session keeps its pair.
        assert_eq!(db.list_correction_pairs(10).unwrap().len(), 1);
    }

    #[test]
    fn session_delete_many_counts_only_rows_that_existed() {
        let mut db = Database::open_in_memory().unwrap();
        assert_eq!(db.delete_sessions(&[]).unwrap(), 0);
        assert_eq!(
            db.delete_sessions(&[Uuid::new_v4().to_string()]).unwrap(),
            0
        );
    }

    #[test]
    fn session_delete_nonexistent_returns_false() {
        let mut db = Database::open_in_memory().unwrap();

        let deleted = db.delete_session(&Uuid::new_v4().to_string()).unwrap();
        assert!(!deleted);
    }

    #[test]
    fn session_delete_idempotent() {
        let mut db = Database::open_in_memory().unwrap();

        let session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now(),
            ended_at: None,
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Active,
            transcript: None,
        };
        db.insert_session(&session).unwrap();

        let id_str = session.id.to_string();
        assert!(db.delete_session(&id_str).unwrap());
        assert!(!db.delete_session(&id_str).unwrap());
    }

    #[test]
    fn correction_pairs_roundtrip_order_and_cascade() {
        let mut db = Database::open_in_memory().unwrap();
        let sid = Uuid::new_v4();
        db.insert_correction_pair(sid, "send there", "send their")
            .unwrap();
        db.insert_correction_pair(sid, "over their", "over there")
            .unwrap();

        let all = db.list_correction_pairs(10).unwrap();
        assert_eq!(all.len(), 2);
        // Newest first; the id tiebreaker makes this deterministic under equal timestamps.
        assert_eq!(all[0], ("over their".to_string(), "over there".to_string()));
        assert_eq!(db.list_correction_pairs(1).unwrap().len(), 1);

        // delete_session cascades to the session's pairs (no session row here,
        // so it returns false, but the pairs must still be gone).
        assert!(!db.delete_session(&sid.to_string()).unwrap());
        assert!(
            db.list_correction_pairs(10).unwrap().is_empty(),
            "deleting the session must remove its correction pairs"
        );
    }

    #[test]
    fn the_startup_sweep_closes_active_rows_without_rewriting_failures() {
        let mut db = Database::open_in_memory().unwrap();
        let interrupted = Uuid::new_v4();
        let failed_at_init = Uuid::new_v4();
        let row = |id, status| DictationSession {
            id,
            started_at: Utc::now(),
            // Both paths leave `ended_at` null: a hard kill never runs teardown,
            // and an init failure records its reason without an end time.
            ended_at: None,
            style: None,
            backend: AsrBackendKind::WhisperCpp,
            status,
            transcript: None,
        };
        db.insert_session(&row(interrupted, SessionStatus::Active))
            .unwrap();
        db.insert_session(&row(
            failed_at_init,
            SessionStatus::Failed("microphone is in use".into()),
        ))
        .unwrap();

        assert_eq!(db.close_orphaned_sessions().unwrap(), 1);
        assert!(
            db.get_session(interrupted)
                .unwrap()
                .unwrap()
                .ended_at
                .is_some(),
            "the abandoned dictation must stop showing as in progress"
        );
        assert!(
            matches!(
                db.get_session(failed_at_init).unwrap().unwrap().status,
                SessionStatus::Failed(reason) if reason == "microphone is in use"
            ),
            "the sweep must not replace a real failure reason with a generic one"
        );
    }

    #[test]
    fn todays_totals_count_only_todays_sessions() {
        let mut db = Database::open_in_memory().unwrap();
        let recent = Utc::now();
        let old = recent - chrono::Duration::days(3);
        for (started, transcript) in [(old, "one two three"), (recent, "four five")] {
            db.insert_session(&DictationSession {
                id: Uuid::new_v4(),
                started_at: started,
                ended_at: Some(started + chrono::Duration::seconds(10)),
                style: None,
                backend: AsrBackendKind::WhisperCpp,
                status: SessionStatus::Completed,
                transcript: Some(transcript.into()),
            })
            .unwrap();
        }

        let cutoff = (recent - chrono::Duration::hours(1)).to_rfc3339();
        let stats = db.session_stats(&cutoff).unwrap();

        assert_eq!(stats.sessions, 2, "all-time counts both");
        assert_eq!(stats.words, 5);
        assert_eq!(stats.sessions_today, 1, "only the recent one is today");
        assert_eq!(stats.words_today, 2);
        // Durations come from a `julianday` subtraction cast to an integer, so
        // each 10-second span can land a second short — and the all-time total
        // sums two of them, so it can be two short. The point of the assertion
        // is which rows were summed, not sub-second accuracy.
        assert!(
            (9..=10).contains(&stats.seconds_today),
            "expected ~10s today, got {}",
            stats.seconds_today
        );
        assert!(
            (18..=20).contains(&stats.seconds),
            "expected ~20s all-time, got {}",
            stats.seconds
        );
    }

    /// The whole point of aggregating in SQL: the client used to derive this
    /// from a capped slice of recent sessions, so a day busier than the cap
    /// under-reported itself.
    #[test]
    fn todays_totals_are_not_capped_by_the_snapshot_slice() {
        let mut db = Database::open_in_memory().unwrap();
        let now = Utc::now();
        for _ in 0..50 {
            db.insert_session(&DictationSession {
                id: Uuid::new_v4(),
                started_at: now,
                ended_at: Some(now + chrono::Duration::seconds(2)),
                style: None,
                backend: AsrBackendKind::WhisperCpp,
                status: SessionStatus::Completed,
                transcript: Some("one two".into()),
            })
            .unwrap();
        }

        let stats = db.session_stats(EPOCH).unwrap();
        assert_eq!(stats.sessions_today, 50);
        assert_eq!(stats.words_today, 100);
    }

    /// The uniqueness rule has to hold on the edit path too. Correction matches
    /// case-insensitively, so two entries spelling the same phrase differently
    /// leave which replacement applies down to row order — invisible to the
    /// user, and reachable by editing rather than adding if only the insert is
    /// guarded.
    #[test]
    fn editing_an_entry_onto_another_phrase_is_refused() {
        let collide = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
        let mut db = Database::open_in_memory().unwrap();
        db.insert_dictionary_entry_unique("github", "GitHub", 0, collide)
            .unwrap()
            .expect("first entry");
        let second = db
            .insert_dictionary_entry_unique("gitlab", "GitLab", 0, collide)
            .unwrap()
            .expect("second entry");

        assert_eq!(
            db.update_dictionary_entry_unique(second, "GitHub", "Git Hub", 0, collide)
                .unwrap(),
            None,
            "an edit must not be the way around the uniqueness rule"
        );
        // Its own phrase is not a collision with itself.
        assert_eq!(
            db.update_dictionary_entry_unique(second, "GitLab", "GitLab Inc", 0, collide)
                .unwrap(),
            Some(true)
        );
        // A phrase nothing else uses still saves; a missing id still reports so.
        assert_eq!(
            db.update_dictionary_entry_unique(second, "sourcehut", "SourceHut", 0, collide)
                .unwrap(),
            Some(true)
        );
        assert_eq!(
            db.update_dictionary_entry_unique(9999, "codeberg", "Codeberg", 0, collide)
                .unwrap(),
            Some(false)
        );
    }

    #[test]
    fn purge_history_clears_sessions_and_their_correction_pairs() {
        let mut db = Database::open_in_memory().unwrap();
        let sid = Uuid::new_v4();
        db.insert_session(&DictationSession {
            id: sid,
            started_at: Utc::now(),
            ended_at: None,
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Active,
            transcript: Some("a stored transcript".into()),
        })
        .unwrap();
        db.insert_correction_pair(sid, "send there", "send their")
            .unwrap();

        assert_eq!(db.purge_history().unwrap(), 1);
        assert!(db.get_session(sid).unwrap().is_none());
        assert!(
            db.list_correction_pairs(10).unwrap().is_empty(),
            "turning history off must clear the transcript pairs too"
        );
        // Idempotent: purging an already-empty history is not an error.
        assert_eq!(db.purge_history().unwrap(), 0);
    }

    #[test]
    fn session_stats_skips_rows_with_malformed_started_at() {
        use diesel::RunQueryDsl;

        let mut db = Database::open_in_memory().unwrap();

        // Insert a valid completed session via the normal API.
        let good = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now() - chrono::Duration::seconds(10),
            ended_at: Some(Utc::now()),
            style: Some("casual".into()),
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Completed,
            transcript: Some("two words".into()),
        };
        db.insert_session(&good).unwrap();

        // Inject a session row with a malformed started_at, bypassing the API.
        diesel::sql_query(
            "INSERT INTO sessions (id, started_at, ended_at, style, backend, status, transcript, created_at) \
             VALUES ('bad-uuid', 'not-a-date', '2024-01-01T00:00:00Z', 'casual', '\"WhisperCpp\"', '\"Completed\"', 'three words here', datetime('now'))",
        )
        .execute(&mut db.conn)
        .unwrap();

        let stats = db.session_stats(EPOCH).unwrap();
        // The malformed row must be skipped entirely — not counted in sessions, words, or seconds.
        assert_eq!(stats.sessions, 1, "only the well-formed session counts");
        assert_eq!(stats.words, 2, "words from malformed row must not be added");
        assert!(
            stats.seconds >= 1,
            "seconds from well-formed session counted"
        );
    }

    /// Rows written before the `word_count` column existed have NULL there and
    /// contribute nothing to the SUM until the one-time backfill runs.
    #[test]
    fn word_count_backfill_fills_pre_migration_rows() {
        use diesel::RunQueryDsl;

        let mut db = Database::open_in_memory().unwrap();
        diesel::sql_query(
            "INSERT INTO sessions (id, started_at, ended_at, style, backend, status, transcript, created_at) \
             VALUES ('legacy-row', '2024-01-01T00:00:00Z', '2024-01-01T00:00:10Z', '', '\"WhisperCpp\"', '\"Completed\"', ' one  two\tthree\nfour ', datetime('now'))",
        )
        .execute(&mut db.conn)
        .unwrap();

        let before = db.session_stats(EPOCH).unwrap();
        assert_eq!(before.sessions, 1, "the row still counts as a session");
        assert_eq!(before.words, 0, "NULL word_count contributes nothing");

        db.backfill_word_counts().unwrap();
        let after = db.session_stats(EPOCH).unwrap();
        assert_eq!(after.words, 4, "backfill uses split_whitespace semantics");

        // Idempotent: a second pass finds no NULLs and changes nothing.
        db.backfill_word_counts().unwrap();
        assert_eq!(db.session_stats(EPOCH).unwrap().words, 4);
    }

    /// `word_count` is maintained, so it has to track a transcript rewrite —
    /// otherwise lifetime totals drift from the text actually stored.
    #[test]
    fn word_count_tracks_transcript_updates() {
        let mut db = Database::open_in_memory().unwrap();
        let mut session = DictationSession {
            id: Uuid::new_v4(),
            started_at: Utc::now() - chrono::Duration::seconds(5),
            ended_at: Some(Utc::now()),
            style: None,
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Completed,
            transcript: Some("one two three".into()),
        };
        db.insert_session(&session).unwrap();
        assert_eq!(db.session_stats(EPOCH).unwrap().words, 3);

        session.transcript = Some("only two".into());
        db.update_session(&session).unwrap();
        assert_eq!(db.session_stats(EPOCH).unwrap().words, 2);
    }

    #[test]
    fn history_pagination_counts_pages_and_filters() {
        let mut db = Database::open_in_memory().unwrap();

        let base = Utc::now() - chrono::Duration::seconds(600);
        // Five well-formed sessions, staggered so DESC-by-start order is
        // deterministic (index 0 oldest, index 4 newest).
        let transcripts = [
            "alpha hello world",
            "bravo hello there",
            "charlie unrelated text",
            "delta hello again",
            "echo final entry",
        ];
        for (i, text) in transcripts.iter().enumerate() {
            let started = base + chrono::Duration::seconds(i as i64 * 10);
            db.insert_session(&DictationSession {
                id: Uuid::new_v4(),
                started_at: started,
                ended_at: Some(started + chrono::Duration::seconds(2)),
                style: None,
                backend: AsrBackendKind::WhisperCpp,
                status: SessionStatus::Completed,
                transcript: Some((*text).into()),
            })
            .unwrap();
        }
        // Rows with no usable transcript (empty, spaces, or other ASCII
        // whitespace) must be excluded from counts and pages.
        for empty in ["", "   ", "\t\n"] {
            db.insert_session(&DictationSession {
                id: Uuid::new_v4(),
                started_at: Utc::now(),
                ended_at: Some(Utc::now()),
                style: None,
                backend: AsrBackendKind::WhisperCpp,
                status: SessionStatus::Completed,
                transcript: Some(empty.into()),
            })
            .unwrap();
        }

        // Total excludes the three whitespace-only rows, and — critically —
        // count_sessions and session_stats agree on the very same row set, so
        // the History total can never drift from the overview total.
        assert_eq!(db.count_sessions(None).unwrap(), 5);
        assert_eq!(db.session_stats(EPOCH).unwrap().sessions, 5);

        // Page 1 (limit 2) is the two newest, in DESC started_at order.
        let page1 = db.list_sessions_page(2, 0, None).unwrap();
        let texts: Vec<_> = page1
            .iter()
            .map(|s| s.transcript.clone().unwrap())
            .collect();
        assert_eq!(texts, vec!["echo final entry", "delta hello again"]);

        // Page 3 (offset 4) holds the single oldest remaining row.
        let page3 = db.list_sessions_page(2, 4, None).unwrap();
        assert_eq!(page3.len(), 1);
        assert_eq!(page3[0].transcript.as_deref(), Some("alpha hello world"));

        // Search filters by transcript substring, case-insensitively, and its
        // count agrees with the returned rows.
        assert_eq!(db.count_sessions(Some("hello")).unwrap(), 3);
        let hits = db.list_sessions_page(10, 0, Some("HELLO")).unwrap();
        assert_eq!(hits.len(), 3);
        assert!(
            hits.iter()
                .all(|s| s.transcript.as_deref().unwrap().contains("hello"))
        );

        // A LIKE metacharacter in the query matches literally, not as a wildcard.
        assert_eq!(db.count_sessions(Some("%")).unwrap(), 0);
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

    /// The history DB holds full dictated transcripts, so `open` must enforce
    /// owner-only permissions even when the pre-existing file is world-readable
    /// (e.g. created by an older build under a permissive umask). Mirrors the
    /// 0600 hardening `settings.rs` applies to the config file.
    #[cfg(unix)]
    #[test]
    fn open_restricts_database_file_to_owner() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lune.db");
        // Simulate a file created under a permissive umask (0644).
        std::fs::write(&path, b"").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        let _db = Database::open(&path).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(
            mode & 0o077,
            0,
            "database file must not be readable/writable by group or other (mode {mode:o})"
        );

        // The WAL/SHM sidecars carry the same transcript content in WAL mode,
        // so any that exist must be locked down too — hardening only the main
        // file would leave the live transcripts world-readable.
        for suffix in ["-wal", "-shm"] {
            let mut sidecar = path.as_os_str().to_owned();
            sidecar.push(suffix);
            let sidecar = std::path::PathBuf::from(sidecar);
            if let Ok(meta) = std::fs::metadata(&sidecar) {
                let mode = meta.permissions().mode();
                assert_eq!(
                    mode & 0o077,
                    0,
                    "{} must not be group/other accessible (mode {mode:o})",
                    sidecar.display()
                );
            }
        }
    }
}
