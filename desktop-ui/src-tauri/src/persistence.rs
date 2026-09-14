use lune_asr::AsrRegistry;
use lune_core::config::{
    ActivationMode, AsrBackendConfig, DevicePolicy, DictationMode, FormatToggles, HistoryRetention,
    HotkeyConfig, KeyCombo, LlmBackendConfig, Modifier, OpenAiLlmProvider, PillAnchor, PillCorner,
    PillOrientation, PillSize, PillVisibility, PrivacyMode, RedactedString, StreamingMode,
    StylePreset, TerminalMode, UserSettings, builtin_registry,
};
use lune_core::session::SessionStatus;
use lune_persist::{
    AppPaths, Database, SettingsStore, SnippetSaveMode, SnippetSaveOutcome, StoredDictionaryEntry,
};

use crate::dto::{
    ActivationModeDto, AsrBackendTypeDto, CreateCustomStyleInput, CreateDictionaryEntryInput,
    DevicePolicyDto, DictationModeDto, DictionaryItemDto, HistoryItemDto, HistoryPageDto,
    LlmBackendTypeDto, PillAnchorDto, PillCornerDto, PillOrientationDto, PillSizeDto,
    PillVisibilityDto, PrivacyModeDto, SaveSnippetInput, SaveWorkspaceSettingsInput,
    SessionStatsDto, SettingsCoreDto, SnippetItemDto, StylePresetDto, TerminalModeDto,
    UpdateCustomStyleInput, UpdateDictionaryEntryInput, WhisperModelDto, WorkspaceSettingsDto,
    WorkspaceSnapshot,
};

/// Page size ceiling for `list_history_page`, which is what the History tab
/// reads.
const HISTORY_LIMIT: u32 = 200;

/// How many recent sessions ride along in the workspace snapshot.
///
/// Small on purpose. The snapshot is rebuilt after every dictation and every
/// settings save, and its only consumer is the Overview card's "recent" list —
/// the History tab pages the table separately and the totals are aggregated in
/// SQL. Shipping 200 full transcripts meant re-serializing and re-parsing most
/// of the database in the same milliseconds the delivery thread was typing.
///
/// Transcripts are still sent whole: the recent rows show a per-session word
/// count and open a detail modal, both of which read the text they were given.
const SNAPSHOT_HISTORY_LIMIT: u32 = 20;
const DICTIONARY_LIMIT: usize = 10_000;
const UNKNOWN_LABEL: &str = "Unknown";

fn is_wellformed_bcp47(tag: &str) -> bool {
    let mut parts = tag.split('-');
    match parts.next() {
        Some(p) if (2..=8).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_alphabetic()) => {}
        _ => return false,
    }
    parts.all(|p| {
        let len = p.len();
        (1..=8).contains(&len) && p.bytes().all(|b| b.is_ascii_alphanumeric())
    })
}

/// Whether the settings driving a destructive action are the user's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingsTrust {
    /// Parsed from the file the user's own saves wrote.
    AsWritten,
    /// The file could not be parsed and was moved aside; these are defaults.
    Quarantined,
}

/// Apply the retention setting to what is already on disk.
///
/// `Days(n)` prunes past the window; `Forever` leaves everything alone; `Off`
/// is "Don't save history" and removes everything, because turning the setting
/// off has to clear what was collected before, not just stop adding to it.
/// Writes are refused separately (`SessionManagerConfig::persist_history`) —
/// this is the half that deals with rows that already exist.
///
/// `SettingsTrust::Quarantined` makes this a no-op. The defaults that a
/// quarantine hands back carry a 30-day window and `Off` is one unknown enum
/// value away, so pruning on them deletes history the user chose to keep, and
/// unlike the settings file itself nothing keeps a copy.
///
/// Best-effort: a failure here is logged, never fatal.
pub(crate) fn enforce_retention(
    database: &mut Database,
    settings: &UserSettings,
    trust: SettingsTrust,
) {
    if trust == SettingsTrust::Quarantined {
        tracing::warn!(
            "settings were quarantined; skipping history retention so the defaults cannot prune it"
        );
        return;
    }
    match settings.history_retention() {
        HistoryRetention::Days(days) => {
            if let Err(e) = database.prune_sessions(days) {
                tracing::warn!(error = %e, days, "history retention prune failed");
            }
        }
        // Nothing to do — that is the whole point of the setting.
        HistoryRetention::Forever => {}
        HistoryRetention::Off => match database.purge_history() {
            Ok(0) => {}
            Ok(removed) => tracing::info!(removed, "history is off — cleared stored sessions"),
            Err(e) => tracing::warn!(error = %e, "clearing stored history failed"),
        },
    }
}

pub fn load_workspace_snapshot() -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let loaded = load_settings_checked(&paths)?;
    let mut database = open_database(&paths)?;
    enforce_retention(&mut database, &loaded.settings, loaded.trust);
    build_snapshot(loaded.settings, &mut database)
}

pub fn save_workspace_settings(
    input: SaveWorkspaceSettingsInput,
) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let mut settings = load_settings(&paths)?;
    let mut database = open_database(&paths)?;
    apply_settings_dto(&mut settings, input, &mut database)?;
    save_settings(&paths, &settings)?;
    // Retention applies the moment it is changed. Switching to "Don't save
    // history" must clear what is already stored rather than wait for the next
    // window load.
    enforce_retention(&mut database, &settings, SettingsTrust::AsWritten);
    build_snapshot(settings, &mut database)
}

use crate::limits::{
    CONFIDENCE_MAX, CONFIDENCE_MIN, DICT_ENTRY_MAX_BYTES, HISTORY_BULK_DELETE_MAX,
    HISTORY_DAYS_FOREVER, HISTORY_DAYS_MAX, HISTORY_DAYS_MIN, PILL_EDGE_MARGIN_MAX,
    SNIPPET_BODY_MAX_BYTES, SNIPPET_TRIGGER_MAX_BYTES, STYLE_DESC_MAX_BYTES,
    STYLE_INSTRUCTIONS_MAX_BYTES, STYLE_NAME_MAX_BYTES,
};

/// Maximum length for dictionary phrase/replacement strings.
const DICT_ENTRY_MAX_LEN: usize = DICT_ENTRY_MAX_BYTES;

/// Maximum length for a snippet trigger phrase.
const SNIPPET_TRIGGER_MAX_LEN: usize = SNIPPET_TRIGGER_MAX_BYTES;
/// Maximum length for a snippet body template.
const SNIPPET_BODY_MAX_LEN: usize = SNIPPET_BODY_MAX_BYTES;

/// Whether `s` carries a Unicode format character that renders as nothing.
///
/// `char::is_control` does not cover these: bidi overrides (U+202E and the
/// isolate family) and zero-width characters are invisible on screen but
/// travel with the text into the user's document, so what a saved snippet or
/// dictionary entry *looks* like can differ from what it actually types.
fn has_invisible_format_char(s: &str) -> bool {
    s.chars().any(|c| {
        matches!(c,
            '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{2069}'
            | '\u{FEFF}')
    })
}

/// Create (`id: None`) or update (`id: Some`) a voice snippet.
pub fn save_snippet(input: SaveSnippetInput) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let trigger = input.trigger_phrase.trim();
    let body = input.body.as_str();
    if trigger.is_empty() || body.trim().is_empty() {
        return Err("trigger phrase and body are required".into());
    }
    if trigger.len() > SNIPPET_TRIGGER_MAX_LEN {
        return Err(format!(
            "snippet triggers must be at most {SNIPPET_TRIGGER_MAX_LEN} characters"
        ));
    }
    if body.len() > SNIPPET_BODY_MAX_LEN {
        return Err(format!(
            "snippet bodies must be at most {SNIPPET_BODY_MAX_LEN} characters"
        ));
    }
    // A trigger with no alphanumerics would normalize to nothing and could
    // never be spoken.
    if !trigger.chars().any(|c| c.is_alphanumeric()) {
        return Err("snippet triggers need at least one letter or digit".into());
    }
    // A trigger starting with a parametric command prefix is unreachable: the
    // router claims the prefix and looks up whatever follows it as the snippet
    // *name*, so the trigger itself never matches. Reject it here rather than
    // saving a snippet that can never fire.
    if let Some(prefix) = lune_dict::commands::reserved_trigger_prefix(trigger) {
        return Err(format!(
            "snippet triggers cannot start with \"{prefix}\" — it is a reserved \
             voice command; say \"{prefix} <name>\" to expand a snippet by name"
        ));
    }
    // Reject control characters (newline/tab stay legal in the body) to
    // prevent injection into the delivery path.
    if trigger.chars().any(|c| c.is_control())
        || body
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err("snippets must not contain control characters".into());
    }
    if has_invisible_format_char(trigger) || has_invisible_format_char(body) {
        return Err("snippets must not contain invisible formatting characters".into());
    }

    let mut database = open_database(&paths)?;
    // Spoken matching normalizes case, punctuation and whitespace, so the
    // collision test has to normalize too: comparing the stored strings would
    // let "Meeting, header." and "meeting header" both exist, and matching
    // would then silently shadow one of them. The check runs inside the write's
    // transaction, so two saves racing cannot both pass it.
    let new_id = uuid::Uuid::new_v4().to_string();
    let mode = match input.id.as_deref() {
        Some(id) => SnippetSaveMode::Update { id },
        None => SnippetSaveMode::Insert { id: &new_id },
    };
    let outcome = database
        .save_snippet_unique(mode, trigger, body, |stored, candidate| {
            lune_dict::snippets::triggers_collide(stored, candidate)
        })
        .map_err(|e| format!("save snippet: {e}"))?;
    match outcome {
        SnippetSaveOutcome::Saved => {}
        SnippetSaveOutcome::TriggerTaken => {
            return Err("another snippet already uses this trigger phrase".into());
        }
        SnippetSaveOutcome::NotFound => return Err("snippet not found".into()),
    }

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn delete_snippet(snippet_id: String) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let mut database = open_database(&paths)?;
    let deleted = database
        .delete_snippet(&snippet_id)
        .map_err(|e| format!("delete snippet: {e}"))?;
    // Match the `delete_session` contract below: a delete that removed nothing
    // is an error, not a silent success (a stale id, or two windows racing).
    if !deleted {
        return Err("snippet not found".into());
    }
    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn create_dictionary_entry(
    input: CreateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let phrase = input.phrase.trim();
    let replacement = input.replacement.trim();
    if phrase.is_empty() || replacement.is_empty() {
        return Err("phrase and replacement are required".into());
    }
    if phrase.len() > DICT_ENTRY_MAX_LEN || replacement.len() > DICT_ENTRY_MAX_LEN {
        return Err(format!(
            "dictionary entries must be at most {DICT_ENTRY_MAX_LEN} characters"
        ));
    }
    // Reject control characters (except newline/tab) to prevent injection.
    if phrase
        .chars()
        .chain(replacement.chars())
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err("dictionary entries must not contain control characters".into());
    }
    if has_invisible_format_char(phrase) || has_invisible_format_char(replacement) {
        return Err("dictionary entries must not contain invisible formatting characters".into());
    }

    let mut database = open_database(&paths)?;
    let inserted = database
        .insert_dictionary_entry_unique(phrase, replacement, input.priority, |stored, candidate| {
            lune_dict::phrases_collide(stored, candidate)
        })
        .map_err(|e| format!("insert dictionary entry: {e}"))?;
    if inserted.is_none() {
        return Err("a dictionary entry already uses this phrase".into());
    }

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn delete_session(session_id: String) -> Result<WorkspaceSnapshot, String> {
    if session_id.is_empty() {
        return Err("session_id must not be empty".into());
    }
    let paths = resolve_paths()?;
    let mut database = open_database(&paths)?;
    let deleted = database
        .delete_session(&session_id)
        .map_err(|e| format!("delete session: {e}"))?;
    if !deleted {
        return Err(format!("session not found: {session_id}"));
    }

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

/// Reject a retention window the backend cannot honour.
///
/// Three choices live in one nullable number: `None` is off, `0` is
/// `HISTORY_DAYS_FOREVER`, and anything else is a real window. The sentinel
/// sits outside the range check on purpose — it is not a zero-day window.
fn validate_history_days(days: Option<u32>) -> Result<(), String> {
    match days {
        None => Ok(()),
        Some(d)
            if d == HISTORY_DAYS_FOREVER || (HISTORY_DAYS_MIN..=HISTORY_DAYS_MAX).contains(&d) =>
        {
            Ok(())
        }
        Some(d) => Err(format!(
            "sessionHistoryDays must be {HISTORY_DAYS_FOREVER} (keep forever) or between \
             {HISTORY_DAYS_MIN} and {HISTORY_DAYS_MAX}, got {d}"
        )),
    }
}

/// Delete several sessions in one call, for the History tab's bulk selection.
///
/// Unlike [`delete_session`], ids that no longer exist are **not** an error:
/// a multi-row selection can easily include something a second window already
/// removed, and failing the whole batch over that would leave the user with no
/// way to clear the rest. An empty list is still a caller bug.
pub fn delete_sessions(session_ids: Vec<String>) -> Result<WorkspaceSnapshot, String> {
    if session_ids.is_empty() {
        return Err("session_ids must not be empty".into());
    }
    if session_ids.len() > HISTORY_BULK_DELETE_MAX {
        return Err(format!(
            "at most {HISTORY_BULK_DELETE_MAX} sessions can be deleted at once, got {}",
            session_ids.len()
        ));
    }
    if session_ids.iter().any(|id| id.is_empty()) {
        return Err("session_ids must not contain an empty id".into());
    }
    let paths = resolve_paths()?;
    let mut database = open_database(&paths)?;
    let removed = database
        .delete_sessions(&session_ids)
        .map_err(|e| format!("delete sessions: {e}"))?;
    tracing::info!(
        removed,
        requested = session_ids.len(),
        "bulk session delete"
    );

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

/// Fetch one page of session history from the database. Unlike the workspace
/// snapshot (which carries only `SNAPSHOT_HISTORY_LIMIT` recent rows), this
/// pages through the entire table so the History tab can reach every session.
/// `page` is 1-based; `search`, when non-empty, filters by transcript substring.
pub fn list_history_page(
    page: u32,
    page_size: u32,
    search: Option<String>,
) -> Result<HistoryPageDto, String> {
    let page = page.max(1);
    let page_size = page_size.clamp(1, HISTORY_LIMIT);
    let offset = (page - 1).saturating_mul(page_size);
    let query = search.as_deref().map(str::trim).filter(|s| !s.is_empty());

    let paths = resolve_paths()?;
    let mut database = open_database(&paths)?;

    let total = database
        .count_sessions(query)
        .map_err(|e| format!("count sessions: {e}"))?;
    let sessions = database
        .list_sessions_page(page_size, offset, query)
        .map_err(|e| format!("list sessions page: {e}"))?
        .into_iter()
        .map(|session| HistoryItemDto {
            id: session.id.to_string(),
            started_at: session.started_at.to_rfc3339(),
            ended_at: session.ended_at.map(|ended| ended.to_rfc3339()),
            transcript: session.transcript.unwrap_or_default(),
            backend: format!("{:?}", session.backend),
            style: session.style.unwrap_or_default(),
            status: session_status_label(&session.status),
        })
        .collect();

    Ok(HistoryPageDto {
        sessions,
        total: total.clamp(0, i64::from(u32::MAX)) as u32,
    })
}

pub fn delete_dictionary_entry(entry_id: i32) -> Result<WorkspaceSnapshot, String> {
    if entry_id <= 0 {
        return Err("entry_id must be a positive integer".into());
    }
    let paths = resolve_paths()?;
    let mut database = open_database(&paths)?;
    let deleted = database
        .delete_dictionary_entry(entry_id)
        .map_err(|e| format!("delete dictionary entry: {e}"))?;
    if !deleted {
        return Err(format!("dictionary entry not found: {entry_id}"));
    }

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn update_dictionary_entry(
    input: UpdateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    if input.id <= 0 {
        return Err("entry id must be a positive integer".into());
    }
    let paths = resolve_paths()?;
    let phrase = input.phrase.trim();
    let replacement = input.replacement.trim();
    if phrase.is_empty() || replacement.is_empty() {
        return Err("phrase and replacement are required".into());
    }
    if phrase.len() > DICT_ENTRY_MAX_LEN || replacement.len() > DICT_ENTRY_MAX_LEN {
        return Err(format!(
            "dictionary entries must be at most {DICT_ENTRY_MAX_LEN} characters"
        ));
    }
    if phrase
        .chars()
        .chain(replacement.chars())
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err("dictionary entries must not contain control characters".into());
    }
    if has_invisible_format_char(phrase) || has_invisible_format_char(replacement) {
        return Err("dictionary entries must not contain invisible formatting characters".into());
    }

    let mut database = open_database(&paths)?;
    // Same uniqueness rule as the create path — an edit must not be the way
    // around it. `None` means another entry already answers to this phrase.
    let updated = database
        .update_dictionary_entry_unique(
            input.id,
            phrase,
            replacement,
            input.priority,
            lune_dict::phrases_collide,
        )
        .map_err(|e| format!("update dictionary entry: {e}"))?;
    match updated {
        Some(true) => {}
        Some(false) => return Err(format!("dictionary entry not found: {}", input.id)),
        None => return Err("a dictionary entry already uses this phrase".into()),
    }

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

// ── Custom style presets ─────────────────────────────────────────────────

const CUSTOM_STYLE_MAX: usize = 50;
const STYLE_NAME_MAX_LEN: usize = STYLE_NAME_MAX_BYTES;
const STYLE_DESC_MAX_LEN: usize = STYLE_DESC_MAX_BYTES;
const STYLE_INSTRUCTIONS_MAX_LEN: usize = STYLE_INSTRUCTIONS_MAX_BYTES;

fn validate_style_fields(name: &str, description: &str, instructions: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > STYLE_NAME_MAX_LEN {
        return Err(format!("name must be 1–{STYLE_NAME_MAX_LEN} characters"));
    }
    if description.len() > STYLE_DESC_MAX_LEN {
        return Err(format!(
            "description must be at most {STYLE_DESC_MAX_LEN} characters"
        ));
    }
    if instructions.is_empty() || instructions.len() > STYLE_INSTRUCTIONS_MAX_LEN {
        return Err(format!(
            "instructions must be 1–{STYLE_INSTRUCTIONS_MAX_LEN} characters"
        ));
    }
    if name
        .chars()
        .chain(description.chars())
        .chain(instructions.chars())
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err("style fields must not contain control characters".into());
    }
    Ok(())
}

pub fn create_custom_style(input: CreateCustomStyleInput) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let settings = load_settings(&paths)?;
    let mut database = open_database(&paths)?;
    let name = input.name.trim().to_owned();
    let description = input.description.trim().to_owned();
    let instructions = input.instructions.trim().to_owned();
    validate_style_fields(&name, &description, &instructions)?;
    let count = database
        .count_custom_styles()
        .map_err(|e| format!("count custom styles: {e}"))?;
    if count as usize >= CUSTOM_STYLE_MAX {
        return Err(format!(
            "maximum of {CUSTOM_STYLE_MAX} custom styles reached"
        ));
    }
    let preset = StylePreset {
        id: format!("custom-{}", uuid::Uuid::new_v4().as_hyphenated()),
        name,
        description,
        style_instructions: instructions,
        few_shot_pairs: Vec::new(),
        formatting: input.formatting.into(),
    };
    // A new style has no examples yet — mark it stale (`dirty = true`) so the UI
    // prompts a generation. The create flow clears this by generating
    // immediately when the styling engine is ready; otherwise the flag persists
    // so the refresh button stays highlighted until the user generates them.
    database
        .insert_custom_style(&preset, true)
        .map_err(|e| format!("insert custom style: {e}"))?;
    build_snapshot(settings, &mut database)
}

pub fn update_custom_style(input: UpdateCustomStyleInput) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let settings = load_settings(&paths)?;
    let mut database = open_database(&paths)?;
    let name = input.name.trim().to_owned();
    let description = input.description.trim().to_owned();
    let instructions = input.instructions.trim().to_owned();
    validate_style_fields(&name, &description, &instructions)?;
    let existing = database
        .get_custom_style(&input.id)
        .map_err(|e| format!("load custom style: {e}"))?
        .ok_or_else(|| format!("custom style not found: {}", input.id))?;
    // The generated examples are seeded from the name + instructions, so mark
    // them stale when either changes; a formatting/description-only edit keeps
    // the existing state (already-stale stays stale).
    let few_shot_stale = existing.preset.name != name
        || existing.preset.style_instructions != instructions
        || existing.few_shot_dirty;
    let formatting: FormatToggles = input.formatting.into();
    database
        .update_custom_style_fields(
            &input.id,
            &name,
            &description,
            &instructions,
            &formatting,
            few_shot_stale,
        )
        .map_err(|e| format!("update custom style: {e}"))?;
    build_snapshot(settings, &mut database)
}

pub fn delete_custom_style(style_id: String) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let mut settings = load_settings(&paths)?;
    let mut database = open_database(&paths)?;
    let removed = database
        .delete_custom_style(&style_id)
        .map_err(|e| format!("delete custom style: {e}"))?;
    if !removed {
        return Err(format!("custom style not found: {style_id}"));
    }
    // If the deleted style was the active selection, clear it — that is a
    // settings change, so persist the TOML too.
    if settings.default_style.as_deref() == Some(style_id.as_str()) {
        settings.default_style = None;
        save_settings(&paths, &settings)?;
    }
    build_snapshot(settings, &mut database)
}

pub fn save_style_few_shot_pairs(
    style_id: &str,
    pairs: Vec<(String, String)>,
) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let settings = load_settings(&paths)?;
    let mut database = open_database(&paths)?;
    // Freshly (re)generated — clear the stale flag.
    let updated = database
        .set_custom_style_few_shot_pairs(style_id, &pairs, false)
        .map_err(|e| format!("save few-shot pairs: {e}"))?;
    if !updated {
        return Err(format!("custom style not found: {style_id}"));
    }
    build_snapshot(settings, &mut database)
}

fn build_snapshot(
    settings: UserSettings,
    database: &mut Database,
) -> Result<WorkspaceSnapshot, String> {
    let history = database
        .list_sessions(SNAPSHOT_HISTORY_LIMIT)
        .map_err(|e| format!("list sessions: {e}"))?
        .into_iter()
        .map(|session| HistoryItemDto {
            id: session.id.to_string(),
            started_at: session.started_at.to_rfc3339(),
            ended_at: session.ended_at.map(|ended| ended.to_rfc3339()),
            transcript: session.transcript.unwrap_or_default(),
            backend: format!("{:?}", session.backend),
            // Empty string = no styling (the session's `style` was `None`).
            style: session.style.unwrap_or_default(),
            status: session_status_label(&session.status),
        })
        .collect();

    let dictionary: Vec<_> = database
        .list_dictionary_entries_with_ids()
        .map_err(|e| format!("list dictionary entries: {e}"))?
        .into_iter()
        .take(DICTIONARY_LIMIT)
        .map(map_dictionary_entry)
        .collect();

    let snippets: Vec<_> = database
        .list_snippets()
        .map_err(|e| format!("list snippets: {e}"))?
        .into_iter()
        .map(|s| SnippetItemDto {
            id: s.id,
            trigger_phrase: s.trigger_phrase,
            body: s.body,
        })
        .collect();

    let mut style_presets: Vec<StylePresetDto> = builtin_registry()
        .all()
        .iter()
        .map(|p| StylePresetDto {
            id: p.id.clone(),
            name: p.name.clone(),
            description: p.description.clone(),
            is_builtin: true,
            // Surface the builtin's real prompt so the read-only view — and
            // duplicating it into a new custom style — shows the actual text.
            instructions: p.style_instructions.clone(),
            formatting: p.formatting.into(),
            few_shot_dirty: false,
        })
        .collect();
    let custom_styles = database
        .list_custom_styles()
        .map_err(|e| format!("list custom styles: {e}"))?;
    for custom in &custom_styles {
        style_presets.push(StylePresetDto {
            id: custom.preset.id.clone(),
            name: custom.preset.name.clone(),
            description: custom.preset.description.clone(),
            is_builtin: false,
            instructions: custom.preset.style_instructions.clone(),
            formatting: custom.preset.formatting.into(),
            few_shot_dirty: custom.few_shot_dirty,
        });
    }

    // Local midnight, as UTC — "today" is the user's day, not the DB's.
    let today_start = chrono::Local::now()
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .and_then(|midnight| {
            midnight
                .and_local_timezone(chrono::Local)
                .single()
                .map(|dt| dt.with_timezone(&chrono::Utc).to_rfc3339())
        })
        // A DST transition can leave local midnight ambiguous or non-existent.
        // Falling back to UTC midnight keeps the stats close rather than wrong.
        .unwrap_or_else(|| {
            chrono::Utc::now()
                .date_naive()
                .and_time(Default::default())
                .and_utc()
                .to_rfc3339()
        });

    let stats = database
        .session_stats(&today_start)
        .map(|s| SessionStatsDto {
            sessions: s.sessions,
            words: s.words,
            seconds: s.seconds,
            sessions_today: s.sessions_today,
            words_today: s.words_today,
            seconds_today: s.seconds_today,
        })
        .unwrap_or_default();

    Ok(WorkspaceSnapshot {
        ipc_flavor: "tauri",
        service_status: "ready".into(),
        settings: settings_to_dto(&settings),
        history,
        dictionary,
        snippets,
        style_presets,
        stats,
    })
}

fn resolve_paths() -> Result<AppPaths, String> {
    let paths = AppPaths::resolve().ok_or("failed to resolve app paths")?;
    paths
        .ensure_dirs()
        .map_err(|e| format!("ensure app dirs: {e}"))?;
    Ok(paths)
}

/// Settings plus whether they came from a file that parsed.
pub(crate) struct CheckedSettings {
    pub settings: UserSettings,
    pub trust: SettingsTrust,
}

fn load_settings(paths: &AppPaths) -> Result<UserSettings, String> {
    Ok(load_settings_checked(paths)?.settings)
}

pub(crate) fn load_settings_checked(paths: &AppPaths) -> Result<CheckedSettings, String> {
    let store = SettingsStore::new(paths.settings_file());
    // Same recovery as the startup path: a malformed file must not make every
    // settings-touching command fail with no way out.
    let loaded = store
        .load_or_quarantine()
        .map_err(|e| format!("load settings: {e}"))?;
    let mut settings = loaded.settings;
    settings.migrate_pipeline_fields();
    Ok(CheckedSettings {
        settings,
        trust: if loaded.quarantined {
            SettingsTrust::Quarantined
        } else {
            SettingsTrust::AsWritten
        },
    })
}

fn save_settings(paths: &AppPaths, settings: &UserSettings) -> Result<(), String> {
    let store = SettingsStore::new(paths.settings_file());
    store
        .save(settings)
        .map_err(|e| format!("save settings: {e}"))
}

fn open_database(paths: &AppPaths) -> Result<Database, String> {
    Database::open(&paths.database_file()).map_err(|e| format!("open database: {e}"))
}

fn settings_to_dto(settings: &UserSettings) -> WorkspaceSettingsDto {
    let backend_defaults = AsrRegistry::default_pipeline_config(settings.asr_backend.kind());
    let (openai_asr_base_url, openai_asr_model, openai_asr_api_key_set) =
        match &settings.asr_backend {
            AsrBackendConfig::OpenAiCompatible {
                base_url,
                model,
                api_key,
                ..
            } => (
                Some(base_url.to_string()),
                Some(model.clone()),
                !api_key.expose().is_empty(),
            ),
            _ => (None, None, false),
        };
    // Display the saved remote provider regardless of the active backend, so the
    // settings UI keeps showing it after Off / engine switches. `openai_llm_active`
    // reports whether that provider is the currently-active backend.
    let openai_llm_base_url = Some(settings.llm_openai.base_url.clone());
    let openai_llm_model = Some(settings.llm_openai.model.clone());
    let openai_llm_api_key_set = !settings.llm_openai.api_key.expose().is_empty();
    let openai_llm_active = matches!(
        settings.llm_backend,
        LlmBackendConfig::OpenAiCompatible { .. }
    );

    WorkspaceSettingsDto {
        core: SettingsCoreDto {
            activation_mode: activation_mode_to_dto(settings.activation_mode),
            dictation_mode: dictation_mode_to_dto(settings.dictation_mode),
            copy_to_clipboard: settings.copy_to_clipboard,
            sound_cues_enabled: settings.sound_cues_enabled,
            sound_cue_volume: settings.sound_cue_volume,
            use_gpu: !matches!(settings.device, DevicePolicy::Cpu),
            device_policy: device_policy_to_dto(settings.device),
            asr_backend_type: settings.asr_backend.kind().into(),
            confidence_threshold: settings
                .pipeline
                .word_confidence_threshold
                .or(backend_defaults.word_confidence_threshold),
            style_preset: settings.default_style.clone(),
            hotkey_toggle: format_key_combo(&settings.hotkey.toggle_dictation),
            hotkey_cancel: format_key_combo(&settings.hotkey.cancel_dictation),
            pill_visibility: pill_visibility_to_dto(settings.pill_visibility),
            pill_anchor: pill_anchor_to_dto(settings.pill_anchor),
            pill_locked: settings.pill_locked,
            pill_edge_margin: settings.pill_edge_margin,
            pill_monitor: settings.pill_monitor.clone(),
            pill_all_workspaces: settings.pill_all_workspaces,
            pill_size: pill_size_to_dto(settings.pill_size),
            pill_orientation: pill_orientation_to_dto(settings.pill_orientation),
            pill_corner: pill_corner_to_dto(settings.pill_corner),
            language: settings.language.clone(),
            dictation_language: settings.dictation_language.clone(),
            model_idle_unload_minutes: settings.model_idle_unload_minutes,
            input_device: settings.input_device.clone(),
            vad_pcen: settings.vad_pcen,
            start_minimized: settings.start_minimized,
            check_updates_on_launch: settings.check_updates_on_launch,
            session_history_days: settings.session_history_days,
            terminal_mode: terminal_mode_to_dto(settings.terminal_mode),
            privacy_mode: privacy_mode_to_dto(settings.privacy_mode),
            llm_enabled: !matches!(settings.llm_backend, LlmBackendConfig::Disabled),
            openai_asr_base_url,
            openai_asr_model,
            openai_llm_base_url,
            openai_llm_model,
        },
        openai_asr_api_key_set,
        openai_llm_api_key_set,
        openai_llm_active,
        whisper_model: match &settings.asr_backend {
            AsrBackendConfig::WhisperCpp { model } => (*model).into(),
            _ => WhisperModelDto::default(),
        },
    }
}

fn apply_settings_dto(
    settings: &mut UserSettings,
    input: SaveWorkspaceSettingsInput,
    database: &mut Database,
) -> Result<(), String> {
    let SaveWorkspaceSettingsInput {
        core: input,
        llm_backend_type,
        openai_asr_api_key,
        openai_llm_api_key,
        whisper_model,
    } = input;
    settings.activation_mode = activation_mode_from_dto(input.activation_mode);
    settings.pill_visibility = pill_visibility_from_dto(input.pill_visibility);
    // A placement change (anchor / edge margin / pinned monitor) clears any
    // dragged position so the new placement is applied visibly, not shadowed
    // by the old drag. Size/orientation/corner keep the dragged spot.
    let edge_margin = input.pill_edge_margin.min(PILL_EDGE_MARGIN_MAX);
    let placement_changed = pill_anchor_from_dto(input.pill_anchor) != settings.pill_anchor
        || edge_margin != settings.pill_edge_margin
        || input.pill_monitor != settings.pill_monitor;
    settings.pill_anchor = pill_anchor_from_dto(input.pill_anchor);
    settings.pill_locked = input.pill_locked;
    settings.pill_edge_margin = edge_margin;
    settings.pill_monitor = input.pill_monitor.clone();
    settings.pill_all_workspaces = input.pill_all_workspaces;
    settings.pill_size = pill_size_from_dto(input.pill_size);
    settings.pill_orientation = pill_orientation_from_dto(input.pill_orientation);
    settings.pill_corner = pill_corner_from_dto(input.pill_corner);
    if placement_changed {
        settings.pill_position = None;
    }
    settings.dictation_mode = dictation_mode_from_dto(input.dictation_mode);
    settings.copy_to_clipboard = input.copy_to_clipboard;
    settings.sound_cues_enabled = input.sound_cues_enabled;
    settings.sound_cue_volume = input.sound_cue_volume.clamp(0.0, 1.0);
    settings.device = resolve_device_policy(input.use_gpu, input.device_policy);
    settings.asr_backend = asr_backend_from_type(
        input.asr_backend_type,
        &settings.asr_backend,
        &input,
        openai_asr_api_key.as_deref(),
        whisper_model,
    )?;

    // Validate numeric thresholds — reject NaN/Inf/out-of-range.
    if let Some(ct) = input.confidence_threshold
        && (!ct.is_finite() || !(CONFIDENCE_MIN..=CONFIDENCE_MAX).contains(&ct))
    {
        return Err(format!(
            "confidenceThreshold must be {CONFIDENCE_MIN}–{CONFIDENCE_MAX}, got {ct}"
        ));
    }
    settings.pipeline.word_confidence_threshold = input.confidence_threshold;

    // Validate style_preset — `None`/empty means no styling (raw transcript);
    // any present value must be a known builtin or custom style.
    settings.default_style = match input.style_preset.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(style) => {
            if style.len() > 128 {
                return Err("stylePreset must be 1–128 characters".into());
            }
            // Skip re-validation when the selection is unchanged from what is
            // already persisted: it was validated when first set, and querying
            // the DB on every unrelated settings save (e.g. a hotkey toggle)
            // would couple all saves to the database. Only a *changed* value
            // needs a fresh lookup. `settings.default_style` still holds the
            // pre-mutation value here (the assignment happens after this match).
            let unchanged = settings.default_style.as_deref() == Some(style);
            let is_known = unchanged
                || builtin_registry().get(style).is_some()
                || database
                    .get_custom_style(style)
                    .map_err(|e| format!("check custom style: {e}"))?
                    .is_some();
            if !is_known {
                return Err(format!("unknown stylePreset: {style}"));
            }
            Some(style.to_owned())
        }
    };

    settings.hotkey = HotkeyConfig {
        toggle_dictation: parse_key_combo(&input.hotkey_toggle)?,
        cancel_dictation: parse_key_combo(&input.hotkey_cancel)?,
    };
    let language = input.language.trim();
    if !is_wellformed_bcp47(language) {
        return Err(
            "language must be a well-formed BCP-47 tag (e.g. \"en\", \"en-US\", \"pt-BR\")".into(),
        );
    }
    settings.language = language.to_owned();

    let dictation_language = input.dictation_language.trim();
    if dictation_language != "auto" && !is_wellformed_bcp47(dictation_language) {
        return Err(
            "dictationLanguage must be \"auto\" or a well-formed BCP-47 tag (e.g. \"en\", \"de\")"
                .into(),
        );
    }
    settings.dictation_language = dictation_language.to_owned();
    // Zero is normalized to "off" rather than "unload the moment a dictation
    // ends": a zero-minute timeout would free the model after every utterance
    // and pay a reload on the next one, which no user means to ask for.
    settings.model_idle_unload_minutes = input.model_idle_unload_minutes.filter(|&m| m > 0);
    settings.input_device = input.input_device.clone();
    settings.vad_pcen = input.vad_pcen;
    settings.start_minimized = input.start_minimized;
    settings.check_updates_on_launch = input.check_updates_on_launch;
    settings.terminal_mode = terminal_mode_from_dto(input.terminal_mode);
    settings.privacy_mode = privacy_mode_from_dto(input.privacy_mode);
    validate_history_days(input.session_history_days)?;
    settings.session_history_days = input.session_history_days;
    // Persist the remote provider config independently of the active backend so
    // it survives Off / switching engines. A blank incoming field keeps the saved
    // value (the api key uses the same "blank = keep the saved key" rule as ASR).
    if let Some(u) = input
        .openai_llm_base_url
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        settings.llm_openai.base_url = u.to_owned();
    }
    if let Some(m) = input.openai_llm_model.as_deref().filter(|s| !s.is_empty()) {
        settings.llm_openai.model = m.to_owned();
    }
    if let Some(k) = openai_llm_api_key.as_deref().filter(|s| !s.is_empty()) {
        settings.llm_openai.api_key = RedactedString::new(k);
    }
    // Off ⇒ Disabled; otherwise build the remote engine from the saved provider.
    settings.llm_backend = if input.llm_enabled {
        llm_backend_from_type(llm_backend_type, &settings.llm_openai)?
    } else {
        LlmBackendConfig::Disabled
    };
    Ok(())
}

// ── DTO ↔ Core conversions ──────────────────────────────────────────────

fn activation_mode_to_dto(mode: ActivationMode) -> ActivationModeDto {
    match mode {
        ActivationMode::Toggle => ActivationModeDto::Toggle,
        ActivationMode::Hold => ActivationModeDto::Hold,
        _ => ActivationModeDto::Toggle,
    }
}

fn activation_mode_from_dto(dto: ActivationModeDto) -> ActivationMode {
    match dto {
        ActivationModeDto::Toggle => ActivationMode::Toggle,
        ActivationModeDto::Hold => ActivationMode::Hold,
    }
}

fn dictation_mode_to_dto(mode: DictationMode) -> DictationModeDto {
    match mode {
        DictationMode::TypeInPlace => DictationModeDto::TypeInPlace,
        DictationMode::Clipboard => DictationModeDto::Clipboard,
        _ => DictationModeDto::Clipboard,
    }
}

fn dictation_mode_from_dto(dto: DictationModeDto) -> DictationMode {
    match dto {
        DictationModeDto::TypeInPlace => DictationMode::TypeInPlace,
        DictationModeDto::Clipboard => DictationMode::Clipboard,
    }
}

fn device_policy_to_dto(policy: DevicePolicy) -> DevicePolicyDto {
    match policy {
        DevicePolicy::Cpu => DevicePolicyDto::Cpu,
        DevicePolicy::Cuda => DevicePolicyDto::Cuda,
        DevicePolicy::Auto => DevicePolicyDto::Auto,
    }
}

fn resolve_device_policy(use_gpu: bool, policy: DevicePolicyDto) -> DevicePolicy {
    if !use_gpu {
        return DevicePolicy::Cpu;
    }
    match policy {
        DevicePolicyDto::Cuda => DevicePolicy::Cuda,
        DevicePolicyDto::Auto | DevicePolicyDto::Cpu => DevicePolicy::Auto,
    }
}

fn pill_visibility_to_dto(vis: PillVisibility) -> PillVisibilityDto {
    match vis {
        PillVisibility::Always => PillVisibilityDto::Always,
        PillVisibility::WhenDictating => PillVisibilityDto::WhenDictating,
        PillVisibility::Never => PillVisibilityDto::Never,
        _ => PillVisibilityDto::Always,
    }
}

pub(crate) fn pill_visibility_from_dto(dto: PillVisibilityDto) -> PillVisibility {
    match dto {
        PillVisibilityDto::Always => PillVisibility::Always,
        PillVisibilityDto::WhenDictating => PillVisibility::WhenDictating,
        PillVisibilityDto::Never => PillVisibility::Never,
    }
}

fn pill_anchor_to_dto(a: PillAnchor) -> PillAnchorDto {
    match a {
        PillAnchor::TopLeft => PillAnchorDto::TopLeft,
        PillAnchor::TopCenter => PillAnchorDto::TopCenter,
        PillAnchor::TopRight => PillAnchorDto::TopRight,
        PillAnchor::BottomLeft => PillAnchorDto::BottomLeft,
        PillAnchor::BottomCenter => PillAnchorDto::BottomCenter,
        PillAnchor::BottomRight => PillAnchorDto::BottomRight,
        _ => PillAnchorDto::BottomCenter,
    }
}

fn pill_anchor_from_dto(dto: PillAnchorDto) -> PillAnchor {
    match dto {
        PillAnchorDto::TopLeft => PillAnchor::TopLeft,
        PillAnchorDto::TopCenter => PillAnchor::TopCenter,
        PillAnchorDto::TopRight => PillAnchor::TopRight,
        PillAnchorDto::BottomLeft => PillAnchor::BottomLeft,
        PillAnchorDto::BottomCenter => PillAnchor::BottomCenter,
        PillAnchorDto::BottomRight => PillAnchor::BottomRight,
    }
}

fn pill_size_to_dto(s: PillSize) -> PillSizeDto {
    match s {
        PillSize::Compact => PillSizeDto::Compact,
        PillSize::Standard => PillSizeDto::Standard,
        PillSize::Large => PillSizeDto::Large,
        _ => PillSizeDto::Standard,
    }
}

fn pill_size_from_dto(dto: PillSizeDto) -> PillSize {
    match dto {
        PillSizeDto::Compact => PillSize::Compact,
        PillSizeDto::Standard => PillSize::Standard,
        PillSizeDto::Large => PillSize::Large,
    }
}

fn pill_orientation_to_dto(o: PillOrientation) -> PillOrientationDto {
    match o {
        PillOrientation::Horizontal => PillOrientationDto::Horizontal,
        PillOrientation::Vertical => PillOrientationDto::Vertical,
        _ => PillOrientationDto::Horizontal,
    }
}

fn pill_orientation_from_dto(dto: PillOrientationDto) -> PillOrientation {
    match dto {
        PillOrientationDto::Horizontal => PillOrientation::Horizontal,
        PillOrientationDto::Vertical => PillOrientation::Vertical,
    }
}

fn pill_corner_to_dto(c: PillCorner) -> PillCornerDto {
    match c {
        PillCorner::Pill => PillCornerDto::Pill,
        PillCorner::Rounded => PillCornerDto::Rounded,
        PillCorner::Square => PillCornerDto::Square,
        _ => PillCornerDto::Rounded,
    }
}

fn pill_corner_from_dto(dto: PillCornerDto) -> PillCorner {
    match dto {
        PillCornerDto::Pill => PillCorner::Pill,
        PillCornerDto::Rounded => PillCorner::Rounded,
        PillCornerDto::Square => PillCorner::Square,
    }
}

fn terminal_mode_to_dto(mode: TerminalMode) -> TerminalModeDto {
    match mode {
        TerminalMode::Auto => TerminalModeDto::Auto,
        TerminalMode::Terminal => TerminalModeDto::Terminal,
        TerminalMode::Normal => TerminalModeDto::Normal,
        _ => TerminalModeDto::Auto,
    }
}

fn terminal_mode_from_dto(dto: TerminalModeDto) -> TerminalMode {
    match dto {
        TerminalModeDto::Auto => TerminalMode::Auto,
        TerminalModeDto::Terminal => TerminalMode::Terminal,
        TerminalModeDto::Normal => TerminalMode::Normal,
    }
}

fn privacy_mode_to_dto(mode: PrivacyMode) -> PrivacyModeDto {
    match mode {
        PrivacyMode::LocalOnly => PrivacyModeDto::LocalOnly,
        PrivacyMode::LocalPreferred => PrivacyModeDto::LocalPreferred,
        PrivacyMode::AllowRemote => PrivacyModeDto::AllowRemote,
        // `PrivacyMode` is `#[non_exhaustive]`; the strictest option is the
        // only safe thing to show for a variant this build does not know.
        _ => PrivacyModeDto::LocalOnly,
    }
}

fn privacy_mode_from_dto(dto: PrivacyModeDto) -> PrivacyMode {
    match dto {
        PrivacyModeDto::LocalOnly => PrivacyMode::LocalOnly,
        PrivacyModeDto::LocalPreferred => PrivacyMode::LocalPreferred,
        PrivacyModeDto::AllowRemote => PrivacyMode::AllowRemote,
    }
}

fn asr_backend_from_type(
    kind: AsrBackendTypeDto,
    current: &AsrBackendConfig,
    input: &SettingsCoreDto,
    api_key: Option<&str>,
    whisper_model: Option<WhisperModelDto>,
) -> Result<AsrBackendConfig, String> {
    Ok(match kind {
        AsrBackendTypeDto::WhisperCpp => {
            // Prefer the incoming selection; else keep the current whisper model
            // if already on WhisperCpp; else the default. The choice lives inside
            // the WhisperCpp variant, so a switch away and back within one session
            // is preserved via `current`, but it is not persisted across a
            // different backend.
            let model = whisper_model
                .map(Into::into)
                .or(match current {
                    AsrBackendConfig::WhisperCpp { model } => Some(*model),
                    _ => None,
                })
                .unwrap_or_default();
            AsrBackendConfig::WhisperCpp { model }
        }
        AsrBackendTypeDto::OpenAiCompatible => {
            // Prefer incoming DTO fields; fall back to current config values if already OpenAI.
            let (cur_url, cur_key, cur_model) = match current {
                AsrBackendConfig::OpenAiCompatible {
                    base_url,
                    api_key,
                    model,
                    ..
                } => (
                    base_url.to_string(),
                    api_key.expose().to_owned(),
                    model.clone(),
                ),
                _ => (
                    "https://api.openai.com/v1".to_owned(),
                    String::new(),
                    "whisper-1".to_owned(),
                ),
            };
            let base_url_str = input.openai_asr_base_url.as_deref().unwrap_or(&cur_url);
            let api_key_str = api_key.unwrap_or(&cur_key);
            let model_str = input.openai_asr_model.as_deref().unwrap_or(&cur_model);
            let base_url: url::Url = base_url_str
                .parse()
                .map_err(|e| format!("invalid openaiAsrBaseUrl: {e}"))?;
            validate_api_url(&base_url, "openaiAsrBaseUrl")?;
            AsrBackendConfig::OpenAiCompatible {
                base_url,
                api_key: RedactedString::new(api_key_str),
                model: model_str.to_owned(),
                streaming: StreamingMode::HttpPolling,
            }
        }
    })
}

/// Build the styling LLM backend for the selected engine. Mirrors
/// `asr_backend_from_type`: for a remote provider, incoming DTO fields win and
/// fall back to the current config, so a `None` (blank/unchanged) api key keeps
/// the saved key.
fn llm_backend_from_type(
    kind: LlmBackendTypeDto,
    provider: &OpenAiLlmProvider,
) -> Result<LlmBackendConfig, String> {
    Ok(match kind {
        LlmBackendTypeDto::OpenAiCompatible => {
            let base_url: url::Url = provider
                .base_url
                .parse()
                .map_err(|e| format!("invalid openaiLlmBaseUrl: {e}"))?;
            validate_api_url(&base_url, "openaiLlmBaseUrl")?;
            LlmBackendConfig::OpenAiCompatible {
                base_url,
                api_key: provider.api_key.clone(),
                model: provider.model.clone(),
                max_tokens: provider.max_tokens,
            }
        }
    })
}

/// Validate that an API URL uses HTTPS (or is a loopback address for local dev).
fn validate_api_url(url: &url::Url, field: &str) -> Result<(), String> {
    match url.scheme() {
        "https" => Ok(()),
        "http" => {
            let host = url.host_str().unwrap_or_default();
            if host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]" {
                Ok(())
            } else {
                Err(format!(
                    "{field}: HTTP is only allowed for localhost; use HTTPS for remote endpoints"
                ))
            }
        }
        scheme => Err(format!(
            "{field}: unsupported URL scheme \"{scheme}\", expected https"
        )),
    }
}

/// Maximum number of modifier+key parts in a hotkey combo.
const MAX_COMBO_PARTS: usize = 8;

fn parse_key_combo(raw: &str) -> Result<KeyCombo, String> {
    let raw_parts: Vec<_> = raw
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if raw_parts.is_empty() {
        return Err("hotkey cannot be empty".into());
    }
    if raw_parts.len() > MAX_COMBO_PARTS {
        return Err(format!(
            "hotkey cannot contain more than {MAX_COMBO_PARTS} parts"
        ));
    }

    let parts = raw_parts;
    let key = parts[parts.len() - 1];
    // Reject key names that are empty or unreasonably long.
    if key.is_empty() || key.len() > 32 {
        return Err(format!("invalid hotkey key name: {key:?}"));
    }

    let mut modifiers = Vec::new();
    let mut seen_modifiers = std::collections::HashSet::new();
    for modifier in &parts[..parts.len() - 1] {
        let parsed = parse_modifier(modifier)?;
        if !seen_modifiers.insert(modifier.to_ascii_lowercase()) {
            return Err(format!("duplicate hotkey modifier: {modifier}"));
        }
        modifiers.push(parsed);
    }

    Ok(KeyCombo {
        modifiers,
        key: key.to_string(),
    })
}

fn parse_modifier(raw: &str) -> Result<Modifier, String> {
    match raw {
        "Ctrl" => Ok(Modifier::Ctrl),
        "Shift" => Ok(Modifier::Shift),
        "Alt" => Ok(Modifier::Alt),
        "Meta" => Ok(Modifier::Meta),
        _ => Err(format!("unsupported hotkey modifier: {raw}")),
    }
}

fn format_key_combo(combo: &KeyCombo) -> String {
    let mut parts: Vec<&str> = combo.modifiers.iter().map(modifier_label).collect();
    parts.push(combo.key.as_str());
    parts.join("+")
}

fn modifier_label(modifier: &Modifier) -> &'static str {
    match modifier {
        Modifier::Ctrl => "Ctrl",
        Modifier::Shift => "Shift",
        Modifier::Alt => "Alt",
        Modifier::Meta => "Meta",
        _ => UNKNOWN_LABEL,
    }
}

fn map_dictionary_entry(entry: StoredDictionaryEntry) -> DictionaryItemDto {
    DictionaryItemDto {
        id: entry.id,
        phrase: entry.phrase,
        replacement: entry.replacement,
        priority: entry.priority,
    }
}

fn session_status_label(status: &SessionStatus) -> String {
    match status {
        SessionStatus::Active => "Active".into(),
        SessionStatus::Completed => "Completed".into(),
        SessionStatus::Failed(message) => format!("Failed: {message}"),
        SessionStatus::Cancelled => "Cancelled".into(),
        _ => UNKNOWN_LABEL.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quarantined settings file leaves us holding defaults, not the user's
    /// choices. Acting on them destroys history the user asked to keep, and
    /// nothing restores it.
    #[test]
    fn retention_does_not_run_on_quarantined_settings() {
        use lune_core::asr::AsrBackendKind;
        use lune_core::session::DictationSession;

        let mut db = Database::open_in_memory().unwrap();
        db.insert_session(&DictationSession {
            id: uuid::Uuid::new_v4(),
            started_at: chrono::Utc::now(),
            ended_at: None,
            style: None,
            backend: AsrBackendKind::WhisperCpp,
            status: SessionStatus::Active,
            transcript: Some("a stored transcript".into()),
        })
        .unwrap();

        // Defaults say "history off" only because the real file could not be
        // read; `Off` would otherwise purge everything.
        let defaults = UserSettings {
            session_history_days: None,
            ..UserSettings::default()
        };
        enforce_retention(&mut db, &defaults, SettingsTrust::Quarantined);
        assert_eq!(db.list_sessions(10).unwrap().len(), 1);

        enforce_retention(&mut db, &defaults, SettingsTrust::AsWritten);
        assert!(db.list_sessions(10).unwrap().is_empty());
    }

    /// The frontend reads every settings field at the top level of the object.
    #[test]
    fn settings_dto_serializes_flat() {
        let dto = serde_json::to_value(settings_to_dto(&UserSettings::default())).unwrap();
        for key in [
            "activationMode",
            "privacyMode",
            "openaiLlmActive",
            "whisperModel",
        ] {
            assert!(dto.get(key).is_some(), "missing top-level {key}: {dto}");
        }
        assert!(dto.get("core").is_none(), "{dto}");
    }

    /// Build a `SaveWorkspaceSettingsInput` from defaults for test convenience.
    fn default_save_input() -> SaveWorkspaceSettingsInput {
        let dto = settings_to_dto(&UserSettings::default());
        SaveWorkspaceSettingsInput {
            core: dto.core,
            llm_backend_type: LlmBackendTypeDto::OpenAiCompatible,
            openai_asr_api_key: None,
            openai_llm_api_key: None,
            whisper_model: Some(dto.whisper_model),
        }
    }

    #[test]
    fn apply_settings_rejects_invalid_style_and_language() {
        let mut settings = UserSettings::default();
        let mut db = Database::open_in_memory().expect("db");

        let mut input = default_save_input();
        input.core.style_preset = Some("not-a-style".into());
        assert!(apply_settings_dto(&mut settings, input, &mut db).is_err());

        let mut input = default_save_input();
        input.core.language = "bad tag".into();
        assert!(apply_settings_dto(&mut settings, input, &mut db).is_err());
    }

    #[test]
    fn apply_settings_persists_confidence_threshold() {
        let mut settings = UserSettings::default();
        let mut db = Database::open_in_memory().expect("db");

        let mut input = default_save_input();
        input.core.confidence_threshold = Some(0.55);
        apply_settings_dto(&mut settings, input, &mut db).unwrap();
        assert_eq!(settings.pipeline.word_confidence_threshold, Some(0.55));

        // Out-of-range is rejected.
        let mut input = default_save_input();
        input.core.confidence_threshold = Some(1.5);
        assert!(apply_settings_dto(&mut settings, input, &mut db).is_err());
    }

    #[test]
    fn dictation_language_accepts_auto_and_bcp47() {
        let mut settings = UserSettings::default();
        let mut db = Database::open_in_memory().expect("db");

        let mut input = default_save_input();
        input.core.dictation_language = "auto".into();
        apply_settings_dto(&mut settings, input, &mut db).unwrap();
        assert_eq!(settings.dictation_language, "auto");

        let mut input = default_save_input();
        input.core.dictation_language = "en".into();
        apply_settings_dto(&mut settings, input, &mut db).unwrap();
        assert_eq!(settings.dictation_language, "en");

        let mut input = default_save_input();
        input.core.dictation_language = "zh".into();
        apply_settings_dto(&mut settings, input, &mut db).unwrap();
        assert_eq!(settings.dictation_language, "zh");
    }

    #[test]
    fn dictation_language_rejects_invalid_tag() {
        let mut settings = UserSettings::default();
        let mut db = Database::open_in_memory().expect("db");

        let mut input = default_save_input();
        input.core.dictation_language = "not valid".into();
        assert!(apply_settings_dto(&mut settings, input, &mut db).is_err());
    }

    #[test]
    fn dictation_language_round_trips_through_dto() {
        let settings = UserSettings {
            dictation_language: "de".into(),
            ..UserSettings::default()
        };

        let dto = settings_to_dto(&settings);
        assert_eq!(dto.core.dictation_language, "de");
    }

    #[test]
    fn device_policy_round_trips_cuda_without_downgrade() {
        let mut settings = UserSettings {
            device: DevicePolicy::Cuda,
            ..UserSettings::default()
        };
        let mut db = Database::open_in_memory().expect("db");
        let mut input = default_save_input();
        input.core.use_gpu = true;
        input.core.device_policy = DevicePolicyDto::Cuda;

        apply_settings_dto(&mut settings, input, &mut db).unwrap();
        assert!(matches!(settings.device, DevicePolicy::Cuda));
    }

    #[test]
    fn settings_to_dto_does_not_expose_api_keys() {
        let settings = UserSettings {
            asr_backend: AsrBackendConfig::OpenAiCompatible {
                base_url: "https://api.openai.com/v1".parse().unwrap(),
                api_key: RedactedString::new("secret-asr"),
                model: "whisper-1".into(),
                streaming: StreamingMode::HttpPolling,
            },
            ..UserSettings::default()
        };

        let dto = settings_to_dto(&settings);
        assert!(dto.openai_asr_api_key_set);
        let json = serde_json::to_string(&dto).unwrap();
        assert!(!json.contains("secret-asr"));
    }

    #[test]
    fn parse_key_combo_rejects_duplicate_modifiers_and_overflow() {
        assert!(parse_key_combo("Ctrl+Ctrl+Space").is_err());
        assert!(parse_key_combo("Ctrl+Alt+Shift+Meta+Fn+Hyper+Super+Caps+A").is_err());
    }

    #[test]
    fn validate_api_url_allows_ipv6_loopback_http() {
        let url = url::Url::parse("http://[::1]:8080/v1").unwrap();
        assert!(validate_api_url(&url, "openaiAsrBaseUrl").is_ok());
    }

    #[test]
    fn delete_session_rejects_empty_id() {
        let result = delete_session(String::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must not be empty"));
    }

    #[test]
    fn delete_sessions_rejects_an_empty_or_oversized_list() {
        assert!(
            delete_sessions(Vec::new())
                .unwrap_err()
                .contains("must not be empty")
        );
        let too_many = vec!["id".to_string(); HISTORY_BULK_DELETE_MAX + 1];
        assert!(delete_sessions(too_many).unwrap_err().contains("at most"));
        assert!(
            delete_sessions(vec![String::new()])
                .unwrap_err()
                .contains("empty id")
        );
    }

    /// The retention setting carries three choices in one nullable number, and
    /// only the last is a prune window. A range check that forgot the sentinel
    /// would reject "keep forever" outright.
    #[test]
    fn history_days_accepts_the_forever_sentinel_but_not_an_out_of_range_window() {
        assert!(validate_history_days(None).is_ok(), "off");
        assert!(
            validate_history_days(Some(HISTORY_DAYS_FOREVER)).is_ok(),
            "keep forever"
        );
        assert!(validate_history_days(Some(HISTORY_DAYS_MIN)).is_ok());
        assert!(validate_history_days(Some(HISTORY_DAYS_MAX)).is_ok());
        assert!(validate_history_days(Some(HISTORY_DAYS_MAX + 1)).is_err());
    }
}
