use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use lune_core::config::{AsrBackendConfig, LlmBackendConfig, RedactedString, UserSettings};
use tracing::warn;

use crate::error::{PersistError, PersistResult};
use crate::secrets::{KeyringVault, SecretSlot, SecretVault};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// True when raw settings TOML still carries the removed `dictation_mode = "Both"`
/// (the legacy "type and copy" mode), so its copy intent can be migrated.
fn dictation_mode_was_both(content: &str) -> bool {
    content.lines().any(|line| {
        let line = line.trim();
        line.starts_with("dictation_mode") && line.contains("\"Both\"")
    })
}

/// True when the raw settings TOML carries a non-empty `api_key = "..."` line —
/// i.e. a plaintext secret that should be migrated into the keyring. Empty
/// (`api_key = ""`, the blanked/keyring-backed state) does not count.
fn has_inline_api_key(content: &str) -> bool {
    content.lines().any(|line| {
        let line = line.trim_start();
        line.strip_prefix("api_key")
            .and_then(|rest| rest.trim_start().strip_prefix('='))
            .map(str::trim)
            .is_some_and(|value| value.starts_with('"') && value != "\"\"")
    })
}

/// True when the raw settings TOML still carries the legacy `[[custom_styles]]`
/// table or its companion `few_shot_dirty` set — data that now lives in the DB.
fn raw_has_legacy_custom_styles(raw: &str) -> bool {
    raw.contains("[[custom_styles]]") || raw.contains("few_shot_dirty")
}

/// Read/write `UserSettings` as TOML with atomic writes.
///
/// API keys are kept out of the plaintext TOML: on `save` they are written to
/// the OS keyring (via [`SecretVault`]) and blanked in the file; on `load` they
/// are restored from the keyring back into the in-memory `UserSettings`. The
/// rest of the app therefore always sees real keys and needs no changes.
pub struct SettingsStore {
    path: PathBuf,
    vault: Box<dyn SecretVault>,
}

impl SettingsStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            vault: Box::new(KeyringVault),
        }
    }

    /// Construct with an explicit secret vault. Used by tests to inject an
    /// in-memory backend instead of touching the real OS keyring.
    #[cfg(test)]
    pub fn with_vault(path: impl Into<PathBuf>, vault: Box<dyn SecretVault>) -> Self {
        Self {
            path: path.into(),
            vault,
        }
    }

    /// Load settings from disk. Returns defaults when the file doesn't exist.
    /// A malformed file produces `PersistError::TomlDeserialize` so callers can
    /// decide between failing the startup, prompting the user, or backing up
    /// the broken file — silently defaulting was discarding real configuration
    /// (including API keys) and hiding the underlying problem.
    pub fn load(&self) -> PersistResult<UserSettings> {
        if !self.path.exists() {
            return Ok(UserSettings::default());
        }
        let content = fs::read_to_string(&self.path)?;
        let mut settings: UserSettings = toml::from_str(&content)?;
        // Migrate the removed "type and copy" (`Both`) mode: the mode itself
        // deserializes to `TypeInPlace`, but its clipboard-copy intent is now the
        // separate `copy_to_clipboard` toggle, so set that for legacy configs.
        if dictation_mode_was_both(&content) {
            settings.copy_to_clipboard = true;
        }
        self.restore_secrets(&mut settings);
        Ok(settings)
    }

    /// Load settings, moving aside a file that cannot be parsed.
    ///
    /// For the startup path, where propagating the error means the process
    /// exits and the user has no way to repair the file from the UI. A
    /// malformed file is renamed to `settings.toml.bad-<unix seconds>` and
    /// defaults are returned, so the app starts and writes a valid file on the
    /// next save while the original stays on disk for recovery. API keys are
    /// unaffected — they live in the keyring, not the file.
    ///
    /// Only a parse failure is recovered. An I/O error still propagates: the
    /// file's contents are probably intact, and silently replacing a
    /// configuration we merely failed to read would be worse than failing.
    pub fn load_or_quarantine(&self) -> PersistResult<UserSettings> {
        let parse_error = match self.load() {
            Err(PersistError::TomlDeserialize(e)) => e,
            other => return other,
        };
        let backup = quarantine_path(&self.path);
        if let Err(e) = fs::rename(&self.path, &backup) {
            warn!(error = %e, "could not move the malformed settings file aside");
            return Err(PersistError::TomlDeserialize(parse_error));
        }
        warn!(
            error = %parse_error,
            backup = %backup.display(),
            "settings file is malformed; moved aside and started from defaults"
        );
        let mut settings = UserSettings::default();
        self.restore_secrets(&mut settings);
        Ok(settings)
    }

    /// Save settings atomically (write to temp, then rename).
    ///
    /// API keys are stored in the keyring and blanked in the written TOML. When
    /// the keyring is unavailable the key stays inline (best-effort security),
    /// so a save never loses it.
    pub fn save(&self, settings: &UserSettings) -> PersistResult<()> {
        let scrubbed = self.persist_secrets(settings);
        let content = toml::to_string_pretty(&scrubbed)?;
        atomic_write(&self.path, content.as_bytes())?;
        Ok(())
    }

    /// Lift any plaintext API keys out of `settings.toml` into the OS keyring.
    ///
    /// Called once at startup so a key never lingers in the file until the
    /// user's next settings change. Idempotent and cheap: it does nothing when
    /// the file holds no inline key (the steady state after migration) and
    /// rewrites the file only when a key actually moved. If the keyring is
    /// unavailable the key stays inline — the file is left as-is, so this is
    /// always safe to call.
    pub fn migrate_secrets_to_keyring(&self) -> PersistResult<()> {
        if !self.path.exists() {
            return Ok(());
        }
        // Harden on every boot, before the inline-key check. `atomic_write`
        // covers files this version wrote, but a config created by an older
        // build (or by a user's editor) still sits at the default umask, and
        // the key it may hold is exactly what this function exists to handle.
        // Best-effort: a permission failure must not stop the app from booting.
        if let Err(e) = restrict_to_owner(&self.path) {
            tracing::debug!(error = %e, "could not restrict settings.toml permissions");
        }
        let raw = fs::read_to_string(&self.path)?;
        if !has_inline_api_key(&raw) {
            return Ok(());
        }
        let settings = self.load()?;
        let scrubbed = self.persist_secrets(&settings);
        let new_content = toml::to_string_pretty(&scrubbed)?;
        if new_content != raw {
            atomic_write(&self.path, new_content.as_bytes())?;
            tracing::info!("migrated plaintext API key(s) from settings.toml into the OS keyring");
        }
        Ok(())
    }

    /// One-time migration: lift legacy `[[custom_styles]]` (and the companion
    /// `few_shot_dirty` set) out of `settings.toml` into the `custom_styles` DB
    /// table, then strip them from the file. Idempotent and convergent — the DB
    /// insert is `INSERT OR IGNORE` inside one transaction, so a retry after a
    /// partial run never errors or duplicates. Mirrors
    /// [`Self::migrate_secrets_to_keyring`]; safe to call on every boot.
    ///
    /// Ordering is load-bearing for data safety: the styles are committed to the
    /// DB *before* the file is rewritten. `UserSettings` no longer carries these
    /// fields, so any `save` drops them — the file therefore must not be
    /// rewritten until the DB durably holds the data. A DB-insert failure
    /// returns `Err` (the caller treats it as fatal) rather than risk a later
    /// save clobbering the un-migrated styles. Stripping the file afterwards is
    /// best-effort: the data is already safe, so a strip failure just retries on
    /// the next boot.
    pub fn migrate_custom_styles_to_db(&self, db: &mut crate::Database) -> PersistResult<()> {
        if !self.path.exists() {
            return Ok(());
        }
        let raw = fs::read_to_string(&self.path)?;
        if !raw_has_legacy_custom_styles(&raw) {
            return Ok(());
        }
        // Parse only the two legacy fields; every other TOML key is ignored, and
        // `StylePreset`'s own removed fields (e.g. `user_prompt_template`) are
        // tolerated since it does not deny unknown fields.
        #[derive(serde::Deserialize, Default)]
        struct LegacyStyles {
            #[serde(default)]
            custom_styles: Vec<lune_core::config::StylePreset>,
            #[serde(default)]
            few_shot_dirty: std::collections::BTreeSet<String>,
        }
        let legacy: LegacyStyles = toml::from_str(&raw)?;
        let batch: Vec<(lune_core::config::StylePreset, bool)> = legacy
            .custom_styles
            .into_iter()
            .map(|style| {
                let dirty = legacy.few_shot_dirty.contains(&style.id);
                (style, dirty)
            })
            .collect();
        // Atomic + convergent: the whole batch lands or nothing does, and ids
        // already copied by a prior partial run are skipped. Any failure here is
        // propagated (fatal) so the file is never stripped with data at risk.
        let inserted = db.insert_custom_styles_migration(&batch)?;
        // The styles are now durably in the DB, so it is safe to rewrite the
        // file without them. A `load → save` round-trip strips the legacy keys
        // (the current `UserSettings` has no field for them). Best-effort: the
        // data is already safe and a re-run just retries, so a write failure is
        // logged, not fatal.
        if let Err(e) = self.load().and_then(|settings| self.save(&settings)) {
            tracing::warn!(
                error = %e,
                "custom styles migrated into the database, but stripping settings.toml failed; will retry next boot"
            );
            return Ok(());
        }
        if inserted > 0 {
            tracing::info!(
                count = inserted,
                "migrated custom styles from settings.toml into the database"
            );
        }
        Ok(())
    }

    /// True when `settings.toml` still carries legacy `[[custom_styles]]` /
    /// `few_shot_dirty` blocks that must be migrated into the DB before the file
    /// can be safely rewritten (a rewrite drops them — they are no longer a
    /// `UserSettings` field). Lets the boot path decide whether a failed DB open
    /// is fatal (legacy data at risk) or harmless (nothing to migrate).
    pub fn has_legacy_custom_styles(&self) -> PersistResult<bool> {
        if !self.path.exists() {
            return Ok(false);
        }
        Ok(raw_has_legacy_custom_styles(&fs::read_to_string(
            &self.path,
        )?))
    }

    /// Populate blanked API keys from the keyring. An inline (non-empty) key in
    /// the TOML always wins — it is either a legacy plaintext config or the
    /// fallback written when no keyring was available — and gets migrated into
    /// the keyring on the next save.
    fn restore_secrets(&self, settings: &mut UserSettings) {
        // ── Remote ASR key ──
        if let AsrBackendConfig::OpenAiCompatible { api_key, .. } = &mut settings.asr_backend
            && api_key.expose().is_empty()
            && let Some(stored) = self.vault.get(SecretSlot::AsrApiKey)
        {
            *api_key = RedactedString::new(stored);
        }

        // ── Remote styling-LLM key ──
        // Canonical home is `llm_openai`; the active `llm_backend` holds a copy
        // when remote. Resolve the provider key first, then mirror it into the
        // active backend so `styling_llm_readiness` and the client builder see it.
        if settings.llm_openai.api_key.expose().is_empty()
            && let Some(stored) = self.vault.get(SecretSlot::LlmApiKey)
        {
            settings.llm_openai.api_key = RedactedString::new(stored);
        }
        let llm_key = {
            let k = settings.llm_openai.api_key.expose();
            (!k.is_empty()).then(|| k.to_owned())
        };
        if let LlmBackendConfig::OpenAiCompatible { api_key, .. } = &mut settings.llm_backend
            && api_key.expose().is_empty()
            && let Some(k) = llm_key
        {
            *api_key = RedactedString::new(k);
        }
    }

    /// Store API keys in the keyring and return a clone with those keys blanked
    /// for TOML serialization. A key the keyring rejects is left inline so it is
    /// never lost; a genuinely empty key clears its keyring slot.
    fn persist_secrets(&self, settings: &UserSettings) -> UserSettings {
        let mut out = settings.clone();

        // ── Remote ASR key ──
        let asr_key = match &out.asr_backend {
            AsrBackendConfig::OpenAiCompatible { api_key, .. } => Some(api_key.expose().to_owned()),
            _ => None,
        };
        match asr_key {
            Some(key) if !key.is_empty() => {
                if self.vault.set(SecretSlot::AsrApiKey, &key)
                    && let AsrBackendConfig::OpenAiCompatible { api_key, .. } = &mut out.asr_backend
                {
                    *api_key = RedactedString::new("");
                }
            }
            // Empty key or no remote ASR backend: drop any stale keyring entry.
            _ => self.vault.delete(SecretSlot::AsrApiKey),
        }

        // ── Remote styling-LLM key ──
        // Canonical home is `llm_openai`, but a legacy config (predating that
        // field) may carry the key only under an `OpenAiCompatible`
        // `llm_backend`. Fall back to it so startup migration never leaves that
        // key in plaintext; blank both once it is safely in the vault.
        let llm_key = {
            let canonical = out.llm_openai.api_key.expose();
            if !canonical.is_empty() {
                canonical.to_owned()
            } else if let LlmBackendConfig::OpenAiCompatible { api_key, .. } = &out.llm_backend {
                api_key.expose().to_owned()
            } else {
                String::new()
            }
        };
        if !llm_key.is_empty() {
            if self.vault.set(SecretSlot::LlmApiKey, &llm_key) {
                out.llm_openai.api_key = RedactedString::new("");
                if let LlmBackendConfig::OpenAiCompatible { api_key, .. } = &mut out.llm_backend {
                    *api_key = RedactedString::new("");
                }
            }
        } else {
            self.vault.delete(SecretSlot::LlmApiKey);
        }

        out
    }
}

/// Write bytes atomically: write to a sibling temp file, then rename.
/// Where a malformed settings file is moved so it is kept but no longer read.
/// The timestamp keeps repeated failures from overwriting each other.
fn quarantine_path(path: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut name = path
        .file_name()
        .unwrap_or_else(|| OsStr::new("settings.toml"))
        .to_os_string();
    name.push(format!(".bad-{stamp}"));
    path.with_file_name(name)
}

fn atomic_write(target: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let pid = std::process::id();
    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = target.with_extension(format!("toml.tmp.{pid}.{seq}"));
    // Create owner-only rather than writing and then chmod-ing: this file holds
    // an inline API key whenever the keyring was unavailable, and `fs::write`
    // would publish it at the default umask for the window between the write
    // and the chmod.
    let publish = write_owner_only(&tmp, data).and_then(|()| fs::rename(&tmp, target));
    if publish.is_err() {
        // Never leave a temp file holding a key behind on a failed write; the
        // rename is what makes it the caller's to clean up.
        let _ = fs::remove_file(&tmp);
    }
    publish
}

/// Create `path` with mode `0600` and write `data` to it.
///
/// Fails if the path already exists — the caller's temp names are per-process
/// and per-call, so a collision means something else is writing there and
/// reusing its permissions would defeat the point.
#[cfg(unix)]
fn write_owner_only(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(data)?;
    // Flush before the caller renames this into place. The rename is what makes
    // the new settings visible; without the flush a power loss between the two
    // can publish a name pointing at nothing, which the quarantine path then
    // has to clean up on the next boot.
    file.sync_all()?;
    // `mode` is masked by the umask, so a restrictive umask could land us below
    // 0600; set it exactly once the bytes are down.
    restrict_to_owner(path)
}

#[cfg(not(unix))]
fn write_owner_only(path: &Path, data: &[u8]) -> std::io::Result<()> {
    fs::write(path, data)
}

/// Owner-only permissions (`0600`) for a file that may hold an API key.
///
/// No-op off Unix: Windows and macOS keyrings are effectively always available,
/// so the inline-key fallback that makes this matter is a Linux concern.
#[cfg(unix)]
fn restrict_to_owner(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store = SettingsStore::new(&path);

        // Defaults when file missing
        let settings = store.load().unwrap();
        assert_eq!(settings.language, "auto");

        // Save and reload
        store.save(&settings).unwrap();
        let reloaded = store.load().unwrap();
        assert_eq!(reloaded.language, settings.language);
        assert_eq!(reloaded.session_history_days, settings.session_history_days);
    }

    #[test]
    fn migrate_custom_styles_moves_toml_into_db_and_strips_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store =
            SettingsStore::with_vault(&path, Box::new(crate::secrets::InMemoryVault::default()));

        // Start from a full valid config, then splice in the legacy fields that
        // used to live in the TOML: top-level `few_shot_dirty` and a custom
        // style block (with the removed `user_prompt_template` key, to prove the
        // parser tolerates it).
        store.save(&UserSettings::default()).unwrap();
        let base = fs::read_to_string(&path).unwrap();
        let with_dirty = base.replacen(
            "language = \"auto\"",
            "language = \"auto\"\nfew_shot_dirty = [\"custom-abc\"]",
            1,
        );
        let legacy = format!(
            "{with_dirty}\n[[custom_styles]]\n\
             id = \"custom-abc\"\n\
             name = \"Enhance Prompt\"\n\
             description = \"d\"\n\
             style_instructions = \"Make it better.\"\n\
             user_prompt_template = \"{{raw_text}}\"\n\
             few_shot_pairs = []\n\n\
             [custom_styles.formatting]\n\
             keybinds = true\n\
             title_abbreviations = false\n\
             auto_capitalization = false\n\
             brand_casing = false\n"
        );
        fs::write(&path, &legacy).unwrap();

        let mut db = crate::Database::open_in_memory().unwrap();
        store.migrate_custom_styles_to_db(&mut db).unwrap();

        // Landed in the DB with its stale flag and formatting preserved.
        let styles = db.list_custom_styles().unwrap();
        assert_eq!(styles.len(), 1);
        assert_eq!(styles[0].preset.id, "custom-abc");
        assert_eq!(styles[0].preset.name, "Enhance Prompt");
        assert!(styles[0].few_shot_dirty);
        assert!(!styles[0].preset.formatting.auto_capitalization);

        // Stripped from the TOML.
        let rewritten = fs::read_to_string(&path).unwrap();
        assert!(!rewritten.contains("[[custom_styles]]"));
        assert!(!rewritten.contains("few_shot_dirty"));

        // Idempotent: a second run is a no-op and does not duplicate.
        store.migrate_custom_styles_to_db(&mut db).unwrap();
        assert_eq!(db.list_custom_styles().unwrap().len(), 1);

        // After a successful migration the file is legacy-free, so the boot
        // path treats a later DB-open failure as harmless.
        assert!(!store.has_legacy_custom_styles().unwrap());
    }

    #[test]
    fn has_legacy_custom_styles_detects_unmigrated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store =
            SettingsStore::with_vault(&path, Box::new(crate::secrets::InMemoryVault::default()));

        // No file yet → nothing to migrate.
        assert!(!store.has_legacy_custom_styles().unwrap());

        // A clean save (no legacy keys) → still nothing to migrate.
        store.save(&UserSettings::default()).unwrap();
        assert!(!store.has_legacy_custom_styles().unwrap());

        // Splice in a legacy block → detected, so a failed DB open is fatal.
        let base = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            format!("{base}\n[[custom_styles]]\nid = \"custom-x\"\n"),
        )
        .unwrap();
        assert!(store.has_legacy_custom_styles().unwrap());
    }

    #[test]
    fn legacy_both_mode_migrates_to_type_plus_copy() {
        use lune_core::config::DictationMode;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store = SettingsStore::new(&path);

        // Write a valid config, then force the removed legacy `Both` value.
        store.save(&UserSettings::default()).unwrap();
        let legacy = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| {
                if l.trim_start().starts_with("dictation_mode") {
                    "dictation_mode = \"Both\"".to_string()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, legacy).unwrap();

        // `Both` is gone, so it must not crash the load, and its copy intent is
        // carried over to the new toggle.
        let loaded = store.load().unwrap();
        assert_eq!(loaded.dictation_mode, DictationMode::TypeInPlace);
        assert!(loaded.copy_to_clipboard, "legacy Both must enable copy");
    }

    #[test]
    fn atomic_write_no_partial() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");

        atomic_write(&path, b"hello").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");

        // Tmp files should be cleaned up after rename.
        let tmp_prefix = "test.toml.tmp.";
        let leftovers = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(tmp_prefix))
            })
            .count();
        assert_eq!(leftovers, 0);
    }

    /// The inline-key fallback (no keyring available) leaves a real API key in
    /// this file, so it must never be written at the default umask.
    #[cfg(unix)]
    #[test]
    fn written_settings_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store = SettingsStore::new(&path);
        store.save(&UserSettings::default()).unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "settings.toml mode was {mode:o}, want 600");
    }

    /// A config written by an older build sits at the default umask. Boot must
    /// tighten it rather than leave a pre-existing key world-readable.
    #[cfg(unix)]
    #[test]
    fn boot_hardens_a_preexisting_loose_settings_file() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        fs::write(&path, "language = \"auto\"\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        SettingsStore::new(&path)
            .migrate_secrets_to_keyring()
            .unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "settings.toml mode was {mode:o}, want 600");
    }

    #[test]
    fn load_propagates_toml_parse_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, b"= not valid = toml ===\n").unwrap();

        let store = SettingsStore::new(&path);
        let err = store
            .load()
            .expect_err("malformed toml must produce an error, not silent defaults");
        assert!(
            matches!(err, crate::error::PersistError::TomlDeserialize(_)),
            "expected TomlDeserialize, got {err:?}",
        );
    }

    #[test]
    fn quarantine_moves_a_malformed_file_aside_and_starts_from_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, b"= not valid = toml ===\n").unwrap();

        let settings = SettingsStore::new(&path)
            .load_or_quarantine()
            .expect("a malformed file must not stop startup");
        assert_eq!(settings.language, UserSettings::default().language);

        assert!(!path.exists(), "the malformed file must not be read again");
        let kept: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("settings.toml.bad-")
            })
            .collect();
        assert_eq!(kept.len(), 1, "the original must be kept for recovery");
        assert_eq!(
            std::fs::read(kept[0].path()).unwrap(),
            b"= not valid = toml ===\n",
            "the quarantined copy must be the user's original bytes",
        );
    }

    #[test]
    fn quarantine_leaves_a_valid_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let want = UserSettings {
            language: "en-GB".into(),
            ..Default::default()
        };
        std::fs::write(&path, toml::to_string_pretty(&want).unwrap()).unwrap();

        let settings = SettingsStore::new(&path).load_or_quarantine().unwrap();
        assert_eq!(settings.language, "en-GB");
        assert!(path.exists(), "a readable file must not be moved");
    }

    #[test]
    fn concurrent_save_does_not_fail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store = std::sync::Arc::new(SettingsStore::new(&path));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = std::sync::Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let settings = UserSettings::default();
                store.save(&settings).unwrap();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let persisted = store.load().unwrap();
        assert_eq!(persisted.language, UserSettings::default().language);
    }

    #[test]
    fn api_keys_go_to_vault_not_toml() {
        use lune_core::config::{OpenAiLlmProvider, StreamingMode};
        use url::Url;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        // One store instance holds the in-memory vault, so save() then load()
        // share the same secret state — exactly like the real keyring.
        let store =
            SettingsStore::with_vault(&path, Box::new(crate::secrets::InMemoryVault::default()));

        let settings = UserSettings {
            asr_backend: AsrBackendConfig::OpenAiCompatible {
                base_url: Url::parse("https://api.example.com/v1").unwrap(),
                api_key: RedactedString::new("secret-asr"),
                model: "whisper-1".into(),
                streaming: StreamingMode::HttpPolling,
            },
            llm_openai: OpenAiLlmProvider {
                base_url: "https://api.example.com/v1".into(),
                api_key: RedactedString::new("secret-llm"),
                model: "gpt-4o-mini".into(),
                max_tokens: 512,
            },
            llm_backend: LlmBackendConfig::OpenAiCompatible {
                base_url: Url::parse("https://api.example.com/v1").unwrap(),
                api_key: RedactedString::new("secret-llm"),
                model: "gpt-4o-mini".into(),
                max_tokens: 512,
            },
            ..UserSettings::default()
        };

        store.save(&settings).unwrap();

        // Neither secret may appear in the plaintext TOML.
        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(
            !on_disk.contains("secret-asr"),
            "ASR key leaked into settings.toml:\n{on_disk}"
        );
        assert!(
            !on_disk.contains("secret-llm"),
            "LLM key leaked into settings.toml:\n{on_disk}"
        );

        // Reloading restores both keys from the vault into memory.
        let loaded = store.load().unwrap();
        match &loaded.asr_backend {
            AsrBackendConfig::OpenAiCompatible { api_key, .. } => {
                assert_eq!(api_key.expose(), "secret-asr");
            }
            other => panic!("asr backend changed shape: {other:?}"),
        }
        assert_eq!(loaded.llm_openai.api_key.expose(), "secret-llm");
        match &loaded.llm_backend {
            LlmBackendConfig::OpenAiCompatible { api_key, .. } => {
                assert_eq!(api_key.expose(), "secret-llm");
            }
            other => panic!("llm backend changed shape: {other:?}"),
        }
    }

    #[test]
    fn unavailable_keyring_keeps_key_inline() {
        // A vault that refuses every write models a headless/locked keyring;
        // the key must stay inline in the TOML so it is never lost.
        struct RejectingVault;
        impl SecretVault for RejectingVault {
            fn get(&self, _: SecretSlot) -> Option<String> {
                None
            }
            fn set(&self, _: SecretSlot, _: &str) -> bool {
                false
            }
            fn delete(&self, _: SecretSlot) {}
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store = SettingsStore::with_vault(&path, Box::new(RejectingVault));

        let settings = UserSettings {
            llm_openai: lune_core::config::OpenAiLlmProvider {
                api_key: RedactedString::new("inline-secret"),
                ..Default::default()
            },
            ..UserSettings::default()
        };
        store.save(&settings).unwrap();

        let on_disk = fs::read_to_string(&path).unwrap();
        assert!(
            on_disk.contains("inline-secret"),
            "key must stay inline when the keyring is unavailable:\n{on_disk}"
        );

        let loaded = store.load().unwrap();
        assert_eq!(loaded.llm_openai.api_key.expose(), "inline-secret");
    }

    #[test]
    fn migrate_lifts_plaintext_key_into_keyring_and_is_idempotent() {
        use lune_core::config::OpenAiLlmProvider;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");

        // Emulate a pre-migration file: a plaintext key written straight to TOML
        // (RedactedString serializes transparently, so the key is inline).
        let legacy = UserSettings {
            llm_openai: OpenAiLlmProvider {
                api_key: RedactedString::new("plaintext-key"),
                ..Default::default()
            },
            ..UserSettings::default()
        };
        let raw = toml::to_string_pretty(&legacy).unwrap();
        assert!(raw.contains("plaintext-key"));
        assert!(has_inline_api_key(&raw));
        fs::write(&path, &raw).unwrap();

        let store =
            SettingsStore::with_vault(&path, Box::new(crate::secrets::InMemoryVault::default()));
        store.migrate_secrets_to_keyring().unwrap();

        // The plaintext key is gone from disk but restored on load.
        let after = fs::read_to_string(&path).unwrap();
        assert!(
            !after.contains("plaintext-key"),
            "plaintext key must be removed from settings.toml:\n{after}"
        );
        assert!(!has_inline_api_key(&after));
        assert_eq!(
            store.load().unwrap().llm_openai.api_key.expose(),
            "plaintext-key"
        );

        // Idempotent: a second migration finds nothing inline and rewrites nothing.
        store.migrate_secrets_to_keyring().unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), after);
    }

    #[test]
    fn migrate_lifts_legacy_llm_backend_key_into_keyring() {
        use lune_core::config::OpenAiLlmProvider;
        use url::Url;

        // Legacy config: the styling key lives only under `llm_backend`, while
        // `llm_openai` (the newer canonical provider) is blank — the exact shape
        // that regressed (key left in plaintext while "migrated" was logged).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let legacy = UserSettings {
            llm_backend: LlmBackendConfig::OpenAiCompatible {
                base_url: Url::parse("https://api.example.com/v1").unwrap(),
                api_key: RedactedString::new("sk-legacy"),
                model: "gpt-4o-mini".into(),
                max_tokens: 512,
            },
            llm_openai: OpenAiLlmProvider {
                api_key: RedactedString::new(""),
                ..Default::default()
            },
            ..UserSettings::default()
        };
        let raw = toml::to_string_pretty(&legacy).unwrap();
        assert!(raw.contains("sk-legacy"));
        fs::write(&path, &raw).unwrap();

        let store =
            SettingsStore::with_vault(&path, Box::new(crate::secrets::InMemoryVault::default()));
        store.migrate_secrets_to_keyring().unwrap();

        let after = fs::read_to_string(&path).unwrap();
        assert!(
            !after.contains("sk-legacy"),
            "legacy llm_backend key must leave the plaintext file:\n{after}"
        );

        // Restored into the canonical provider AND the active backend on reload.
        let loaded = store.load().unwrap();
        assert_eq!(loaded.llm_openai.api_key.expose(), "sk-legacy");
        match &loaded.llm_backend {
            LlmBackendConfig::OpenAiCompatible { api_key, .. } => {
                assert_eq!(api_key.expose(), "sk-legacy");
            }
            other => panic!("llm backend changed shape: {other:?}"),
        }
    }

    #[test]
    fn llm_key_persists_when_styling_disabled() {
        use lune_core::config::OpenAiLlmProvider;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let store =
            SettingsStore::with_vault(&path, Box::new(crate::secrets::InMemoryVault::default()));

        // Styling turned OFF (Disabled backend) but a saved remote provider key —
        // the "provider persists across turning styling off" contract.
        let settings = UserSettings {
            llm_backend: LlmBackendConfig::Disabled,
            llm_openai: OpenAiLlmProvider {
                api_key: RedactedString::new("saved-key"),
                ..Default::default()
            },
            ..UserSettings::default()
        };
        store.save(&settings).unwrap();
        assert!(!fs::read_to_string(&path).unwrap().contains("saved-key"));

        let loaded = store.load().unwrap();
        assert!(matches!(loaded.llm_backend, LlmBackendConfig::Disabled));
        assert_eq!(
            loaded.llm_openai.api_key.expose(),
            "saved-key",
            "saved provider key must survive styling being off"
        );
    }
}
