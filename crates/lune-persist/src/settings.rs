use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use lune_core::config::UserSettings;

use crate::error::PersistResult;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// True when raw settings TOML still carries the removed `dictation_mode = "Both"`
/// (the legacy "type and copy" mode), so its copy intent can be migrated.
fn dictation_mode_was_both(content: &str) -> bool {
    content.lines().any(|line| {
        let line = line.trim();
        line.starts_with("dictation_mode") && line.contains("\"Both\"")
    })
}

/// Read/write `UserSettings` as TOML with atomic writes.
pub struct SettingsStore {
    path: PathBuf,
}

impl SettingsStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
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
        Ok(settings)
    }

    /// Save settings atomically (write to temp, then rename).
    pub fn save(&self, settings: &UserSettings) -> PersistResult<()> {
        let content = toml::to_string_pretty(settings)?;
        atomic_write(&self.path, content.as_bytes())?;
        Ok(())
    }
}

/// Write bytes atomically: write to a sibling temp file, then rename.
fn atomic_write(target: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let pid = std::process::id();
    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = target.with_extension(format!("toml.tmp.{pid}.{seq}"));
    fs::write(&tmp, data)?;
    fs::rename(&tmp, target)?;
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
}
