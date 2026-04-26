use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use lune_core::config::UserSettings;

use crate::error::PersistResult;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Read/write `UserSettings` as TOML with atomic writes.
pub struct SettingsStore {
    path: PathBuf,
}

impl SettingsStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Load settings from disk. Returns defaults if file doesn't exist
    /// or cannot be parsed (pre-release migration: old keys like `output_mode`
    /// will fail to parse, so we fall back gracefully).
    pub fn load(&self) -> PersistResult<UserSettings> {
        if !self.path.exists() {
            return Ok(UserSettings::default());
        }
        let content = fs::read_to_string(&self.path)?;
        match toml::from_str(&content) {
            Ok(settings) => Ok(settings),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %self.path.display(),
                    "failed to parse settings, using defaults"
                );
                Ok(UserSettings::default())
            }
        }
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
