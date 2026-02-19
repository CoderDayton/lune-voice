use std::fs;
use std::path::{Path, PathBuf};

use lune_core::config::UserSettings;

use crate::error::PersistResult;

/// Read/write `UserSettings` as TOML with atomic writes.
pub struct SettingsStore {
    path: PathBuf,
}

impl SettingsStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Load settings from disk. Returns defaults if file doesn't exist.
    pub fn load(&self) -> PersistResult<UserSettings> {
        if !self.path.exists() {
            return Ok(UserSettings::default());
        }
        let content = fs::read_to_string(&self.path)?;
        let settings = toml::from_str(&content)?;
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
    let tmp = target.with_extension("toml.tmp");
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
        assert_eq!(settings.language, "en-US");

        // Save and reload
        store.save(&settings).unwrap();
        let reloaded = store.load().unwrap();
        assert_eq!(reloaded.language, settings.language);
        assert_eq!(
            reloaded.session_history_days,
            settings.session_history_days
        );
    }

    #[test]
    fn atomic_write_no_partial() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");

        atomic_write(&path, b"hello").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello");

        // Tmp file should be cleaned up
        assert!(!path.with_extension("toml.tmp").exists());
    }
}
