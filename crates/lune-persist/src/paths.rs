use std::path::PathBuf;

/// Resolved XDG-compliant application paths.
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
}

impl AppPaths {
    const APP_NAME: &str = "lune-voice";

    /// Resolve paths using XDG conventions via the `dirs` crate.
    pub fn resolve() -> Option<Self> {
        let config_dir = dirs::config_dir()?.join(Self::APP_NAME);
        let data_dir = dirs::data_dir()?.join(Self::APP_NAME);
        Some(Self {
            config_dir,
            data_dir,
        })
    }

    /// Create from explicit paths (for testing).
    pub fn from_dirs(config_dir: impl Into<PathBuf>, data_dir: impl Into<PathBuf>) -> Self {
        Self {
            config_dir: config_dir.into(),
            data_dir: data_dir.into(),
        }
    }

    pub fn settings_file(&self) -> PathBuf {
        self.config_dir.join("settings.toml")
    }

    pub fn database_file(&self) -> PathBuf {
        self.data_dir.join("lune.db")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    pub fn adaptive_vad_file(&self) -> PathBuf {
        self.data_dir.join("adaptive_vad.json")
    }

    /// Ensure all directories exist.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config_dir)?;
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(self.models_dir())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    #[test]
    fn paths_from_dirs() {
        let paths = AppPaths::from_dirs("/tmp/config", "/tmp/data");
        assert_eq!(
            paths.settings_file(),
            Path::new("/tmp/config/settings.toml")
        );
        assert_eq!(paths.database_file(), Path::new("/tmp/data/lune.db"));
    }

    #[test]
    fn resolve_returns_some() {
        // Should work on any system with a home directory
        let paths = AppPaths::resolve();
        assert!(paths.is_some());
    }

    #[test]
    fn adaptive_vad_file_path() {
        let paths = AppPaths::from_dirs("/tmp/config", "/tmp/data");
        assert_eq!(
            paths.adaptive_vad_file(),
            Path::new("/tmp/data/adaptive_vad.json")
        );
    }
}
