pub type PersistResult<T> = std::result::Result<T, PersistError>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PersistError {
    #[error("diesel error: {0}")]
    Diesel(#[from] diesel::result::Error),

    #[error("diesel connection error: {0}")]
    Connection(#[from] diesel::ConnectionError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("toml serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("toml parse error: {0}")]
    TomlDeserialize(#[from] toml::de::Error),

    #[error("migration error: {0}")]
    Migration(String),

    #[error("row decode error: {0}")]
    RowDecode(String),

    #[error("model download: {0}")]
    ModelDownload(String),

    #[error("not found: {0}")]
    NotFound(String),
}
