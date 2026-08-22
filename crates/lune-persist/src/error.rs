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

    #[error("row encode error: {0}")]
    RowEncode(String),

    #[error("model download: {0}")]
    ModelDownload(String),

    /// The caller asked for the transfer to stop. Distinct from
    /// [`Self::ModelDownload`] so a user who pressed *Cancel* is not shown a
    /// failure for the thing they just asked for.
    #[error("model download cancelled")]
    ModelDownloadCancelled,

    #[error("not found: {0}")]
    NotFound(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_encode_display() {
        let e = PersistError::RowEncode("status serialize: bad".into());
        assert_eq!(e.to_string(), "row encode error: status serialize: bad");
    }
}
