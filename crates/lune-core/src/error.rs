use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("audio error: {0}")]
    Audio(String),

    #[error("ASR error: {0}")]
    Asr(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("persistence error: {0}")]
    Persistence(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("model file not found: {}", .0.display())]
    ModelNotFound(PathBuf),

    #[error("session error: {0}")]
    Session(String),

    #[error("command error: {0}")]
    Command(String),

    #[error("output error: {0}")]
    Output(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}
