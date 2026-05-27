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

impl Error {
    pub fn audio(context: impl Into<String>) -> Self {
        Self::Audio(context.into())
    }
    pub fn asr(context: impl Into<String>) -> Self {
        Self::Asr(context.into())
    }
    pub fn llm(context: impl Into<String>) -> Self {
        Self::Llm(context.into())
    }
    pub fn persistence(context: impl Into<String>) -> Self {
        Self::Persistence(context.into())
    }
    pub fn config(context: impl Into<String>) -> Self {
        Self::Config(context.into())
    }
    pub fn session(context: impl Into<String>) -> Self {
        Self::Session(context.into())
    }
    pub fn command(context: impl Into<String>) -> Self {
        Self::Command(context.into())
    }
    pub fn output(context: impl Into<String>) -> Self {
        Self::Output(context.into())
    }
    pub fn other(context: impl Into<String>) -> Self {
        Self::Other(context.into())
    }
    pub fn model_not_found(path: impl Into<PathBuf>) -> Self {
        Self::ModelNotFound(path.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_constructor_renders() {
        let err = Error::asr("missing model");
        assert_eq!(err.to_string(), "ASR error: missing model");
    }

    #[test]
    fn model_not_found_renders_path() {
        let err = Error::model_not_found("/tmp/missing.onnx");
        assert!(err.to_string().contains("/tmp/missing.onnx"));
    }
}
