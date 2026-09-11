pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("ASR error: {0}")]
    Asr(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("session error: {0}")]
    Session(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

macro_rules! string_constructors {
    ($($method:ident => $variant:ident),+ $(,)?) => {
        impl Error {
            $(
                #[doc = concat!("Construct an [`Error::", stringify!($variant), "`] from a context message.")]
                pub fn $method(context: impl Into<String>) -> Self {
                    Self::$variant(context.into())
                }
            )+
        }
    };
}

string_constructors! {
    asr => Asr,
    llm => Llm,
    config => Config,
    session => Session,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_constructor_renders() {
        let err = Error::asr("missing model");
        assert_eq!(err.to_string(), "ASR error: missing model");
    }
}
