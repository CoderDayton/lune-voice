//! ASR backend implementations — runtime-dispatched inference engines.

pub mod openai;
pub mod registry;
pub mod whisper_cpp;

pub use openai::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend};
pub use registry::{AsrRegistry, AsrSessionHandle};
pub use whisper_cpp::{
    WhisperCppBackend, WhisperCppConfig, check_ggml_file, install_logging_hooks,
};
