//! ASR backend implementations — runtime-dispatched inference engines.

pub mod openai;
pub mod registry;
pub mod whisper_cpp;

pub use openai::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
pub use registry::{AsrRegistry, AsrSessionHandle};
pub use whisper_cpp::{WhisperCppBackend, WhisperCppConfig};
