//! ASR backend implementations — runtime-dispatched inference engines.

pub mod openai;
pub mod parakeet;
pub mod qwen3;
pub mod registry;

pub use openai::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
pub use parakeet::{ParakeetTdtBackend, ParakeetTdtConfig};
pub use qwen3::{Qwen3AsrBackend, Qwen3AsrConfig};
pub use registry::{AsrRegistry, AsrSessionHandle};
