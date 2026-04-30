//! ASR backend implementations — runtime-dispatched inference engines.

pub mod openai;
pub mod qwen3;
pub mod registry;

pub use openai::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
pub use qwen3::{Qwen3AsrBackend, Qwen3AsrConfig, Qwen3AsrProfile};
pub use registry::{AsrRegistry, AsrSessionHandle};
