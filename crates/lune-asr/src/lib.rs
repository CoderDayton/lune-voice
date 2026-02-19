mod backend;
pub mod config;
mod mel;
pub mod model;
mod openai_backend;
mod segment;

pub use backend::{VoxtralConfig, VoxtralLocalBackend};
pub use openai_backend::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
