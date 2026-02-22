mod openai_backend;
mod parakeet_backend;
mod segment;

pub use openai_backend::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
pub use parakeet_backend::{ParakeetTdtBackend, ParakeetTdtConfig};
