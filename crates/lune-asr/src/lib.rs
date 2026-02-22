mod mel;
mod openai_backend;
mod parakeet_backend;
pub mod segment;
mod tdt_decoder;
mod vocab;

pub use openai_backend::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
pub use parakeet_backend::{ParakeetTdtBackend, ParakeetTdtConfig};
