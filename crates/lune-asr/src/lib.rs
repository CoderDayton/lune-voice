pub mod mel;
mod openai_backend;
#[cfg(feature = "ort-qwen")]
mod ort_qwen_backend;
#[cfg(feature = "ort-qwen")]
pub(crate) mod ort_qwen_stream;
pub mod ort_utils;
mod parakeet_backend;
#[cfg(feature = "legacy-qwen")]
mod qwen_backend;
mod tdt_decoder;
mod vocab;

pub use mel::extract_features;
pub use openai_backend::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
#[cfg(feature = "ort-qwen")]
pub use ort_qwen_backend::{OrtQwenAsrBackend, OrtQwenAsrConfig};
pub use parakeet_backend::{ParakeetTdtBackend, ParakeetTdtConfig};
#[cfg(feature = "legacy-qwen")]
pub use qwen_backend::{QwenAsrBackend, QwenAsrConfig};
