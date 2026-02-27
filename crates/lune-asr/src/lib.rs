pub mod mel;
mod openai_backend;
#[cfg(feature = "ort-qwen")]
mod ort_forced_aligner;
#[cfg(feature = "ort-qwen")]
mod ort_qwen_backend;
#[cfg(feature = "ort-qwen")]
pub mod ort_qwen_stream;
pub mod ort_utils;
mod parakeet_backend;
#[cfg(feature = "legacy-qwen")]
mod qwen_backend;
mod tdt_decoder;
mod vocab;

pub use mel::extract_features;
pub use openai_backend::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
#[cfg(feature = "ort-qwen")]
pub use ort_forced_aligner::{OrtForcedAligner, WordTimestamp};
#[cfg(feature = "ort-qwen")]
pub use ort_qwen_backend::{OrtQwenAsrBackend, OrtQwenAsrConfig};
#[cfg(feature = "ort-qwen")]
pub use ort_qwen_stream::compute_qwen_mel;
pub use parakeet_backend::{ParakeetTdtBackend, ParakeetTdtConfig};
#[cfg(feature = "legacy-qwen")]
pub use qwen_backend::{QwenAsrBackend, QwenAsrConfig};
