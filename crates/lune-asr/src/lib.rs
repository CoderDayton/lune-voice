pub mod confidence;
pub mod confidence_calibrator;
mod lcs_merge;
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
mod registry;
mod tdt_decoder;
mod vocab;

pub use confidence_calibrator::IsotonicCalibrator;
pub use mel::extract_features;
pub use openai_backend::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend, encode_wav_16bit};
#[cfg(feature = "ort-qwen")]
pub use ort_forced_aligner::{OrtForcedAligner, WordTimestamp};
#[cfg(feature = "ort-qwen")]
pub use ort_qwen_backend::{OrtQwenAsrBackend, OrtQwenAsrConfig};
#[cfg(feature = "ort-qwen")]
pub use ort_qwen_stream::compute_qwen_mel;
pub use parakeet_backend::{ParakeetTdtBackend, ParakeetTdtConfig};
pub use registry::{AsrRegistry, AsrSessionHandle};
