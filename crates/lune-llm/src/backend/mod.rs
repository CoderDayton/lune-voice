//! LLM backend implementations.

pub mod null;
pub mod openai;
pub mod ort_style;

// Re-export public items for API stability
pub use null::NullPostProcessor;
pub use openai::{OpenAiLlmBackend, OpenAiLlmConfig};
pub use ort_style::{OrtStyleLlm, OrtStyleLlmConfig};
