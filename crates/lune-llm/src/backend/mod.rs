//! LLM backend implementations.

pub mod null;
pub mod openai;

// Re-export public items for API stability
pub use null::NullPostProcessor;
pub use openai::{OpenAiLlmBackend, OpenAiLlmConfig};
