//! LLM backend implementations.

pub mod llama_cpp;
pub mod null;
pub mod openai;

// Re-export public items for API stability
pub use llama_cpp::{LlamaCppBackend, LlamaCppConfig};
pub use null::NullPostProcessor;
pub use openai::{OpenAiLlmBackend, OpenAiLlmConfig};
