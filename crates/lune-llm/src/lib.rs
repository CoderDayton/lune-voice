mod backend;
mod openai_backend;
mod prompt;

pub use backend::{LocalLlmBackend, LocalLlmConfig};
pub use openai_backend::{OpenAiLlmBackend, OpenAiLlmConfig};
