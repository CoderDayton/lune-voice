// SPDX-License-Identifier: GPL-3.0-or-later

//! LLM backend implementations.

pub mod null;
pub mod openai;

// Re-export public items for API stability
pub use null::NullPostProcessor;
pub use openai::{OpenAiLlmBackend, OpenAiLlmConfig};
