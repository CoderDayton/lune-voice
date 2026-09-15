// SPDX-License-Identifier: GPL-3.0-or-later

pub mod backend;
pub mod few_shot;
pub mod pipeline;
pub mod prompt;

pub use backend::{NullPostProcessor, OpenAiLlmBackend, OpenAiLlmConfig};
pub use pipeline::StylePipeline;
