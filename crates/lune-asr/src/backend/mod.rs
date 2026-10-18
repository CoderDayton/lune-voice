// SPDX-License-Identifier: GPL-3.0-or-later

//! ASR backend implementations — runtime-dispatched inference engines.

mod local;
pub mod openai;
pub mod parakeet_cpp;
pub mod registry;
pub mod whisper_cpp;

pub use local::{check_ggml_file, install_logging_hooks};
pub use openai::{OpenAiAsrConfig, OpenAiCompatibleAsrBackend};
pub use parakeet_cpp::{ParakeetCppBackend, ParakeetCppConfig};
pub use registry::{AsrRegistry, AsrSessionHandle};
pub use whisper_cpp::{LeadIn, WhisperCppBackend, WhisperCppConfig};
