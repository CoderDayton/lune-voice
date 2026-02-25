use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use url::Url;

use super::hotkey::HotkeyConfig;
use super::modes::{ActivationMode, DictationMode, PrivacyMode};
use super::redacted::RedactedString;
use super::style::StyleId;

/// Top-level user settings, persisted as TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    pub asr_backend: AsrBackendConfig,
    pub llm_backend: LlmBackendConfig,
    pub hotkey: HotkeyConfig,
    pub default_style: StyleId,
    /// BCP-47 tag, e.g. "en-US"
    pub language: String,
    pub privacy_mode: PrivacyMode,
    pub dictation_mode: DictationMode,
    pub activation_mode: ActivationMode,
    /// None = session history disabled
    pub session_history_days: Option<u32>,
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            asr_backend: AsrBackendConfig::OrtQwenAsr {
                model_dir: PathBuf::new(),
                use_gpu: true,
                stream_chunk_sec: default_qwen_stream_chunk_sec(),
                stream_rollback: default_qwen_stream_rollback(),
                stream_unfixed_chunks: default_qwen_stream_unfixed_chunks(),
                stream_max_new_tokens: default_qwen_stream_max_new_tokens(),
                past_text_conditioning: default_qwen_past_text_conditioning(),
            },
            llm_backend: LlmBackendConfig::OrtLocal {
                model_dir: PathBuf::new(),
                use_gpu: true,
                n_ctx: 2048,
                max_output_tokens: 256,
                temperature: 0.0,
                top_p: 0.8,
                top_k: 20,
                repeat_penalty: 1.0,
                repeat_last_n: 64,
            },
            hotkey: HotkeyConfig::default(),
            default_style: "general".into(),
            language: "en-US".into(),
            privacy_mode: PrivacyMode::LocalOnly,
            dictation_mode: DictationMode::default(),
            activation_mode: ActivationMode::default(),
            session_history_days: Some(30),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AsrBackendConfig {
    ParakeetTdt {
        model_dir: PathBuf,
        use_gpu: bool,
    },
    QwenAsr {
        model_dir: PathBuf,
        /// Streaming chunk size in seconds for `qwen-asr`.
        #[serde(default = "default_qwen_stream_chunk_sec")]
        stream_chunk_sec: f32,
        /// Number of rollback tokens kept for boundary stability.
        #[serde(default = "default_qwen_stream_rollback")]
        stream_rollback: i32,
        /// Number of initial "unfixed" chunks before stable commit.
        #[serde(default = "default_qwen_stream_unfixed_chunks")]
        stream_unfixed_chunks: i32,
        /// Decoder cap per stream update.
        #[serde(default = "default_qwen_stream_max_new_tokens")]
        stream_max_new_tokens: i32,
        /// Whether stream chunks are conditioned on prior committed text.
        #[serde(default = "default_qwen_past_text_conditioning")]
        past_text_conditioning: bool,
    },
    OrtQwenAsr {
        model_dir: PathBuf,
        use_gpu: bool,
        /// Streaming chunk size in seconds.
        #[serde(default = "default_qwen_stream_chunk_sec")]
        stream_chunk_sec: f32,
        /// Number of rollback tokens kept for boundary stability.
        #[serde(default = "default_qwen_stream_rollback")]
        stream_rollback: i32,
        /// Number of initial "unfixed" chunks before stable commit.
        #[serde(default = "default_qwen_stream_unfixed_chunks")]
        stream_unfixed_chunks: i32,
        /// Decoder cap per stream update.
        #[serde(default = "default_qwen_stream_max_new_tokens")]
        stream_max_new_tokens: i32,
        /// Whether stream chunks are conditioned on prior committed text.
        #[serde(default = "default_qwen_past_text_conditioning")]
        past_text_conditioning: bool,
    },
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        streaming: StreamingMode,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum LlmBackendConfig {
    OrtLocal {
        model_dir: PathBuf,
        use_gpu: bool,
        #[serde(default = "default_n_ctx")]
        n_ctx: u32,
        #[serde(default = "default_max_output_tokens")]
        max_output_tokens: u32,
        #[serde(default)]
        temperature: f32,
        #[serde(default = "default_top_p")]
        top_p: f32,
        #[serde(default = "default_top_k")]
        top_k: i32,
        #[serde(default = "default_repeat_penalty")]
        repeat_penalty: f32,
        #[serde(default = "default_repeat_last_n")]
        repeat_last_n: i32,
    },
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        max_tokens: u32,
    },
}

// Serde default functions for LlmBackendConfig::Local fields.
fn default_n_ctx() -> u32 {
    2048
}
fn default_max_output_tokens() -> u32 {
    256
}
fn default_top_p() -> f32 {
    0.8
}
fn default_top_k() -> i32 {
    20
}
fn default_repeat_penalty() -> f32 {
    1.15
}
fn default_repeat_last_n() -> i32 {
    64
}

fn default_qwen_stream_chunk_sec() -> f32 {
    0.5
}

fn default_qwen_stream_rollback() -> i32 {
    5
}

fn default_qwen_stream_unfixed_chunks() -> i32 {
    2
}

fn default_qwen_stream_max_new_tokens() -> i32 {
    32
}

fn default_qwen_past_text_conditioning() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamingMode {
    WebSocket,
    HttpPolling,
}
