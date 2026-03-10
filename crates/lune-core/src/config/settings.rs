use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use url::Url;

use super::hotkey::HotkeyConfig;
use super::modes::{ActivationMode, DictationMode, PrivacyMode};
use super::pipeline::PipelineOverrides;
use super::redacted::RedactedString;
use super::style::StyleId;

/// Configuration for the optional forced-aligner pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignerConfig {
    pub enabled: bool,
    /// Directory containing `model.onnx` and `fa_config.json`.
    pub model_dir: PathBuf,
}

/// Top-level user settings, persisted as TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    /// Compute device for local inference: `Cpu`, `Cuda`, or `Auto`.
    pub device: DevicePolicy,
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
    /// Per-word confidence threshold for LLM gating.
    /// **Deprecated**: use `pipeline.word_confidence_threshold` instead.
    /// Migrated automatically when `[pipeline]` section is absent.
    #[serde(default = "default_word_confidence_threshold")]
    pub word_confidence_threshold: f32,
    /// Max speech duration in seconds before auto-ending session.
    /// **Deprecated**: use `pipeline.max_speech_secs` instead.
    #[serde(default = "default_max_speech_secs")]
    pub max_speech_secs: u32,
    /// Optional forced-aligner for post-utterance word-order correction.
    /// When `enabled = true`, the aligner runs after each silence flush and
    /// emits `CorrectedText` if words were pasted out of order.
    #[serde(default)]
    pub aligner: Option<AlignerConfig>,
    /// Enable adaptive VAD silence thresholds.
    /// **Deprecated**: use `pipeline.adaptive_vad` instead.
    #[serde(default)]
    pub adaptive_vad: bool,
    /// Per-backend pipeline overrides. All fields are optional;
    /// `None` inherits the active backend's recommended default.
    #[serde(default)]
    pub pipeline: PipelineOverrides,
}

impl UserSettings {
    /// Migrate legacy top-level pipeline fields into `PipelineOverrides`.
    ///
    /// Called once after deserialization. If the `[pipeline]` section is
    /// empty (all `None`), legacy fields are promoted into it. If the
    /// user has already set `[pipeline]` fields, legacy fields are ignored.
    pub fn migrate_pipeline_fields(&mut self) {
        if self.pipeline != PipelineOverrides::default() {
            return; // User has explicit [pipeline] section — don't overwrite.
        }
        if self.word_confidence_threshold != default_word_confidence_threshold() {
            self.pipeline.word_confidence_threshold = Some(self.word_confidence_threshold);
        }
        if self.max_speech_secs != default_max_speech_secs() {
            self.pipeline.max_speech_secs = Some(self.max_speech_secs);
        }
        if self.adaptive_vad {
            self.pipeline.adaptive_vad = Some(true);
        }
    }
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            device: DevicePolicy::Auto,
            asr_backend: AsrBackendConfig::OrtQwenAsr {
                model_dir: PathBuf::new(),
            },
            llm_backend: LlmBackendConfig::LlamaCppLocal {
                model_path: PathBuf::new(),
                n_gpu_layers: default_n_gpu_layers(),
                n_ctx: default_n_ctx(),
                max_output_tokens: default_llm_max_output_tokens(),
            },
            hotkey: HotkeyConfig::default(),
            default_style: "casual".into(),
            language: "en-US".into(),
            privacy_mode: PrivacyMode::LocalOnly,
            dictation_mode: DictationMode::default(),
            activation_mode: ActivationMode::default(),
            session_history_days: Some(30),
            word_confidence_threshold: default_word_confidence_threshold(),
            max_speech_secs: default_max_speech_secs(),
            aligner: None,
            adaptive_vad: false,
            pipeline: PipelineOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AsrBackendConfig {
    ParakeetTdt {
        model_dir: PathBuf,
    },
    OrtQwenAsr {
        model_dir: PathBuf,
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
    LlamaCppLocal {
        model_path: PathBuf,
        #[serde(default = "default_n_gpu_layers")]
        n_gpu_layers: u32,
        #[serde(default = "default_n_ctx")]
        n_ctx: u32,
        #[serde(default = "default_llm_max_output_tokens")]
        max_output_tokens: u32,
    },
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        max_tokens: u32,
    },
}

fn default_n_gpu_layers() -> u32 {
    1000
}
fn default_n_ctx() -> u32 {
    2048
}
fn default_llm_max_output_tokens() -> u32 {
    256
}

fn default_word_confidence_threshold() -> f32 {
    0.65
}

fn default_max_speech_secs() -> u32 {
    60
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamingMode {
    WebSocket,
    HttpPolling,
}

/// Compute device policy for local model inference.
///
/// Serializes as lowercase TOML strings: `"auto"`, `"cuda"`, `"cpu"`.
/// The default is `"auto"`, which tries CUDA at runtime and falls back to CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DevicePolicy {
    /// Force CPU-only inference.
    Cpu,
    /// Prefer CUDA GPU inference. Falls back to CPU if CUDA is not available
    /// at runtime or the `cuda` feature was not compiled in.
    Cuda,
    /// Try CUDA when compiled with the `cuda` feature; fall back to CPU.
    /// This is the recommended setting: zero-config on GPU machines, safe on CPU-only.
    #[default]
    Auto,
}

impl DevicePolicy {
    /// Resolve to a boolean for backend `use_gpu` fields.
    ///
    /// - `Cpu` → always `false`
    /// - `Cuda` → always `true` (runtime init still falls back if CUDA unavailable)
    /// - `Auto` → `true` when `cuda` feature is compiled in (`cuda_available`),
    ///   otherwise `false`. The `cuda` feature is enabled by default.
    #[inline]
    pub fn use_gpu(self, cuda_available: bool) -> bool {
        match self {
            Self::Cpu => false,
            Self::Cuda => true,
            Self::Auto => cuda_available,
        }
    }
}
