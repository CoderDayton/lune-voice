use serde::{Deserialize, Serialize};
use url::Url;

use super::hotkey::HotkeyConfig;
use super::modes::{ActivationMode, DictationMode, PillVisibility, PrivacyMode, TerminalMode};
use super::pipeline::PipelineOverrides;
use super::redacted::RedactedString;
use super::style_llm::{StyleId, StylePreset};

/// Deserialize `dictation_mode`, tolerating the removed legacy `"Both"` value
/// (the old "type and copy" mode) by mapping it to `TypeInPlace` via `FromStr`.
/// The copy behaviour is restored separately by the `copy_to_clipboard`
/// migration in `lune-persist`, so old configs neither break nor silently lose
/// their intent.
fn deserialize_dictation_mode<'de, D>(d: D) -> Result<DictationMode, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    s.parse().map_err(serde::de::Error::custom)
}

/// Top-level user settings, persisted as TOML.
///
/// # Layering rule (where does a new field belong?)
///
/// A new setting belongs in **`pipeline: PipelineOverrides`** (and the
/// underlying `PipelineConfig`) when it is a *runtime tuning knob* whose
/// sensible default depends on the active ASR backend — e.g. silence
/// thresholds, segment limits, confidence gating. These flow through
/// `AsrRegistry::pipeline_config().with_overrides(&user.pipeline)` so each
/// backend can ship its own defaults and the user can override them.
///
/// A new setting belongs **here at top level** when it is a *binary
/// feature toggle or session-level preference* that the session reads
/// directly at start time and that has no per-backend default —
/// e.g. `mic_prewarm`, `vad_pcen`, hotkey config, language. These bypass
/// the pipeline merge and are read straight from `UserSettings` in
/// `SessionManager` / `AppRuntime`.
///
/// When in doubt: if Qwen3 and OpenAI would want different defaults, it
/// belongs in `PipelineOverrides`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    /// Compute device for local inference: `Cpu`, `Cuda`, or `Auto`.
    pub device: DevicePolicy,
    pub asr_backend: AsrBackendConfig,
    #[serde(default)]
    pub llm_backend: LlmBackendConfig,
    pub hotkey: HotkeyConfig,
    pub default_style: StyleId,
    #[serde(default)]
    pub custom_styles: Vec<StylePreset>,
    /// BCP-47 tag, e.g. "en-US"
    pub language: String,
    /// Dictation language for ASR/correction. "auto" = detect from speech.
    #[serde(default = "default_dictation_language")]
    pub dictation_language: String,
    pub privacy_mode: PrivacyMode,
    #[serde(deserialize_with = "deserialize_dictation_mode")]
    pub dictation_mode: DictationMode,
    /// Keep the dictated text on the clipboard after delivery. Orthogonal to
    /// `dictation_mode` (works with either Paste or Type). Off by default.
    #[serde(default)]
    pub copy_to_clipboard: bool,
    /// Play short sound cues when dictation starts, finishes, or errors.
    #[serde(default = "default_true")]
    pub sound_cues_enabled: bool,
    /// Sound cue playback volume in [0.0, 1.0].
    #[serde(default = "default_sound_cue_volume")]
    pub sound_cue_volume: f32,
    pub activation_mode: ActivationMode,
    /// When to show the floating dictation pill overlay.
    #[serde(default)]
    pub pill_visibility: PillVisibility,
    /// How to detect terminal windows for paste chord selection.
    /// `Auto` probes the compositor (works on Hyprland, Sway, KDE, GNOME).
    /// Set `Terminal` on COSMIC or other compositors without focus detection.
    #[serde(default)]
    pub terminal_mode: TerminalMode,
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
    /// Enable adaptive VAD silence thresholds.
    /// **Deprecated**: use `pipeline.adaptive_vad` instead.
    #[serde(default)]
    pub adaptive_vad: bool,
    /// Per-backend pipeline overrides. All fields are optional;
    /// `None` inherits the active backend's recommended default.
    #[serde(default)]
    pub pipeline: PipelineOverrides,
    /// Keep the microphone input stream open across sessions so the first
    /// chunk arrives ~70 ms sooner after a hotkey press. The mic remains
    /// powered/visible to the OS while the app runs; turn this off if you
    /// prefer the indicator only appearing during dictation.
    #[serde(default = "default_true")]
    pub mic_prewarm: bool,
    /// Preferred microphone input device, by name. `None` (default) uses the
    /// system default input. A saved device that is no longer present falls
    /// back to the default at capture-open time — never fatal.
    #[serde(default)]
    pub input_device: Option<String>,
    /// Enable the PCEN-style adaptive AGC frontend before the VAD.
    /// Improves robustness on quiet mics and noisy environments. Default on.
    #[serde(default = "default_true")]
    pub vad_pcen: bool,
    /// Launch the desktop app with the main window hidden to the system
    /// tray instead of shown. Useful when Lune is autostarted on login.
    #[serde(default)]
    pub start_minimized: bool,
}

fn default_true() -> bool {
    true
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
            asr_backend: AsrBackendConfig::Qwen3Asr {},
            llm_backend: LlmBackendConfig::Disabled,
            hotkey: HotkeyConfig::default(),
            default_style: "minimal".into(),
            custom_styles: Vec::new(),
            language: "auto".into(),
            dictation_language: default_dictation_language(),
            privacy_mode: PrivacyMode::LocalOnly,
            dictation_mode: DictationMode::default(),
            copy_to_clipboard: false,
            sound_cues_enabled: true,
            sound_cue_volume: 1.0,
            activation_mode: ActivationMode::default(),
            pill_visibility: PillVisibility::default(),
            terminal_mode: TerminalMode::default(),
            session_history_days: Some(30),
            word_confidence_threshold: default_word_confidence_threshold(),
            max_speech_secs: default_max_speech_secs(),
            adaptive_vad: false,
            pipeline: PipelineOverrides::default(),
            mic_prewarm: true,
            input_device: None,
            vad_pcen: true,
            start_minimized: false,
        }
    }
}

/// Which whisper.cpp GGML model the local Whisper backend loads.
///
/// This is the single source of truth tying a user-facing selection to its
/// on-disk directory, its HuggingFace filename, and its display name; both the
/// downloader (`lune-persist`) and the backend loader (`lune-asr`) read it so
/// the mapping never drifts. All three are the multilingual q8_0 quantizations
/// from `ggerganov/whisper.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum WhisperModel {
    /// ~264 MB. Fast, lowest accuracy of the three.
    Small,
    /// ~823 MB. Balanced accuracy/speed.
    #[default]
    Medium,
    /// ~874 MB. Best accuracy; turbo decoder keeps it fast.
    LargeV3Turbo,
}

impl WhisperModel {
    /// The HuggingFace repo hosting every GGML whisper model.
    pub const REPO_ID: &'static str = "ggerganov/whisper.cpp";

    /// Every selectable model, in ascending size order. Drives the downloader's
    /// spec list and the settings selector.
    pub const ALL: [WhisperModel; 3] = [Self::Small, Self::Medium, Self::LargeV3Turbo];

    /// Local `models/` subdirectory holding this model's GGML file.
    pub fn dir(self) -> &'static str {
        match self {
            Self::Small => crate::models::DIR_WHISPER_SMALL,
            Self::Medium => crate::models::DIR_WHISPER_MEDIUM,
            Self::LargeV3Turbo => crate::models::DIR_WHISPER_LARGE_V3_TURBO,
        }
    }

    /// Filename to download from [`WhisperModel::REPO_ID`] and to load from disk.
    pub fn ggml_file(self) -> &'static str {
        match self {
            Self::Small => "ggml-small-q8_0.bin",
            Self::Medium => "ggml-medium-q8_0.bin",
            Self::LargeV3Turbo => "ggml-large-v3-turbo-q8_0.bin",
        }
    }

    /// Human-readable name for UI and model-status cards.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Small => "Whisper Small (q8_0)",
            Self::Medium => "Whisper Medium (q8_0)",
            Self::LargeV3Turbo => "Whisper Large v3 Turbo (q8_0)",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AsrBackendConfig {
    Qwen3Asr {},
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        streaming: StreamingMode,
    },
    /// Local whisper.cpp inference via the bundled GGML model `model`.
    WhisperCpp {
        #[serde(default)]
        model: WhisperModel,
    },
}

impl AsrBackendConfig {
    /// Return the [`AsrBackendKind`] discriminant for this config.
    pub fn kind(&self) -> crate::asr::AsrBackendKind {
        use crate::asr::AsrBackendKind;
        match self {
            Self::Qwen3Asr { .. } => AsrBackendKind::Qwen3Asr,
            Self::OpenAiCompatible { .. } => AsrBackendKind::OpenAiCompatible,
            Self::WhisperCpp { .. } => AsrBackendKind::WhisperCpp,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum LlmBackendConfig {
    /// Use the local ONNX style LLM (Qwen3 0.6B) if downloaded.
    Local {},
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        max_tokens: u32,
    },
    #[serde(other)]
    Disabled,
}

impl Default for LlmBackendConfig {
    fn default() -> Self {
        Self::Local {}
    }
}

fn default_dictation_language() -> String {
    "auto".into()
}

fn default_sound_cue_volume() -> f32 {
    1.0
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
    pub fn use_gpu(self, cuda_available: bool) -> bool {
        match self {
            Self::Cpu => false,
            Self::Cuda => true,
            Self::Auto => cuda_available,
        }
    }

    /// True when the user explicitly chose CUDA. ORT EP registration should
    /// fail loudly in this case instead of silently falling back to CPU —
    /// the surrounding code is expected to forward this to
    /// `ep::CUDA::default().build().error_on_failure()`.
    pub fn strict_cuda(self) -> bool {
        matches!(self, Self::Cuda)
    }
}
