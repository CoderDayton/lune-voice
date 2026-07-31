use serde::{Deserialize, Serialize};
use url::Url;

use super::hotkey::HotkeyConfig;
use super::modes::{ActivationMode, DictationMode, PillVisibility, PrivacyMode, TerminalMode};
use super::pipeline::PipelineOverrides;
use super::redacted::RedactedString;
use super::style_llm::StyleId;

/// Persisted pill overlay position, in logical pixels from the monitor's
/// top-left corner. Stored when the user drags the pill so it reopens where
/// they left it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PillPosition {
    pub x: i32,
    pub y: i32,
}

/// Where the pill's *default* placement anchors on the monitor. Only applies
/// while the user hasn't dragged it (`pill_position: None`); picking a new
/// anchor clears any dragged position so the change is visible immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum PillAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    BottomLeft,
    #[default]
    BottomCenter,
    BottomRight,
}

/// Overall footprint of the pill bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum PillSize {
    Compact,
    #[default]
    Standard,
    Large,
}

/// Long-axis direction of the pill bar. `Vertical` swaps width and height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum PillOrientation {
    #[default]
    Horizontal,
    Vertical,
}

/// Corner rounding of the pill bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum PillCorner {
    /// Fully round end caps (radius = half the short side).
    Pill,
    /// Softly rounded rectangle (the original look).
    #[default]
    Rounded,
    /// Nearly-square corners.
    Square,
}

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
/// When in doubt: if two backends would want different defaults, it
/// belongs in `PipelineOverrides`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    /// Compute device for local inference: `Cpu`, `Cuda`, or `Auto`.
    pub device: DevicePolicy,
    pub asr_backend: AsrBackendConfig,
    #[serde(default)]
    pub llm_backend: LlmBackendConfig,
    /// Saved remote styling-LLM provider, kept independently of `llm_backend`
    /// so the endpoint/key/model persist across turning styling off or
    /// switching engines. Applied when the active backend is `OpenAiCompatible`.
    #[serde(default)]
    pub llm_openai: OpenAiLlmProvider,
    pub hotkey: HotkeyConfig,
    /// Selected styling preset id, or `None` for no styling (raw transcript).
    /// Legacy configs storing the removed `"minimal"` sentinel migrate to `None`.
    #[serde(default)]
    pub default_style: Option<StyleId>,
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
    /// Persisted pill overlay position (logical px from the monitor's
    /// top-left). `None` = default anchor placement; set when the user
    /// drags the pill.
    #[serde(default)]
    pub pill_position: Option<PillPosition>,
    /// Default placement anchor for the pill when it hasn't been dragged.
    #[serde(default)]
    pub pill_anchor: PillAnchor,
    /// Prevent repositioning the pill by dragging.
    #[serde(default)]
    pub pill_locked: bool,
    /// Gap (logical px) between the pill and the screen edge in its default
    /// anchor placement — clears docks/taskbars.
    #[serde(default = "default_pill_edge_margin")]
    pub pill_edge_margin: u32,
    /// Pin the pill to a specific monitor by connector name (e.g. "DP-1").
    /// `None` = automatic (current/primary monitor).
    #[serde(default)]
    pub pill_monitor: Option<String>,
    /// Show the pill on every workspace/virtual desktop instead of only the
    /// one it was opened on.
    #[serde(default = "default_true")]
    pub pill_all_workspaces: bool,
    /// Footprint of the pill bar.
    #[serde(default)]
    pub pill_size: PillSize,
    /// Long-axis direction of the pill bar.
    #[serde(default)]
    pub pill_orientation: PillOrientation,
    /// Corner rounding of the pill bar.
    #[serde(default)]
    pub pill_corner: PillCorner,
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
        // The "minimal" preset (a styling-off sentinel) was removed; "off" is now
        // the absence of a selection. Map the legacy value to `None`.
        if self.default_style.as_deref() == Some("minimal") {
            self.default_style = None;
        }
        // Back-fill the saved provider from a remote backend that was configured
        // before `llm_openai` existed, so it survives being toggled off later.
        if let LlmBackendConfig::OpenAiCompatible {
            base_url,
            api_key,
            model,
            max_tokens,
        } = &self.llm_backend
        {
            self.llm_openai = OpenAiLlmProvider {
                base_url: base_url.to_string(),
                api_key: api_key.clone(),
                model: model.clone(),
                max_tokens: *max_tokens,
            };
        }
    }
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            device: DevicePolicy::Auto,
            asr_backend: AsrBackendConfig::WhisperCpp {
                model: WhisperModel::default(),
            },
            llm_backend: LlmBackendConfig::Disabled,
            llm_openai: OpenAiLlmProvider::default(),
            hotkey: HotkeyConfig::default(),
            default_style: None,
            language: "auto".into(),
            dictation_language: default_dictation_language(),
            privacy_mode: PrivacyMode::LocalOnly,
            dictation_mode: DictationMode::default(),
            copy_to_clipboard: false,
            sound_cues_enabled: true,
            sound_cue_volume: 1.0,
            activation_mode: ActivationMode::default(),
            pill_visibility: PillVisibility::default(),
            pill_position: None,
            pill_anchor: PillAnchor::default(),
            pill_locked: false,
            pill_edge_margin: default_pill_edge_margin(),
            pill_monitor: None,
            pill_all_workspaces: true,
            pill_size: PillSize::default(),
            pill_orientation: PillOrientation::default(),
            pill_corner: PillCorner::default(),
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
    /// ~190 MB. Fast, lowest accuracy of the three.
    Small,
    /// ~539 MB. Balanced accuracy/speed.
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
            Self::Small => "ggml-small-q5_1.bin",
            Self::Medium => "ggml-medium-q5_0.bin",
            Self::LargeV3Turbo => "ggml-large-v3-turbo-q8_0.bin",
        }
    }

    /// Human-readable name for UI and model-status cards.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Small => "Whisper Small (q5_1)",
            Self::Medium => "Whisper Medium (q5_0)",
            Self::LargeV3Turbo => "Whisper Large v3 Turbo (q8_0)",
        }
    }

    /// Size/accuracy-specific blurb for model-status cards. All three run
    /// fully on-device via whisper.cpp; audio never leaves the machine.
    pub fn description(self) -> &'static str {
        match self {
            Self::Small => {
                "Fastest, lightest Whisper model (~190 MB). \
                            Lowest accuracy of the three — best for quick \
                            notes or low-end hardware. Runs fully on-device."
            }
            Self::Medium => {
                "Balanced Whisper model (~539 MB). Good accuracy \
                             at moderate speed — the recommended default for \
                             most machines. Runs fully on-device."
            }
            Self::LargeV3Turbo => {
                "Most accurate Whisper model (~874 MB). The \
                                   turbo decoder keeps it fast despite its \
                                   size — best for accents, jargon, or noisy \
                                   audio. Runs fully on-device."
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AsrBackendConfig {
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

/// Forward/backward-compatible deserialize. An unknown `type` tag — notably
/// the removed `"Qwen3Asr"` ONNX backend still written in older
/// `settings.toml` files — maps to the current default (`WhisperCpp`) instead
/// of failing the entire settings load.
impl<'de> Deserialize<'de> for AsrBackendConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "type")]
        enum Repr {
            OpenAiCompatible {
                base_url: Url,
                api_key: RedactedString,
                model: String,
                streaming: StreamingMode,
            },
            WhisperCpp {
                #[serde(default)]
                model: WhisperModel,
            },
            #[serde(other)]
            Unknown,
        }

        Ok(match Repr::deserialize(deserializer)? {
            Repr::OpenAiCompatible {
                base_url,
                api_key,
                model,
                streaming,
            } => AsrBackendConfig::OpenAiCompatible {
                base_url,
                api_key,
                model,
                streaming,
            },
            Repr::WhisperCpp { model } => AsrBackendConfig::WhisperCpp { model },
            Repr::Unknown => AsrBackendConfig::WhisperCpp {
                model: WhisperModel::default(),
            },
        })
    }
}

impl AsrBackendConfig {
    /// Return the [`AsrBackendKind`] discriminant for this config.
    pub fn kind(&self) -> crate::asr::AsrBackendKind {
        use crate::asr::AsrBackendKind;
        match self {
            Self::OpenAiCompatible { .. } => AsrBackendKind::OpenAiCompatible,
            Self::WhisperCpp { .. } => AsrBackendKind::WhisperCpp,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum LlmBackendConfig {
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        max_tokens: u32,
    },
    #[serde(other)]
    #[default]
    Disabled,
}

/// Whether `url` points at the user's own machine or network rather than a
/// third-party service.
///
/// This is the line an API key is protecting. A hosted provider needs one, and
/// sending transcripts to one without a key means shipping them off the machine
/// to get a 401 back. A self-hosted OpenAI-compatible server — Ollama,
/// llama-server, LM Studio — usually has no key at all, and requiring one there
/// would block the setup this app is built around.
///
/// Loopback and the RFC 1918 / RFC 4193 private ranges count, as do `.local`
/// and `.localhost` names: a box on the user's LAN is still theirs. Anything
/// that resolves through DNS to the public internet does not.
pub fn is_local_endpoint(url: &Url) -> bool {
    use url::Host;
    match url.host() {
        Some(Host::Domain(name)) => {
            let name = name.trim_end_matches('.').to_ascii_lowercase();
            name == "localhost" || name.ends_with(".localhost") || name.ends_with(".local")
        }
        Some(Host::Ipv4(ip)) => {
            ip.is_loopback() || ip.is_private() || ip.is_link_local() || ip.is_unspecified()
        }
        // `Ipv6Addr::is_unique_local` is still unstable, so match fc00::/7 by
        // hand; `fe80::/10` is link-local.
        Some(Host::Ipv6(ip)) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || (ip.segments()[0] & 0xfe00) == 0xfc00
                || (ip.segments()[0] & 0xffc0) == 0xfe80
        }
        None => false,
    }
}

/// Remote OpenAI-compatible styling-LLM provider settings, persisted
/// independently of the active `llm_backend` so the endpoint/key/model survive
/// turning styling off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiLlmProvider {
    pub base_url: String,
    pub api_key: RedactedString,
    pub model: String,
    pub max_tokens: u32,
}

impl Default for OpenAiLlmProvider {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: RedactedString::new(""),
            model: "gpt-4o-mini".into(),
            max_tokens: 512,
        }
    }
}

fn default_dictation_language() -> String {
    "auto".into()
}

/// Matches the pill's original hard-coded bottom gap so existing installs
/// keep their exact placement.
fn default_pill_edge_margin() -> u32 {
    72
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

#[cfg(test)]
mod tests {
    use super::*;

    fn local(url: &str) -> bool {
        is_local_endpoint(&Url::parse(url).expect("valid url"))
    }

    #[test]
    fn self_hosted_endpoints_are_local() {
        // The setups this app is usually pointed at: Ollama, llama-server,
        // LM Studio — on this machine or elsewhere on the user's own network.
        for url in [
            "http://localhost:11434/v1",
            "http://LOCALHOST:8080/v1",
            "http://127.0.0.1:1234/v1",
            "http://[::1]:8080/v1",
            "http://192.168.1.50:11434/v1",
            "http://10.0.0.4:8080/v1",
            "http://172.16.3.9:8080/v1",
            "http://workstation.local:11434/v1",
            "http://ollama.localhost/v1",
        ] {
            assert!(local(url), "{url} should be local");
        }
    }

    #[test]
    fn hosted_providers_are_not_local() {
        for url in [
            "https://api.openai.com/v1",
            "https://openrouter.ai/api/v1",
            // Close to a private range but routable, so it is a third party.
            "http://172.32.0.1/v1",
            "http://11.0.0.1/v1",
            // A public host whose name merely mentions localhost.
            "https://localhost.example.com/v1",
        ] {
            assert!(!local(url), "{url} should not be local");
        }
    }

    #[test]
    fn legacy_qwen3_asr_backend_migrates_to_whisper_default() {
        // Old persisted configs carry `type = "Qwen3Asr"`. With the ONNX backend
        // removed, deserializing such a config must not fail — it falls back to
        // the current default (`WhisperCpp`) so settings still load.
        let cfg: AsrBackendConfig = serde_json::from_str(r#"{"type":"Qwen3Asr"}"#).unwrap();
        match cfg {
            AsrBackendConfig::WhisperCpp { model } => {
                assert_eq!(model, WhisperModel::default());
            }
            other => panic!("expected WhisperCpp default, got {other:?}"),
        }
    }

    #[test]
    fn whisper_cpp_backend_round_trips() {
        let cfg: AsrBackendConfig =
            serde_json::from_str(r#"{"type":"WhisperCpp","model":"Small"}"#).unwrap();
        match cfg {
            AsrBackendConfig::WhisperCpp { model } => assert_eq!(model, WhisperModel::Small),
            other => panic!("expected WhisperCpp, got {other:?}"),
        }
    }

    #[test]
    fn default_settings_use_whisper_cpp() {
        assert_eq!(
            UserSettings::default().asr_backend.kind(),
            crate::asr::AsrBackendKind::WhisperCpp
        );
    }
}
