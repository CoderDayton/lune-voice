// SPDX-License-Identifier: GPL-3.0-or-later

use serde::{Deserialize, Serialize};
use url::Url;

use super::hotkey::HotkeyConfig;
use super::modes::{ActivationMode, DictationMode, PillVisibility, PrivacyMode, TerminalMode};
use super::pipeline::PipelineOverrides;
use super::redacted::RedactedString;
use super::style_llm::StyleId;

/// Persisted pill overlay position, in the desktop's own coordinate space —
/// logical pixels on Linux, device pixels on Windows and macOS, matching what
/// the platform reports for monitor geometry. Stored when the user drags the
/// pill so it reopens where they left it.
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
/// their intent. Any other unrecognised value falls back to the default, for
/// the same reason [`lenient`] exists.
fn deserialize_dictation_mode<'de, D>(d: D) -> Result<DictationMode, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Ok(s.parse().unwrap_or_default())
}

/// Deserialize `T`, falling back to `T::default()` for a value this build does
/// not recognise.
///
/// A settings file written by a newer version can name an enum variant that
/// does not exist here. Serde treats that as a parse failure for the whole
/// file, which costs the user every other setting in it — and, downstream, the
/// history that the restored default retention window then prunes. One unknown
/// field is worth one default, not the file.
fn lenient<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Lenient<T> {
        Known(T),
        Unknown(serde::de::IgnoredAny),
    }

    Ok(match Lenient::<T>::deserialize(deserializer)? {
        Lenient::Known(value) => value,
        Lenient::Unknown(_) => T::default(),
    })
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
/// e.g. `vad_pcen`, `model_idle_unload_minutes`, hotkey config, language.
/// These bypass
/// the pipeline merge and are read straight from `UserSettings` in
/// `SessionManager` / `AppRuntime`.
///
/// When in doubt: if two backends would want different defaults, it
/// belongs in `PipelineOverrides`.
// `serde(default)` at the container level: a settings file written by a build
// that predates a field keeps every value it does carry, instead of failing to
// parse and costing the user every other setting. Writes are atomic
// (`lune-persist::settings::atomic_write`), so a half-written file is not the
// case this covers; a file that is malformed rather than merely incomplete
// still fails to parse and is quarantined by `SettingsStore::load_or_quarantine`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UserSettings {
    /// Compute device for local inference: `Cpu`, `Cuda`, or `Auto`.
    #[serde(deserialize_with = "lenient")]
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
    #[serde(deserialize_with = "lenient")]
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
    #[serde(deserialize_with = "lenient")]
    pub activation_mode: ActivationMode,
    /// When to show the floating dictation pill overlay.
    #[serde(default, deserialize_with = "lenient")]
    pub pill_visibility: PillVisibility,
    /// Persisted pill overlay position — see [`PillPosition`] for the
    /// coordinate space. `None` = default anchor placement; set when the user
    /// drags the pill.
    #[serde(default)]
    pub pill_position: Option<PillPosition>,
    /// Default placement anchor for the pill when it hasn't been dragged.
    #[serde(default, deserialize_with = "lenient")]
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
    /// Show the pill on every workspace instead of only the one it was opened
    /// on. Linux and macOS; Windows virtual desktops cannot be addressed
    /// per-window, so the flag is ignored there.
    #[serde(default = "default_true")]
    pub pill_all_workspaces: bool,
    /// Footprint of the pill bar.
    #[serde(default, deserialize_with = "lenient")]
    pub pill_size: PillSize,
    /// Long-axis direction of the pill bar.
    #[serde(default, deserialize_with = "lenient")]
    pub pill_orientation: PillOrientation,
    /// Corner rounding of the pill bar.
    #[serde(default, deserialize_with = "lenient")]
    pub pill_corner: PillCorner,
    /// How to detect terminal windows for paste chord selection.
    /// `Auto` probes the compositor (works on Hyprland, Sway, KDE, GNOME).
    /// Set `Terminal` on COSMIC or other compositors without focus detection.
    #[serde(default, deserialize_with = "lenient")]
    pub terminal_mode: TerminalMode,
    /// How long a dictation stays in the history, in days.
    ///
    /// `None` disables history outright: nothing is written for a dictation —
    /// no session row, no transcript, no correction pair — and anything already
    /// stored is cleared when the setting changes. It is not "keep forever";
    /// `Some(0)` is. Decode it with [`UserSettings::history_retention`] rather
    /// than matching on the raw field.
    ///
    /// Field-level `default` on purpose, overriding the container's: TOML has
    /// no null, so `None` is written as an *absent* key. Falling back to
    /// `UserSettings::default()` there would read that absence back as
    /// `Some(30)` and silently re-enable history on the next launch.
    #[serde(default)]
    pub session_history_days: Option<u32>,
    /// Per-backend pipeline overrides. All fields are optional;
    /// `None` inherits the active backend's recommended default.
    #[serde(default)]
    pub pipeline: PipelineOverrides,
    /// Free the loaded ASR model after this many minutes without a dictation,
    /// reloading it on the next hotkey press. `None` (the default) keeps it
    /// resident for the life of the process.
    ///
    /// The model is the process's whole memory footprint — 514 MB for the
    /// default Medium — so unloading gives nearly all of it back, in single-digit
    /// milliseconds. What it costs is on the way back in: a reload is ~160 ms
    /// when the model file is still in the OS page cache and ~750 ms when it is
    /// not (measured on Medium; `cgs bench --env TYPE=modelload`). That is paid
    /// on the next hotkey press, which is why the start path buffers microphone
    /// audio across the reload instead of opening the mic after it.
    #[serde(default)]
    pub model_idle_unload_minutes: Option<u32>,
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
    /// Ask the release endpoint for a newer version at startup. Off by
    /// default: dictation is fully local, so the app should not reach the
    /// network unless the user asks it to.
    #[serde(default)]
    pub check_updates_on_launch: bool,
    /// Apply canonical brand casing to dictated names ("github" -> "GitHub").
    /// A style's own `formatting.brand_casing` overrides this when set.
    #[serde(default = "default_true")]
    pub brand_casing: bool,
}

fn default_true() -> bool {
    true
}

/// What `session_history_days` actually means, decoded once so no caller has
/// to re-interpret the raw field.
///
/// The stored field stays `Option<u32>` for wire and file compatibility, but it
/// carries three choices, and two of them are *not* a prune window. Reading it
/// through this enum is what keeps `Forever` from being handed to
/// `prune_sessions` as "0 days", which would delete everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryRetention {
    /// Don't save history: nothing is written, and anything stored is cleared.
    Off,
    /// Keep every session, never prune.
    Forever,
    /// Keep sessions younger than this many days.
    Days(u32),
}

impl HistoryRetention {
    /// Whether finished dictations are recorded at all.
    pub fn records(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl UserSettings {
    /// Decode `session_history_days` into the choice the user actually made.
    ///
    /// `None` = off, `Some(0)` = keep forever, `Some(n)` = an `n`-day window.
    /// Zero is the sentinel because TOML has no null and the field predates the
    /// third choice; a zero-day window would be meaningless anyway.
    pub fn history_retention(&self) -> HistoryRetention {
        match self.session_history_days {
            None => HistoryRetention::Off,
            Some(0) => HistoryRetention::Forever,
            Some(days) => HistoryRetention::Days(days),
        }
    }

    /// Migrate legacy field spellings after deserialization.
    pub fn migrate_pipeline_fields(&mut self) {
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
            pipeline: PipelineOverrides::default(),
            model_idle_unload_minutes: None,
            input_device: None,
            vad_pcen: true,
            start_minimized: false,
            check_updates_on_launch: false,
            brand_casing: true,
        }
    }
}

/// Which whisper.cpp GGML model the local Whisper backend loads.
///
/// This is the single source of truth tying a user-facing selection to its
/// on-disk directory, its HuggingFace filename, and its display name; both the
/// downloader (`lune-persist`) and the backend loader (`lune-asr`) read it so
/// the mapping never drifts. All are multilingual quantizations from
/// `ggerganov/whisper.cpp`: q8_0 for Base and Large v3 Turbo, q5 for Small and
/// Medium.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum WhisperModel {
    /// ~78 MB. For low-end hardware; lowest accuracy.
    Base,
    /// ~190 MB. Fast, less accurate than Medium.
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
    pub const ALL: [WhisperModel; 4] = [Self::Base, Self::Small, Self::Medium, Self::LargeV3Turbo];

    /// Local `models/` subdirectory holding this model's GGML file.
    pub fn dir(self) -> &'static str {
        match self {
            Self::Base => crate::models::DIR_WHISPER_BASE,
            Self::Small => crate::models::DIR_WHISPER_SMALL,
            Self::Medium => crate::models::DIR_WHISPER_MEDIUM,
            Self::LargeV3Turbo => crate::models::DIR_WHISPER_LARGE_V3_TURBO,
        }
    }

    /// Filename to download from [`WhisperModel::REPO_ID`] and to load from disk.
    pub fn ggml_file(self) -> &'static str {
        match self {
            Self::Base => "ggml-base-q8_0.bin",
            Self::Small => "ggml-small-q5_1.bin",
            Self::Medium => "ggml-medium-q5_0.bin",
            Self::LargeV3Turbo => "ggml-large-v3-turbo-q8_0.bin",
        }
    }

    /// Human-readable name for UI and model-status cards.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Base => "Whisper Base (q8_0)",
            Self::Small => "Whisper Small (q5_1)",
            Self::Medium => "Whisper Medium (q5_0)",
            Self::LargeV3Turbo => "Whisper Large v3 Turbo (q8_0)",
        }
    }

    /// Size/accuracy-specific blurb for model-status cards. All of them run
    /// fully on-device via whisper.cpp; audio never leaves the machine.
    pub fn description(self) -> &'static str {
        match self {
            Self::Base => {
                "Smallest Whisper model (~78 MB). Lowest accuracy — \
                 for older or low-power computers where the larger \
                 models are too slow. Runs fully on-device."
            }
            Self::Small => {
                "Fast, light Whisper model (~190 MB). Less accurate \
                            than Medium — good for quick notes on modest \
                            hardware. Runs fully on-device."
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
                #[serde(default, deserialize_with = "lenient")]
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
/// that resolves through DNS to the public internet does not, and neither does
/// a link-local or unspecified address.
///
/// # Tradeoff: this also gates TLS
///
/// Both remote backends (`lune-asr` and `lune-llm`) reuse this test to decide
/// whether plaintext `http://` is acceptable, so "local" here means the API key
/// and every dictated transcript may cross the user's LAN in cleartext. That is
/// deliberate: self-hosted inference servers ship without TLS, and demanding
/// HTTPS would break the LAN-GPU-box setup this app is built around. The
/// exposure is bounded by the LAN — anyone able to sniff it is already inside
/// the user's network — and it never applies to a public host, which still
/// requires HTTPS. Narrowing this to loopback-only is a deliberate posture
/// change, not a bug fix: it would refuse ASR at construction and silently
/// degrade styling to passthrough for anyone running a model on another box.
pub fn is_local_endpoint(url: &Url) -> bool {
    use url::Host;
    match url.host() {
        Some(Host::Domain(name)) => {
            let name = name.trim_end_matches('.').to_ascii_lowercase();
            name == "localhost" || name.ends_with(".localhost") || name.ends_with(".local")
        }
        // Link-local is deliberately excluded: `169.254.0.0/16` carries the
        // cloud metadata address `169.254.169.254`, and an autoconfigured
        // address is not the deliberate "box on my LAN" this carve-out exists
        // for. The unspecified address is not a destination at all.
        Some(Host::Ipv4(ip)) => ip.is_loopback() || ip.is_private(),
        // `Ipv6Addr::is_unique_local` is still unstable, so match fc00::/7 by
        // hand. `fe80::/10` is excluded for the same reason as IPv4 link-local.
        Some(Host::Ipv6(ip)) => ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00,
        None => false,
    }
}

/// Gate a user-configured remote endpoint on TLS, with the local carve-out of
/// [`is_local_endpoint`].
///
/// `what` names the caller in the message (`"ASR"`, `"LLM"`). Returns the
/// rejection message rather than an [`Error`](crate::Error) so each backend
/// keeps its own error variant.
pub fn require_tls_or_local(url: &Url, what: &str) -> Result<(), String> {
    let scheme = url.scheme();
    if scheme == "https" || is_local_endpoint(url) {
        return Ok(());
    }
    Err(format!(
        "OpenAI {what} base_url must use HTTPS (got {scheme}://); \
         plaintext only allowed for loopback and private local addresses"
    ))
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_is_the_smallest_whisper_model_offered() {
        assert_eq!(WhisperModel::ALL[0], WhisperModel::Base);
        assert_eq!(WhisperModel::Base.ggml_file(), "ggml-base-q8_0.bin");
        assert_eq!(WhisperModel::Base.dir(), "whisper-base-q8_0");
    }

    fn local(url: &str) -> bool {
        is_local_endpoint(&Url::parse(url).expect("valid url"))
    }

    /// The accept/reject matrix both remote backends share. Each backend keeps
    /// only a wiring test against its own constructor.
    #[test]
    fn tls_gate_accepts_https_and_local_plaintext_only() {
        let gate = |url: &str| require_tls_or_local(&Url::parse(url).expect("valid url"), "ASR");

        for url in [
            "https://api.example.com/v1",
            "http://127.0.0.1:11434/v1",
            "http://[::1]:8080/v1",
            "http://localhost:1234/v1",
            "http://box.local/v1",
            "http://192.168.1.50:11434/v1",
            "http://10.0.0.7/v1",
        ] {
            assert!(gate(url).is_ok(), "{url} should be allowed");
        }

        for url in [
            "http://api.example.com/v1",
            // Link-local carries the cloud metadata address; not a LAN box.
            "http://169.254.169.254/v1",
        ] {
            let err = gate(url).expect_err("{url} should be refused");
            assert!(err.contains("HTTPS"), "{url}: {err}");
        }
    }

    /// A file written by a newer build names enum variants this build has
    /// never heard of. Each such field falls back to its own default; every
    /// other value in the file survives.
    #[test]
    fn unknown_enum_values_fall_back_instead_of_failing_the_load() {
        let raw = r#"
language = "en-GB"
session_history_days = 90
device = "npu"
privacy_mode = "FutureMode"
terminal_mode = "FutureMode"
activation_mode = "FutureMode"
pill_visibility = "FutureMode"
pill_anchor = "FutureMode"
pill_size = "Gigantic"
pill_orientation = "Diagonal"
pill_corner = "FutureMode"
dictation_mode = "FutureMode"

[asr_backend]
type = "WhisperCpp"
model = "LargeV4"
"#;
        let settings: UserSettings = toml::from_str(raw).expect("a newer file must still load");

        assert_eq!(settings.language, "en-GB");
        assert_eq!(settings.session_history_days, Some(90));

        let d = UserSettings::default();
        assert_eq!(settings.device, d.device);
        assert_eq!(settings.privacy_mode, d.privacy_mode);
        assert_eq!(settings.terminal_mode, d.terminal_mode);
        assert_eq!(settings.activation_mode, d.activation_mode);
        assert_eq!(settings.pill_visibility, d.pill_visibility);
        assert_eq!(settings.pill_anchor, d.pill_anchor);
        assert_eq!(settings.pill_size, d.pill_size);
        assert_eq!(settings.pill_orientation, d.pill_orientation);
        assert_eq!(settings.pill_corner, d.pill_corner);
        assert_eq!(settings.dictation_mode, d.dictation_mode);
        assert!(matches!(
            settings.asr_backend,
            AsrBackendConfig::WhisperCpp {
                model: WhisperModel::Medium
            }
        ));
    }

    #[test]
    fn brand_casing_is_on_by_default_and_when_absent_from_the_file() {
        assert!(UserSettings::default().brand_casing);
        let settings: UserSettings = toml::from_str("language = \"en-GB\"\n").unwrap();
        assert!(settings.brand_casing);
        let off: UserSettings = toml::from_str("brand_casing = false\n").unwrap();
        assert!(!off.brand_casing);
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
    fn auto_device_follows_the_runtime_cuda_answer() {
        // `Auto` is the default, and the argument it defers to must stay a
        // runtime fact. Wiring a compile-time feature flag in here is what made
        // the downloadable CUDA plugin unreachable in `whisper-cpp` builds.
        assert!(DevicePolicy::Auto.use_gpu(true));
        assert!(!DevicePolicy::Auto.use_gpu(false));
        // The explicit choices ignore it in both directions.
        assert!(DevicePolicy::Cuda.use_gpu(false));
        assert!(!DevicePolicy::Cpu.use_gpu(true));
    }

    #[test]
    fn link_local_and_unspecified_addresses_are_not_local() {
        // These are the addresses the plaintext carve-out must not cover: an
        // autoconfigured link-local address was never the deliberate "box on my
        // LAN" it exists for, and `169.254.169.254` is the cloud metadata
        // endpoint. The unspecified address is not a destination at all.
        for url in [
            "http://169.254.169.254/v1",
            "http://169.254.1.1/v1",
            "http://0.0.0.0:8080/v1",
            "http://[fe80::1]:8080/v1",
            "http://[::]:8080/v1",
        ] {
            assert!(!local(url), "{url} must not be treated as local");
        }
    }

    #[test]
    fn history_off_survives_a_settings_round_trip() {
        // TOML omits a `None` field entirely, so "Don't save history" is
        // written as an absent key. The container-level `serde(default)` would
        // read that back as the 30-day default and silently start recording
        // transcripts again on the next launch.
        let off = UserSettings {
            session_history_days: None,
            ..Default::default()
        };
        let text = toml::to_string_pretty(&off).expect("serialize settings");
        let back: UserSettings = toml::from_str(&text).expect("deserialize settings");
        assert_eq!(back.session_history_days, None, "history must stay off");

        // A value still round-trips, and a file that predates the field still
        // gets the default rather than failing to parse.
        let kept = UserSettings {
            session_history_days: Some(7),
            ..Default::default()
        };
        let text = toml::to_string_pretty(&kept).expect("serialize settings");
        let back: UserSettings = toml::from_str(&text).expect("deserialize settings");
        assert_eq!(back.session_history_days, Some(7));
        assert_eq!(
            UserSettings::default().session_history_days,
            Some(30),
            "an unset install still keeps a month of history"
        );
    }

    #[test]
    fn retention_reads_the_three_choices_out_of_one_field() {
        let with = |days| UserSettings {
            session_history_days: days,
            ..Default::default()
        };
        assert_eq!(with(None).history_retention(), HistoryRetention::Off);
        assert_eq!(with(Some(0)).history_retention(), HistoryRetention::Forever);
        assert_eq!(
            with(Some(30)).history_retention(),
            HistoryRetention::Days(30)
        );
        // "Keep forever" still records dictations — only `None` stops writes.
        assert!(with(Some(0)).history_retention().records());
        assert!(!with(None).history_retention().records());
    }

    #[test]
    fn keep_forever_survives_a_settings_round_trip() {
        let forever = UserSettings {
            session_history_days: Some(0),
            ..Default::default()
        };
        let text = toml::to_string_pretty(&forever).expect("serialize settings");
        let back: UserSettings = toml::from_str(&text).expect("deserialize settings");
        assert_eq!(back.history_retention(), HistoryRetention::Forever);
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
