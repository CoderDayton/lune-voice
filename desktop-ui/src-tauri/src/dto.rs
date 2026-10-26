// SPDX-License-Identifier: GPL-3.0-or-later

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// Generated TS bindings live alongside the hand-rolled types under
/// `desktop-ui/src/lib/backend/generated/`. Run `cargo test -p lune-ui` to
/// regenerate after editing any DTO. The `types.ts` barrel re-exports the
/// generated names so downstream imports stay stable.
const _GENERATED_EXPORT_DIR: &str = "../../src/lib/backend/generated/";

// ── DTO enums ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum ActivationModeDto {
    Toggle,
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum DictationModeDto {
    TypeInPlace,
    Clipboard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum DevicePolicyDto {
    Cpu,
    Vulkan,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum AsrBackendTypeDto {
    OpenAiCompatible,
    WhisperCpp,
    /// NVIDIA Parakeet TDT, on-device; the quantization rides in
    /// `parakeet_model`.
    ParakeetCpp,
}

/// Which styling-LLM engine the user picked. `Disabled` is carried by
/// `llm_enabled == false`. Remote (OpenAI-compatible) is the only engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum LlmBackendTypeDto {
    OpenAiCompatible,
}

impl From<lune_core::asr::AsrBackendKind> for AsrBackendTypeDto {
    fn from(kind: lune_core::asr::AsrBackendKind) -> Self {
        use lune_core::asr::AsrBackendKind;
        match kind {
            AsrBackendKind::OpenAiCompatible => Self::OpenAiCompatible,
            AsrBackendKind::WhisperCpp => Self::WhisperCpp,
            AsrBackendKind::ParakeetCpp => Self::ParakeetCpp,
            // `AsrBackendKind` is `#[non_exhaustive]`. The catch-all fires only
            // when a new lune-core variant ships without a matching
            // `AsrBackendTypeDto` variant + arm here. Loud in dev, fall back to
            // WhisperCpp in release so end users keep a working selector.
            other => {
                debug_assert!(
                    false,
                    "unhandled AsrBackendKind variant {other:?}: add a matching \
                     AsrBackendTypeDto variant and From arm before shipping"
                );
                Self::WhisperCpp
            }
        }
    }
}

impl From<AsrBackendTypeDto> for lune_core::asr::AsrBackendKind {
    fn from(dto: AsrBackendTypeDto) -> Self {
        match dto {
            AsrBackendTypeDto::OpenAiCompatible => Self::OpenAiCompatible,
            AsrBackendTypeDto::WhisperCpp => Self::WhisperCpp,
            AsrBackendTypeDto::ParakeetCpp => Self::ParakeetCpp,
        }
    }
}

/// Which local whisper.cpp model the WhisperCpp backend uses. Mirrors
/// [`lune_core::config::WhisperModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum WhisperModelDto {
    Base,
    Small,
    #[default]
    Medium,
    LargeV3Turbo,
}

impl From<lune_core::config::WhisperModel> for WhisperModelDto {
    fn from(model: lune_core::config::WhisperModel) -> Self {
        use lune_core::config::WhisperModel;
        match model {
            WhisperModel::Base => Self::Base,
            WhisperModel::Small => Self::Small,
            WhisperModel::Medium => Self::Medium,
            WhisperModel::LargeV3Turbo => Self::LargeV3Turbo,
            // `WhisperModel` is `#[non_exhaustive]`: fall back loudly in dev,
            // safely in release if a new variant ships without an arm here.
            other => {
                debug_assert!(false, "unhandled WhisperModel variant {other:?}");
                Self::Medium
            }
        }
    }
}

impl From<WhisperModelDto> for lune_core::config::WhisperModel {
    fn from(dto: WhisperModelDto) -> Self {
        match dto {
            WhisperModelDto::Base => Self::Base,
            WhisperModelDto::Small => Self::Small,
            WhisperModelDto::Medium => Self::Medium,
            WhisperModelDto::LargeV3Turbo => Self::LargeV3Turbo,
        }
    }
}

/// Which Parakeet quantization the ParakeetCpp backend uses. Mirrors
/// [`lune_core::config::ParakeetModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum ParakeetModelDto {
    #[default]
    TdtV3Q8,
    TdtV3Q4K,
}

impl From<lune_core::config::ParakeetModel> for ParakeetModelDto {
    fn from(model: lune_core::config::ParakeetModel) -> Self {
        use lune_core::config::ParakeetModel;
        match model {
            ParakeetModel::TdtV3Q8 => Self::TdtV3Q8,
            ParakeetModel::TdtV3Q4K => Self::TdtV3Q4K,
            // `ParakeetModel` is `#[non_exhaustive]`: fall back loudly in dev,
            // safely in release if a new variant ships without an arm here.
            other => {
                debug_assert!(false, "unhandled ParakeetModel variant {other:?}");
                Self::TdtV3Q8
            }
        }
    }
}

impl From<ParakeetModelDto> for lune_core::config::ParakeetModel {
    fn from(dto: ParakeetModelDto) -> Self {
        match dto {
            ParakeetModelDto::TdtV3Q8 => Self::TdtV3Q8,
            ParakeetModelDto::TdtV3Q4K => Self::TdtV3Q4K,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum PillVisibilityDto {
    Always,
    WhenDictating,
    Never,
}

/// Default placement anchor for the pill overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum PillAnchorDto {
    TopLeft,
    TopCenter,
    TopRight,
    BottomLeft,
    #[default]
    BottomCenter,
    BottomRight,
}

/// Footprint of the pill bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum PillSizeDto {
    Compact,
    #[default]
    Standard,
    Large,
}

/// Long-axis direction of the pill bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum PillOrientationDto {
    #[default]
    Horizontal,
    Vertical,
}

/// Corner rounding of the pill bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum PillCornerDto {
    Pill,
    #[default]
    Rounded,
    Square,
}

/// One connected monitor, for the pill's monitor picker.
#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct MonitorInfoDto {
    /// Connector name (e.g. "DP-1") — the persisted pin key.
    pub name: String,
    /// Logical size, for the picker label.
    pub width: u32,
    pub height: u32,
    pub is_primary: bool,
}

/// Mic gain control before the recognizer. Mirrors
/// [`lune_core::config::AgcMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum AgcModeDto {
    Off,
    #[default]
    Auto,
    Always,
}

/// Paste-shortcut selection strategy for the output stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum TerminalModeDto {
    Auto,
    Terminal,
    Normal,
}

/// Whether remote services may be used at all. Mirrors
/// [`lune_core::config::PrivacyMode`].
///
/// `LocalOnly` is the default and it *overrides* a configured remote backend:
/// styling silently degrades to the raw transcript and remote transcription
/// is refused outright. That makes it the one setting that can make a
/// correctly-configured API key do nothing, so it has to be visible. The UI
/// shows it as a single switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum PrivacyModeDto {
    /// Never contact a remote service.
    #[default]
    LocalOnly,
    /// Remote services may be used; each still needs to be configured.
    AllowRemote,
}

// ── Read DTOs (Serialize → frontend) ─────────────────────────────────────

/// Per-style deterministic formatting toggles, surfaced to the styles page.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct FormatTogglesDto {
    pub keybinds: bool,
    pub title_abbreviations: bool,
    pub auto_capitalization: bool,
    /// Null follows the global `brandCasing` setting.
    #[serde(default)]
    pub brand_casing: Option<bool>,
}

impl Default for FormatTogglesDto {
    fn default() -> Self {
        lune_core::config::FormatToggles::default().into()
    }
}

impl From<lune_core::config::FormatToggles> for FormatTogglesDto {
    fn from(f: lune_core::config::FormatToggles) -> Self {
        Self {
            keybinds: f.keybinds,
            title_abbreviations: f.title_abbreviations,
            auto_capitalization: f.auto_capitalization,
            brand_casing: f.brand_casing,
        }
    }
}

impl From<FormatTogglesDto> for lune_core::config::FormatToggles {
    fn from(d: FormatTogglesDto) -> Self {
        Self {
            keybinds: d.keybinds,
            title_abbreviations: d.title_abbreviations,
            auto_capitalization: d.auto_capitalization,
            brand_casing: d.brand_casing,
        }
    }
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct StylePresetDto {
    pub id: String,
    pub name: String,
    pub description: String,
    pub is_builtin: bool,
    pub instructions: String,
    pub formatting: FormatTogglesDto,
    /// Custom styles only: their few-shot examples are stale (prompt/name
    /// changed since last generated). Always false for builtins.
    pub few_shot_dirty: bool,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSnapshot {
    #[ts(type = "string")]
    pub ipc_flavor: &'static str,
    pub service_status: String,
    pub capabilities: PlatformCapabilitiesDto,
    pub settings: WorkspaceSettingsDto,
    pub history: Vec<HistoryItemDto>,
    pub dictionary: Vec<DictionaryItemDto>,
    pub snippets: Vec<SnippetItemDto>,
    pub style_presets: Vec<StylePresetDto>,
    pub stats: SessionStatsDto,
}

/// What the running platform's window system can do. A capability that is
/// false means the setting exists but has no effect there, so the UI hides the
/// control rather than offering one that silently does nothing.
#[derive(Debug, Clone, Copy, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct PlatformCapabilitiesDto {
    /// A window can be pinned to every workspace / virtual desktop. Windows has
    /// no such call, so `pill_all_workspaces` is inert there.
    pub pill_all_workspaces: bool,
}

impl PlatformCapabilitiesDto {
    pub const fn current() -> Self {
        Self {
            pill_all_workspaces: !cfg!(target_os = "windows"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct SessionStatsDto {
    pub sessions: u32,
    pub words: u32,
    pub seconds: u32,
    /// The same totals for sessions started today, aggregated in SQL over every
    /// row rather than over the capped `history` slice — which stops being the
    /// whole day as soon as the day runs past the cap.
    pub sessions_today: u32,
    pub words_today: u32,
    pub seconds_today: u32,
}

/// Settings fields the frontend reads and writes with the same shape.
///
/// Flattened into [`WorkspaceSettingsDto`] and [`SaveWorkspaceSettingsInput`],
/// so a shared setting is declared once and both directions stay in step. The
/// `serde(default)`s apply when a save omits the field.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct SettingsCoreDto {
    pub activation_mode: ActivationModeDto,
    pub dictation_mode: DictationModeDto,
    /// Keep the dictated text on the clipboard after delivery (orthogonal to the
    /// send method).
    pub copy_to_clipboard: bool,
    /// Play short sound cues on dictation start/stop/error.
    pub sound_cues_enabled: bool,
    /// Sound cue playback volume in [0.0, 1.0].
    pub sound_cue_volume: f32,
    pub use_gpu: bool,
    pub device_policy: DevicePolicyDto,
    pub asr_backend_type: AsrBackendTypeDto,
    pub confidence_threshold: Option<f32>,
    pub style_preset: Option<String>,
    pub hotkey_toggle: String,
    pub hotkey_cancel: String,
    pub pill_visibility: PillVisibilityDto,
    #[serde(default)]
    pub pill_anchor: PillAnchorDto,
    #[serde(default)]
    pub pill_locked: bool,
    #[serde(default = "default_pill_edge_margin_dto")]
    pub pill_edge_margin: u32,
    /// Monitor connector name the pill is pinned to; null = automatic.
    #[serde(default)]
    pub pill_monitor: Option<String>,
    #[serde(default = "default_true_dto")]
    pub pill_all_workspaces: bool,
    #[serde(default)]
    pub pill_size: PillSizeDto,
    #[serde(default)]
    pub pill_orientation: PillOrientationDto,
    #[serde(default)]
    pub pill_corner: PillCornerDto,
    pub language: String,
    pub dictation_language: String,
    /// Minutes without a dictation before the ASR model is freed; null = never.
    #[serde(default)]
    pub model_idle_unload_minutes: Option<u32>,
    /// Preferred microphone input device by name; null = system default.
    #[serde(default)]
    pub input_device: Option<String>,
    pub agc_mode: AgcModeDto,
    pub start_minimized: bool,
    /// Check the release endpoint for a newer version at startup.
    #[serde(default)]
    pub check_updates_on_launch: bool,
    /// Capitalize known names ("github" -> "GitHub"). A style can override it.
    #[serde(default = "default_true_dto")]
    pub brand_casing: bool,
    /// How long to keep dictation history, in days. `None` disables history.
    pub session_history_days: Option<u32>,
    /// Paste-shortcut selection strategy for the output stage.
    pub terminal_mode: TerminalModeDto,
    /// Whether remote services may be used at all.
    #[serde(default)]
    pub privacy_mode: PrivacyModeDto,
    /// Whether the remote styling LLM backend is enabled.
    pub llm_enabled: bool,
    /// OpenAI-compatible ASR endpoint and model (meaningful when
    /// `asr_backend_type` is `OpenAiCompatible`).
    pub openai_asr_base_url: Option<String>,
    pub openai_asr_model: Option<String>,
    /// Saved remote styling provider. It persists when styling is off; on save,
    /// a blank value keeps the saved one.
    pub openai_llm_base_url: Option<String>,
    pub openai_llm_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSettingsDto {
    #[serde(flatten)]
    pub core: SettingsCoreDto,
    /// Whether an ASR API key is saved. The key itself never leaves the backend.
    pub openai_asr_api_key_set: bool,
    /// Whether a styling API key is saved.
    pub openai_llm_api_key_set: bool,
    /// Whether the *active* styling backend is the remote provider (the provider
    /// fields persist even when it isn't active).
    pub openai_llm_active: bool,
    /// Selected local whisper model (meaningful when asr_backend_type == WhisperCpp).
    pub whisper_model: WhisperModelDto,
    /// Selected Parakeet model (meaningful when asr_backend_type == ParakeetCpp).
    pub parakeet_model: ParakeetModelDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct PillSnapshotDto {
    pub service_status: String,
    pub activation_mode: ActivationModeDto,
    pub pill_visibility: PillVisibilityDto,
    /// Dragging disabled — the pill webview skips its drag handlers.
    pub pill_locked: bool,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct HistoryItemDto {
    pub id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub transcript: String,
    pub backend: String,
    pub style: String,
    pub status: String,
}

/// One page of session history plus the total number of sessions matching the
/// query (ignoring the page window), so the UI can render "N of TOTAL" and the
/// page count without loading every row.
#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct HistoryPageDto {
    pub sessions: Vec<HistoryItemDto>,
    pub total: u32,
}

/// A voice snippet: a spoken trigger phrase that expands to a text template
/// (with `{date}`, `{time}`, `{datetime}`, `{clipboard}` placeholders).
#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct SnippetItemDto {
    pub id: String,
    pub trigger_phrase: String,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct DictionaryItemDto {
    pub id: i32,
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

/// What GPU acceleration can do on this machine right now.
#[derive(Debug, Clone, Copy, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "kebab-case")]
pub enum GpuBackendStateDto {
    /// Built without the native ASR backend; there is nothing to accelerate.
    Unsupported,
    DriverMissing,
    DriverOutdated,
    /// Driver present, plugin not downloaded yet. The offer state.
    NotInstalled,
    Ready,
    /// Plugin installed, waiting for the next launch to register its device.
    /// Not a failure, and told apart from one so the UI can ask for a restart
    /// in the user's own language rather than relaying an English `reason`.
    NeedsRestart,
    Failed,
}

/// Streamed while the GPU plugin archive downloads.
#[derive(Debug, Clone, Copy, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct GpuInstallProgressDto {
    /// `u32` rather than `u64` so ts-rs emits `number` instead of `bigint`:
    /// the archive is a few hundred MB, nowhere near the 4 GB ceiling, and
    /// `bigint` would force the progress arithmetic in the UI to be awkward.
    pub downloaded_bytes: u32,
    /// `0` when the server sends no `Content-Length`.
    pub total_bytes: u32,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct GpuBackendStatusDto {
    pub state: GpuBackendStateDto,
    /// Compute devices ggml registered, so the UI can name the GPU rather than
    /// claim success generically. Empty unless `state` is `ready`.
    pub devices: Vec<String>,
    /// User-facing explanation when `state` is `failed`.
    pub reason: Option<String>,
    /// Download size of the GPU plugin archive, for the confirmation prompt.
    /// `u32` for the same reason as [`GpuInstallProgressDto`]: `number`, not
    /// `bigint`, on the TypeScript side.
    pub download_bytes: u32,
    /// Whether the plugin archive for this platform has actually been
    /// published. False on a build whose release digest is still the
    /// `UNPUBLISHED` sentinel, where the install would refuse — the UI hides
    /// the download rather than offering a button that cannot succeed.
    pub download_available: bool,
}

/// Where the update flow is right now, as the in-app modal renders it.
///
/// Every outcome the user can be told about is a variant here rather than a
/// native message box: a system dialog is drawn by the desktop environment,
/// looks nothing like the rest of the app, and on Linux can land behind the
/// window that asked for it. Reporting through the app's own modal keeps the
/// answer where the click was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "kebab-case")]
pub enum UpdatePhaseDto {
    /// Nothing to show: no check running and nothing waiting on an answer.
    Idle,
    /// A check the user asked for is in flight.
    Checking,
    /// The check finished and this build is current.
    UpToDate,
    /// A newer version exists and is waiting for the user to accept it.
    Available,
    /// A newer version exists, but this copy is not a package the updater can
    /// replace (the portable archive). The user downloads it themselves.
    ManualDownload,
    /// The accepted update is downloading.
    Downloading,
    /// Download finished; the package is being applied. The app restarts next.
    Installing,
    /// The check itself could not complete (offline, bad endpoint, bad config).
    CheckFailed,
    /// This build does not check for updates at all. Not a failure — there is
    /// nothing to retry — so it must not render as one.
    Disabled,
    /// The download or install failed. The offer survives so it can be retried.
    InstallFailed,
}

/// Everything the update modal needs to render, in one snapshot.
///
/// Also the *stored* state, not just an event payload: a launch check can find
/// an update while the main window is closed, and an event emitted at a window
/// that does not exist is simply lost. The frontend reads this snapshot on
/// mount, so the offer survives until it is answered.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct UpdateStateDto {
    pub phase: UpdatePhaseDto,
    /// Version this build reports — the same string the check compares.
    pub current_version: String,
    /// Version the release server offers, once a check has found one.
    pub available_version: Option<String>,
    /// Release notes from the manifest, when it carried any.
    pub notes: Option<String>,
    /// User-facing failure text for the two failure phases; `None` otherwise.
    pub message: Option<String>,
    /// `u32` for the same reason as [`GpuInstallProgressDto`]: `number`, not
    /// `bigint`, on the TypeScript side.
    pub downloaded_bytes: u32,
    /// `0` until the server sends a `Content-Length`.
    pub total_bytes: u32,
}

impl UpdateStateDto {
    /// A state carrying nothing but a phase and the running version — the shape
    /// every transition except the two progress-bearing ones needs.
    pub fn phase(phase: UpdatePhaseDto, current_version: &str) -> Self {
        Self {
            phase,
            current_version: current_version.to_string(),
            available_version: None,
            notes: None,
            message: None,
            downloaded_bytes: 0,
            total_bytes: 0,
        }
    }
}

// ── Write DTOs (Deserialize ← frontend) ──────────────────────────────────

#[derive(Debug, Clone, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SaveWorkspaceSettingsInput {
    #[serde(flatten)]
    pub core: SettingsCoreDto,
    /// Which styling engine to use when enabled.
    pub llm_backend_type: LlmBackendTypeDto,
    /// A `None` key keeps the saved one.
    pub openai_asr_api_key: Option<String>,
    /// A `None` key keeps the saved one.
    pub openai_llm_api_key: Option<String>,
    /// Selected local whisper model (sent when asr_backend_type == WhisperCpp).
    #[serde(default)]
    pub whisper_model: Option<WhisperModelDto>,
    /// Selected Parakeet model (sent when asr_backend_type == ParakeetCpp).
    #[serde(default)]
    pub parakeet_model: Option<ParakeetModelDto>,
}

fn default_pill_edge_margin_dto() -> u32 {
    72
}

fn default_true_dto() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveSnippetInput {
    /// `None` creates a new snippet; `Some(id)` updates the existing one.
    pub id: Option<String>,
    pub trigger_phrase: String,
    pub body: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDictionaryEntryInput {
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateDictionaryEntryInput {
    pub id: i32,
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CreateCustomStyleInput {
    pub name: String,
    pub description: String,
    pub instructions: String,
    #[serde(default)]
    pub formatting: FormatTogglesDto,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UpdateCustomStyleInput {
    pub id: String,
    pub name: String,
    pub description: String,
    pub instructions: String,
    #[serde(default)]
    pub formatting: FormatTogglesDto,
}

// ── Stream events ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SessionStreamEvent {
    ServiceStatus {
        status: String,
    },
    SessionStarted {
        #[serde(rename = "sessionId")]
        session_id: String,
    },
    Processing,
    FinalizedText {
        text: String,
    },
    /// Raw transcript from a capture-only session (snippet trigger recorder).
    /// Never delivered to the focused app; empty text = no speech heard.
    CaptureText {
        text: String,
    },
    AudioLevel {
        level: f32,
    },
    CommandDetected {
        payload: String,
    },
    SessionStopped,
    SessionError {
        message: String,
    },
    /// Non-fatal anomaly (e.g. a persist write failed but dictation is
    /// still progressing). Frontend should surface as a transient warning;
    /// no terminal/teardown semantics.
    SessionWarning {
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::{AsrBackendTypeDto, SaveWorkspaceSettingsInput, SessionStreamEvent};

    /// Every field the save input requires, and nothing else.
    fn minimal_save_input() -> serde_json::Value {
        serde_json::json!({
            "activationMode": "Toggle",
            "dictationMode": "TypeInPlace",
            "copyToClipboard": false,
            "soundCuesEnabled": true,
            "soundCueVolume": 0.5,
            "useGpu": true,
            "devicePolicy": "Auto",
            "asrBackendType": "WhisperCpp",
            "hotkeyToggle": "Ctrl+Space",
            "hotkeyCancel": "Escape",
            "pillVisibility": "Always",
            "language": "en",
            "dictationLanguage": "auto",
            "agcMode": "Auto",
            "startMinimized": false,
            "terminalMode": "Auto",
            "llmEnabled": false,
            "llmBackendType": "OpenAiCompatible",
        })
    }

    /// A renamed or misspelled frontend field must fail the save loudly rather
    /// than be dropped. A read-only field sent back is the likeliest mistake.
    #[test]
    fn save_settings_input_rejects_unknown_fields() {
        serde_json::from_value::<SaveWorkspaceSettingsInput>(minimal_save_input())
            .expect("minimal input deserializes");

        let mut input = minimal_save_input();
        input["openaiAsrApiKeySet"] = true.into();
        let err = serde_json::from_value::<SaveWorkspaceSettingsInput>(input).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn session_stream_events_match_frontend_shape() {
        let started = serde_json::to_value(SessionStreamEvent::SessionStarted {
            session_id: "session-1".into(),
        })
        .unwrap();
        assert_eq!(started["type"], "sessionStarted");
        assert_eq!(started["sessionId"], "session-1");
        assert!(started.get("session_id").is_none());

        let status = serde_json::to_value(SessionStreamEvent::ServiceStatus {
            status: "ready".into(),
        })
        .unwrap();
        assert_eq!(status["type"], "serviceStatus");
        assert_eq!(status["status"], "ready");
    }

    #[test]
    fn asr_backend_kind_dto_round_trip() {
        use lune_core::asr::AsrBackendKind;
        for kind in [
            AsrBackendKind::WhisperCpp,
            AsrBackendKind::ParakeetCpp,
            AsrBackendKind::OpenAiCompatible,
        ] {
            let dto: AsrBackendTypeDto = kind.into();
            let back: AsrBackendKind = dto.into();
            assert_eq!(back, kind);
        }
    }
}
