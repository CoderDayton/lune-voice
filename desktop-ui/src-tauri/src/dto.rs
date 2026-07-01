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
    Cuda,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum AsrBackendTypeDto {
    OpenAiCompatible,
    WhisperCpp,
}

impl From<lune_core::asr::AsrBackendKind> for AsrBackendTypeDto {
    fn from(kind: lune_core::asr::AsrBackendKind) -> Self {
        use lune_core::asr::AsrBackendKind;
        match kind {
            AsrBackendKind::OpenAiCompatible => Self::OpenAiCompatible,
            AsrBackendKind::WhisperCpp => Self::WhisperCpp,
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
        }
    }
}

/// Which local whisper.cpp model the WhisperCpp backend uses. Mirrors
/// [`lune_core::config::WhisperModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum WhisperModelDto {
    Small,
    #[default]
    Medium,
    LargeV3Turbo,
}

impl From<lune_core::config::WhisperModel> for WhisperModelDto {
    fn from(model: lune_core::config::WhisperModel) -> Self {
        use lune_core::config::WhisperModel;
        match model {
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
            WhisperModelDto::Small => Self::Small,
            WhisperModelDto::Medium => Self::Medium,
            WhisperModelDto::LargeV3Turbo => Self::LargeV3Turbo,
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

/// Paste-shortcut selection strategy for the output stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub enum TerminalModeDto {
    Auto,
    Terminal,
    Normal,
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
    pub brand_casing: bool,
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
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSnapshot {
    #[ts(type = "string")]
    pub ipc_flavor: &'static str,
    pub service_status: String,
    pub settings: WorkspaceSettingsDto,
    pub history: Vec<HistoryItemDto>,
    pub dictionary: Vec<DictionaryItemDto>,
    pub style_presets: Vec<StylePresetDto>,
    pub stats: SessionStatsDto,
}

#[derive(Debug, Clone, Copy, Default, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct SessionStatsDto {
    pub sessions: u32,
    pub words: u32,
    pub seconds: u32,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSettingsDto {
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
    /// Clean-segment confidence bound for bypassing the correction LLM.
    pub llm_skip_threshold: Option<f32>,
    pub adaptive_vad: bool,
    pub speech_threshold: f32,
    pub speech_threshold_end: f32,
    pub style_preset: Option<String>,
    pub hotkey_toggle: String,
    pub hotkey_cancel: String,
    pub pill_visibility: PillVisibilityDto,
    pub language: String,
    pub dictation_language: String,
    pub mic_prewarm: bool,
    /// Preferred microphone input device by name; null = system default.
    pub input_device: Option<String>,
    pub vad_pcen: bool,
    pub start_minimized: bool,
    /// How long to keep dictation history, in days. `None` disables history.
    pub session_history_days: Option<u32>,
    /// Paste-shortcut selection strategy for the output stage.
    pub terminal_mode: TerminalModeDto,
    /// Whether the correction LLM backend is enabled (Local or remote).
    pub llm_enabled: bool,
    // OpenAI ASR fields (populated when asr_backend_type == OpenAiCompatible)
    pub openai_asr_base_url: Option<String>,
    pub openai_asr_model: Option<String>,
    pub openai_asr_api_key_set: bool,
    /// Selected local whisper model (meaningful when asr_backend_type == WhisperCpp).
    pub whisper_model: WhisperModelDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct PillSnapshotDto {
    pub service_status: String,
    pub activation_mode: ActivationModeDto,
    pub pill_visibility: PillVisibilityDto,
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

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct DictionaryItemDto {
    pub id: i32,
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

/// Whether GPU (CUDA) acceleration was compiled into this build. GPU support
/// is a build-time decision — the native backends compile CUDA in only under
/// the `whisper-cuda`/`llama-cuda` cargo features — so at runtime "GPU
/// available" reduces to "was a CUDA feature compiled in". No `camelCase`
/// rename: the single field stays `compiled_in` across the IPC boundary.
#[derive(Debug, Clone, Copy, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
pub struct GpuSupportDto {
    pub compiled_in: bool,
}

// ── Write DTOs (Deserialize ← frontend) ──────────────────────────────────

#[derive(Debug, Clone, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SaveWorkspaceSettingsInput {
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
    /// Clean-segment confidence bound for bypassing the correction LLM.
    pub llm_skip_threshold: Option<f32>,
    pub adaptive_vad: bool,
    pub speech_threshold: f32,
    pub speech_threshold_end: f32,
    pub style_preset: Option<String>,
    pub hotkey_toggle: String,
    pub hotkey_cancel: String,
    pub pill_visibility: PillVisibilityDto,
    pub language: String,
    pub dictation_language: String,
    pub mic_prewarm: bool,
    /// Preferred microphone input device by name; null = system default.
    #[serde(default)]
    pub input_device: Option<String>,
    pub vad_pcen: bool,
    pub start_minimized: bool,
    /// How long to keep dictation history, in days. `None` disables history.
    pub session_history_days: Option<u32>,
    /// Paste-shortcut selection strategy for the output stage.
    pub terminal_mode: TerminalModeDto,
    /// Whether the correction LLM backend should be enabled.
    pub llm_enabled: bool,
    // OpenAI ASR fields (sent when asr_backend_type == OpenAiCompatible)
    pub openai_asr_base_url: Option<String>,
    pub openai_asr_api_key: Option<String>,
    pub openai_asr_model: Option<String>,
    /// Selected local whisper model (sent when asr_backend_type == WhisperCpp).
    #[serde(default)]
    pub whisper_model: Option<WhisperModelDto>,
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
    PartialText,
    Processing,
    FinalizedText {
        text: String,
    },
    CorrectedText {
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationDiagnosticsDto {
    pub noise_p95: f32,
    pub speech_p10: f32,
    pub pause_p20: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CalibrationStreamEvent {
    PhaseChanged {
        phase: String,
    },
    Frame {
        phase: String,
        #[serde(rename = "vadScore")]
        vad_score: f32,
        energy: f32,
    },
    Result {
        #[serde(rename = "speechThreshold")]
        speech_threshold: f32,
        #[serde(rename = "speechThresholdEnd")]
        speech_threshold_end: f32,
        diagnostics: CalibrationDiagnosticsDto,
    },
    Error {
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        AsrBackendTypeDto, CalibrationDiagnosticsDto, CalibrationStreamEvent, SessionStreamEvent,
    };

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
    fn calibration_stream_events_match_frontend_shape() {
        // Use values that round-trip exactly through f32→f64 (powers of 2).
        let frame = serde_json::to_value(CalibrationStreamEvent::Frame {
            phase: "noise".into(),
            vad_score: 0.5,
            energy: 0.25,
        })
        .unwrap();
        assert_eq!(frame["type"], "frame");
        assert_eq!(frame["phase"], "noise");
        assert_eq!(frame["vadScore"].as_f64(), Some(0.5));
        assert_eq!(frame["energy"].as_f64(), Some(0.25));
        // Snake_case must NOT leak across the IPC boundary — the rename is
        // load-bearing for the frontend types.
        assert!(
            frame.get("vad_score").is_none(),
            "snake_case leaked: vad_score"
        );

        let result = serde_json::to_value(CalibrationStreamEvent::Result {
            speech_threshold: 0.5,
            speech_threshold_end: 0.25,
            diagnostics: CalibrationDiagnosticsDto {
                noise_p95: 0.0625,
                speech_p10: 0.75,
                pause_p20: 0.125,
            },
        })
        .unwrap();
        assert_eq!(result["type"], "result");
        assert_eq!(result["speechThreshold"].as_f64(), Some(0.5));
        assert_eq!(result["speechThresholdEnd"].as_f64(), Some(0.25));
        assert!(result.get("speech_threshold").is_none());
        assert!(result.get("speech_threshold_end").is_none());

        let diag = &result["diagnostics"];
        assert_eq!(diag["noiseP95"].as_f64(), Some(0.0625));
        assert_eq!(diag["speechP10"].as_f64(), Some(0.75));
        assert_eq!(diag["pauseP20"].as_f64(), Some(0.125));
        assert!(diag.get("noise_p95").is_none());
    }

    #[test]
    fn asr_backend_kind_dto_round_trip() {
        use lune_core::asr::AsrBackendKind;
        for kind in [AsrBackendKind::WhisperCpp, AsrBackendKind::OpenAiCompatible] {
            let dto: AsrBackendTypeDto = kind.into();
            let back: AsrBackendKind = dto.into();
            assert_eq!(back, kind);
        }
    }
}
