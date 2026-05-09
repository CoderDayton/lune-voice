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
    Both,
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
    Qwen3Asr,
    OpenAiCompatible,
}

impl From<lune_core::asr::AsrBackendKind> for AsrBackendTypeDto {
    fn from(kind: lune_core::asr::AsrBackendKind) -> Self {
        use lune_core::asr::AsrBackendKind;
        match kind {
            AsrBackendKind::Qwen3Asr => Self::Qwen3Asr,
            AsrBackendKind::OpenAiCompatible => Self::OpenAiCompatible,
            // `AsrBackendKind` is `#[non_exhaustive]`. The catch-all fires only
            // when a new lune-core variant ships without a matching
            // `AsrBackendTypeDto` variant + arm here. Loud in dev, fall back to
            // Qwen3 in release so end users keep a working selector.
            other => {
                debug_assert!(
                    false,
                    "unhandled AsrBackendKind variant {other:?}: add a matching \
                     AsrBackendTypeDto variant and From arm before shipping"
                );
                Self::Qwen3Asr
            }
        }
    }
}

impl From<AsrBackendTypeDto> for lune_core::asr::AsrBackendKind {
    fn from(dto: AsrBackendTypeDto) -> Self {
        match dto {
            AsrBackendTypeDto::Qwen3Asr => Self::Qwen3Asr,
            AsrBackendTypeDto::OpenAiCompatible => Self::OpenAiCompatible,
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

// ── Read DTOs (Serialize → frontend) ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct StylePresetDto {
    pub id: String,
    pub name: String,
    pub description: String,
    pub is_builtin: bool,
    pub instructions: String,
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
    pub use_gpu: bool,
    pub device_policy: DevicePolicyDto,
    pub asr_backend_type: AsrBackendTypeDto,
    pub confidence_threshold: Option<f32>,
    pub adaptive_vad: bool,
    pub speech_threshold: f32,
    pub speech_threshold_end: f32,
    pub style_preset: String,
    pub hotkey_toggle: String,
    pub hotkey_cancel: String,
    pub pill_visibility: PillVisibilityDto,
    pub language: String,
    pub dictation_language: String,
    pub mic_prewarm: bool,
    pub vad_pcen: bool,
    // OpenAI ASR fields (populated when asr_backend_type == OpenAiCompatible)
    pub openai_asr_base_url: Option<String>,
    pub openai_asr_model: Option<String>,
    pub openai_asr_api_key_set: bool,
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

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase")]
pub struct DictionaryItemDto {
    pub id: i32,
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

// ── Write DTOs (Deserialize ← frontend) ──────────────────────────────────

#[derive(Debug, Clone, Deserialize, TS)]
#[ts(export, export_to = "../../src/lib/backend/generated/")]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SaveWorkspaceSettingsInput {
    pub activation_mode: ActivationModeDto,
    pub dictation_mode: DictationModeDto,
    pub use_gpu: bool,
    pub device_policy: DevicePolicyDto,
    pub asr_backend_type: AsrBackendTypeDto,
    pub confidence_threshold: Option<f32>,
    pub adaptive_vad: bool,
    pub speech_threshold: f32,
    pub speech_threshold_end: f32,
    pub style_preset: String,
    pub hotkey_toggle: String,
    pub hotkey_cancel: String,
    pub pill_visibility: PillVisibilityDto,
    pub language: String,
    pub dictation_language: String,
    pub mic_prewarm: bool,
    pub vad_pcen: bool,
    // OpenAI ASR fields (sent when asr_backend_type == OpenAiCompatible)
    pub openai_asr_base_url: Option<String>,
    pub openai_asr_api_key: Option<String>,
    pub openai_asr_model: Option<String>,
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
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UpdateCustomStyleInput {
    pub id: String,
    pub name: String,
    pub description: String,
    pub instructions: String,
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
        for kind in [AsrBackendKind::Qwen3Asr, AsrBackendKind::OpenAiCompatible] {
            let dto: AsrBackendTypeDto = kind.into();
            let back: AsrBackendKind = dto.into();
            assert_eq!(back, kind);
        }
    }
}
