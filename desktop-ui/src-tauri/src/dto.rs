use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSnapshot {
    pub ipc_flavor: &'static str,
    pub service_status: String,
    pub settings: WorkspaceSettingsDto,
    pub history: Vec<HistoryItemDto>,
    pub dictionary: Vec<DictionaryItemDto>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSettingsDto {
    pub activation_mode: String,
    pub dictation_mode: String,
    pub use_gpu: bool,
    pub asr_backend_type: String,
    pub llm_backend_type: String,
    pub confidence_threshold: Option<f32>,
    pub adaptive_vad: bool,
    pub speech_threshold: f32,
    pub style_preset: String,
    pub hotkey_toggle: String,
    pub hotkey_cancel: String,
    pub aligner_enabled: bool,
    // OpenAI ASR fields (populated when asr_backend_type == "OpenAiCompatible")
    pub openai_asr_base_url: Option<String>,
    pub openai_asr_api_key: Option<String>,
    pub openai_asr_model: Option<String>,
    // OpenAI LLM fields (populated when llm_backend_type == "OpenAiCompatible")
    pub openai_llm_base_url: Option<String>,
    pub openai_llm_api_key: Option<String>,
    pub openai_llm_model: Option<String>,
}

// Redact API keys from Debug output to prevent accidental log exposure.
impl std::fmt::Debug for WorkspaceSettingsDto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceSettingsDto")
            .field("asr_backend_type", &self.asr_backend_type)
            .field("llm_backend_type", &self.llm_backend_type)
            .field(
                "openai_asr_api_key",
                &self.openai_asr_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field(
                "openai_llm_api_key",
                &self.openai_llm_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DictionaryItemDto {
    pub id: i32,
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDictionaryEntryInput {
    pub phrase: String,
    pub replacement: String,
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SessionStreamEvent {
    ServiceStatus { status: String },
    SessionStarted { session_id: String },
    PartialText,
    Processing,
    FinalizedText { text: String },
    CorrectedText { text: String },
    CommandDetected { payload: String },
    SessionStopped,
    SessionError { message: String },
}
