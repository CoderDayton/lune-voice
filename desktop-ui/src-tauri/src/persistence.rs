use std::{path::PathBuf, str::FromStr};

use lune_core::config::{
    ActivationMode, AlignerConfig, AsrBackendConfig, DevicePolicy, DictationMode, HotkeyConfig,
    KeyCombo, LlmBackendConfig, Modifier, PillVisibility, RedactedString, StreamingMode,
    UserSettings,
};
use lune_core::session::SessionStatus;
use lune_persist::{AppPaths, Database, SettingsStore, StoredDictionaryEntry};

use crate::dto::{
    CreateDictionaryEntryInput, DictionaryItemDto, HistoryItemDto, WorkspaceSettingsDto,
    WorkspaceSnapshot,
};

const HISTORY_LIMIT: u32 = 200;
const DICTIONARY_LIMIT: usize = 10_000;
const UNKNOWN_LABEL: &str = "Unknown";

pub fn load_workspace_snapshot() -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let settings = load_settings(&paths)?;
    let mut database = open_database(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn save_workspace_settings(input: WorkspaceSettingsDto) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let mut settings = load_settings(&paths)?;
    apply_settings_dto(&mut settings, input)?;
    save_settings(&paths, &settings)?;

    let mut database = open_database(&paths)?;
    build_snapshot(settings, &mut database)
}

/// Maximum length for dictionary phrase/replacement strings.
const DICT_ENTRY_MAX_LEN: usize = 1024;

pub fn create_dictionary_entry(
    input: CreateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let phrase = input.phrase.trim();
    let replacement = input.replacement.trim();
    if phrase.is_empty() || replacement.is_empty() {
        return Err("phrase and replacement are required".into());
    }
    if phrase.len() > DICT_ENTRY_MAX_LEN || replacement.len() > DICT_ENTRY_MAX_LEN {
        return Err(format!(
            "dictionary entries must be at most {DICT_ENTRY_MAX_LEN} characters"
        ));
    }
    // Reject control characters (except newline/tab) to prevent injection.
    if phrase
        .chars()
        .chain(replacement.chars())
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err("dictionary entries must not contain control characters".into());
    }

    let mut database = open_database(&paths)?;
    database
        .insert_dictionary_entry(phrase, replacement, input.priority)
        .map_err(|e| format!("insert dictionary entry: {e}"))?;

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn delete_dictionary_entry(entry_id: i32) -> Result<WorkspaceSnapshot, String> {
    if entry_id <= 0 {
        return Err("entry_id must be a positive integer".into());
    }
    let paths = resolve_paths()?;
    let mut database = open_database(&paths)?;
    let deleted = database
        .delete_dictionary_entry(entry_id)
        .map_err(|e| format!("delete dictionary entry: {e}"))?;
    if !deleted {
        return Err(format!("dictionary entry not found: {entry_id}"));
    }

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

fn build_snapshot(
    settings: UserSettings,
    database: &mut Database,
) -> Result<WorkspaceSnapshot, String> {
    let history = database
        .list_sessions(HISTORY_LIMIT)
        .map_err(|e| format!("list sessions: {e}"))?
        .into_iter()
        .map(|session| HistoryItemDto {
            id: session.id.to_string(),
            started_at: session.started_at.to_rfc3339(),
            ended_at: session.ended_at.map(|ended| ended.to_rfc3339()),
            transcript: session.transcript.unwrap_or_default(),
            backend: format!("{:?}", session.backend),
            style: session.style,
            status: session_status_label(&session.status),
        })
        .collect();

    let dictionary: Vec<_> = database
        .list_dictionary_entries_with_ids()
        .map_err(|e| format!("list dictionary entries: {e}"))?
        .into_iter()
        .take(DICTIONARY_LIMIT)
        .map(map_dictionary_entry)
        .collect();

    Ok(WorkspaceSnapshot {
        ipc_flavor: "tauri",
        service_status: "ready".into(),
        settings: settings_to_dto(&settings),
        history,
        dictionary,
    })
}

fn resolve_paths() -> Result<AppPaths, String> {
    let paths = AppPaths::resolve().ok_or("failed to resolve app paths")?;
    paths
        .ensure_dirs()
        .map_err(|e| format!("ensure app dirs: {e}"))?;
    Ok(paths)
}

fn load_settings(paths: &AppPaths) -> Result<UserSettings, String> {
    let store = SettingsStore::new(paths.settings_file());
    store.load().map_err(|e| format!("load settings: {e}"))
}

fn save_settings(paths: &AppPaths, settings: &UserSettings) -> Result<(), String> {
    let store = SettingsStore::new(paths.settings_file());
    store
        .save(settings)
        .map_err(|e| format!("save settings: {e}"))
}

fn open_database(paths: &AppPaths) -> Result<Database, String> {
    Database::open(&paths.database_file()).map_err(|e| format!("open database: {e}"))
}

fn settings_to_dto(settings: &UserSettings) -> WorkspaceSettingsDto {
    let (openai_asr_base_url, openai_asr_api_key, openai_asr_model) = match &settings.asr_backend {
        AsrBackendConfig::OpenAiCompatible {
            base_url,
            api_key,
            model,
            ..
        } => (
            Some(base_url.to_string()),
            Some(api_key.expose().to_owned()),
            Some(model.clone()),
        ),
        _ => (None, None, None),
    };

    let (openai_llm_base_url, openai_llm_api_key, openai_llm_model) = match &settings.llm_backend {
        LlmBackendConfig::OpenAiCompatible {
            base_url,
            api_key,
            model,
            ..
        } => (
            Some(base_url.to_string()),
            Some(api_key.expose().to_owned()),
            Some(model.clone()),
        ),
        _ => (None, None, None),
    };

    WorkspaceSettingsDto {
        activation_mode: settings.activation_mode.to_string(),
        dictation_mode: settings.dictation_mode.to_string(),
        use_gpu: !matches!(settings.device, DevicePolicy::Cpu),
        asr_backend_type: asr_backend_type(&settings.asr_backend).to_owned(),
        llm_backend_type: llm_backend_type(&settings.llm_backend).to_owned(),
        confidence_threshold: settings.pipeline.word_confidence_threshold,
        adaptive_vad: settings
            .pipeline
            .adaptive_vad
            .unwrap_or(settings.adaptive_vad),
        speech_threshold: settings.pipeline.speech_threshold.unwrap_or(0.5),
        style_preset: settings.default_style.clone(),
        hotkey_toggle: format_key_combo(&settings.hotkey.toggle_dictation),
        hotkey_cancel: format_key_combo(&settings.hotkey.cancel_dictation),
        aligner_enabled: settings.aligner.as_ref().is_some_and(|cfg| cfg.enabled),
        pill_visibility: settings.pill_visibility.to_string(),
        language: settings.language.clone(),
        openai_asr_base_url,
        openai_asr_api_key,
        openai_asr_model,
        openai_llm_base_url,
        openai_llm_api_key,
        openai_llm_model,
    }
}

fn apply_settings_dto(
    settings: &mut UserSettings,
    input: WorkspaceSettingsDto,
) -> Result<(), String> {
    settings.activation_mode = ActivationMode::from_str(input.activation_mode.trim())
        .map_err(|e| format!("activationMode: {e}"))?;
    settings.pill_visibility = PillVisibility::from_str(input.pill_visibility.trim())
        .map_err(|e| format!("pillVisibility: {e}"))?;
    settings.dictation_mode = DictationMode::from_str(input.dictation_mode.trim())
        .map_err(|e| format!("dictationMode: {e}"))?;
    settings.device = if input.use_gpu {
        DevicePolicy::Auto
    } else {
        DevicePolicy::Cpu
    };
    settings.asr_backend =
        asr_backend_from_type(&input.asr_backend_type, &settings.asr_backend, &input)?;
    settings.llm_backend =
        llm_backend_from_type(&input.llm_backend_type, &settings.llm_backend, &input)?;

    // Validate numeric thresholds — reject NaN/Inf/out-of-range.
    if let Some(ct) = input.confidence_threshold
        && (!ct.is_finite() || !(0.0..=1.0).contains(&ct))
    {
        return Err(format!("confidenceThreshold must be 0.0–1.0, got {ct}"));
    }
    if !input.speech_threshold.is_finite() || !(0.0..=1.0).contains(&input.speech_threshold) {
        return Err(format!(
            "speechThreshold must be 0.0–1.0, got {}",
            input.speech_threshold
        ));
    }

    settings.pipeline.word_confidence_threshold = input.confidence_threshold;
    settings.word_confidence_threshold = input.confidence_threshold.unwrap_or(0.5);
    settings.pipeline.adaptive_vad = Some(input.adaptive_vad);
    settings.adaptive_vad = input.adaptive_vad;
    settings.pipeline.speech_threshold = Some(input.speech_threshold);

    // Validate style_preset — reject empty or excessively long values.
    let style = input.style_preset.trim();
    if style.is_empty() || style.len() > 128 {
        return Err("stylePreset must be 1–128 characters".into());
    }
    settings.default_style = style.to_owned();

    settings.hotkey = HotkeyConfig {
        toggle_dictation: parse_key_combo(&input.hotkey_toggle)?,
        cancel_dictation: parse_key_combo(&input.hotkey_cancel)?,
    };
    if let Some(ref mut cfg) = settings.aligner {
        cfg.enabled = input.aligner_enabled;
    } else if input.aligner_enabled {
        settings.aligner = Some(AlignerConfig {
            enabled: true,
            model_dir: UserSettings::default()
                .aligner
                .map(|c| c.model_dir)
                .unwrap_or_default(),
        });
    }
    settings.language = input.language.trim().to_owned();
    Ok(())
}

fn asr_backend_type(config: &AsrBackendConfig) -> &'static str {
    match config {
        AsrBackendConfig::ParakeetTdt { .. } => "ParakeetTdt",
        AsrBackendConfig::OrtQwenAsr { .. } => "OrtQwenAsr",
        AsrBackendConfig::OpenAiCompatible { .. } => "OpenAiCompatible",
        _ => UNKNOWN_LABEL,
    }
}

fn llm_backend_type(config: &LlmBackendConfig) -> &'static str {
    match config {
        LlmBackendConfig::LlamaCppLocal { .. } => "LlamaCppLocal",
        LlmBackendConfig::OpenAiCompatible { .. } => "OpenAiCompatible",
        _ => UNKNOWN_LABEL,
    }
}

fn asr_backend_from_type(
    kind: &str,
    current: &AsrBackendConfig,
    input: &WorkspaceSettingsDto,
) -> Result<AsrBackendConfig, String> {
    Ok(match kind {
        "ParakeetTdt" => AsrBackendConfig::ParakeetTdt {
            model_dir: current_parakeet_model_dir(current),
        },
        "OrtQwenAsr" => AsrBackendConfig::OrtQwenAsr {
            model_dir: current_qwen_model_dir(current),
        },
        "OpenAiCompatible" => {
            // Prefer incoming DTO fields; fall back to current config values if already OpenAI.
            let (cur_url, cur_key, cur_model) = match current {
                AsrBackendConfig::OpenAiCompatible {
                    base_url,
                    api_key,
                    model,
                    ..
                } => (
                    base_url.to_string(),
                    api_key.expose().to_owned(),
                    model.clone(),
                ),
                _ => (
                    "https://api.openai.com/v1".to_owned(),
                    String::new(),
                    "whisper-1".to_owned(),
                ),
            };
            let base_url_str = input.openai_asr_base_url.as_deref().unwrap_or(&cur_url);
            let api_key_str = input.openai_asr_api_key.as_deref().unwrap_or(&cur_key);
            let model_str = input.openai_asr_model.as_deref().unwrap_or(&cur_model);
            let base_url: url::Url = base_url_str
                .parse()
                .map_err(|e| format!("invalid openaiAsrBaseUrl: {e}"))?;
            validate_api_url(&base_url, "openaiAsrBaseUrl")?;
            AsrBackendConfig::OpenAiCompatible {
                base_url,
                api_key: RedactedString::new(api_key_str),
                model: model_str.to_owned(),
                streaming: StreamingMode::HttpPolling,
            }
        }
        _ => return Err(format!("unsupported asrBackendType: {kind}")),
    })
}

fn llm_backend_from_type(
    kind: &str,
    current: &LlmBackendConfig,
    input: &WorkspaceSettingsDto,
) -> Result<LlmBackendConfig, String> {
    Ok(match kind {
        "LlamaCppLocal" if matches!(current, LlmBackendConfig::LlamaCppLocal { .. }) => {
            current.clone()
        }
        "LlamaCppLocal" => UserSettings::default().llm_backend,
        "OpenAiCompatible" => {
            // Prefer incoming DTO fields; fall back to current config values if already OpenAI.
            let (cur_url, cur_key, cur_model, cur_max_tokens) = match current {
                LlmBackendConfig::OpenAiCompatible {
                    base_url,
                    api_key,
                    model,
                    max_tokens,
                } => (
                    base_url.to_string(),
                    api_key.expose().to_owned(),
                    model.clone(),
                    *max_tokens,
                ),
                _ => (
                    "https://api.openai.com/v1".to_owned(),
                    String::new(),
                    "gpt-4o-mini".to_owned(),
                    512u32,
                ),
            };
            let base_url_str = input.openai_llm_base_url.as_deref().unwrap_or(&cur_url);
            let api_key_str = input.openai_llm_api_key.as_deref().unwrap_or(&cur_key);
            let model_str = input.openai_llm_model.as_deref().unwrap_or(&cur_model);
            let base_url: url::Url = base_url_str
                .parse()
                .map_err(|e| format!("invalid openaiLlmBaseUrl: {e}"))?;
            validate_api_url(&base_url, "openaiLlmBaseUrl")?;
            LlmBackendConfig::OpenAiCompatible {
                base_url,
                api_key: RedactedString::new(api_key_str),
                model: model_str.to_owned(),
                max_tokens: cur_max_tokens,
            }
        }
        _ => return Err(format!("unsupported llmBackendType: {kind}")),
    })
}

/// Validate that an API URL uses HTTPS (or is a loopback address for local dev).
fn validate_api_url(url: &url::Url, field: &str) -> Result<(), String> {
    match url.scheme() {
        "https" => Ok(()),
        "http" => {
            let host = url.host_str().unwrap_or_default();
            if host == "localhost" || host == "127.0.0.1" || host == "[::1]" {
                Ok(())
            } else {
                Err(format!(
                    "{field}: HTTP is only allowed for localhost; use HTTPS for remote endpoints"
                ))
            }
        }
        scheme => Err(format!(
            "{field}: unsupported URL scheme \"{scheme}\", expected https"
        )),
    }
}

fn current_parakeet_model_dir(current: &AsrBackendConfig) -> PathBuf {
    match current {
        AsrBackendConfig::ParakeetTdt { model_dir } => model_dir.clone(),
        _ => PathBuf::new(),
    }
}

fn current_qwen_model_dir(current: &AsrBackendConfig) -> PathBuf {
    match current {
        AsrBackendConfig::OrtQwenAsr { model_dir } => model_dir.clone(),
        _ => PathBuf::new(),
    }
}

/// Maximum number of modifier+key parts in a hotkey combo.
const MAX_COMBO_PARTS: usize = 8;

fn parse_key_combo(raw: &str) -> Result<KeyCombo, String> {
    let parts: Vec<_> = raw
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .take(MAX_COMBO_PARTS)
        .collect();
    if parts.is_empty() {
        return Err("hotkey cannot be empty".into());
    }

    let key = parts[parts.len() - 1];
    // Reject key names that are empty or unreasonably long.
    if key.is_empty() || key.len() > 32 {
        return Err(format!("invalid hotkey key name: {key:?}"));
    }

    let mut modifiers = Vec::new();
    for modifier in &parts[..parts.len() - 1] {
        modifiers.push(parse_modifier(modifier)?);
    }

    Ok(KeyCombo {
        modifiers,
        key: key.to_string(),
    })
}

fn parse_modifier(raw: &str) -> Result<Modifier, String> {
    match raw {
        "Ctrl" => Ok(Modifier::Ctrl),
        "Shift" => Ok(Modifier::Shift),
        "Alt" => Ok(Modifier::Alt),
        "Meta" => Ok(Modifier::Meta),
        _ => Err(format!("unsupported hotkey modifier: {raw}")),
    }
}

fn format_key_combo(combo: &KeyCombo) -> String {
    let mut parts: Vec<&str> = combo.modifiers.iter().map(modifier_label).collect();
    parts.push(combo.key.as_str());
    parts.join("+")
}

fn modifier_label(modifier: &Modifier) -> &'static str {
    match modifier {
        Modifier::Ctrl => "Ctrl",
        Modifier::Shift => "Shift",
        Modifier::Alt => "Alt",
        Modifier::Meta => "Meta",
        _ => UNKNOWN_LABEL,
    }
}

fn map_dictionary_entry(entry: StoredDictionaryEntry) -> DictionaryItemDto {
    DictionaryItemDto {
        id: entry.id,
        phrase: entry.phrase,
        replacement: entry.replacement,
        priority: entry.priority,
    }
}

fn session_status_label(status: &SessionStatus) -> String {
    match status {
        SessionStatus::Active => "Active".into(),
        SessionStatus::Completed => "Completed".into(),
        SessionStatus::Failed(message) => format!("Failed: {message}"),
        SessionStatus::Cancelled => "Cancelled".into(),
        _ => UNKNOWN_LABEL.into(),
    }
}
