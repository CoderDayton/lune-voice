use std::str::FromStr;

use lune_core::config::{
    ActivationMode, AsrBackendConfig, DevicePolicy, DictationMode, HotkeyConfig, KeyCombo,
    LlmBackendConfig, Modifier, UserSettings,
};
use lune_core::session::SessionStatus;
use lune_persist::{AppPaths, Database, SettingsStore, StoredDictionaryEntry};

use crate::dto::{
    CreateDictionaryEntryInput, DictionaryItemDto, HistoryItemDto, WorkspaceSettingsDto,
    WorkspaceSnapshot,
};

const HISTORY_LIMIT: u32 = 200;

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

pub fn create_dictionary_entry(
    input: CreateDictionaryEntryInput,
) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let phrase = input.phrase.trim();
    let replacement = input.replacement.trim();
    if phrase.is_empty() || replacement.is_empty() {
        return Err("phrase and replacement are required".into());
    }

    let mut database = open_database(&paths)?;
    database
        .insert_dictionary_entry(phrase, replacement, input.priority)
        .map_err(|e| format!("insert dictionary entry: {e}"))?;

    let settings = load_settings(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn delete_dictionary_entry(entry_id: i32) -> Result<WorkspaceSnapshot, String> {
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

    let dictionary = database
        .list_dictionary_entries_with_ids()
        .map_err(|e| format!("list dictionary entries: {e}"))?
        .into_iter()
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
    }
}

fn apply_settings_dto(
    settings: &mut UserSettings,
    input: WorkspaceSettingsDto,
) -> Result<(), String> {
    settings.activation_mode = ActivationMode::from_str(&input.activation_mode)
        .map_err(|e| format!("activationMode: {e}"))?;
    settings.dictation_mode = DictationMode::from_str(&input.dictation_mode)
        .map_err(|e| format!("dictationMode: {e}"))?;
    settings.device = if input.use_gpu {
        DevicePolicy::Auto
    } else {
        DevicePolicy::Cpu
    };
    settings.asr_backend = asr_backend_from_type(&input.asr_backend_type, &settings.asr_backend)?;
    settings.llm_backend = llm_backend_from_type(&input.llm_backend_type, &settings.llm_backend)?;
    settings.pipeline.word_confidence_threshold = input.confidence_threshold;
    settings.word_confidence_threshold = input.confidence_threshold.unwrap_or(0.5);
    settings.pipeline.adaptive_vad = Some(input.adaptive_vad);
    settings.adaptive_vad = input.adaptive_vad;
    settings.pipeline.speech_threshold = Some(input.speech_threshold);
    settings.default_style = input.style_preset;
    settings.hotkey = HotkeyConfig {
        toggle_dictation: parse_key_combo(&input.hotkey_toggle)?,
        cancel_dictation: parse_key_combo(&input.hotkey_cancel)?,
    };
    Ok(())
}

fn asr_backend_type(config: &AsrBackendConfig) -> &'static str {
    match config {
        AsrBackendConfig::ParakeetTdt { .. } => "ParakeetTdt",
        AsrBackendConfig::OrtQwenAsr { .. } => "OrtQwenAsr",
        AsrBackendConfig::OpenAiCompatible { .. } => "OpenAiCompatible",
        _ => "Unknown",
    }
}

fn llm_backend_type(config: &LlmBackendConfig) -> &'static str {
    match config {
        LlmBackendConfig::LlamaCppLocal { .. } => "LlamaCppLocal",
        LlmBackendConfig::OpenAiCompatible { .. } => "OpenAiCompatible",
        _ => "Unknown",
    }
}

fn asr_backend_from_type(
    kind: &str,
    current: &AsrBackendConfig,
) -> Result<AsrBackendConfig, String> {
    Ok(match kind {
        "ParakeetTdt" => match current {
            AsrBackendConfig::ParakeetTdt { model_dir } => AsrBackendConfig::ParakeetTdt {
                model_dir: model_dir.clone(),
            },
            _ => AsrBackendConfig::ParakeetTdt {
                model_dir: std::path::PathBuf::new(),
            },
        },
        "OrtQwenAsr" => match current {
            AsrBackendConfig::OrtQwenAsr { model_dir } => AsrBackendConfig::OrtQwenAsr {
                model_dir: model_dir.clone(),
            },
            _ => AsrBackendConfig::OrtQwenAsr {
                model_dir: std::path::PathBuf::new(),
            },
        },
        "OpenAiCompatible" => match current {
            AsrBackendConfig::OpenAiCompatible { .. } => current.clone(),
            _ => return Err("OpenAiCompatible ASR cannot be created from this UI yet".into()),
        },
        _ => return Err(format!("unsupported asrBackendType: {kind}")),
    })
}

fn llm_backend_from_type(
    kind: &str,
    current: &LlmBackendConfig,
) -> Result<LlmBackendConfig, String> {
    Ok(match kind {
        "LlamaCppLocal" => match current {
            LlmBackendConfig::LlamaCppLocal { .. } => current.clone(),
            _ => UserSettings::default().llm_backend,
        },
        "OpenAiCompatible" => match current {
            LlmBackendConfig::OpenAiCompatible { .. } => current.clone(),
            _ => return Err("OpenAiCompatible LLM cannot be created from this UI yet".into()),
        },
        _ => return Err(format!("unsupported llmBackendType: {kind}")),
    })
}

fn parse_key_combo(raw: &str) -> Result<KeyCombo, String> {
    let parts: Vec<_> = raw
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        return Err("hotkey cannot be empty".into());
    }

    let mut modifiers = Vec::new();
    for modifier in &parts[..parts.len() - 1] {
        modifiers.push(parse_modifier(modifier)?);
    }

    Ok(KeyCombo {
        modifiers,
        key: parts[parts.len() - 1].to_string(),
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
        _ => "Unknown",
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
        _ => "Unknown".into(),
    }
}
