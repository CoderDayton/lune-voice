use std::path::PathBuf;

use lune_core::config::{
    ActivationMode, AlignerConfig, AsrBackendConfig, DevicePolicy, DictationMode, HotkeyConfig,
    KeyCombo, LlmBackendConfig, Modifier, PillVisibility, RedactedString, StreamingMode,
    UserSettings, builtin_registry,
};
use lune_core::session::SessionStatus;
use lune_persist::{AppPaths, Database, SettingsStore, StoredDictionaryEntry};

use crate::dto::{
    ActivationModeDto, AsrBackendTypeDto, CreateDictionaryEntryInput, DevicePolicyDto,
    DictationModeDto, DictionaryItemDto, HistoryItemDto, LlmBackendTypeDto, PillVisibilityDto,
    SaveWorkspaceSettingsInput, WorkspaceSettingsDto, WorkspaceSnapshot,
};

const HISTORY_LIMIT: u32 = 200;
const DICTIONARY_LIMIT: usize = 10_000;
const UNKNOWN_LABEL: &str = "Unknown";

fn is_wellformed_bcp47(tag: &str) -> bool {
    let mut parts = tag.split('-');
    match parts.next() {
        Some(p) if (2..=8).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_alphabetic()) => {}
        _ => return false,
    }
    parts.all(|p| {
        let len = p.len();
        (1..=8).contains(&len) && p.bytes().all(|b| b.is_ascii_alphanumeric())
    })
}

pub fn load_workspace_snapshot() -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let settings = load_settings(&paths)?;
    let mut database = open_database(&paths)?;
    build_snapshot(settings, &mut database)
}

pub fn save_workspace_settings(
    input: SaveWorkspaceSettingsInput,
) -> Result<WorkspaceSnapshot, String> {
    let paths = resolve_paths()?;
    let mut settings = load_settings(&paths)?;
    apply_settings_dto(&mut settings, input)?;
    let mut database = open_database(&paths)?;
    save_settings(&paths, &settings)?;
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
    let mut settings = store.load().map_err(|e| format!("load settings: {e}"))?;
    settings.migrate_pipeline_fields();
    Ok(settings)
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
    let (openai_asr_base_url, openai_asr_model, openai_asr_api_key_set) =
        match &settings.asr_backend {
            AsrBackendConfig::OpenAiCompatible {
                base_url,
                model,
                api_key,
                ..
            } => (
                Some(base_url.to_string()),
                Some(model.clone()),
                !api_key.expose().is_empty(),
            ),
            _ => (None, None, false),
        };

    let (openai_llm_base_url, openai_llm_model, openai_llm_api_key_set) =
        match &settings.llm_backend {
            LlmBackendConfig::OpenAiCompatible {
                base_url,
                model,
                api_key,
                ..
            } => (
                Some(base_url.to_string()),
                Some(model.clone()),
                !api_key.expose().is_empty(),
            ),
            _ => (None, None, false),
        };

    WorkspaceSettingsDto {
        activation_mode: activation_mode_to_dto(settings.activation_mode),
        dictation_mode: dictation_mode_to_dto(settings.dictation_mode),
        use_gpu: !matches!(settings.device, DevicePolicy::Cpu),
        device_policy: device_policy_to_dto(settings.device),
        asr_backend_type: asr_backend_type_to_dto(&settings.asr_backend),
        llm_backend_type: llm_backend_type_to_dto(&settings.llm_backend),
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
        pill_visibility: pill_visibility_to_dto(settings.pill_visibility),
        language: settings.language.clone(),
        openai_asr_base_url,
        openai_asr_model,
        openai_asr_api_key_set,
        openai_llm_base_url,
        openai_llm_model,
        openai_llm_api_key_set,
    }
}

fn apply_settings_dto(
    settings: &mut UserSettings,
    input: SaveWorkspaceSettingsInput,
) -> Result<(), String> {
    settings.activation_mode = activation_mode_from_dto(input.activation_mode);
    settings.pill_visibility = pill_visibility_from_dto(input.pill_visibility);
    settings.dictation_mode = dictation_mode_from_dto(input.dictation_mode);
    settings.device = resolve_device_policy(input.use_gpu, input.device_policy);
    settings.asr_backend =
        asr_backend_from_type(input.asr_backend_type, &settings.asr_backend, &input)?;
    settings.llm_backend =
        llm_backend_from_type(input.llm_backend_type, &settings.llm_backend, &input)?;

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
    if builtin_registry().get(style).is_none() {
        return Err(format!("unknown stylePreset: {style}"));
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
    let language = input.language.trim();
    if !is_wellformed_bcp47(language) {
        return Err(
            "language must be a well-formed BCP-47 tag (e.g. \"en\", \"en-US\", \"pt-BR\")".into(),
        );
    }
    settings.language = language.to_owned();
    Ok(())
}

// ── DTO ↔ Core conversions ──────────────────────────────────────────────

fn activation_mode_to_dto(mode: ActivationMode) -> ActivationModeDto {
    match mode {
        ActivationMode::Toggle => ActivationModeDto::Toggle,
        ActivationMode::Hold => ActivationModeDto::Hold,
        _ => ActivationModeDto::Toggle,
    }
}

fn activation_mode_from_dto(dto: ActivationModeDto) -> ActivationMode {
    match dto {
        ActivationModeDto::Toggle => ActivationMode::Toggle,
        ActivationModeDto::Hold => ActivationMode::Hold,
    }
}

fn dictation_mode_to_dto(mode: DictationMode) -> DictationModeDto {
    match mode {
        DictationMode::TypeInPlace => DictationModeDto::TypeInPlace,
        DictationMode::Clipboard => DictationModeDto::Clipboard,
        DictationMode::Both => DictationModeDto::Both,
        _ => DictationModeDto::Clipboard,
    }
}

fn dictation_mode_from_dto(dto: DictationModeDto) -> DictationMode {
    match dto {
        DictationModeDto::TypeInPlace => DictationMode::TypeInPlace,
        DictationModeDto::Clipboard => DictationMode::Clipboard,
        DictationModeDto::Both => DictationMode::Both,
    }
}

fn device_policy_to_dto(policy: DevicePolicy) -> DevicePolicyDto {
    match policy {
        DevicePolicy::Cpu => DevicePolicyDto::Cpu,
        DevicePolicy::Cuda => DevicePolicyDto::Cuda,
        DevicePolicy::Auto => DevicePolicyDto::Auto,
    }
}

fn resolve_device_policy(use_gpu: bool, policy: DevicePolicyDto) -> DevicePolicy {
    if !use_gpu {
        return DevicePolicy::Cpu;
    }
    match policy {
        DevicePolicyDto::Cuda => DevicePolicy::Cuda,
        DevicePolicyDto::Auto | DevicePolicyDto::Cpu => DevicePolicy::Auto,
    }
}

fn asr_backend_type_to_dto(config: &AsrBackendConfig) -> AsrBackendTypeDto {
    match config {
        AsrBackendConfig::ParakeetTdt { .. } => AsrBackendTypeDto::ParakeetTdt,
        AsrBackendConfig::OpenAiCompatible { .. } => AsrBackendTypeDto::OpenAiCompatible,
        _ => AsrBackendTypeDto::ParakeetTdt,
    }
}

fn llm_backend_type_to_dto(config: &LlmBackendConfig) -> LlmBackendTypeDto {
    match config {
        LlmBackendConfig::LlamaCppLocal { .. } => LlmBackendTypeDto::LlamaCppLocal,
        LlmBackendConfig::OpenAiCompatible { .. } => LlmBackendTypeDto::OpenAiCompatible,
        _ => LlmBackendTypeDto::LlamaCppLocal,
    }
}

fn pill_visibility_to_dto(vis: PillVisibility) -> PillVisibilityDto {
    match vis {
        PillVisibility::Always => PillVisibilityDto::Always,
        PillVisibility::WhenDictating => PillVisibilityDto::WhenDictating,
        PillVisibility::Never => PillVisibilityDto::Never,
        _ => PillVisibilityDto::Always,
    }
}

fn pill_visibility_from_dto(dto: PillVisibilityDto) -> PillVisibility {
    match dto {
        PillVisibilityDto::Always => PillVisibility::Always,
        PillVisibilityDto::WhenDictating => PillVisibility::WhenDictating,
        PillVisibilityDto::Never => PillVisibility::Never,
    }
}

fn asr_backend_from_type(
    kind: AsrBackendTypeDto,
    current: &AsrBackendConfig,
    input: &SaveWorkspaceSettingsInput,
) -> Result<AsrBackendConfig, String> {
    Ok(match kind {
        AsrBackendTypeDto::ParakeetTdt => AsrBackendConfig::ParakeetTdt {
            model_dir: current_parakeet_model_dir(current),
        },
        AsrBackendTypeDto::OpenAiCompatible => {
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
    })
}

fn llm_backend_from_type(
    kind: LlmBackendTypeDto,
    current: &LlmBackendConfig,
    input: &SaveWorkspaceSettingsInput,
) -> Result<LlmBackendConfig, String> {
    Ok(match kind {
        LlmBackendTypeDto::LlamaCppLocal
            if matches!(current, LlmBackendConfig::LlamaCppLocal { .. }) =>
        {
            current.clone()
        }
        LlmBackendTypeDto::LlamaCppLocal => UserSettings::default().llm_backend,
        LlmBackendTypeDto::OpenAiCompatible => {
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
    })
}

/// Validate that an API URL uses HTTPS (or is a loopback address for local dev).
fn validate_api_url(url: &url::Url, field: &str) -> Result<(), String> {
    match url.scheme() {
        "https" => Ok(()),
        "http" => {
            let host = url.host_str().unwrap_or_default();
            if host == "localhost" || host == "127.0.0.1" || host == "::1" || host == "[::1]" {
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

/// Maximum number of modifier+key parts in a hotkey combo.
const MAX_COMBO_PARTS: usize = 8;

fn parse_key_combo(raw: &str) -> Result<KeyCombo, String> {
    let raw_parts: Vec<_> = raw
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if raw_parts.is_empty() {
        return Err("hotkey cannot be empty".into());
    }
    if raw_parts.len() > MAX_COMBO_PARTS {
        return Err(format!(
            "hotkey cannot contain more than {MAX_COMBO_PARTS} parts"
        ));
    }

    let parts = raw_parts;
    let key = parts[parts.len() - 1];
    // Reject key names that are empty or unreasonably long.
    if key.is_empty() || key.len() > 32 {
        return Err(format!("invalid hotkey key name: {key:?}"));
    }

    let mut modifiers = Vec::new();
    let mut seen_modifiers = std::collections::HashSet::new();
    for modifier in &parts[..parts.len() - 1] {
        let parsed = parse_modifier(modifier)?;
        if !seen_modifiers.insert(modifier.to_ascii_lowercase()) {
            return Err(format!("duplicate hotkey modifier: {modifier}"));
        }
        modifiers.push(parsed);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_settings_migrates_legacy_pipeline_fields() {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::from_dirs(dir.path().join("config"), dir.path().join("data"));

        let mut legacy = UserSettings::default();
        legacy.pipeline = Default::default();
        legacy.word_confidence_threshold = 0.77;
        legacy.max_speech_secs = 42;
        legacy.adaptive_vad = true;
        save_settings(&paths, &legacy).unwrap();

        let loaded = load_settings(&paths).unwrap();
        assert_eq!(loaded.pipeline.word_confidence_threshold, Some(0.77));
        assert_eq!(loaded.pipeline.max_speech_secs, Some(42));
        assert_eq!(loaded.pipeline.adaptive_vad, Some(true));
    }

    /// Build a `SaveWorkspaceSettingsInput` from defaults for test convenience.
    fn default_save_input() -> SaveWorkspaceSettingsInput {
        let dto = settings_to_dto(&UserSettings::default());
        SaveWorkspaceSettingsInput {
            activation_mode: dto.activation_mode,
            dictation_mode: dto.dictation_mode,
            use_gpu: dto.use_gpu,
            device_policy: dto.device_policy,
            asr_backend_type: dto.asr_backend_type,
            llm_backend_type: dto.llm_backend_type,
            confidence_threshold: dto.confidence_threshold,
            adaptive_vad: dto.adaptive_vad,
            speech_threshold: dto.speech_threshold,
            style_preset: dto.style_preset,
            hotkey_toggle: dto.hotkey_toggle,
            hotkey_cancel: dto.hotkey_cancel,
            aligner_enabled: dto.aligner_enabled,
            pill_visibility: dto.pill_visibility,
            language: dto.language,
            openai_asr_base_url: dto.openai_asr_base_url,
            openai_asr_api_key: None,
            openai_asr_model: dto.openai_asr_model,
            openai_llm_base_url: dto.openai_llm_base_url,
            openai_llm_api_key: None,
            openai_llm_model: dto.openai_llm_model,
        }
    }

    #[test]
    fn apply_settings_rejects_invalid_style_and_language() {
        let mut settings = UserSettings::default();

        let mut input = default_save_input();
        input.style_preset = "not-a-style".into();
        assert!(apply_settings_dto(&mut settings, input).is_err());

        let mut input = default_save_input();
        input.language = "bad tag".into();
        assert!(apply_settings_dto(&mut settings, input).is_err());
    }

    #[test]
    fn device_policy_round_trips_cuda_without_downgrade() {
        let mut settings = UserSettings {
            device: DevicePolicy::Cuda,
            ..UserSettings::default()
        };
        let mut input = default_save_input();
        input.use_gpu = true;
        input.device_policy = DevicePolicyDto::Cuda;

        apply_settings_dto(&mut settings, input).unwrap();
        assert!(matches!(settings.device, DevicePolicy::Cuda));
    }

    #[test]
    fn settings_to_dto_does_not_expose_api_keys() {
        let settings = UserSettings {
            asr_backend: AsrBackendConfig::OpenAiCompatible {
                base_url: "https://api.openai.com/v1".parse().unwrap(),
                api_key: RedactedString::new("secret-asr"),
                model: "whisper-1".into(),
                streaming: StreamingMode::HttpPolling,
            },
            llm_backend: LlmBackendConfig::OpenAiCompatible {
                base_url: "https://api.openai.com/v1".parse().unwrap(),
                api_key: RedactedString::new("secret-llm"),
                model: "gpt-4o-mini".into(),
                max_tokens: 128,
            },
            ..UserSettings::default()
        };

        let dto = settings_to_dto(&settings);
        // Keys are not in the read DTO — only the boolean flags.
        assert!(dto.openai_asr_api_key_set);
        assert!(dto.openai_llm_api_key_set);
        // Verify the serialized JSON contains no actual key values.
        let json = serde_json::to_string(&dto).unwrap();
        assert!(!json.contains("secret-asr"));
        assert!(!json.contains("secret-llm"));
    }

    #[test]
    fn parse_key_combo_rejects_duplicate_modifiers_and_overflow() {
        assert!(parse_key_combo("Ctrl+Ctrl+Space").is_err());
        assert!(parse_key_combo("Ctrl+Alt+Shift+Meta+Fn+Hyper+Super+Caps+A").is_err());
    }

    #[test]
    fn validate_api_url_allows_ipv6_loopback_http() {
        let url = url::Url::parse("http://[::1]:8080/v1").unwrap();
        assert!(validate_api_url(&url, "openaiAsrBaseUrl").is_ok());
    }
}
