// SPDX-License-Identifier: GPL-3.0-or-later

mod hotkey;
mod modes;
mod pipeline;
mod redacted;
mod settings;
mod style_llm;

// Re-export everything so `use lune_core::config::X` paths remain valid.
pub use hotkey::{HotkeyConfig, KeyCombo, Modifier};
pub use modes::{
    ActivationMode, AgcMode, DictationMode, PillVisibility, PrivacyMode, TerminalMode,
};
pub use pipeline::{PipelineConfig, PipelineOverrides};
pub use redacted::RedactedString;
pub use settings::{
    AsrBackendConfig, DevicePolicy, HistoryRetention, LlmBackendConfig, OpenAiLlmProvider,
    ParakeetModel, UserSettings, WhisperModel, is_local_endpoint, require_tls_or_local,
};
pub use settings::{PillAnchor, PillCorner, PillOrientation, PillPosition, PillSize};
pub use style_llm::{
    BASE_SYSTEM_PROMPT, FormatToggles, StyleId, StylePreset, StyleRegistry, builtin_registry,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_settings_toml_round_trip() {
        let settings = UserSettings::default();
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        let parsed: UserSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.language, "auto");
        assert_eq!(parsed.privacy_mode, PrivacyMode::LocalOnly);
        assert_eq!(parsed.dictation_mode, DictationMode::Clipboard);
        assert_eq!(parsed.activation_mode, ActivationMode::Toggle);
    }

    #[test]
    fn hotkey_config_serde_round_trip() {
        let config = HotkeyConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: HotkeyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.toggle_dictation.key, "Space");
        assert_eq!(parsed.toggle_dictation.modifiers, vec![Modifier::Ctrl]);
    }

    #[test]
    fn style_preset_serde_round_trip() {
        let preset = StylePreset::casual();
        let json = serde_json::to_string(&preset).unwrap();
        let parsed: StylePreset = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "casual");
        assert!(!parsed.few_shot_pairs.is_empty());
    }

    #[test]
    fn privacy_mode_non_exhaustive() {
        let json = r#""LocalOnly""#;
        let mode: PrivacyMode = serde_json::from_str(json).unwrap();
        assert_eq!(mode, PrivacyMode::LocalOnly);
    }

    #[test]
    fn builtin_presets_unique_ids() {
        let presets = StylePreset::all_builtin();
        assert_eq!(presets.len(), 5);
        let ids: Vec<&str> = presets.iter().map(|p| p.id.as_str()).collect();
        assert!(ids.contains(&"casual"));
        assert!(ids.contains(&"professional"));
        assert!(ids.contains(&"formal"));
        assert!(ids.contains(&"technical"));
        assert!(ids.contains(&"writing"));
        // All IDs are unique.
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(unique.len(), presets.len());
    }

    #[test]
    fn builtin_presets_have_few_shots() {
        for preset in StylePreset::all_builtin() {
            assert!(
                !preset.few_shot_pairs.is_empty(),
                "preset '{}' has no few-shot pairs",
                preset.id
            );
        }
    }

    #[test]
    fn all_builtin_returns_same_ptr() {
        let a = StylePreset::all_builtin().as_ptr();
        let b = StylePreset::all_builtin().as_ptr();
        assert_eq!(a, b);
    }

    #[test]
    fn registry_get_by_id() {
        let reg = builtin_registry();
        assert!(reg.get("casual").is_some());
        assert!(reg.get("technical").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn registry_len() {
        assert_eq!(builtin_registry().all().len(), 5);
    }

    #[test]
    fn dictation_mode_fromstr_err_is_cold() {
        let err = "bad".parse::<DictationMode>().unwrap_err();
        assert!(err.contains("TypeInPlace|Clipboard"));
    }

    #[test]
    fn activation_mode_fromstr_err_is_cold() {
        let err = "bad".parse::<ActivationMode>().unwrap_err();
        assert!(err.contains("Toggle|Hold"));
    }

    #[test]
    fn example_settings_toml_parses() {
        let example = include_str!("../../../../config/settings.toml");
        let parsed: UserSettings = toml::from_str(example)
            .expect("config/settings.toml must parse against current schema");
        assert_eq!(parsed.default_style, None);
        assert_eq!(parsed.language, "auto");
        assert_eq!(parsed.dictation_language, "auto");
        assert_eq!(parsed.agc_mode, AgcMode::Auto);
        assert!(matches!(
            parsed.asr_backend,
            AsrBackendConfig::WhisperCpp { .. }
        ));
        assert!(matches!(parsed.llm_backend, LlmBackendConfig::Disabled));
    }

    #[test]
    fn a_settings_file_with_removed_keys_and_values_still_loads() {
        let parsed: UserSettings = toml::from_str(
            r#"
language = "en"
mic_prewarm = true
vad_pcen = false
privacy_mode = "LocalPreferred"
dictation_mode = "Both"
device = "cuda"

[asr_backend]
type = "Qwen3Asr"
"#,
        )
        .expect("removed keys and values must not fail the load");
        assert_eq!(parsed.language, "en");
        assert_eq!(parsed.privacy_mode, PrivacyMode::LocalOnly);
        assert_eq!(parsed.dictation_mode, DictationMode::default());
        assert!(!parsed.copy_to_clipboard);
        assert_eq!(parsed.device, DevicePolicy::Auto);
        assert_eq!(parsed.agc_mode, AgcMode::Auto);
        assert!(matches!(
            parsed.asr_backend,
            AsrBackendConfig::WhisperCpp { .. }
        ));
    }

    #[test]
    fn a_remote_asr_backend_with_a_removed_streaming_key_still_loads() {
        let parsed: UserSettings = toml::from_str(
            r#"
[asr_backend]
type = "OpenAiCompatible"
base_url = "https://api.openai.com/v1"
api_key = "k"
model = "whisper-1"
streaming = "HttpPolling"
"#,
        )
        .expect("a removed key must be ignored");
        assert!(matches!(
            parsed.asr_backend,
            AsrBackendConfig::OpenAiCompatible { .. }
        ));
    }
}
