use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use url::Url;
use zeroize::Zeroize;

/// A string that redacts its value in Debug output and zeroizes on drop.
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RedactedString(String);

impl RedactedString {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for RedactedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl Drop for RedactedString {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

/// Top-level user settings, persisted as TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSettings {
    pub asr_backend: AsrBackendConfig,
    pub llm_backend: LlmBackendConfig,
    pub hotkey: HotkeyConfig,
    pub default_style: StyleId,
    /// BCP-47 tag, e.g. "en-US"
    pub language: String,
    pub privacy_mode: PrivacyMode,
    pub output_mode: OutputMode,
    /// None = session history disabled
    pub session_history_days: Option<u32>,
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            asr_backend: AsrBackendConfig::ParakeetTdt {
                model_dir: PathBuf::new(),
                use_gpu: true,
                context_secs: 30.0,
            },
            llm_backend: LlmBackendConfig::Local {
                model_path: PathBuf::new(),
                use_gpu: true,
                n_ctx: 2048,
            },
            hotkey: HotkeyConfig::default(),
            default_style: "general".into(),
            language: "en-US".into(),
            privacy_mode: PrivacyMode::LocalOnly,
            output_mode: OutputMode::LiveTyping,
            session_history_days: Some(30),
        }
    }
}

pub type StyleId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AsrBackendConfig {
    ParakeetTdt {
        model_dir: PathBuf,
        use_gpu: bool,
        /// Sliding context window size in seconds.
        context_secs: f32,
    },
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        streaming: StreamingMode,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum LlmBackendConfig {
    Local {
        model_path: PathBuf,
        use_gpu: bool,
        n_ctx: u32,
    },
    OpenAiCompatible {
        base_url: Url,
        api_key: RedactedString,
        model: String,
        max_tokens: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamingMode {
    WebSocket,
    HttpPolling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PrivacyMode {
    /// Reject any remote backend usage.
    LocalOnly,
    /// Use remote only if explicitly configured + consented.
    LocalPreferred,
    /// Use whatever is configured.
    AllowRemote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OutputMode {
    /// Stream partials with correction.
    LiveTyping,
    /// Emit final text only when session ends.
    TypeOnStop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub toggle_dictation: KeyCombo,
    pub cancel_dictation: KeyCombo,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            toggle_dictation: KeyCombo {
                modifiers: vec![Modifier::Ctrl],
                key: "Space".into(),
            },
            cancel_dictation: KeyCombo {
                modifiers: vec![],
                key: "Escape".into(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyCombo {
    pub modifiers: Vec<Modifier>,
    pub key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Modifier {
    Ctrl,
    Shift,
    Alt,
    Meta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylePreset {
    pub id: StyleId,
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    /// Template with `{raw_text}`, `{context}` slots.
    pub user_prompt_template: String,
    pub few_shot_pairs: Vec<(String, String)>,
}

impl StylePreset {
    pub fn all_builtin() -> Vec<Self> {
        vec![Self::general(), Self::email(), Self::slack(), Self::meeting_notes(), Self::code_comments()]
    }

    /// Built-in "General" preset: grammar + disfluency removal.
    pub fn general() -> Self {
        Self {
            id: "general".into(),
            name: "General".into(),
            description: "Clean grammar and remove disfluencies".into(),
            system_prompt: "You are a text post-processor. Clean up the following dictated text: fix grammar, remove filler words (um, uh, like), fix punctuation. Preserve the original meaning and tone. Output only the cleaned text.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "so um I was thinking that we should like go to the store".into(),
                    "I was thinking that we should go to the store.".into(),
                ),
                (
                    "the the meeting is uh scheduled for tomorrow at 3".into(),
                    "The meeting is scheduled for tomorrow at 3.".into(),
                ),
            ],
        }
    }

    pub fn email() -> Self {
        Self {
            id: "email".into(),
            name: "Email".into(),
            description: "Formal email with proper greeting and sign-off".into(),
            system_prompt: "You are a text post-processor for dictated emails. Fix grammar, \
                remove filler words, ensure formal register. Add paragraph breaks where natural. \
                Preserve the sender's intent exactly. Output only the cleaned text.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "hey uh just wanted to follow up on the thing we discussed last week about the project timeline".into(),
                    "I wanted to follow up on our discussion last week regarding the project timeline.".into(),
                ),
                (
                    "so um thanks for getting back to me I will uh review the document and send my feedback by friday".into(),
                    "Thank you for getting back to me. I will review the document and send my feedback by Friday.".into(),
                ),
            ],
        }
    }

    pub fn slack() -> Self {
        Self {
            id: "slack".into(),
            name: "Slack".into(),
            description: "Casual Slack message, concise and direct".into(),
            system_prompt: "You are a text post-processor for Slack messages. Fix grammar and \
                remove filler words, but keep the tone casual and conversational. \
                Keep it concise. Output only the cleaned text.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "hey um can you take a look at the PR when you get a chance".into(),
                    "hey, can you take a look at the PR when you get a chance?".into(),
                ),
                (
                    "so the build is failing on main uh someone probably needs to fix that".into(),
                    "the build is failing on main — someone needs to fix that".into(),
                ),
            ],
        }
    }

    pub fn meeting_notes() -> Self {
        Self {
            id: "meeting_notes".into(),
            name: "Meeting Notes".into(),
            description: "Structured meeting notes with bullets and action items".into(),
            system_prompt: "You are a text post-processor for meeting notes. Fix grammar and \
                remove filler words. Format as clear bullet points. Identify action items \
                and label them with 'Action:'. Output only the cleaned, formatted text.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "so we talked about the Q3 roadmap and uh John is going to handle the API redesign and Sarah needs to review the mockups by next Tuesday".into(),
                    "- Discussed Q3 roadmap\n- Action: John — API redesign\n- Action: Sarah — review mockups by next Tuesday".into(),
                ),
                (
                    "the main blocker is um the dependency on the auth service which is owned by the platform team".into(),
                    "- Main blocker: dependency on the auth service (owned by Platform team)".into(),
                ),
            ],
        }
    }

    pub fn code_comments() -> Self {
        Self {
            id: "code_comments".into(),
            name: "Code Comments".into(),
            description: "Technical code comments, terse and precise".into(),
            system_prompt: "You are a text post-processor for code comments and documentation. \
                Fix grammar, remove filler words. Keep language technical and precise. \
                Use imperative mood for doc comments. No emojis. \
                Output only the cleaned text.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "so this function basically um takes a slice and uh returns the sorted version without modifying the original".into(),
                    "Returns a sorted copy of the slice without modifying the original.".into(),
                ),
                (
                    "uh we need to handle the case where the cache is empty or like the key doesn't exist".into(),
                    "Handle the case where the cache is empty or the key does not exist.".into(),
                ),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_settings_toml_round_trip() {
        let settings = UserSettings::default();
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        let parsed: UserSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.language, "en-US");
        assert_eq!(parsed.privacy_mode, PrivacyMode::LocalOnly);
        assert_eq!(parsed.output_mode, OutputMode::LiveTyping);
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
        let preset = StylePreset::general();
        let json = serde_json::to_string(&preset).unwrap();
        let parsed: StylePreset = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "general");
        assert_eq!(parsed.few_shot_pairs.len(), 2);
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
        assert!(ids.contains(&"general"));
        assert!(ids.contains(&"email"));
        assert!(ids.contains(&"slack"));
        assert!(ids.contains(&"meeting_notes"));
        assert!(ids.contains(&"code_comments"));
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
}
