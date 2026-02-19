# 04 – Data Design

← [Components](./03-components.md) | → [External Interfaces](./05-external-interfaces.md)

---

## 6.1 Core Data Structures

```rust
// ─── Settings ────────────────────────────────────────────────────────────────

pub struct UserSettings {
    pub asr_backend:   AsrBackendConfig,
    pub llm_backend:   LlmBackendConfig,
    pub hotkey:        HotkeyConfig,
    pub default_style: StyleId,
    pub language:      String,           // BCP-47 tag, e.g. "en-US"
    pub privacy_mode:  PrivacyMode,
    pub output_mode:   OutputMode,       // LiveTyping | TypeOnStop
    pub session_history_days: Option<u32>, // None = disabled
}

pub enum AsrBackendConfig {
    VoxtralLocal {
        model_path:    PathBuf,
        use_gpu:       bool,
        context_secs:  f32,             // sliding window size
    },
    OpenAiCompatible {
        base_url:      Url,
        api_key:       Secret<String>,
        model:         String,
        streaming:     StreamingMode,   // WebSocket | HttpPolling
    },
}

pub enum LlmBackendConfig {
    Local {
        model_path:   PathBuf,
        use_gpu:      bool,
        n_ctx:        u32,
    },
    OpenAiCompatible {
        base_url:     Url,
        api_key:      Secret<String>,
        model:        String,
        max_tokens:   u32,
    },
}

// ─── Style Presets ───────────────────────────────────────────────────────────

pub type StyleId = String;  // e.g. "email_formal", "slack_casual"

pub struct StylePreset {
    pub id:                  StyleId,
    pub name:                String,
    pub description:         String,
    pub system_prompt:       String,
    pub user_prompt_template: String,     // {raw_text}, {context} slots
    pub few_shot_pairs:      Vec<(String, String)>, // (input, expected_output)
}

// ─── Dictionary & Snippets ───────────────────────────────────────────────────

pub struct DictionaryEntry {
    pub phrase:      String,
    pub replacement: String,
    pub priority:    u8,
}

pub struct Snippet {
    pub id:             String,
    pub trigger_phrase: String,
    pub body:           String,    // may contain {date}, {name}, etc.
}

// ─── Session ─────────────────────────────────────────────────────────────────

pub struct DictationSession {
    pub id:         Uuid,
    pub started_at: DateTime<Utc>,
    pub ended_at:   Option<DateTime<Utc>>,
    pub style:      StyleId,
    pub backend:    AsrBackendKind,
    pub status:     SessionStatus,
    pub transcript: Option<String>,    // final post-processed text
}

pub enum SessionStatus {
    Active,
    Completed,
    Failed(String),    // error message
    Cancelled,
}

pub enum AsrBackendKind {
    VoxtralLocal,
    OpenAiCompatible,
}

// ─── ASR Output ──────────────────────────────────────────────────────────────

pub struct AsrSegment {
    pub id:       u64,
    pub text:     String,
    pub is_final: bool,          // immutable once true
    pub start_ms: Option<u64>,
    pub end_ms:   Option<u64>,
}

// ─── Hotkey ──────────────────────────────────────────────────────────────────

pub struct HotkeyConfig {
    pub toggle_dictation: KeyCombo,
    pub cancel_dictation: KeyCombo,
}

pub struct KeyCombo {
    pub modifiers: Vec<Modifier>,   // Ctrl, Shift, Alt, Meta
    pub key:       String,          // key name, e.g. "Space", "F13"
}

// ─── Privacy ─────────────────────────────────────────────────────────────────

pub enum PrivacyMode {
    LocalOnly,          // reject any remote backend usage
    LocalPreferred,     // use remote only if explicitly configured + consented
    AllowRemote,        // use whatever is configured
}

pub enum OutputMode {
    LiveTyping,         // stream partials with correction
    TypeOnStop,         // emit final text only when session ends
}
```

---

## 6.2 Text Processing Pipeline

```
1. Raw ASR segments appended to session rolling buffer
        │
        │  (partial segments may update in-place)
        ▼
2. Stabilization logic
   - Track segment by ID
   - Mark is_final=true after configurable silence window (default 500 ms)
   - Once final, segment is immutable
        │
        ▼
3. Finalized segment batch assembled
   - Aggregate consecutive final segments into a processing window
   - Attach: StylePreset metadata, dictionary hints, prev_context
        │
        ▼
4. VoiceCommandRouter pass
   - Scan batch for command phrases
   - Dispatch commands; remove trigger phrases from text
        │
        ▼
5. TextPostProcessor.transform(PostProcessInput)
   - LLM cleans grammar, removes disfluencies, applies style
   - Returns PostProcessOutput { cleaned_text, metadata }
        │
        ▼
6. Diff against last emitted text
   - Character-level diff to find minimal edit
   - Emit: backspace N chars + new text  (or full replace in TypeOnStop mode)
        │
        ▼
7. TextOutputController emits to sink
   - Notes view (Tauri event), or
   - OS keyboard injection
```

---

## 6.3 State Transitions

```
Session lifecycle:
  [Idle] ──hotkey──▶ [Recording]
  [Recording] ──hotkey / "stop dictation"──▶ [Processing]
  [Processing] ──transform complete──▶ [Idle]
  [Recording / Processing] ──error──▶ [Error] ──auto-recover──▶ [Idle]
  [Recording] ──"cancel" / Escape──▶ [Cancelled] ──▶ [Idle]
```

---

## Section Checklist

### Data Structures
- [ ] All structs implement `serde::Serialize` + `serde::Deserialize`
- [ ] `Secret<String>` (e.g., via `secrecy` crate) used for API keys — never plain `String`
- [ ] `Url` type used for base URLs (validated at parse time, not as raw `String`)
- [ ] `Uuid` sourced from `uuid` crate with `v4` feature
- [ ] `DateTime<Utc>` sourced from `chrono` crate
- [ ] All enums marked `#[non_exhaustive]` where future variants are expected

### Pipeline
- [ ] Stabilization window value documented in settings schema with units (milliseconds)
- [ ] Rolling buffer has bounded capacity (configurable max segments)
- [ ] Diff algorithm tested: verify minimal backspace count for typical edit scenarios
- [ ] `PostProcessInput` never contains raw audio — text only
- [ ] Pipeline stages each log at appropriate `tracing` level (debug/info/warn)

### Sessions
- [ ] Session ID generated with `Uuid::new_v4()` — never sequential integers
- [ ] `ended_at` set atomically on session stop
- [ ] Failed sessions record error message; partial transcript preserved if available
- [ ] Session records written to persistence only when history is enabled
