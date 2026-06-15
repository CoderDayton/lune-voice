use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use serde::{Deserialize, Serialize};

pub type StyleId = String;

/// System prompt for the ASR correction pass.
///
/// Language-agnostic: the policy is "smallest edits to a natural written form
/// in the same language as the input." The `{lang}` placeholder used by older
/// versions has been removed because the value piped through from the ASR was
/// inconsistent (sometimes a BCP-47 code, sometimes a full English language
/// name), which produced ungrammatical headers and a strong English bias.
/// Existing `.replace("{lang}", ..)` call sites become no-ops and remain safe.
pub const BASE_SYSTEM_PROMPT: &str = "\
You are an ASR transcript corrector.\n\
Output only the corrected transcript, in the same language as the input. \
Make the smallest edits that yield natural written text.\n\n\
Apply when clearly warranted:\n\
- Fix misspellings, fused or split words, and obvious ASR substitutions.\n\
- Fix homophones only when the surrounding context makes the right word unambiguous.\n\
- Insert short missing function words and fix simple agreement, plurals, and contractions.\n\
- If a dictionary list is provided below, use the exact target spelling for any matched phrase.\n\n\
Punctuation rules:\n\
- Preserve any punctuation already present in the input.\n\
- Inside one sentence, use commas for clause boundaries, list separators, and natural pauses — not periods.\n\
- Split into a new sentence only when the content clearly shifts to a separate complete thought.\n\
- End each sentence with exactly one terminal mark (. ! ?). Use ? only for actual questions; reserve ! for clear emphasis.\n\
- Capitalize the first word of each sentence and proper nouns. Do not invent capitalization mid-sentence.\n\n\
Do not:\n\
- Translate, paraphrase, or reorder content.\n\
- Replace a word with an unrelated word, or invent facts, names, or details.\n\
- Censor, soften, or remove profanity or sensitive terms.\n\
- Answer questions or reply conversationally — the input is a transcript, not a request.\n\
- Insert interior periods, semicolons, em-dashes, ellipses, or quotation marks unless the input clearly demands them.\n\n\
If a word looks wrong but the correct alternative is unclear, keep the original.\n\n\
Examples (English; the same minimal-edit policy applies in any language):\n\
Input: thanks\n\
Output: Thanks.\n\
Input: we should meat at noon\n\
Output: We should meet at noon.\n\
Input: pick up bread eggs and milk on the way home\n\
Output: Pick up bread, eggs, and milk on the way home.\n\
Input: did you see john where is he\n\
Output: Did you see John? Where is he?\n\
Input: i finished the report send it over when you get a sec ill be at lunch till one\n\
Output: I finished the report. Send it over when you get a sec. I'll be at lunch till one.\n\
Input: hey just checking in on the deploy did the migration finish if not we should roll back before the demo at three im on slack let me know\n\
Output: Hey, just checking in on the deploy. Did the migration finish? If not, we should roll back before the demo at three. I'm on Slack, let me know.";

/// Per-style toggles for the deterministic formatting transforms that run in
/// the correction path, before and independently of the LLM styling pass. All
/// default on; a style can switch any of them off.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct FormatToggles {
    /// Join spoken keyboard chords: "Super H" -> "Super+H".
    #[serde(default = "default_format_toggle")]
    pub keybinds: bool,
    /// Abbreviate spoken titles: "Mister Thompson" -> "Mr. Thompson".
    #[serde(default = "default_format_toggle")]
    pub title_abbreviations: bool,
    /// Capitalize the first word of each sentence.
    #[serde(default = "default_format_toggle")]
    pub auto_capitalization: bool,
    /// Apply canonical brand casing: "github" -> "GitHub", "iphone" -> "iPhone".
    #[serde(default = "default_format_toggle")]
    pub brand_casing: bool,
}

fn default_format_toggle() -> bool {
    true
}

impl Default for FormatToggles {
    fn default() -> Self {
        Self {
            keybinds: true,
            title_abbreviations: true,
            auto_capitalization: true,
            brand_casing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylePreset {
    pub id: StyleId,
    pub name: String,
    pub description: String,
    /// Style-specific instructions passed in the user prompt.
    /// Empty string means the base prompt is sufficient on its own.
    pub style_instructions: String,
    /// Template with `{raw_text}` slot.
    pub user_prompt_template: String,
    pub few_shot_pairs: Vec<(String, String)>,
    /// When `true`, the styling LLM pass is skipped — only the correction pass
    /// runs. The natural carrier for "this preset is correction-only".
    /// Decouples behavior from the historical `id == "minimal"` sentinel.
    #[serde(default)]
    pub skip_styling: bool,
    /// Per-style deterministic formatting toggles (keybinds, title
    /// abbreviations, auto-capitalization, brand casing). Default: all on.
    #[serde(default)]
    pub formatting: FormatToggles,
}

/// O(1) lookup registry for style presets.
///
/// Backed by a `HashMap` for fast `get()` and a `Vec` to preserve insertion
/// order for `all()`.
pub struct StyleRegistry {
    presets: HashMap<String, Arc<StylePreset>>,
    ordered: Vec<Arc<StylePreset>>,
}

impl StyleRegistry {
    /// Look up a preset by ID. Returns `Arc::clone` (atomic increment, no deep copy).
    pub fn get(&self, id: &str) -> Option<Arc<StylePreset>> {
        self.presets.get(id).cloned()
    }

    /// All presets in insertion order.
    #[inline]
    pub fn all(&self) -> &[Arc<StylePreset>] {
        &self.ordered
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }
}

static REGISTRY: LazyLock<StyleRegistry> = LazyLock::new(|| {
    let builtins = [
        StylePreset::minimal(),
        StylePreset::casual(),
        StylePreset::professional(),
        StylePreset::formal(),
        StylePreset::technical(),
        StylePreset::writing(),
    ];
    let mut presets = HashMap::with_capacity(builtins.len());
    let mut ordered = Vec::with_capacity(builtins.len());
    for preset in builtins {
        let arc = Arc::new(preset);
        presets.insert(arc.id.clone(), Arc::clone(&arc));
        ordered.push(arc);
    }
    StyleRegistry { presets, ordered }
});

/// Returns the singleton builtin style registry.
pub fn builtin_registry() -> &'static StyleRegistry {
    &REGISTRY
}

impl StylePreset {
    /// Backward-compat shim: returns all builtin presets as a static slice of Arcs.
    pub fn all_builtin() -> &'static [Arc<StylePreset>] {
        builtin_registry().all()
    }

    /// Whether this preset requires a separate styling pass.
    /// Driven by `skip_styling` so behavior is not coupled to a magic ID.
    pub fn needs_styling(&self) -> bool {
        !self.skip_styling
    }

    /// Build a self-contained user prompt for the styling-only pass.
    /// No system prompt needed — everything is in the user message.
    pub fn build_styling_prompt(&self, corrected_text: &str) -> String {
        use std::fmt::Write;
        let mut prompt = String::with_capacity(512);
        let _ = writeln!(
            prompt,
            "Rewrite the transcription with the following style. Output only the rewritten text."
        );
        let _ = writeln!(prompt, "\nStyle: {}", self.name);
        let _ = writeln!(prompt, "Rules: {}", self.style_instructions);
        if !self.few_shot_pairs.is_empty() {
            let _ = writeln!(prompt, "\nExamples:");
            for (input, output) in &self.few_shot_pairs {
                let _ = writeln!(prompt, "Input: \"{input}\"");
                let _ = writeln!(prompt, "Output: \"{output}\"");
            }
        }
        let _ = writeln!(prompt, "\nTranscript:");
        let _ = writeln!(prompt, "{corrected_text}");
        prompt.push_str("\n/no_think");
        prompt
    }

    /// Default preset: everyday speech with minimal formatting.
    /// Casual — texting, messaging, chat. Lowercase, contractions, minimal punctuation.
    pub fn casual() -> Self {
        Self {
            id: "casual".into(),
            name: "Casual".into(),
            description: "Texting, messaging, chat".into(),
            style_instructions: "Replace every colon with a comma. Replace every semicolon with a period. Remove all quotation marks. Remove exclamation marks and dashes. Lowercase everything except proper nouns. Use contractions.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "She said: \"I will be there\"; however, I do not think she meant it.".into(),
                    "she said she'd be there, but i don't think she meant it.".into(),
                ),
                (
                    "The plan was simple: cut costs; raise prices; hire fewer people.".into(),
                    "the plan was simple, cut costs, raise prices, hire fewer people.".into(),
                ),
                (
                    "He asked: \"Are you coming?\" I replied -- \"Yes!\"".into(),
                    "he asked if i was coming. i said yes.".into(),
                ),
            ],
            skip_styling: false,
            formatting: FormatToggles::default(),
        }
    }

    /// Workplace communication: clear, direct, properly formatted.
    pub fn professional() -> Self {
        Self {
            id: "professional".into(),
            name: "Professional".into(),
            description: "Clear workplace communication".into(),
            style_instructions: "Use standard capitalization and grammar. Break run-on sentences. Keep it direct and clear. Do not use contractions.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "so yeah the project's on track and we'll hit the deadline, Sarah's team is handling the backend.".into(),
                    "The project is on track, and we will hit the deadline. Sarah's team is handling the backend.".into(),
                ),
                (
                    "i talked to the client and they're fine with the delay but they want weekly updates going forward.".into(),
                    "I spoke with the client, and they are fine with the delay. They want weekly updates going forward.".into(),
                ),
                (
                    "the deploy broke last night, it was a config issue, Mike fixed it around 2 AM.".into(),
                    "The deploy broke last night due to a configuration issue. Mike fixed it around 2 AM.".into(),
                ),
            ],
            skip_styling: false,
            formatting: FormatToggles::default(),
        }
    }

    /// Official correspondence: formal register, no contractions.
    pub fn formal() -> Self {
        Self {
            id: "formal".into(),
            name: "Formal".into(),
            description: "Official correspondence and documents".into(),
            style_instructions: "Use formal register. Expand all contractions. Replace casual words with formal equivalents. Write complete, structured sentences.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "Thanks for getting back to me. I'll review the document and send my feedback by Friday.".into(),
                    "Thank you for your response. I will review the document and provide my feedback by Friday.".into(),
                ),
                (
                    "We can't do the meeting on Monday, let's push it to Wednesday.".into(),
                    "We are unable to accommodate the meeting on Monday. I would suggest rescheduling to Wednesday.".into(),
                ),
                (
                    "Just wanted to check in on the project. How's it going?".into(),
                    "I am writing to inquire about the current status of the project.".into(),
                ),
            ],
            skip_styling: false,
            formatting: FormatToggles::default(),
        }
    }

    /// Minimal — fix only clearly wrong words, preserve everything else verbatim.
    pub fn minimal() -> Self {
        Self {
            id: "minimal".into(),
            name: "Minimal".into(),
            description: "Error fixes only, no AI styling".into(),
            style_instructions:
                "Only fix clearly wrong words. Do not add, remove, or rephrase anything.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "I was going to the store and I seen a dog".into(),
                    "I was going to the store and I saw a dog.".into(),
                ),
                (
                    "she don't even know what happened its crazy".into(),
                    "She don't even know what happened, it's crazy.".into(),
                ),
                (
                    "the whether is nice today I think we should go outside".into(),
                    "The weather is nice today, I think we should go outside.".into(),
                ),
            ],
            skip_styling: true,
            formatting: FormatToggles::default(),
        }
    }

    /// Code and developer communication with backtick formatting.
    pub fn technical() -> Self {
        Self {
            id: "technical".into(),
            name: "Technical".into(),
            description: "Code and developer communication".into(),
            style_instructions: "Wrap code identifiers, function names, types, and keywords in backticks. Apply PascalCase or snake_case as context dictates. Keep technical jargon.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "Let's make an async function called sync data that awaits fetch to API and then logs status.".into(),
                    "Let's make an `async` function called `sync_data` that `await`s fetch to API and then logs status.".into(),
                ),
                (
                    "The user profile class needs a get by ID method that returns an optional user.".into(),
                    "The `UserProfile` class needs a `get_by_id` method that returns an `Optional<User>`.".into(),
                ),
                (
                    "If the boolean is true then we push to the state array and trigger a re-render.".into(),
                    "If the `boolean` is `true` then we push to the `state` array and trigger a re-render.".into(),
                ),
            ],
            skip_styling: false,
            formatting: FormatToggles::default(),
        }
    }

    /// Long-form writing — articles, docs, blog posts, essays.
    pub fn writing() -> Self {
        Self {
            id: "writing".into(),
            name: "Writing".into(),
            description: "Long-form content — articles, docs, essays".into(),
            style_instructions: "Write polished prose. Break run-on sentences. Vary sentence length. Remove filler words like 'so', 'just', 'basically', 'like'. Keep the author's voice.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "So the main idea is that we need to rethink testing because right now it's kind of a mess and nobody really knows what's covered.".into(),
                    "We need to rethink our approach to testing. Right now it's a mess, and nobody knows what's covered.".into(),
                ),
                (
                    "Basically the problem is that it just scales poorly and once you hit about ten thousand users things start to break down.".into(),
                    "The problem is that it scales poorly. Once you hit ten thousand users, things start to break down.".into(),
                ),
            ],
            skip_styling: false,
            formatting: FormatToggles::default(),
        }
    }
}
