use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use serde::{Deserialize, Serialize};

pub type StyleId = String;

/// Universal rules shared by every ASR correction pass.
/// Language placeholder `{lang}` is replaced at prompt-build time.
pub const BASE_SYSTEM_PROMPT: &str = "\
You are a speech-to-text correction engine for {lang}.\n\
Correct ASR transcription errors with the smallest possible edits.\n\n\
Examples:\n\
Input: 'i hav eto go to the store'\n\
Output: 'I have to go to the store.'\n\
Input: 'how you doing today'\n\
Output: 'How are you doing today?'\n\
Input: 'I went store and brought some groceries'\n\
Output: 'I went to the store and bought some groceries.'\n\
Input: 'let go to the mall and bye some cloth'\n\
Output: 'Let\\'s go to the mall and buy some clothes.'\n\
Input: 'i want two go home'\n\
Output: 'I want to go home.'\n\
Input: 'we should meat at noon'\n\
Output: 'We should meet at noon.'\n\n\
Rules:\n\
- Preserve the speaker's intended meaning.\n\
- Make the fewest edits needed to produce natural written {lang}.\n\
- Fix obvious spelling errors, broken words, and ASR word substitutions.\n\
- Fix homophones only when the sentence context clearly supports the replacement.\n\
- Add only small missing function words such as a, an, the, to, of, are, is, and do.\n\
- Fix basic singular/plural forms, conjunctions, and contractions.\n\
- Capitalize sentences and add ending punctuation.\n\
- Never translate or change the language of the input.\n\
- Never censor, remove, or soften any words including profanity.\n\
- When uncertain about a word, keep the original.\n\
- Do not replace a word with an unrelated word.\n\
- Do not add new facts, names, objects, or actions.\n\
- Do not answer questions or respond conversationally.\n\
- Output only the corrected transcription.";

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
    /// "minimal" only corrects errors — no restyling needed.
    pub fn needs_styling(&self) -> bool {
        self.id != "minimal"
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
        }
    }

    /// Minimal — fix only clearly wrong words, preserve everything else verbatim.
    pub fn minimal() -> Self {
        Self {
            id: "minimal".into(),
            name: "Minimal".into(),
            description: "Fix errors only — no style changes".into(),
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
        }
    }
}
