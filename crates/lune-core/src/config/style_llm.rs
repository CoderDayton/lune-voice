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

    /// Default preset: everyday speech with minimal formatting.
    /// Casual — texting, messaging, chat. Lowercase, contractions, minimal punctuation.
    pub fn casual() -> Self {
        Self {
            id: "casual".into(),
            name: "Casual".into(),
            description: "Texting, messaging, chat".into(),
            style_instructions: "Lowercase everything except proper nouns. Use contractions. Minimal punctuation — skip periods at end of single sentences.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "I am going to the store do you want anything".into(),
                    "i'm going to the store do you want anything".into(),
                ),
                (
                    "That is really cool I can not believe it worked".into(),
                    "that's really cool i can't believe it worked".into(),
                ),
                (
                    "Hey are you coming to Sarah's party tonight or not".into(),
                    "hey are you coming to sarah's party tonight or not".into(),
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
            style_instructions: "Apply standard grammar and capitalization. Capitalize proper nouns. Write clear, direct sentences.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "I wanted to follow up on the thing we discussed last week about project timeline".into(),
                    "I wanted to follow up on what we discussed last week about the project timeline.".into(),
                ),
                (
                    "the main blocker is dependency on oth service which is owned by platform team".into(),
                    "The main blocker is a dependency on the auth service, which is owned by the Platform team.".into(),
                ),
                (
                    "she finished report ahead of schedule and her manager approved it without revisions and client was notified".into(),
                    "She finished the report ahead of schedule, and her manager approved it without revisions. The client was notified.".into(),
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
            style_instructions: "Apply formal register. Expand all contractions. Write complete sentences with proper capitalization.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "I wanted to follow up on the thing we discussed last week about project timeline".into(),
                    "I would like to follow up on our discussion last week regarding the project timeline.".into(),
                ),
                (
                    "thanks for getting back to me I will review document and send my feed back by friday".into(),
                    "Thank you for your response. I will review the document and provide my feedback by Friday.".into(),
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
            style_instructions: "Wrap code identifiers in backticks. Apply PascalCase or snake_case as context dictates. Retain developer jargon.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "let's make an async funk shin called sink data that awaits fetch to api and then logs status".into(),
                    "Let's make an `async` function called `sync_data` that `await`s fetch to API and then logs status.".into(),
                ),
                (
                    "so we gotta nuke the old user profile class and replace it with a new interface called I user repo that has a get by id method".into(),
                    "So we gotta nuke the old `UserProfile` class and replace it with a new interface called `IUserRepo` that has a `get_by_id` method.".into(),
                ),
                (
                    "if the bullion is true then we push to the state array and trigger a re render".into(),
                    "If the boolean is true then we push to the `state` array and trigger a re-render.".into(),
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
            style_instructions: "Use complete sentences with proper grammar and punctuation. Break long run-on sentences into shorter ones. Keep the author's voice.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "so the main idea here is that we need to rethink how we approach testing because right now its kind of a mess and nobody really knows what's covered".into(),
                    "The main idea here is that we need to rethink how we approach testing. Right now it's kind of a mess, and nobody really knows what's covered.".into(),
                ),
                (
                    "the problem with the current approach is that it scales poorly and once you hit about ten thousand users things start to break down".into(),
                    "The problem with the current approach is that it scales poorly. Once you hit about ten thousand users, things start to break down.".into(),
                ),
            ],
        }
    }
}
