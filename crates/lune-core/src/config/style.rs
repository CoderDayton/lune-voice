use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use serde::{Deserialize, Serialize};

pub type StyleId = String;

/// Universal rules shared by every style preset. Assembled once in the system
/// message; individual presets only carry style-specific additions.
///
/// Span-correction mode: uncertain words are marked with `[?word?]`. The LLM
/// fixes only flagged words and preserves all unmarked text verbatim.
pub const BASE_SYSTEM_PROMPT: &str = "\
You are a post-ASR transcript correction engine.
You receive a transcript where uncertain words are wrapped in [?word?] markers.
Rules:
1. Replace each [?word?] with the correct word based on context. Most flagged words are already correct — output them unchanged if they fit.
2. Fix word boundary errors adjacent to markers (e.g. [?leisure?] ly → leisurely).
3. Remove filler words: um, uh, like, you know.
4. Keep ALL unmarked words exactly as-is. Never reorder, merge, or restructure sentences.
5. Output ONLY the corrected transcript — no markers, no explanation, no extra text.";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylePreset {
    pub id: StyleId,
    pub name: String,
    pub description: String,
    /// Style-specific additions appended after [`BASE_SYSTEM_PROMPT`].
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
    #[inline]
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
        StylePreset::casual(),
        StylePreset::professional(),
        StylePreset::formal(),
        StylePreset::email(),
        StylePreset::technical(),
        StylePreset::notes(),
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
#[inline]
pub fn builtin_registry() -> &'static StyleRegistry {
    &REGISTRY
}

impl StylePreset {
    /// Backward-compat shim: returns all builtin presets as a static slice of Arcs.
    #[inline]
    pub fn all_builtin() -> &'static [Arc<StylePreset>] {
        builtin_registry().all()
    }

    /// Default preset: everyday speech with minimal formatting.
    pub fn casual() -> Self {
        Self {
            id: "casual".into(),
            name: "Casual".into(),
            description: "Everyday speech — messaging, texting, chat".into(),
            style_instructions: "Capitalize sentence-initial words. Retain contractions and informal tone.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "I'm gonna go to the store and buy some more [?flower?] for the cake".into(),
                    "I'm gonna go to the store and buy some more flour for the cake.".into(),
                ),
                (
                    "hey can you take a look at this when you get a chance".into(),
                    "Hey, can you take a look at this when you get a chance?".into(),
                ),
                (
                    "it's totally [?ridiculus?] that the [?where?] is this bad after a month just fix the darn thing".into(),
                    "It's totally ridiculous that the wear is this bad after a month, just fix the darn thing.".into(),
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
                    "I wanted to follow up on the [?thing?] we discussed last week about the project timeline".into(),
                    "I wanted to follow up on what we discussed last week about the project timeline.".into(),
                ),
                (
                    "the main blocker is the dependency on the [?oth?] service which is owned by the platform team".into(),
                    "The main blocker is the dependency on the auth service, which is owned by the Platform team.".into(),
                ),
                (
                    "she finished the report ahead of schedule and her manager approved it without revisions and the client was notified".into(),
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
                    "I wanted to follow up on the [?thing?] we discussed last week about the project timeline".into(),
                    "I would like to follow up on our discussion last week regarding the project timeline.".into(),
                ),
                (
                    "thanks for getting back to me I will review the document and send my [?feed back?] by friday".into(),
                    "Thank you for your response. I will review the document and provide my feedback by Friday.".into(),
                ),
            ],
        }
    }

    /// Email and letter formatting with paragraph structure.
    pub fn email() -> Self {
        Self {
            id: "email".into(),
            name: "Email".into(),
            description: "Email and letter formatting".into(),
            style_instructions: "Apply formal register. Expand all contractions. Insert paragraph breaks at natural sentence boundaries.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "I wanted to let you know the [?quarterly?] report is ready for review and I've attached the updated figures".into(),
                    "I wanted to let you know that the quarterly report is ready for review. I have attached the updated figures.".into(),
                ),
                (
                    "thanks for getting back to me so quickly I appreciate it and I'll have the [?revisions?] done by end of day friday".into(),
                    "Thank you for your prompt response. I will have the revisions completed by end of day Friday.".into(),
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
            style_instructions: "Wrap code identifiers in backticks. Apply PascalCase or snake_case as context dictates. Resolve software homophones: sink→sync, wait→await, bull→bool, funk shin→function. Retain developer jargon.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "let's make an async [?funk shin?] called [?sink?] data that awaits a fetch to the api and then logs the status".into(),
                    "Let's make an `async` function called `sync_data` that `await`s a fetch to the API and then logs the status.".into(),
                ),
                (
                    "so we gotta nuke the old user profile class and replace it with a new interface called [?I user repo?] that has a get by id method".into(),
                    "So we gotta nuke the old `UserProfile` class and replace it with a new interface called `IUserRepo` that has a `get_by_id` method.".into(),
                ),
                (
                    "if the [?bullion?] is true then we just push to the state array and trigger a re render".into(),
                    "If the boolean is true then we just push to the `state` array and trigger a re-render.".into(),
                ),
            ],
        }
    }

    /// Structured notes with bullet points and action items.
    pub fn notes() -> Self {
        Self {
            id: "notes".into(),
            name: "Notes".into(),
            description: "Structured notes with bullets and action items".into(),
            style_instructions: "Reformat output as bullet points. Prefix each action item with 'Action:'. Omit filler sentences.".into(),
            user_prompt_template: "{raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "we talked about the Q3 [?road map?] and John is going to handle the API redesign and Sarah needs to review the mockups by next Tuesday".into(),
                    "- Discussed Q3 roadmap\n- Action: John — API redesign\n- Action: Sarah — review mockups by next Tuesday".into(),
                ),
                (
                    "the main blocker is the dependency on the [?oth?] service which is owned by the platform team".into(),
                    "- Main blocker: dependency on the auth service (owned by Platform team)".into(),
                ),
            ],
        }
    }
}
