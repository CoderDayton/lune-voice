use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use serde::{Deserialize, Serialize};

pub type StyleId = String;

/// Universal rules shared by every style preset. Assembled once in the system
/// message; individual presets only carry style-specific additions.
pub const BASE_SYSTEM_PROMPT: &str = "\
You are a speech-to-text post-processor. You receive raw ASR transcriptions \
and output ONLY the cleaned version. Rules:
1. Fix clear ASR misheard words (homophones, garbled tokens).
2. Remove verbal fillers (um, uh, like, you know).
3. Preserve the speaker's original intent and meaning exactly.
4. Do NOT add words, respond conversationally, or interpret the text.
5. If the input is already clean, output it unchanged.
6. Output ONLY the cleaned transcription — nothing else.
7. When prior context is provided, reorder ASR fragments if needed so \
the output continues naturally from it.";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StylePreset {
    pub id: StyleId,
    pub name: String,
    pub description: String,
    /// Style-specific additions appended after [`BASE_SYSTEM_PROMPT`].
    /// Empty string means the base prompt is sufficient on its own.
    pub style_instructions: String,
    /// Template with `{raw_text}`, `{context}` slots.
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
        StylePreset::general(),
        StylePreset::coding(),
        StylePreset::email(),
        StylePreset::slack(),
        StylePreset::meeting_notes(),
        StylePreset::code_comments(),
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

    /// Built-in "General" preset: grammar + disfluency removal.
    pub fn general() -> Self {
        Self {
            id: "general".into(),
            name: "General".into(),
            description: "Clean grammar and remove disfluencies".into(),
            style_instructions: String::new(),
            user_prompt_template: "{context}ASR: {raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "ASR: so um I'm gonna go to the store and buy some more flower for the cake".into(),
                    "So I'm gonna go to the store and buy some more flour for the cake.".into(),
                ),
                (
                    "ASR: it's like totally ridiculous that the where is this bad after a month you know just fix the darn thing".into(),
                    "It's totally ridiculous that the wear is this bad after a month, just fix the darn thing.".into(),
                ),
                (
                    "ASR: um hello world uh".into(),
                    "Hello world.".into(),
                ),
                (
                    "ASR: the thermal dissipation is like uh roughly 450 watts and we gotta ensure the substrate doesn't de-grate".into(),
                    "The thermal dissipation is roughly 450 watts and we gotta ensure the substrate doesn't degrade.".into(),
                ),
            ],
        }
    }

    pub fn coding() -> Self {
        Self {
            id: "coding".into(),
            name: "Vibe Coding".into(),
            description: "High-intelligence technical polishing with semantic code-element detection".into(),
            style_instructions: "Format technical entities in `backticks` / PascalCase. Correct homophones for software context (sink→async, wait→await, post→POST). Maintain developer slang.".into(),
            user_prompt_template: "{context}ASR: {raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "ASR: um let's make an async funk shin called sink data that awaits a fetch to the api and then logs the status".into(),
                    "Let's make an `async` function called `sink_data` that `await`s a fetch to the API and then logs the status.".into(),
                ),
                (
                    "ASR: so we gotta nuke the old user profile class and replace it with a new interface called I user repo that has a get by id method".into(),
                    "So we gotta nuke the old `UserProfile` class and replace it with a new interface called `IUserRepo` that has a `get_by_id` method.".into(),
                ),
                (
                    "ASR: if the bullion is true then we just push to the state array and trigger a re render you know".into(),
                    "If the boolean is true then we just push to the `state` array and trigger a re-render.".into(),
                ),
            ],
        }
    }

    pub fn email() -> Self {
        Self {
            id: "email".into(),
            name: "Email".into(),
            description: "Formal email with proper greeting and sign-off".into(),
            style_instructions: "Ensure formal register. Add paragraph breaks where natural.".into(),
            user_prompt_template: "{context}ASR: {raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "ASR: hey uh just wanted to follow up on the thing we discussed last week about the project timeline".into(),
                    "I wanted to follow up on our discussion last week regarding the project timeline.".into(),
                ),
                (
                    "ASR: so um thanks for getting back to me I will uh review the document and send my feedback by friday".into(),
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
            style_instructions: "Keep the tone casual and concise.".into(),
            user_prompt_template: "{context}ASR: {raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "ASR: hey um can you take a look at the PR when you get a chance".into(),
                    "hey, can you take a look at the PR when you get a chance?".into(),
                ),
                (
                    "ASR: so the build is failing on main uh someone probably needs to fix that"
                        .into(),
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
            style_instructions: "Format as clear bullet points. Label action items with 'Action:'.".into(),
            user_prompt_template: "{context}ASR: {raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "ASR: so we talked about the Q3 roadmap and uh John is going to handle the API redesign and Sarah needs to review the mockups by next Tuesday".into(),
                    "- Discussed Q3 roadmap\n- Action: John — API redesign\n- Action: Sarah — review mockups by next Tuesday".into(),
                ),
                (
                    "ASR: the main blocker is um the dependency on the auth service which is owned by the platform team".into(),
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
            style_instructions: "Keep language technical and precise. Use imperative mood. No emojis.".into(),
            user_prompt_template: "{context}ASR: {raw_text}".into(),
            few_shot_pairs: vec![
                (
                    "ASR: so this function basically um takes a slice and uh returns the sorted version without modifying the original".into(),
                    "Returns a sorted copy of the slice without modifying the original.".into(),
                ),
                (
                    "ASR: uh we need to handle the case where the cache is empty or like the key doesn't exist".into(),
                    "Handle the case where the cache is empty or the key does not exist.".into(),
                ),
            ],
        }
    }
}
