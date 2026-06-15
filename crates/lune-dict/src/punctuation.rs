//! Deterministic punctuation normalization + LLM-gate detection.
//!
//! ASR output (Qwen3-ASR in particular) often arrives lowercase and without
//! terminal punctuation even when every *word* is high-confidence. The word
//! heuristic in [`crate::corrector`] resolves spelling, not punctuation, so a
//! fully-resolved segment would otherwise be emitted verbatim — "turn on the
//! light then close the door" — and bypass the LLM entirely.
//!
//! This module mirrors the "resolve deterministically, escalate the residual"
//! philosophy of the corrector, applied to punctuation:
//!
//! * **Tier 1 — deterministic, no LLM:** capitalize the first letter; append a
//!   terminal period when the segment is a single resolved declarative. These
//!   fixes are unambiguous, so they never need the model.
//! * **Tier 2 — ambiguous, trips the LLM gate:** comma-less run-ons and
//!   interrogative-leading segments with no end mark. The right fix here
//!   (internal commas, a sentence split, `?` vs `.` vs `!`) cannot be guessed
//!   safely, so [`PunctuationOutcome::needs_llm`] is set and the caller routes
//!   the text through the LLM punctuation pass.
//!
//! Non-Latin scripts (CJK, Hangul, etc.) use different punctuation and casing
//! conventions, so they pass through untouched and never trip the gate.

use lune_core::config::FormatToggles;

/// Word count at or above which a separator-free segment is treated as a
/// run-on candidate the deterministic pass cannot safely punctuate. Tuned for
/// the "balanced" sensitivity: high enough to ignore ordinary short sentences,
/// low enough to catch genuinely jarring run-ons.
const RUN_ON_WORD_THRESHOLD: usize = 12;

/// Wh-words that mark a segment as interrogative on their own. When one of
/// these opens a segment that lacks an end mark, statement-vs-question
/// punctuation is ambiguous, so it is deferred to the LLM rather than guessed.
const WH_LEADS: &[&str] = &[
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
];

/// Auxiliaries that open a question only via subject-auxiliary inversion
/// ("can you…", "did he…"). Alone they are imperative/declarative-prone
/// ("have a great day", "do not touch that", "will check later"), so they
/// count as interrogative only when a question subject follows.
const AUX_LEADS: &[&str] = &[
    "is", "are", "am", "was", "were", "do", "does", "did", "can", "could", "would", "should",
    "will", "shall", "has", "have", "had", "may", "might", "must",
];

/// Subjects that, following an auxiliary, signal genuine inversion — pronouns
/// plus the existential/demonstrative heads ("is there…", "was that…").
const QUESTION_SUBJECTS: &[&str] = &[
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "it",
    "this",
    "that",
    "there",
    "anyone",
    "anybody",
    "someone",
    "somebody",
    "everyone",
    "everybody",
    "nobody",
];

/// Result of [`normalize`].
pub struct PunctuationOutcome {
    /// Text with deterministic, unambiguous fixes applied: a leading capital
    /// and, for resolved declaratives, a terminal period. When `needs_llm` is
    /// true the terminal mark is left to the LLM, so only capitalization is
    /// applied here.
    pub text: String,
    /// True when residual punctuation is ambiguous enough to warrant the LLM
    /// (run-on candidate, or interrogative-leading with no end mark).
    pub needs_llm: bool,
}

/// Normalize punctuation on a single dictation segment.
///
/// Returns the deterministically-fixed text plus whether the residual
/// punctuation should be handed to the LLM. The returned text is always safe to
/// emit as-is *and* safe to feed to the LLM as input.
pub fn normalize(text: &str) -> PunctuationOutcome {
    normalize_with(text, &FormatToggles::default())
}

/// Like [`normalize`], but with per-style control over the deterministic
/// formatting transforms — keybind joins, title abbreviations, and
/// auto-capitalization. The terminal-period fix and the run-on/interrogative LLM
/// gate are always applied.
pub fn normalize_with(text: &str, fmt: &FormatToggles) -> PunctuationOutcome {
    let trimmed = text.trim();
    if trimmed.is_empty() || !is_latin(trimmed) {
        return PunctuationOutcome {
            text: text.to_owned(),
            needs_llm: false,
        };
    }

    let abbreviated = if fmt.title_abbreviations {
        abbreviate_titles(trimmed)
    } else {
        trimmed.to_owned()
    };
    let hotkeyed = if fmt.keybinds {
        join_hotkey_chords(&abbreviated)
    } else {
        abbreviated
    };
    let trimmed = hotkeyed.as_str();

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let ends_terminal = ends_with_terminal(trimmed);
    let interrogative = is_interrogative_lead(&words);

    // Run-on: long stretch with no internal separator or sentence break. Such a
    // segment likely needs internal commas or a split — the deterministic pass
    // can only touch the ends, so escalate it.
    let run_on = words.len() >= RUN_ON_WORD_THRESHOLD && !has_internal_separator(trimmed);

    // Interrogative lead without an end mark is ambiguous (`?` vs `.` vs `!`),
    // so let the LLM decide rather than guessing a question mark.
    let needs_llm = run_on || (interrogative && !ends_terminal);

    // Tier 1: capitalize the first word of the segment and of every internal
    // sentence (ASR emits sentence breaks from pause prosody, and the words
    // after them arrive lowercase).
    let mut out = if fmt.auto_capitalization {
        capitalize_sentence_starts(trimmed)
    } else {
        trimmed.to_owned()
    };

    // Tier 1: append a terminal period only for resolved declaratives. When
    // `needs_llm` is set the interrogative/run-on branch owns the end mark.
    if !ends_terminal && !needs_llm {
        out.push('.');
    }

    PunctuationOutcome {
        text: out,
        needs_llm,
    }
}

/// Spoken-form honorifics ASR emits verbatim ("Mister Thompson") where
/// written English universally abbreviates ("Mr. Thompson"). Applied only
/// when the next word is capitalized, so "hey mister" and "the doctor said"
/// pass through untouched.
const TITLE_ABBREVIATIONS: &[(&str, &str)] =
    &[("mister", "Mr."), ("missus", "Mrs."), ("doctor", "Dr.")];

/// Rewrite spoken honorifics to their written abbreviation when followed by
/// a capitalized word. Words carrying any edge punctuation are left alone —
/// "mister," mid-list is vocative, not a title. Rebuilds with single spaces,
/// which dictation text already uses.
fn abbreviate_titles(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    for (i, w) in words.iter().enumerate() {
        let next_capitalized = words.get(i + 1).is_some_and(|n| {
            n.chars()
                .find(|c| c.is_alphabetic())
                .is_some_and(|c| c.is_uppercase())
        });
        let abbrev = TITLE_ABBREVIATIONS
            .iter()
            .find(|(spoken, _)| w.eq_ignore_ascii_case(spoken))
            .filter(|_| next_capitalized)
            .map(|(_, abbr)| *abbr);
        out.push(abbrev.map_or_else(|| (*w).to_string(), |a| a.to_string()));
    }
    out.join(" ")
}

/// Modifier words that begin a keyboard chord. Matched case-insensitively but
/// only fire when written capitalized, which keeps ordinary prose ("control
/// room", "shift focus") from being joined.
const HOTKEY_MODIFIERS: &[&str] = &[
    "super", "control", "ctrl", "alt", "shift", "command", "cmd", "meta", "win", "windows",
    "option", "hyper", "fn",
];

/// Single-token named keys that can terminate a chord. Matched only when written
/// capitalized (like the modifiers), so ordinary prose words stay untouched.
const HOTKEY_NAMED_KEYS: &[&str] = &[
    "tab",
    "enter",
    "return",
    "escape",
    "esc",
    "space",
    "spacebar",
    "delete",
    "del",
    "backspace",
    "home",
    "end",
    "insert",
    "up",
    "down",
    "left",
    "right",
];

/// Join spoken keyboard chords into `Modifier+Key` notation: "Super H" →
/// "Super+H", "Control Shift P" → "Control+Shift+P", "Alt Tab" → "Alt+Tab".
///
/// Deliberately conservative: a chord is one or more *capitalized* modifier
/// words followed by exactly one terminal key — a single uppercase letter, a
/// digit, a function key (F1–F24), or a *capitalized* named key (Tab, Enter,
/// Escape, Space, Delete, Backspace, Home, End, Insert, the arrows). The
/// capitalization requirement keeps ordinary prose ("Shift focus", "Command
/// line", "alt tab") from being joined. Trailing punctuation on the key
/// ("Super H." → "Super+H.") is preserved.
fn join_hotkey_chords(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let mut j = i;
        while j < words.len() && is_hotkey_modifier(words[j]) {
            j += 1;
        }
        // Need at least one modifier (j > i) and exactly one terminal key after.
        if j > i && j < words.len() {
            let (key_core, suffix) = split_trailing_punct(words[j]);
            if is_hotkey_key(key_core) {
                let mut chord = words[i..j].join("+");
                chord.push('+');
                chord.push_str(key_core);
                chord.push_str(suffix);
                out.push(chord);
                i = j + 1;
                continue;
            }
        }
        out.push(words[i].to_string());
        i += 1;
    }
    out.join(" ")
}

/// A capitalized, punctuation-free modifier word.
fn is_hotkey_modifier(word: &str) -> bool {
    word.starts_with(|c: char| c.is_ascii_uppercase())
        && HOTKEY_MODIFIERS.contains(&word.to_ascii_lowercase().as_str())
}

/// Split a token into its alphanumeric core and trailing punctuation
/// ("H." → ("H", ".")).
fn split_trailing_punct(word: &str) -> (&str, &str) {
    let end = word
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(0, |(i, c)| i + c.len_utf8());
    word.split_at(end)
}

/// A chord-terminating key: a single uppercase letter or digit, a function key
/// F1–F24, or a capitalized named key (Tab, Enter, the arrows, …). Requiring
/// capitalization on named keys keeps common words from being mistaken for keys.
fn is_hotkey_key(core: &str) -> bool {
    if core.len() == 1 {
        let b = core.as_bytes()[0];
        return b.is_ascii_uppercase() || b.is_ascii_digit();
    }
    if let Some(rest) = core.strip_prefix('F')
        && (1..=2).contains(&rest.len())
        && rest.bytes().all(|b| b.is_ascii_digit())
        && rest.parse::<u8>().is_ok_and(|n| (1..=24).contains(&n))
    {
        return true;
    }
    core.starts_with(|c: char| c.is_ascii_uppercase())
        && HOTKEY_NAMED_KEYS.contains(&core.to_ascii_lowercase().as_str())
}

/// True when every alphabetic character is in a Latin range. Accented European
/// text counts as Latin; CJK, Hangul, Cyrillic, Greek, Arabic, etc. do not.
fn is_latin(text: &str) -> bool {
    text.chars().filter(|c| c.is_alphabetic()).all(|c| {
        c.is_ascii_alphabetic() || matches!(c, '\u{00C0}'..='\u{024F}') // Latin-1 Supplement + Latin Extended-A/B
    })
}

/// Closing quotes/brackets that may trail a sentence terminator.
const CLOSING: &[char] = &['"', '\'', ')', ']', '}', '»', '”', '’'];
/// Sentence-terminating marks.
const TERMINATORS: &[char] = &['.', '!', '?', '…'];

/// Whether the segment already ends in a sentence terminator, ignoring trailing
/// closing quotes/brackets (e.g. `he said "go".` or `(done!)`).
fn ends_with_terminal(text: &str) -> bool {
    text.trim_end_matches(CLOSING).ends_with(TERMINATORS)
}

/// Whether the body carries any internal separator or sentence break (a comma,
/// semicolon, colon, dash, or a non-final terminator). A trailing terminator
/// does not count.
fn has_internal_separator(text: &str) -> bool {
    // Drop a trailing terminator (and any closing quote/bracket) so it is not
    // mistaken for internal structure.
    let body = text.trim_end_matches(CLOSING).trim_end_matches(TERMINATORS);
    body.chars()
        .any(|c| matches!(c, ',' | ';' | ':' | '—' | '–' | '.' | '!' | '?'))
}

/// Trailing-`.` tokens that mark an abbreviation, not a sentence end. `Mr.`,
/// `Mrs.`, `Dr.` are produced by [`abbreviate_titles`] above; the rest are
/// common in dictated text. Dotted abbreviations (`e.g.`, `U.S.`) need no
/// listing — any interior dot already disqualifies a token.
const NON_TERMINAL_ABBREVIATIONS: &[&str] = &["mr", "mrs", "dr", "st", "etc", "vs", "prof"];

/// Capitalize the first word of every sentence: the segment start plus every
/// word following a sentence terminator (`.` `!` `?` `…`), skipping periods
/// that belong to abbreviations (`Mr.`, `e.g.`, `U.S.`) or single-letter
/// initials (`J.`). Tokens that already carry any uppercase (`iPhone`,
/// `GitHub`, mid-name caps) are never altered. Whitespace is preserved
/// byte-for-byte; non-Latin text passes through untouched.
pub fn capitalize_sentence_starts(text: &str) -> String {
    if !is_latin(text) {
        return text.to_owned();
    }

    let mut out = String::with_capacity(text.len());
    let mut at_sentence_start = true;
    let mut rest = text;
    while !rest.is_empty() {
        let ws_end = rest
            .find(|c: char| !c.is_whitespace())
            .unwrap_or(rest.len());
        out.push_str(&rest[..ws_end]);
        rest = &rest[ws_end..];
        if rest.is_empty() {
            break;
        }
        let tok_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let token = &rest[..tok_end];
        if at_sentence_start {
            push_capitalized(&mut out, token);
        } else {
            out.push_str(token);
        }
        at_sentence_start = token_ends_sentence(token);
        rest = &rest[tok_end..];
    }
    out
}

/// Lowercase the leading capital of a mid-sentence word the lexicon knows to be
/// a common word (low capitalization rate). Fixes ASR/heuristic over-caps like
/// "Make Sure" → "Make sure" and "So Make" → "So make" while leaving sentence
/// starts, proper nouns ("John"), brands ("GitHub"), and acronyms ("NASA")
/// untouched. English only — the caller gates on language. This runs on the
/// always-on (fast-path + LLM-input) text so casing is normalized once,
/// regardless of whether the gated LLM pass runs.
pub fn decase_common_words(text: &str) -> String {
    if !is_latin(text) {
        return text.to_owned();
    }
    // (leading-whitespace, token) segments, so a casing decision can see its
    // neighbours; whitespace is preserved byte-for-byte.
    let mut segs: Vec<(&str, &str)> = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let ws_end = rest
            .find(|c: char| !c.is_whitespace())
            .unwrap_or(rest.len());
        let ws = &rest[..ws_end];
        rest = &rest[ws_end..];
        if rest.is_empty() {
            segs.push((ws, ""));
            break;
        }
        let tok_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        segs.push((ws, &rest[..tok_end]));
        rest = &rest[tok_end..];
    }

    // Sentence-start flag per token: a sentence-initial capital is legitimate
    // (owned by capitalize_sentence_starts) and is never a proper-noun-run cue.
    let mut starts = vec![false; segs.len()];
    let mut at_start = true;
    for (i, (_, token)) in segs.iter().enumerate() {
        starts[i] = at_start;
        if !token.is_empty() {
            at_start = token_ends_sentence(token);
        }
    }

    let mut out = String::with_capacity(text.len());
    for i in 0..segs.len() {
        let (ws, token) = segs[i];
        out.push_str(ws);
        if token.is_empty() {
            continue;
        }
        // Proper-noun-run guard: a capital flanked by ANOTHER (non-sentence-
        // initial) capital — "Visual Studio Code", "Lune Voice" — is almost
        // always a multi-word proper noun, so never lowercase it. Neighbours are
        // read in the original casing.
        let prev_cap = i > 0 && !starts[i - 1] && starts_upper(segs[i - 1].1);
        let next_cap = i + 1 < segs.len() && !starts[i + 1] && starts_upper(segs[i + 1].1);
        if starts[i] || prev_cap || next_cap {
            out.push_str(token);
        } else {
            out.push_str(&decase_token(token));
        }
    }
    out
}

/// Whether `token`'s first alphabetic character is uppercase.
fn starts_upper(token: &str) -> bool {
    token
        .chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(|c| c.is_uppercase())
}

/// Lowercase a single token's leading capital iff it is a plain Capitalized
/// common word: first alphabetic char uppercase, NO interior uppercase (so
/// "iPhone"/"GitHub"/"NASA" are spared), and `en_p_cap` positively classifies
/// the word as common (below the stricter decase threshold). Unknown words (a
/// proper noun the table does not know) are left capitalized.
fn decase_token(token: &str) -> String {
    let core = word_core(token);
    if core.is_empty() {
        return token.to_string();
    }
    let mut alphas = core.chars().filter(|c| c.is_alphabetic());
    if !alphas.next().is_some_and(|c| c.is_uppercase()) {
        return token.to_string(); // not a leading capital
    }
    if alphas.any(|c| c.is_uppercase()) {
        return token.to_string(); // interior uppercase = brand/acronym
    }
    let is_common =
        crate::caprate::en_p_cap(core).is_some_and(|p| p < crate::caprate::CAP_DECASE_THRESHOLD);
    if !is_common {
        return token.to_string();
    }
    // Lowercase only the first alphabetic character, preserving punctuation.
    let mut done = false;
    let mut result = String::with_capacity(token.len());
    for c in token.chars() {
        if !done && c.is_alphabetic() {
            result.extend(c.to_lowercase());
            done = true;
        } else {
            result.push(c);
        }
    }
    result
}

/// Restore words an ASR fused inconsistently — the `MakeSure` / `Makesure`
/// artifact where a *repeated* phrase is rendered run-together once. Split a
/// single token into two ONLY when (1) the fused form is not itself a real word
/// (freq guard, so "into"/"cannot" are never touched), and (2) the two pieces
/// appear as an adjacent word pair ELSEWHERE in the same text (self-consistency,
/// so "GitHub" and one-off fusions are left alone). This is deliberately
/// conservative: it fixes only fusions the speaker themselves disambiguated.
/// English only — the caller gates on language.
pub fn defuse_inconsistent(text: &str) -> String {
    use std::collections::HashMap;
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return text.to_owned();
    }
    let cores: Vec<String> = words.iter().map(|w| fusion_core(w)).collect();
    // Adjacent word-pair concatenations → their two lowercase pieces. A fused
    // token matches only a pair that occurs as two SEPARATE adjacent tokens
    // somewhere; a single fused token never contributes its own split.
    let mut bigrams: HashMap<String, (String, String)> = HashMap::new();
    for i in 0..words.len() - 1 {
        let (a, b) = (&cores[i], &cores[i + 1]);
        if a.is_empty() || b.is_empty() {
            continue;
        }
        bigrams
            .entry(format!("{a}{b}"))
            .or_insert_with(|| (a.clone(), b.clone()));
    }
    let en = crate::freq::dict_for("en");
    let mut out: Vec<String> = Vec::with_capacity(words.len() + 4);
    for (i, w) in words.iter().enumerate() {
        let core = &cores[i];
        let split = (core.len() >= 4 && !en.contains(core))
            .then(|| bigrams.get(core))
            .flatten();
        if let Some((a, b)) = split {
            let (prefix, suffix) = edge_punct(w);
            out.push(format!("{prefix}{a}"));
            out.push(format!("{b}{suffix}"));
        } else {
            out.push((*w).to_string());
        }
    }
    out.join(" ")
}

/// Lowercase, alphanumeric-only form of a token ("Make," → "make").
fn fusion_core(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Leading and trailing non-alphanumeric runs of a token, so a split can carry
/// the original edge punctuation: "Makesure!" → ("", "!").
fn edge_punct(word: &str) -> (&str, &str) {
    let start = word
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(word.len(), |(i, _)| i);
    let end = word
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(start, |(i, c)| i + c.len_utf8());
    (&word[..start], &word[end..])
}

/// Append `token` with its first alphabetic character uppercased — unless the
/// token already contains any uppercase, in which case its casing is treated
/// as deliberate (brand/proper casing) and preserved verbatim.
fn push_capitalized(out: &mut String, token: &str) {
    if token.chars().any(|c| c.is_uppercase()) {
        out.push_str(token);
        return;
    }
    let mut done = false;
    for c in token.chars() {
        if !done && c.is_alphabetic() {
            out.extend(c.to_uppercase());
            done = true;
        } else {
            out.push(c);
        }
    }
}

/// Whether `token` ends a sentence for capitalization purposes. `!`, `?`, and
/// `…` always do; `.` does unless the token is an abbreviation, a dotted
/// acronym, or a single-letter initial.
fn token_ends_sentence(token: &str) -> bool {
    let t = token.trim_end_matches(CLOSING);
    let Some(last) = t.chars().last() else {
        return false;
    };
    if !TERMINATORS.contains(&last) {
        return false;
    }
    if last != '.' {
        return true;
    }
    let bare = t
        .trim_end_matches(TERMINATORS)
        .trim_start_matches(|c: char| !c.is_alphanumeric());
    if bare.is_empty() {
        return true;
    }
    if bare.contains('.') {
        return false;
    }
    if NON_TERMINAL_ABBREVIATIONS.contains(&bare.to_lowercase().as_str()) {
        return false;
    }
    // Single-letter initials ("J. Smith").
    let mut chars = bare.chars();
    if let (Some(c), None) = (chars.next(), chars.next())
        && c.is_alphabetic()
    {
        return false;
    }
    true
}

/// Whether the segment opens an interrogative: a wh-word lead, or an
/// auxiliary lead followed by a question subject (subject-auxiliary
/// inversion). Bare auxiliary leads are imperative-prone and do not count.
fn is_interrogative_lead(words: &[&str]) -> bool {
    let Some(first) = words.first().map(|w| word_core(w)) else {
        return false;
    };
    if WH_LEADS.iter().any(|lead| first.eq_ignore_ascii_case(lead)) {
        return true;
    }
    if AUX_LEADS
        .iter()
        .any(|lead| first.eq_ignore_ascii_case(lead))
    {
        return words.get(1).map(|w| word_core(w)).is_some_and(|second| {
            QUESTION_SUBJECTS
                .iter()
                .any(|s| second.eq_ignore_ascii_case(s))
        });
    }
    false
}

/// `word` with edge punctuation stripped ("you," → "you").
fn word_core(word: &str) -> &str {
    word.trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim_end_matches(|c: char| !c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_and_periods_plain_declarative() {
        let out = normalize("turn on the light");
        assert_eq!(out.text, "Turn on the light.");
        assert!(!out.needs_llm);
    }

    #[test]
    fn decase_lowercases_isolated_midsentence_common_word() {
        // An isolated over-capitalized common word is still reverted.
        assert_eq!(
            decase_common_words("I want to Make dinner"),
            "I want to make dinner"
        );
    }

    #[test]
    fn decase_preserves_multiword_proper_noun() {
        // A capital flanked by another capital is a proper-noun run — keep it,
        // even though "visual"/"studio"/"code" are individually common words.
        assert_eq!(
            decase_common_words("Open Visual Studio Code today"),
            "Open Visual Studio Code today"
        );
    }

    #[test]
    fn decase_spares_high_caprate_word() {
        // "Firefox" (p_cap ~0.73) is above the decase threshold even isolated.
        assert_eq!(
            decase_common_words("Then open Firefox now"),
            "Then open Firefox now"
        );
    }

    #[test]
    fn hotkey_joins_modifier_and_letter() {
        assert_eq!(
            join_hotkey_chords("set the hotkey to Super H"),
            "set the hotkey to Super+H"
        );
    }

    #[test]
    fn hotkey_joins_multiple_modifiers() {
        assert_eq!(
            join_hotkey_chords("press Control Shift P"),
            "press Control+Shift+P"
        );
    }

    #[test]
    fn hotkey_joins_function_key_and_keeps_trailing_punct() {
        assert_eq!(join_hotkey_chords("hit Alt F4 now."), "hit Alt+F4 now.");
        assert_eq!(
            join_hotkey_chords("bind it to Super H."),
            "bind it to Super+H."
        );
    }

    #[test]
    fn hotkey_leaves_prose_alone() {
        // Common words after a modifier are not keys; a lowercase modifier and
        // the article "a" never trigger a chord.
        for s in [
            "Control room access",
            "Shift focus to the goal",
            "super heroes assemble",
            "Control a list of items",
        ] {
            assert_eq!(join_hotkey_chords(s), s, "should be unchanged: {s}");
        }
    }

    #[test]
    fn hotkey_join_runs_inside_normalize() {
        assert_eq!(
            normalize("set the hotkey to Super H").text,
            "Set the hotkey to Super+H."
        );
    }

    #[test]
    fn hotkey_joins_named_keys() {
        assert_eq!(join_hotkey_chords("press Alt Tab"), "press Alt+Tab");
        assert_eq!(join_hotkey_chords("Control Enter"), "Control+Enter");
        assert_eq!(join_hotkey_chords("Super Space"), "Super+Space");
        assert_eq!(join_hotkey_chords("Control Shift Up"), "Control+Shift+Up");
    }

    #[test]
    fn hotkey_named_key_requires_capital() {
        // A lowercase named key after a capitalized modifier is left alone, so
        // ordinary phrasing ("shift down a gear", "control end of file") is safe.
        assert_eq!(join_hotkey_chords("Shift down a gear"), "Shift down a gear");
        assert_eq!(
            join_hotkey_chords("Control end of file"),
            "Control end of file"
        );
    }

    #[test]
    fn normalize_with_respects_toggles() {
        // All formatting transforms off: no keybind join, no title abbreviation,
        // no auto-capitalization. (The terminal period is not a toggle.)
        let off = FormatToggles {
            keybinds: false,
            title_abbreviations: false,
            auto_capitalization: false,
            brand_casing: false,
        };
        assert_eq!(
            normalize_with("set the hotkey to Super H", &off).text,
            "set the hotkey to Super H."
        );
        // Default (all on) joins the chord and capitalizes.
        assert_eq!(
            normalize_with("set the hotkey to Super H", &FormatToggles::default()).text,
            "Set the hotkey to Super+H."
        );
    }

    #[test]
    fn decase_fixes_double_cap_after_sentence_break() {
        assert_eq!(
            decase_common_words("Make sure it works. Make Sure it holds."),
            "Make sure it works. Make sure it holds."
        );
    }

    #[test]
    fn decase_preserves_proper_noun() {
        assert_eq!(decase_common_words("I met John today"), "I met John today");
    }

    #[test]
    fn decase_preserves_brand_and_acronym() {
        assert_eq!(
            decase_common_words("I use GitHub at NASA"),
            "I use GitHub at NASA"
        );
    }

    #[test]
    fn decase_preserves_sentence_start_common_word() {
        assert_eq!(decase_common_words("Sure it works."), "Sure it works.");
    }

    #[test]
    fn decase_keeps_pronoun_contractions() {
        // Regression: "I'm"/"I'll"/"I'd"/"I've" mid-sentence must stay
        // capitalized — the apostrophe-stripping key bug lowercased them.
        assert_eq!(
            decase_common_words("yes I'm sure and I'll go once I'd know and I've seen"),
            "yes I'm sure and I'll go once I'd know and I've seen"
        );
    }

    #[test]
    fn decase_lowercases_midsentence_lowercase_contractions() {
        // Contractions that are normally lowercase mid-sentence still revert.
        assert_eq!(
            decase_common_words("I know It's fine and we Don't care"),
            "I know it's fine and we don't care"
        );
    }

    #[test]
    fn decase_preserves_pronoun_i() {
        assert_eq!(decase_common_words("you and I agree"), "you and I agree");
    }

    #[test]
    fn decase_leaves_unknown_capitalized_word() {
        assert_eq!(
            decase_common_words("I saw Zylestra there"),
            "I saw Zylestra there"
        );
    }

    #[test]
    fn decase_handles_trailing_punctuation() {
        assert_eq!(
            decase_common_words("we should, Make sure."),
            "we should, make sure."
        );
    }

    #[test]
    fn defuse_splits_camel_fusion_when_phrase_recurs() {
        // "make sure" appears unfused, so the run-together "MakeSure" is the
        // ASR's own inconsistency and is restored.
        assert_eq!(
            defuse_inconsistent("make sure it works. MakeSure it holds."),
            "make sure it works. make sure it holds."
        );
    }

    #[test]
    fn defuse_splits_lowercase_fusion_when_phrase_recurs() {
        assert_eq!(
            defuse_inconsistent("please make sure of it. makesure of that."),
            "please make sure of it. make sure of that."
        );
    }

    #[test]
    fn defuse_preserves_real_word_even_if_split_recurs() {
        // "in to" appears adjacent, but "into" is a real word — never split it.
        assert_eq!(
            defuse_inconsistent("walk in to the room then go into it"),
            "walk in to the room then go into it"
        );
    }

    #[test]
    fn defuse_preserves_brand_without_recurrence() {
        assert_eq!(
            defuse_inconsistent("I use GitHub every day at work today"),
            "I use GitHub every day at work today"
        );
    }

    #[test]
    fn defuse_preserves_lone_fusion() {
        // No unfused "make sure" elsewhere — no evidence, so leave it.
        assert_eq!(
            defuse_inconsistent("just MakeSure it works fine"),
            "just MakeSure it works fine"
        );
    }

    #[test]
    fn defuse_keeps_trailing_punctuation_on_split() {
        assert_eq!(
            defuse_inconsistent("make sure of it. makesure!"),
            "make sure of it. make sure!"
        );
    }

    /// End-to-end proof of the Qwen3-ASR "Make Sure" fix: the pipeline order
    /// from session.rs (defuse → normalize → decase) turns every inconsistent
    /// rendering of a repeated phrase back into the clean form. Covers all four
    /// artifact shapes observed in the DB: correct, double-cap, CamelCase
    /// fusion, and lowercase fusion.
    #[test]
    fn full_pipeline_fixes_all_makesure_variants() {
        let raw = "make sure it works. Make Sure it holds. \
                   MakeSure it ends. makesure it stops.";
        let defused = defuse_inconsistent(raw);
        let normalized = normalize(&defused).text;
        let final_text = decase_common_words(&normalized);
        assert_eq!(
            final_text,
            "Make sure it works. Make sure it holds. \
             Make sure it ends. Make sure it stops."
        );
    }

    #[test]
    fn leaves_already_punctuated_text_alone() {
        let out = normalize("Turn on the light.");
        assert_eq!(out.text, "Turn on the light.");
        assert!(!out.needs_llm);
    }

    #[test]
    fn respects_existing_terminal_question_mark() {
        let out = normalize("what time is it?");
        assert_eq!(out.text, "What time is it?");
        assert!(!out.needs_llm);
    }

    #[test]
    fn interrogative_lead_without_mark_defers_to_llm() {
        let out = normalize("what time is it");
        // Capitalized but no terminal forced — the LLM owns `?` vs `.`.
        assert_eq!(out.text, "What time is it");
        assert!(out.needs_llm);
    }

    #[test]
    fn long_comma_less_run_defers_to_llm() {
        let out = normalize("turn on the light then close the door it is getting dark in here now");
        assert!(out.needs_llm, "12+ word separator-free run should escalate");
        assert!(out.text.starts_with("Turn"));
    }

    #[test]
    fn long_run_with_comma_stays_deterministic() {
        let out = normalize("turn on the light, then close the door it is getting dark");
        assert!(
            !out.needs_llm,
            "a comma signals existing structure, no escalation"
        );
        assert!(out.text.ends_with('.'));
    }

    #[test]
    fn terminal_detection_ignores_trailing_quote() {
        let out = normalize("he said \"go.\"");
        assert!(!out.needs_llm);
        assert!(out.text.ends_with("\"go.\""));
    }

    #[test]
    fn non_latin_text_passes_through_untouched() {
        let out = normalize("你好世界");
        assert_eq!(out.text, "你好世界");
        assert!(!out.needs_llm);
    }

    #[test]
    fn empty_text_passes_through() {
        let out = normalize("   ");
        assert!(!out.needs_llm);
    }

    #[test]
    fn abbreviates_spoken_titles_before_names() {
        let out = normalize("dear mister Thompson, I am writing to confirm");
        assert!(out.text.starts_with("Dear Mr. Thompson,"));
        let out = normalize("Missus Robinson arrived.");
        assert_eq!(out.text, "Mrs. Robinson arrived.");
        let out = normalize("Doctor Chen will see you now.");
        assert_eq!(out.text, "Dr. Chen will see you now.");
    }

    #[test]
    fn leaves_non_title_honorifics_alone() {
        let out = normalize("hey mister");
        assert_eq!(out.text, "Hey mister.");
        let out = normalize("the doctor said it was fine.");
        assert_eq!(out.text, "The doctor said it was fine.");
        let out = normalize("listen here mister you can't park there.");
        assert!(
            out.text.contains("mister you"),
            "lowercase next word → no abbreviation"
        );
    }

    #[test]
    fn imperative_aux_leads_stay_deterministic() {
        for text in [
            "have a great day",
            "do not touch that",
            "will check later tonight",
            "must remember to call mom",
            "can confirm the fix works",
            "would love to see that",
        ] {
            let out = normalize(text);
            assert!(!out.needs_llm, "imperative escalated: {text}");
            assert!(out.text.ends_with('.'), "no terminal period: {text}");
        }
    }

    #[test]
    fn aux_with_question_subject_defers_to_llm() {
        for text in [
            "can you hear me",
            "do you want coffee",
            "is there a way around it",
            "did he call back",
            "am i wrong about this",
        ] {
            let out = normalize(text);
            assert!(out.needs_llm, "question not escalated: {text}");
            assert!(
                !out.text.ends_with('.'),
                "period forced on question: {text}"
            );
        }
    }

    #[test]
    fn capitalizes_after_internal_sentence_breaks() {
        let out = normalize("okay that works. now let me think");
        assert_eq!(out.text, "Okay that works. Now let me think.");
        let out = normalize("hello there. how are you? great!");
        assert_eq!(out.text, "Hello there. How are you? Great!");
    }

    #[test]
    fn sentence_caps_skip_abbreviations_and_decimals() {
        assert_eq!(
            capitalize_sentence_starts("we used e.g. the fast path."),
            "We used e.g. the fast path."
        );
        assert_eq!(
            capitalize_sentence_starts("it costs 3.5 dollars. then we left."),
            "It costs 3.5 dollars. Then we left."
        );
        assert_eq!(
            capitalize_sentence_starts("j. smith signed it."),
            "J. smith signed it."
        );
        assert_eq!(
            capitalize_sentence_starts("the U.S. team won. great job."),
            "The U.S. team won. Great job."
        );
    }

    #[test]
    fn sentence_caps_respect_terminal_quotes_and_brand_casing() {
        assert_eq!(
            capitalize_sentence_starts("he said \"go.\" then he left."),
            "He said \"go.\" Then he left."
        );
        // Tokens with deliberate casing are never altered, even sentence-initially.
        assert_eq!(
            capitalize_sentence_starts("iPhone is great. iPhone is fine."),
            "iPhone is great. iPhone is fine."
        );
    }

    #[test]
    fn sentence_caps_preserve_whitespace_and_non_latin() {
        assert_eq!(
            capitalize_sentence_starts("  spaced   out. fine. "),
            "  Spaced   out. Fine. "
        );
        assert_eq!(
            capitalize_sentence_starts("你好世界. 再见"),
            "你好世界. 再见"
        );
    }

    #[test]
    fn mister_abbreviation_does_not_end_sentence() {
        let out = normalize("dear mister Thompson, I am writing");
        assert_eq!(out.text, "Dear Mr. Thompson, I am writing.");
    }

    #[test]
    fn short_interrogative_with_period_not_flagged() {
        // Declarative-ending despite an interrogative lead — not ambiguous.
        let out = normalize("how about that.");
        assert_eq!(out.text, "How about that.");
        assert!(!out.needs_llm);
    }
}
