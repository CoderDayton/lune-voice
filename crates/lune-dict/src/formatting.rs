// SPDX-License-Identifier: GPL-3.0-or-later

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
//!   terminal period when the segment is a single resolved declarative; append
//!   a `?` when it is a subject-auxiliary inversion question ("did we ship it");
//!   and demote a pause-driven over-split (`. But` → `, but`) back to a comma.
//!   These fixes are unambiguous, so they never need the model.
//! * **Tier 2 — ambiguous, trips the LLM gate:** comma-less run-ons and
//!   bare wh-leading segments with no end mark. The right fix here
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
    let hotkeyed = drop_dangling_separator(&hotkeyed);
    let trimmed = hotkeyed.as_str();

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let ends_terminal = ends_with_terminal(trimmed);
    // A colon, or a comma or semicolon left after the drop above, ends a short
    // opener (ignoring trailing closing quotes/brackets). It is the ASR's own
    // mark; keep it verbatim and never force a period on top of it — see the
    // terminal block below.
    let ends_open = trimmed.trim_end_matches(CLOSING).ends_with([':', ',', ';']);
    let interrogative = is_interrogative_lead(&words);

    // Run-on: long stretch with no internal separator or sentence break. Such a
    // segment likely needs internal commas or a split — the deterministic pass
    // can only touch the ends, so escalate it.
    let run_on = words.len() >= RUN_ON_WORD_THRESHOLD && !has_internal_separator(trimmed);

    // A subject-auxiliary inversion ("did we ship it", "is there time") is
    // grammatically unambiguous, so resolve it to `?` deterministically: the
    // styling LLM is optional and off by default, so a question must not depend
    // on it for punctuation. A bare wh-lead ("what a mess" vs "what time is it")
    // stays ambiguous and is still deferred.
    let confident_question =
        interrogative && !ends_terminal && !ends_open && is_aux_inversion_question(&words);

    // A remaining interrogative lead without an end mark is ambiguous
    // (`?` vs `.` vs `!`), so let the LLM decide rather than guessing.
    let needs_llm = run_on || (interrogative && !ends_terminal && !confident_question);

    // Tier 1: capitalize the first word of the segment and of every internal
    // sentence (ASR emits sentence breaks from pause prosody, and the words
    // after them arrive lowercase).
    let mut out = if fmt.auto_capitalization {
        capitalize_sentence_starts(trimmed)
    } else {
        trimmed.to_owned()
    };

    // Tier 1: append a terminal period for resolved declaratives — including
    // run-ons, so the deterministic (no-LLM) path is never left without an end
    // mark (`needs_llm` is unchanged: a run-on still routes to the LLM for
    // internal commas when it runs). Two guards: a one-word dictation (a bare
    // word, command, or fragment) gets no period, and an interrogative lead with
    // no end mark is deferred to the LLM (`?` vs `.` can't be guessed safely).
    // A trailing colon, comma or semicolon (`ends_open`) is left exactly as the
    // ASR produced it: no period is appended, so we never emit a double mark
    // such as ":." or ",." and never force a period where the true terminator
    // might be a question mark.
    if words.len() > 1 && !ends_terminal && !ends_open {
        if confident_question {
            out.push('?');
        } else if !interrogative {
            out.push('.');
        }
        // An ambiguous interrogative lead gets no mark here — `needs_llm` owns it.
    }

    // Undo pause-driven over-splitting: a thinking pause the ASR read as a
    // sentence break before a coordinating conjunction ("…so far. But I…")
    // becomes a comma ("…so far, but I…"). Always applied — gate-independent.
    let out = demote_oversplit_conjunctions(&out);

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
    let mut out: Vec<&str> = Vec::with_capacity(words.len());
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
        out.push(abbrev.unwrap_or(w));
    }
    out.join(" ")
}

/// Modifier words that begin a keyboard chord, matched case-insensitively.
/// Whether a match actually joins is decided by [`chord_fires`]: unambiguous
/// chords join regardless of casing, while a single modifier + an ordinary key
/// needs the writer's capitalization to avoid joining prose ("control room").
const HOTKEY_MODIFIERS: &[&str] = &[
    "super", "control", "ctrl", "alt", "shift", "command", "cmd", "meta", "win", "windows",
    "option", "hyper", "fn",
];

/// Single-token named keys that can terminate a chord, matched case-insensitively.
/// After a *single* modifier a named key still requires capitalization (via
/// [`chord_fires`]) so prose like "shift down a gear" is left untouched.
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

/// Join spoken keyboard chords into canonical `Modifier+Key` notation: "Super H"
/// → "Super+H", "control shift p" → "Ctrl+Shift+P", "alt f4" → "Alt+F4",
/// "Alt Tab" → "Alt+Tab".
///
/// A chord is one or more modifier words followed by exactly one terminal key.
/// Firing is split by ambiguity (see [`chord_fires`]): unambiguous chords — two
/// or more modifiers, or a function key — join regardless of casing (ASR emits
/// them lowercase), while a single modifier + an ordinary key still needs the
/// writer's capitalization so prose ("Control a list", "shift focus") is left
/// alone. The joined chord is canonicalized — modifiers Title-cased, letters
/// upper-cased, named keys Title-cased — so the output is uniform whatever the
/// input casing. Trailing punctuation on the key ("Super H." → "Super+H.") is
/// preserved.
fn join_hotkey_chords(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let mut j = i;
        while j < words.len() && is_modifier_word(words[j]) {
            j += 1;
        }
        // One-or-more modifiers (j > i) followed by exactly one terminal key.
        if j > i && j < words.len() {
            let (key_core, suffix) = split_trailing_punct(words[j]);
            if let Some(kind) = classify_key(key_core)
                && chord_fires(j - i, words[i], key_core, kind)
            {
                let mut chord = String::new();
                for (k, w) in words[i..j].iter().enumerate() {
                    if k > 0 {
                        chord.push('+');
                    }
                    chord.push_str(&canonical_modifier(w));
                }
                chord.push('+');
                chord.push_str(&canonical_key(key_core, kind));
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

/// The kind of token that can terminate a keyboard chord.
#[derive(Clone, Copy, PartialEq)]
enum KeyKind {
    Letter,
    Digit,
    Function,
    Named,
}

/// A punctuation-free modifier word, matched case-insensitively ("Control",
/// "ctrl", "shift"). Any edge punctuation ("Control,") disqualifies it and ends
/// the modifier run.
fn is_modifier_word(word: &str) -> bool {
    !word.is_empty()
        && word.bytes().all(|b| b.is_ascii_alphabetic())
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

/// Classify a chord-terminating key, or `None` when the token is not key-shaped:
/// a single letter, a single digit, a function key (F1–F24), or a named key
/// (Tab, Enter, the arrows, …). Casing is ignored here — [`chord_fires`] applies
/// any capitalization requirement.
fn classify_key(core: &str) -> Option<KeyKind> {
    if core.len() == 1 {
        let b = core.as_bytes()[0];
        if b.is_ascii_alphabetic() {
            return Some(KeyKind::Letter);
        }
        if b.is_ascii_digit() {
            return Some(KeyKind::Digit);
        }
        return None;
    }
    if is_function_key(core) {
        return Some(KeyKind::Function);
    }
    HOTKEY_NAMED_KEYS
        .contains(&core.to_ascii_lowercase().as_str())
        .then_some(KeyKind::Named)
}

/// A function key F1–F24, case-insensitive on the leading `F`.
fn is_function_key(core: &str) -> bool {
    let Some(rest) = core.strip_prefix(['F', 'f']) else {
        return false;
    };
    (1..=2).contains(&rest.len())
        && rest.bytes().all(|b| b.is_ascii_digit())
        && rest.parse::<u8>().is_ok_and(|n| (1..=24).contains(&n))
}

/// Whether a modifier run + terminal key should join. Unambiguous chords — two
/// or more modifiers, or a function key — fire regardless of casing, since they
/// never form English prose and ASR emits them lowercase. A single modifier plus
/// an ordinary key stays ambiguous with prose ("control a list", "shift up a
/// gear"), so it requires the writer's own capitalization as the cue. A lowercase
/// "a"/"i" is the article/pronoun and is never treated as a key.
fn chord_fires(modifier_count: usize, first_modifier: &str, key_core: &str, kind: KeyKind) -> bool {
    if kind == KeyKind::Letter && matches!(key_core, "a" | "i") {
        return false;
    }
    if modifier_count >= 2 || kind == KeyKind::Function {
        return true;
    }
    let modifier_capitalized = first_modifier.starts_with(|c: char| c.is_ascii_uppercase());
    let key_capitalized = match kind {
        KeyKind::Letter | KeyKind::Named => key_core.starts_with(|c: char| c.is_ascii_uppercase()),
        KeyKind::Digit | KeyKind::Function => true,
    };
    modifier_capitalized && key_capitalized
}

/// The canonical printed form of a modifier word. "control"/"ctrl" always
/// collapse to the short "Ctrl"; every other modifier is Title-cased as spoken.
fn canonical_modifier(word: &str) -> String {
    if word.eq_ignore_ascii_case("control") || word.eq_ignore_ascii_case("ctrl") {
        return "Ctrl".to_owned();
    }
    title_case_ascii(word)
}

/// The canonical printed form of a chord key: letters upper-cased, function keys
/// `F`+digits, named keys Title-cased, digits unchanged.
fn canonical_key(core: &str, kind: KeyKind) -> String {
    match kind {
        KeyKind::Letter => core.to_ascii_uppercase(),
        KeyKind::Digit => core.to_owned(),
        KeyKind::Function => format!("F{}", &core[1..]),
        KeyKind::Named => title_case_ascii(core),
    }
}

/// ASCII Title-case: first character upper, the rest lower ("ctrl" → "Ctrl",
/// "ENTER" → "Enter").
fn title_case_ascii(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => {
            let mut s = String::with_capacity(word.len());
            s.push(first.to_ascii_uppercase());
            s.push_str(&chars.as_str().to_ascii_lowercase());
            s
        }
        None => String::new(),
    }
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

/// Drop a comma or semicolon that dangles at the end of a segment of three or
/// more words (`we shipped it,`), keeping any closing quote or bracket after
/// it. The ASR emits one when a segment is cut off mid-clause, and a filler
/// stripped from the end (`…today, um`) leaves one behind. Removing it lets the
/// terminal-mark logic choose `.` or `?` instead of stacking a period on the
/// comma. A shorter segment (`hi team,`, `yes,`) is an opener or a vocative, so
/// it is left as spoken; see `ends_open` in [`normalize_with`].
fn drop_dangling_separator(text: &str) -> String {
    let body = text.trim_end_matches(CLOSING);
    let tail = &text[body.len()..];
    match body.strip_suffix([',', ';']) {
        Some(stem) if stem.split_whitespace().count() >= 3 => {
            format!("{}{}", stem.trim_end(), tail)
        }
        _ => text.to_owned(),
    }
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

/// Disfluency fillers stripped as whole tokens. English-only: "um" is a real
/// word in other languages (Portuguese "one/a"), so the caller language-gates
/// this and the multilingual ASR is never touched.
const FILLERS: &[&str] = &["um", "umm", "uhm", "uh", "uhh", "er", "erm", "hmm", "hm"];

/// Remove standalone hesitation fillers ("um", "uh", …) that Qwen3-ASR
/// transcribes verbatim, including mid-sentence ("the sidebar um, in its open
/// state" → "the sidebar in its open state"). A token is dropped only when its
/// punctuation-stripped, lowercased core is exactly a filler, so "umbrella",
/// "uh-huh", and ordinary words are never touched. Whitespace is rejoined with
/// single spaces (dictation text already uses them). If stripping would empty
/// the text — a pure-filler utterance — the original is returned unchanged.
/// English-only; the caller language-gates. Runs on the always-on text (fast
/// path + LLM input) so fillers are gone whether or not the LLM pass runs.
pub fn strip_fillers(text: &str) -> String {
    if !is_latin(text) {
        return text.to_owned();
    }
    let kept: Vec<&str> = text
        .split_whitespace()
        .filter(|tok| !FILLERS.contains(&word_core(tok).to_ascii_lowercase().as_str()))
        .collect();
    if kept.is_empty() {
        return text.to_owned();
    }
    kept.join(" ")
}

/// If the ENTIRE text is wrapped in a single matched pair of double quotes —
/// straight (`"…"`) or curly (`“…”`) — remove them. Whisper sometimes quotes a
/// whole utterance when it detects the speaker is reading/quoting aloud; the
/// speaker dictated the content, not the quote marks. Conservative: only a
/// matched wrapping pair with non-empty content and NO further copy of the
/// closing quote inside is stripped, so `"a" and "b"` and one-sided quotes are
/// left alone. Single quotes are intentionally excluded (apostrophes).
pub fn strip_wrapping_quotes(text: &str) -> String {
    let t = text.trim();
    let (open, close) = match (t.chars().next(), t.chars().next_back()) {
        (Some(o @ '"'), Some(c @ '"')) => (o, c),
        (Some(o @ '\u{201C}'), Some(c @ '\u{201D}')) => (o, c),
        _ => return text.to_owned(),
    };
    // Slice bounds from each quote's OWN byte width, so an open/close pair with
    // differing lengths would still land on char boundaries.
    let start = open.len_utf8();
    let end = t.len() - close.len_utf8();
    if end > start && !t[start..end].contains(close) {
        t[start..end].trim().to_owned()
    } else {
        text.to_owned()
    }
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
/// "iPhone"/"GitHub"/"NASA" are spared), NOT a curated brand (so "React"/"Rust"
/// are spared), and `en_p_cap` positively classifies the word as common (below
/// the stricter decase threshold). Unknown words (a proper noun the table does
/// not know) are left capitalized.
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
    // A curated brand whose canonical is capitalized (React, Rust, Redux) keeps
    // its casing even though the bare word is a common English word — p_cap alone
    // cannot tell brand "React" from verb "react", so the vocab is the authority.
    if crate::lexicon::is_capitalized_brand(&core.to_lowercase()) {
        return token.to_string();
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
        .flat_map(char::to_lowercase)
        .collect()
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
    // An interior dot normally means an abbreviation ("U.S.", "e.g."), which
    // does not end a sentence. An all-digit dotted token has the same shape
    // but genuinely ends one — prices and ratings ("that's a 4.5. Overall…"),
    // and by extension version numbers and IPs ("we shipped 1.2.3. The next…"),
    // are common in dictation, and misreading them swallowed the break.
    let is_dotted_number = bare
        .split('.')
        .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    if bare.contains('.') && !is_dotted_number {
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
    is_aux_inversion_question(words)
}

/// Whether the segment is a subject-auxiliary inversion question ("did we ship
/// it", "is there time") — an auxiliary lead immediately followed by a question
/// subject. Inversion unambiguously marks a question, so unlike a bare wh-lead
/// this can be resolved to a terminal `?` deterministically rather than deferred
/// to the (optional) LLM.
fn is_aux_inversion_question(words: &[&str]) -> bool {
    let Some(first) = words.first().map(|w| word_core(w)) else {
        return false;
    };
    if !AUX_LEADS
        .iter()
        .any(|lead| first.eq_ignore_ascii_case(lead))
    {
        return false;
    }
    words.get(1).map(|w| word_core(w)).is_some_and(|second| {
        QUESTION_SUBJECTS
            .iter()
            .any(|s| second.eq_ignore_ascii_case(s))
    })
}

/// `word` with edge punctuation stripped ("you," → "you").
fn word_core(word: &str) -> &str {
    word.trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim_end_matches(|c: char| !c.is_alphanumeric())
}

/// Words that, opening a pause-split fragment, continue the previous thought
/// rather than start a new sentence — so a period the recognizer inserted before
/// them is demoted to a comma. Two grammatically comma-joinable groups:
///   - coordinating conjunctions: and, but, so, or, yet
///   - concessive / contrast subordinators: though, although, whereas
///
/// Deliberately excluded — adding these would be *wrong*, not just cautious:
///   - "for" / "nor": comma-joinable in theory, but "For"/"Nor" almost always
///     open a real sentence (preposition / negative-inversion), so demoting
///     would fuse unrelated sentences.
///   - conjunctive adverbs (however, therefore, then, thus, hence, moreover,
///     furthermore, nevertheless, nonetheless, meanwhile, otherwise, instead,
///     besides, consequently, accordingly, anyway, still, also): joining two
///     clauses with a *comma* before these is a comma splice — they require a
///     period or semicolon — and "then"/"meanwhile" mark genuine sequences that
///     must stay separate.
const CONTINUATION_WORDS: &[&str] = &[
    "and", "but", "so", "or", "yet", "though", "although", "whereas",
];

/// Upper bound on how many words one sentence may grow to via demotion. Past
/// this, a candidate break is kept rather than spliced, so an adversarial or
/// rambling dictation ("…. So …. So …. So …" without end) can never fuse into a
/// single unbounded comma-spliced wall. Set well above any natural sentence
/// (~40 words) so it never constrains real dictation — purely a safety guard.
const MAX_CLAUSE_WORDS: usize = 48;

/// Demote an over-split sentence break back to a comma when the ASR turned a
/// thinking pause into a period right before a [`CONTINUATION_WORDS`] token that
/// continues the thought ("…so far. But I want…" → "…so far, but I want…").
///
/// The acoustic pause length the model uses to choose `.` vs `,` is baked into
/// the ASR weights — there is no threshold of ours to raise — so this corrects
/// the over-read deterministically after the fact. Only a plain word ending in
/// a single period qualifies as the break, so abbreviations ("etc."), dotted
/// forms ("U.S."), initials ("J."), and acronym sentence-ends ("NASA.") are
/// left intact; and a genuine new sentence opens on "Then"/"Finally"/a fresh
/// subject — never a continuation word — so real breaks stay split.
///
/// Two linear passes over borrowed token spans (no per-token allocation): one
/// marks the boundaries to demote while tracking the running sentence length
/// for the [`MAX_CLAUSE_WORDS`] guard, one rebuilds the string. O(n) in input.
fn demote_oversplit_conjunctions(text: &str) -> String {
    if !is_latin(text) {
        return text.to_owned();
    }

    // Borrowed (leading-whitespace, token) spans so exact spacing survives the
    // rewrite with no String allocated per token. A trailing whitespace-only
    // run is captured as a final span with an empty token.
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

    // Pass 1: mark each conjunction token whose preceding token's terminal
    // period should become a comma. `clause_words` accumulates across demoted
    // boundaries (the clause keeps growing) and resets at every kept break.
    let mut demote = vec![false; segs.len()];
    let mut clause_words = 0usize;
    for i in 0..segs.len() {
        let tok = segs[i].1;
        if tok.is_empty() {
            continue;
        }
        clause_words += 1;
        let oversplit = is_oversplit_period_word(tok);
        let next_is_conj = segs
            .get(i + 1)
            .map(|s| s.1)
            .filter(|n| !n.is_empty())
            .is_some_and(|n| {
                CONTINUATION_WORDS.contains(&word_core(n).to_ascii_lowercase().as_str())
            });
        if oversplit && next_is_conj && clause_words < MAX_CLAUSE_WORDS {
            demote[i + 1] = true; // splice: the clause continues, do not reset
        } else if oversplit || token_ends_sentence(tok) {
            clause_words = 0; // a kept sentence break starts a new clause
        }
    }

    // Pass 2: rebuild. A token preceding a demoted conjunction loses its
    // terminal period to a comma; a demoted conjunction is fully lowercased
    // ("But," / "SO" → "but" / "so") with a now-redundant trailing comma dropped.
    let mut out = String::with_capacity(text.len() + 4);
    for (idx, (ws, tok)) in segs.iter().enumerate() {
        out.push_str(ws);
        if tok.is_empty() {
            continue;
        }
        let demote_this = demote[idx];
        let comma_after = demote.get(idx + 1).copied().unwrap_or(false);
        if !demote_this && !comma_after {
            out.push_str(tok);
            continue;
        }
        let mut piece = if demote_this {
            let lowered = tok.to_lowercase();
            lowered
                .strip_suffix(',')
                .map(str::to_owned)
                .unwrap_or(lowered)
        } else {
            (*tok).to_owned()
        };
        if comma_after && piece.ends_with('.') {
            piece.pop();
            piece.push(',');
        }
        out.push_str(&piece);
    }
    out
}

/// Whether `tok` is a plain word ending in exactly one sentence period — the
/// only shape treated as an over-split break. Requires a lowercase Latin letter
/// immediately before the `.`, no interior `.`, and not a known non-terminal
/// abbreviation, so "U.S.", "etc.", "J.", and "NASA." never qualify.
fn is_oversplit_period_word(tok: &str) -> bool {
    let Some(stem) = tok.strip_suffix('.') else {
        return false;
    };
    if stem.is_empty() || stem.contains('.') {
        return false;
    }
    let last = stem.chars().next_back().unwrap();
    if !(last.is_alphabetic() && last.is_lowercase()) {
        return false;
    }
    !NON_TERMINAL_ABBREVIATIONS.contains(&stem.to_ascii_lowercase().as_str())
}

/// Deterministic shaping of a corrected transcript, applied in a fixed order:
/// English-only de-fusing and filler stripping, wrapping-quote removal,
/// punctuation and casing normalization, then English-only mid-sentence
/// de-casing.
///
/// `strip_wrapping_quotes` must run before `normalize_with`: normalize appends
/// terminal punctuation outside a closing quote (`"…".`), and the strip, which
/// needs the quote to be the final character, can no longer find the pair.
///
/// `lang` is a display name from `lune_core::lang::display_name`. The
/// English-only passes rely on English capitalization and frequency tables, and
/// on "um" being a filler, which it is not in every language.
pub fn shape(text: &str, lang: &str, fmt: &FormatToggles) -> String {
    let english = lang == "English";
    let text = if english {
        strip_fillers(&defuse_inconsistent(text))
    } else {
        text.to_owned()
    };
    let text = strip_wrapping_quotes(&text);
    let text = normalize_with(&text, fmt).text;
    if english {
        decase_common_words(&text)
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_strips_wrapping_quotes_before_punctuating() {
        let fmt = FormatToggles::default();
        assert_eq!(
            shape("\"the results were mixed\"", "English", &fmt),
            "The results were mixed."
        );
    }

    #[test]
    fn shape_never_puts_a_period_after_a_trailing_comma() {
        let fmt = FormatToggles::default();
        for (raw, want) in [
            ("we shipped it today,", "We shipped it today."),
            ("we shipped it today;", "We shipped it today."),
            ("we shipped it today , ", "We shipped it today."),
            ("did we ship it today,", "Did we ship it today?"),
            ("he said \"we shipped it,\"", "He said \"we shipped it\"."),
            ("we shipped it today, um", "We shipped it today."),
            (
                "we shipped it today, and then",
                "We shipped it today, and then.",
            ),
        ] {
            assert_eq!(shape(raw, "English", &fmt), want, "{raw}");
        }
    }

    #[test]
    fn shape_keeps_a_short_opener_as_spoken() {
        let fmt = FormatToggles::default();
        for (raw, want) in [
            ("yes,", "Yes,"),
            ("hi team,", "Hi team,"),
            ("thanks everyone,", "Thanks everyone,"),
            ("he left;", "He left;"),
        ] {
            assert_eq!(shape(raw, "English", &fmt), want, "{raw}");
        }
    }

    #[test]
    fn shape_strips_english_fillers() {
        let fmt = FormatToggles::default();
        assert_eq!(
            shape("um the results were mixed", "English", &fmt),
            "The results were mixed."
        );
    }

    #[test]
    fn shape_keeps_english_only_passes_off_other_languages() {
        // "um" is a real word outside English, so it must survive.
        let fmt = FormatToggles::default();
        let shaped = shape("um dia bonito", "Portuguese", &fmt);
        assert!(shaped.starts_with("Um dia"), "{shaped}");
    }

    #[test]
    fn strip_wrapping_quotes_removes_whole_text_pair() {
        assert_eq!(strip_wrapping_quotes("\"Hello world.\""), "Hello world.");
        assert_eq!(
            strip_wrapping_quotes(" \"Read this aloud\" "),
            "Read this aloud"
        );
        // Curly quotes.
        assert_eq!(strip_wrapping_quotes("\u{201C}Quoted\u{201D}"), "Quoted");
    }

    #[test]
    fn strip_wrapping_quotes_leaves_partial_or_internal_quotes() {
        // Not wrapped.
        assert_eq!(strip_wrapping_quotes("no quotes here"), "no quotes here");
        // One-sided.
        assert_eq!(strip_wrapping_quotes("\"only open"), "\"only open");
        // Internal quotes: not a single wrapping span, so left intact.
        assert_eq!(strip_wrapping_quotes("\"a\" and \"b\""), "\"a\" and \"b\"");
        // A quoted phrase mid-sentence is untouched (doesn't wrap the whole text).
        assert_eq!(
            strip_wrapping_quotes("she said \"hi\" to me"),
            "she said \"hi\" to me"
        );
        // Degenerate inputs.
        assert_eq!(strip_wrapping_quotes("\"\""), "\"\"");
        assert_eq!(strip_wrapping_quotes("\""), "\"");
    }

    #[test]
    fn wrapping_quotes_must_be_stripped_before_normalize() {
        // Ordering contract: strip runs BEFORE punctuation normalization.
        // If it ran after, normalize appends the terminal period outside the
        // closing quote (`"…".`) and the pair is no longer detectable.
        let corrected = strip_wrapping_quotes("\"the results were mixed\"");
        assert_eq!(normalize(&corrected).text, "The results were mixed.");
    }

    #[test]
    fn capitalizes_and_periods_plain_declarative() {
        let out = normalize("turn on the light");
        assert_eq!(out.text, "Turn on the light.");
        assert!(!out.needs_llm);
    }

    #[test]
    fn one_word_dictation_gets_no_period() {
        // A bare word / command / fragment is left without a terminal period.
        assert_eq!(normalize("voice").text, "Voice");
        assert_eq!(normalize("okay").text, "Okay");
    }

    #[test]
    fn trailing_colon_is_kept_without_added_period() {
        // A colon ending an utterance is the ASR's own mark: keep it verbatim,
        // with no appended period (which could be wrong — the true terminator is
        // sometimes a question mark) and no double mark ":.".
        assert_eq!(
            normalize("here are the reasons:").text,
            "Here are the reasons:"
        );
        assert_eq!(normalize("the reasons:").text, "The reasons:");
        assert_eq!(normalize("how do we fix this:").text, "How do we fix this:");
        assert_eq!(normalize("note:").text, "Note:");
    }

    #[test]
    fn run_on_declarative_still_gets_terminal_period() {
        // A long comma-less declarative still routes to the LLM (needs_llm) for
        // internal commas, but the deterministic path now closes it so the
        // no-LLM (minimal) output isn't left period-less.
        let out =
            normalize("you might have to make a new token that is in between those two colors");
        assert!(
            out.needs_llm,
            "long comma-less declarative must still route to the LLM"
        );
        assert!(
            out.text.ends_with('.'),
            "deterministic path must still close the run-on: {:?}",
            out.text
        );
    }

    #[test]
    fn demotes_oversplit_coordinating_conjunction_to_comma() {
        // A pause the ASR over-read as a sentence break leaves ". But" / ". So"
        // where the thought continues — demote the period to a comma.
        let out = normalize("everything looks stable so far. But I want to keep an eye on it");
        assert!(
            out.text.contains("so far, but I want"),
            "expected period->comma demotion: {:?}",
            out.text
        );
        assert!(!out.text.contains("so far. But"), "{:?}", out.text);

        let out = normalize("we should weigh both. So we decided to wait a bit");
        assert!(out.text.contains("both, so we decided"), "{:?}", out.text);
    }

    #[test]
    fn demotion_leaves_genuine_sentence_breaks_intact() {
        // "Then" is not a coordinating conjunction — a real break stays split.
        let out = normalize("run the start script. Then open the config file");
        assert!(
            out.text.contains("script. Then open"),
            "genuine break must stay split: {:?}",
            out.text
        );
    }

    #[test]
    fn demotion_excludes_for_and_nor() {
        // "for"/"nor" are too ambiguous (preposition / inversion) to splice.
        let out = normalize("i waited a while. For the bus was running late again");
        assert!(out.text.contains(". For the bus"), "{:?}", out.text);
    }

    #[test]
    fn demotion_skips_abbreviations_and_acronyms() {
        // A trailing-period abbreviation or an acronym sentence-end is not a
        // plain-word period, so the break is left intact.
        let out = normalize("bring pens, paper, etc. And the laptop too please");
        assert!(out.text.contains("etc. And the"), "{:?}", out.text);

        let out = normalize("i used to work at NASA. And i miss it sometimes now");
        assert!(out.text.contains("NASA. And i"), "{:?}", out.text);
    }

    #[test]
    fn demotion_collapses_redundant_conjunction_comma() {
        // "…it. But, it failed" → one comma, not "…it, but, it failed".
        let out = normalize("we shipped it. But, it failed badly in production");
        assert!(out.text.contains("it, but it failed"), "{:?}", out.text);
        assert!(!out.text.contains("but,"), "{:?}", out.text);
    }

    #[test]
    fn demotion_no_op_when_already_comma_joined() {
        // No period before the conjunction — nothing to demote.
        let out = normalize("send it over whenever you're ready, and I'll review it later");
        assert!(
            out.text.contains("ready, and I'll review"),
            "{:?}",
            out.text
        );
    }

    #[test]
    fn demotes_trailing_concessive_though() {
        // "though" trailing a clause continues the thought — "It's nice. Though."
        // is a pause artifact, not two sentences. Demote to "It's nice, though."
        let out = normalize("the toggle is pretty nice. Though.");
        assert!(out.text.contains("nice, though"), "{:?}", out.text);
        assert!(!out.text.contains("nice. Though"), "{:?}", out.text);
    }

    #[test]
    fn demotes_concessive_subordinators() {
        // "although"/"whereas" take a comma when the clause follows.
        let out = normalize("the api works fine. Although it is a little slow today");
        assert!(out.text.contains("fine, although it"), "{:?}", out.text);
        let out = normalize("sales rose in the north. Whereas the south stayed flat");
        assert!(out.text.contains("north, whereas the"), "{:?}", out.text);
    }

    #[test]
    fn demotion_excludes_conjunctive_adverbs() {
        // A comma before a conjunctive adverb is a comma splice, and "then"
        // marks a real sequence — these breaks must survive untouched.
        assert!(
            normalize("it failed. However we recovered quickly")
                .text
                .contains(". However"),
        );
        assert!(
            normalize("do this first. Then run the install script")
                .text
                .contains(". Then"),
        );
        assert!(
            normalize("it is getting late. Therefore we should stop now")
                .text
                .contains(". Therefore"),
        );
        assert!(
            normalize("we tried really hard. Anyway it did not work out")
                .text
                .contains(". Anyway"),
        );
    }

    #[test]
    fn demotion_allows_multiple_within_one_normal_sentence() {
        // Two over-splits in one ordinary-length sentence both demote.
        let out = normalize("we shipped it. But it broke. So we rolled back quickly");
        assert!(
            out.text.contains("it, but it broke, so we rolled back"),
            "{:?}",
            out.text
        );
    }

    #[test]
    fn demotion_lowercases_all_caps_conjunction() {
        // An all-caps conjunction must fully lowercase ("SO" → "so"), not "sO".
        let out = normalize("we waited a while. SO we left without them");
        assert!(out.text.contains("while, so we left"), "{:?}", out.text);
        assert!(!out.text.contains("sO"), "{:?}", out.text);
    }

    #[test]
    fn demotion_bounds_pathological_chaining() {
        // Adversarial: dozens of clauses each ending ". So <x>". The clause-length
        // guard must keep some breaks rather than fuse everything into one
        // unbounded comma-spliced wall.
        let mut s = String::from("alpha beta gamma");
        for _ in 0..40 {
            s.push_str(". So delta epsilon");
        }
        let out = normalize(&s).text;
        assert!(
            out.contains(". So"),
            "clause-length cap must keep some sentence breaks"
        );
    }

    #[test]
    fn demotion_handles_adversarial_inputs_without_panicking() {
        // None of these may panic (slicing, UTF-8 boundaries, empty stems).
        for s in [
            "",
            "   ",
            ". . . And . So .",
            "but",
            "So",
            "done. So",
            ".And",
            "café. So déjà vu continúa",
            "a. SO b",
            "...... and",
            "U.S. And the rest",
            "wait. Nor did I agree",
        ] {
            let _ = normalize(s);
        }
        // Acronym/abbreviation period is never spliced; unicode stems survive.
        assert!(
            normalize("i used to work at U.S. And the rest of it now")
                .text
                .contains("U.S. And"),
        );
        assert!(
            normalize("café looks good so far. But déjà vu sets in now")
                .text
                .contains("far, but déjà"),
        );
    }

    #[test]
    fn strip_fillers_removes_standalone_disfluencies() {
        // Real transcripts from dictation history (leading + mid-sentence).
        assert_eq!(
            strip_fillers("Um, yeah, you should have had it there"),
            "yeah, you should have had it there"
        );
        assert_eq!(
            strip_fillers("the sidebar um, in its open state"),
            "the sidebar in its open state"
        );
        assert_eq!(
            strip_fillers("you might have to uh make a new token"),
            "you might have to make a new token"
        );
    }

    #[test]
    fn strip_fillers_spares_real_words_and_pure_filler() {
        // Substrings and hyphenated agreement are not fillers.
        assert_eq!(
            strip_fillers("the umbrella is huge"),
            "the umbrella is huge"
        );
        assert_eq!(strip_fillers("uh-huh I agree"), "uh-huh I agree");
        // A pure-filler utterance is left intact rather than emptied.
        assert_eq!(strip_fillers("um uh"), "um uh");
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
    fn decase_spares_known_brand_collision() {
        // Common-word dev brands (React, Rust) sit in the vocab with a
        // capitalized canonical, so decase keeps their casing mid-sentence even
        // though p_cap classifies the bare word as common (react=0.08, rust=0.10).
        assert_eq!(
            decase_common_words("we use React and Rust here"),
            "we use React and Rust here"
        );
        // The brand spare is targeted, not blanket: an ordinary over-capitalized
        // common word is still reverted.
        assert_eq!(
            decase_common_words("I want to Make dinner"),
            "I want to make dinner"
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
            "press Ctrl+Shift+P"
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
        assert_eq!(join_hotkey_chords("Control Enter"), "Ctrl+Enter");
        assert_eq!(join_hotkey_chords("Super Space"), "Super+Space");
        assert_eq!(join_hotkey_chords("Control Shift Up"), "Ctrl+Shift+Up");
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
    fn hotkey_joins_lowercase_unambiguous_chords() {
        // Two-plus modifiers never form prose, so they join even all-lowercase —
        // the common real-ASR case the capital gate used to miss.
        assert_eq!(
            join_hotkey_chords("press control shift p"),
            "press Ctrl+Shift+P"
        );
        assert_eq!(join_hotkey_chords("control alt delete"), "Ctrl+Alt+Delete");
        assert_eq!(join_hotkey_chords("ctrl alt del"), "Ctrl+Alt+Del");
        // A function key is unmistakable, so a single lowercase modifier joins.
        assert_eq!(join_hotkey_chords("hit alt f4"), "hit Alt+F4");
        assert_eq!(join_hotkey_chords("super f1"), "Super+F1");
    }

    #[test]
    fn hotkey_canonicalizes_output_casing() {
        // Whatever the input casing, the chord prints canonically.
        assert_eq!(join_hotkey_chords("CONTROL SHIFT P"), "Ctrl+Shift+P");
        assert_eq!(join_hotkey_chords("ctrl shift tab"), "Ctrl+Shift+Tab");
        assert_eq!(join_hotkey_chords("ALT F4"), "Alt+F4");
    }

    #[test]
    fn hotkey_control_always_shortens_to_ctrl() {
        // "control"/"ctrl"/"CONTROL" all collapse to the short "Ctrl".
        assert_eq!(join_hotkey_chords("control shift p"), "Ctrl+Shift+P");
        assert_eq!(join_hotkey_chords("Control Enter"), "Ctrl+Enter");
        assert_eq!(join_hotkey_chords("CONTROL ALT DELETE"), "Ctrl+Alt+Delete");
    }

    #[test]
    fn hotkey_single_lowercase_modifier_stays_prose() {
        // A single lowercase modifier + an ordinary key is ambiguous with prose,
        // so without the writer's capitalization it is left untouched.
        for s in ["press control p now", "super 2 birds", "shift tab order"] {
            assert_eq!(join_hotkey_chords(s), s, "should be unchanged: {s}");
        }
        // The writer's capital is the cue that promotes it to a chord.
        assert_eq!(join_hotkey_chords("press Super 2"), "press Super+2");
    }

    #[test]
    fn hotkey_guards_lowercase_article_and_pronoun() {
        // A lone lowercase "a"/"i" is the article/pronoun, never a key — even
        // with multiple modifiers in front of it.
        assert_eq!(join_hotkey_chords("control command a"), "control command a");
        assert_eq!(join_hotkey_chords("control shift i"), "control shift i");
        // An explicit capital letter is a genuine key reference.
        assert_eq!(join_hotkey_chords("hit Control A"), "hit Ctrl+A");
    }

    #[test]
    fn hotkey_function_key_bounds() {
        assert_eq!(join_hotkey_chords("super f24"), "Super+F24");
        // Out-of-range / malformed F-keys are not keys, so nothing joins.
        for s in ["Super F25", "Super F0", "Super FA"] {
            assert_eq!(join_hotkey_chords(s), s, "should be unchanged: {s}");
        }
    }

    #[test]
    fn hotkey_preserves_trailing_punct_on_lowercase_chord() {
        assert_eq!(
            join_hotkey_chords("press control shift p."),
            "press Ctrl+Shift+P."
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
            brand_casing: Some(false),
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
    fn aux_inversion_questions_get_deterministic_question_mark() {
        // Subject-auxiliary inversion is unambiguously a question, so it is
        // resolved to `?` deterministically (no LLM) instead of being left
        // unpunctuated when the optional styling LLM is off.
        for text in [
            "can you hear me",
            "do you want coffee",
            "is there a way around it",
            "did he call back",
            "am i wrong about this",
        ] {
            let out = normalize(text);
            assert!(!out.needs_llm, "resolved deterministically: {text}");
            assert!(out.text.ends_with('?'), "no question mark: {text}");
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

    /// A decimal ending a sentence has the same shape as an abbreviation, but
    /// it really is a sentence break — the next word must be capitalized.
    #[test]
    fn sentence_caps_break_after_a_sentence_final_decimal() {
        assert_eq!(
            capitalize_sentence_starts("the price is 3.5. next one is cheaper."),
            "The price is 3.5. Next one is cheaper."
        );
        assert_eq!(
            capitalize_sentence_starts("that's a 4.5. overall i'd recommend it."),
            "That's a 4.5. Overall i'd recommend it."
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
