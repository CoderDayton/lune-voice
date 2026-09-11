use std::collections::HashMap;

use crate::lexicon::{Lexicon, PhraseMatch, WordSource};

pub fn has_dict(language: &str) -> bool {
    crate::freq::has_dict(language)
}

/// Risk score (Heuristic Word Confidence Estimate) at which a word is treated
/// as "uncertain" for the session gate and gate-decision event.
pub const HWCE_FLAG_THRESHOLD: f32 = 0.40;

/// Word risk at/above which a FUZZY (edit-distance-1) dictionary match is
/// skipped: the word is too uncertain (≈ confidence ≤ 0.45) to trust a
/// near-match, so a garbled ASR onset isn't rewritten into a coincidentally
/// close dictionary term. Independent of `HWCE_FLAG_THRESHOLD` so corrector
/// tuning doesn't change dictionary safety. Exact/phrase matches ignore this.
const FUZZY_DICT_RISK_GATE: f32 = 0.55;

/// Built-in always-on corrections: a specific ASR mishear (left, the
/// lowercased form to match) that is ALWAYS rewritten to its canonical form
/// (right) — unconditionally, with no fuzzy-similarity check and no risk gate,
/// exactly like a hand-entered EXACT dictionary entry. A tiny curated safety
/// net for a few high-value terms the bundled vocab can only re-case when
/// spoken correctly but (since bundled-vocab fuzzy was removed) can no longer
/// repair a mishear of. Keep it short and exact. A user-dictionary entry for
/// the same term overrides these on collision.
const ALWAYS_REPLACE: &[(&str, &str)] = &[
    // Project proper nouns (formerly data/vocab/personal.txt, now curated here):
    // for each, the correctly-spoken form re-cased to canonical AND the observed
    // ASR mishear. Applied VERBATIM (never softened/gated), so a key must be
    // safe to force unconditionally — never map a common English word to a
    // different word, or correct text gets corrupted. (Standalone "loon"->"Lune"
    // is deliberately absent: "loon" is a common word/bird — only the
    // unambiguous "loon voice" phrase maps to the product.)
    ("lune voice", "Lune Voice"),
    ("loon voice", "Lune Voice"),
    ("lune", "Lune"),
    ("dayton", "Dayton"),
    ("tayton", "Dayton"),
];

/// Maximum risk at which a known high-frequency word is allowed to pass
/// through unchanged. Above this, the word is marked uncertain even if the
/// lexicon recognizes it.
pub const CANONICAL_OVERRIDE_RISK_CEIL: f32 = 0.75;

/// Maximum risk at which an unknown but proper-shaped token (capitalized
/// word, acronym, CamelCase) is preserved verbatim. Above this, the word is
/// flagged regardless of shape.
pub const PROPER_NOUN_RISK_CEIL: f32 = 0.80;

#[derive(Debug)]
pub struct WordScore {
    pub word: String,
    pub position: usize,
    pub candidate_score: f32,
}

/// Result of processing a single word during correction.
/// Contains the word to include in output and optional unresolved entry.
struct ProcessedWord {
    corrected_word: String,
    unresolved: Option<WordScore>,
}

pub struct CorrectionResult {
    pub corrected_text: String,
    pub word_risk_positions: Vec<(usize, f32)>,
    pub unresolved_words: Vec<WordScore>,
    pub all_resolved: bool,
    /// Risk-score entries that did not align to any text token. Non-zero
    /// indicates ASR word boundaries diverged from `split_whitespace`
    /// (punctuation splits, contractions, duplicate-without-match).
    pub dropped_risk_entries: usize,
}

/// Process a single word during correction.
///
/// Returns the word to include in the output and optionally a `WordScore`
/// for unresolved words that will be passed to the LLM.
#[allow(clippy::too_many_arguments)] // per-word inputs + the brand-casing toggle
fn process_single_word(
    word: &str,
    i: usize,
    stripped: &str,
    lexicon_info: &Option<crate::lexicon::WordInfo>,
    risk_score: f32,
    flag_threshold: f32,
    dict_lookup: &HashMap<String, (&str, bool, bool)>,
    brand_casing: bool,
) -> ProcessedWord {
    // Empty or digit-containing words pass through unchanged
    if stripped.is_empty() || stripped.chars().any(|c| c.is_ascii_digit()) {
        return ProcessedWord {
            corrected_word: word.to_owned(),
            unresolved: None,
        };
    }

    let is_unknown = !lexicon_info.as_ref().is_some_and(|info| {
        info.freq >= Lexicon::CONFIDENT_FREQ || info.source == WordSource::Vocab
    });
    let flagged = risk_score >= flag_threshold;

    // Dictionary replacements take priority. An empty dictionary (the common
    // case) skips the per-word lowercase + probe entirely.
    let dict_repl = (!dict_lookup.is_empty())
        .then(|| dict_lookup.get(crate::fold_ci(stripped).as_str()).copied())
        .flatten();
    if let Some((repl, fuzzy, verbatim)) = dict_repl {
        // A FUZZY (edit-distance-1) match on a flagged/low-confidence word is
        // skipped: the word is likely a garbled ASR onset, so the near-match is
        // coincidental rather than the dictionary term. Exact and multi-word
        // matches (fuzzy == false) always apply; confident mishears stay above
        // the gate and are still corrected.
        if !(fuzzy && risk_score >= FUZZY_DICT_RISK_GATE) {
            // Title-case-only replacements of ordinary words ("Dev", "Server")
            // must not force a mid-sentence capital: apply the spelling in
            // lowercase and let the sentence-initial capitalizer own position
            // casing. Brand casing (GitHub, iPhone, NASA) applies verbatim.
            // A `verbatim` entry (built-in ALWAYS_REPLACE, curated casing) is
            // never softened — e.g. "Lune" stays "Lune", not "lune".
            let softened = if verbatim {
                None
            } else {
                soften_forced_title_case(repl)
            };
            return ProcessedWord {
                corrected_word: restore_punctuation(word, softened.as_deref().unwrap_or(repl)),
                unresolved: None,
            };
        }
    }

    // Known, unflagged words pass through
    if !is_unknown && !flagged {
        return ProcessedWord {
            corrected_word: word.to_owned(),
            unresolved: None,
        };
    }

    // Apply canonical casing from the vocab (e.g. "github" → "GitHub").
    // Only override when the canonical itself carries uppercase information —
    // otherwise we'd flatten correct sentence-initial capitalization (e.g.
    // "All" → "all") because freq-dict canonicals fall back to lowercase.
    if let Some(info) = lexicon_info.as_ref() {
        let canonical_has_upper = info.canonical.chars().any(|c| c.is_ascii_uppercase());
        // Title-case-only canonicals ("Calibration", "Transformers") carry no
        // brand-casing information when the lowercase form is ordinary
        // English — they are sentence-start artifacts in harvested vocab
        // lists, and enforcing them mid-sentence capitalizes common nouns.
        // Casing authority is limited to canonicals with uppercase beyond
        // the first character (GitHub, NASA, iPhone) or words that are not
        // ordinary dictionary words. Ambiguous proper-noun casing is the
        // LLM's job, not the heuristic's.
        let title_case_only = {
            let mut chars = info.canonical.chars();
            chars.next().is_some_and(|c| c.is_ascii_uppercase())
                && !chars.any(|c| c.is_ascii_uppercase())
        };
        // Whether the lowercase form is an ordinary (normally-lowercase) word,
        // so a Title-case-only canonical must NOT force a mid-sentence capital.
        // The capitalization rate is the authority when the cap-rate table knows
        // the word (a high rate means a genuine proper noun whose canonical
        // capital is correct); fall back to dictionary membership otherwise.
        let lowercase_is_common = match info.p_cap {
            Some(p) => p < crate::caprate::CAP_KEEP_THRESHOLD,
            None => {
                info.source == WordSource::FreqMap
                    || wordnik_list::word_exists(&stripped.to_lowercase())
            }
        };
        if brand_casing
            && canonical_has_upper
            && info.canonical.as_ref() != stripped
            && !(title_case_only && lowercase_is_common)
        {
            return ProcessedWord {
                corrected_word: restore_punctuation(word, &info.canonical),
                unresolved: None,
            };
        }

        let is_high_freq = info.freq >= Lexicon::CONFIDENT_FREQ;
        // A Wordnik hit is a valid dictionary word that merely lacks a
        // frequency entry. Treat it like a known word and preserve it
        // verbatim under the risk ceiling instead of flagging it as a
        // suspected out-of-vocabulary error (e.g. "unmounted", "reparented").
        let is_valid_dictionary_word = info.source == WordSource::Wordnik;
        if info.source == WordSource::Vocab
            || ((is_high_freq || is_valid_dictionary_word)
                && risk_score < CANONICAL_OVERRIDE_RISK_CEIL)
        {
            return ProcessedWord {
                corrected_word: word.to_owned(),
                unresolved: None,
            };
        }
    }

    // Proper nouns and acronyms pass through unless very high risk
    let looks_proper = stripped.starts_with(|c: char| c.is_ascii_uppercase())
        || (stripped.len() >= 2 && stripped.chars().all(|c| c.is_ascii_uppercase()));
    // Mixed-case identifier: any uppercase + any lowercase, length >= 3.
    // Looser than "≥2 uppercase" so leading-lowercase forms like "iPhone",
    // "eBay", "macOS" are recognized as proper-shaped tokens.
    let is_camel_case = stripped.len() >= 3
        && stripped.chars().any(|c| c.is_ascii_uppercase())
        && stripped.chars().any(|c| c.is_ascii_lowercase());

    // Proper-shaped tokens are preserved verbatim under the risk ceiling
    // regardless of dictionary membership: a known word that *also* looks
    // like a proper noun (e.g. brand names or sentence-initial capitals)
    // shouldn't be rewritten by the heuristic at moderate risk.
    if (looks_proper || is_camel_case) && risk_score < PROPER_NOUN_RISK_CEIL {
        return ProcessedWord {
            corrected_word: word.to_owned(),
            unresolved: None,
        };
    }

    // Flagged words become uncertain for the session gate/event.
    let unresolved = if flagged {
        Some(WordScore {
            word: stripped.to_owned(),
            position: i,
            candidate_score: 0.0,
        })
    } else {
        None
    };

    ProcessedWord {
        corrected_word: word.to_owned(),
        unresolved,
    }
}

/// Lowercased form of `replacement` when it is Title-case-only ("Dev",
/// "Server") and its lowercase form is an ordinary English word — i.e. when
/// applying it verbatim would force a capital that is wrong mid-sentence.
/// Brand-cased ("GitHub", "iPhone", acronyms) and genuinely proper
/// ("Kubernetes", "Roblox" — not dictionary words) replacements return
/// `None` and apply verbatim. Mirrors the lexicon path's
/// `title_case_only && lowercase_is_common` canonical guard.
pub fn soften_forced_title_case(replacement: &str) -> Option<String> {
    // Unicode-aware so accented capitals ("Élan") are detected as title case;
    // the wordnik list is ASCII-only, so accented words still fail the
    // commonness lookup below and apply verbatim — the safe outcome.
    let mut chars = replacement.chars();
    let title_only =
        chars.next().is_some_and(|c| c.is_uppercase()) && !chars.any(|c| c.is_uppercase());
    if !title_only {
        return None;
    }
    let lower = replacement.to_lowercase();
    wordnik_list::word_exists(&lower).then_some(lower)
}

/// Lexical gate: applies the user dictionary + brand canonical casing and flags
/// low-confidence words for the LLM. It does no spelling repair of its own.
pub struct LexicalGate;

impl Default for LexicalGate {
    fn default() -> Self {
        Self::new()
    }
}

impl LexicalGate {
    pub fn new() -> Self {
        Self
    }

    /// `word_risk_scores` entries are `(word, risk, mean_confidence)` —
    /// `risk` is the existing MIN-token-derived risk, `mean_confidence` is
    /// the MEAN token confidence across the same word (see
    /// `WordConfidence::mean_confidence`). Pass `1.0 - risk` for the third
    /// element when no separate mean is available (neutral: no corroboration).
    pub fn correct<S: AsRef<str>>(
        &self,
        text: &str,
        word_risk_scores: &[(S, f32, f32)],
        dict_replacements: &[crate::DictMatch],
        lang: &str,
    ) -> CorrectionResult {
        self.correct_with_risk_threshold(
            text,
            word_risk_scores,
            dict_replacements,
            lang,
            HWCE_FLAG_THRESHOLD,
            true,
        )
    }

    pub fn correct_with_risk_threshold<S: AsRef<str>>(
        &self,
        text: &str,
        word_risk_scores: &[(S, f32, f32)],
        dict_replacements: &[crate::DictMatch],
        lang: &str,
        flag_threshold: f32,
        brand_casing: bool,
    ) -> CorrectionResult {
        let words: Vec<&str> = text.split_whitespace().collect();
        let aligned = align_word_risk_positions(&words, word_risk_scores);
        let word_risk_positions = aligned.positions;
        let mean_confidence_lookup: HashMap<usize, f32> =
            aligned.mean_confidence.into_iter().collect();
        let dropped_risk_entries = aligned.dropped;

        if !has_dict(lang) {
            return CorrectionResult {
                corrected_text: text.to_owned(),
                word_risk_positions,
                unresolved_words: Vec::new(),
                all_resolved: true,
                dropped_risk_entries,
            };
        }

        let lexicon = Lexicon::for_lang(lang);
        // Index risk positions for O(1) lookup. The original linear scan was
        // O(words) inside the per-word loop → O(words²) overall.
        let risk_lookup: HashMap<usize, f32> = word_risk_positions.iter().copied().collect();
        // Index replacements for O(1) case-insensitive lookup, mirroring
        // risk_lookup. The per-word scan in process_single_word was
        // O(words × replacements); `.rev()` makes the first phrase win on a
        // case-insensitive collision, matching the old `iter().find()`.
        let mut dict_lookup: HashMap<String, (&str, bool, bool)> = dict_replacements
            .iter()
            .rev()
            .map(|m| {
                (
                    crate::fold_ci(&m.matched),
                    // (replacement, fuzzy, verbatim) — user entries are not
                    // verbatim: their casing still goes through soften.
                    (m.replacement.as_str(), m.fuzzy, false),
                )
            })
            .collect();
        // Built-in always-on single-word corrections (see ALWAYS_REPLACE),
        // applied as exact (never risk-gated) matches. A user-dictionary entry
        // for the same word is already present and wins via `or_insert`.
        for &(from, to) in ALWAYS_REPLACE {
            if !from.contains(' ') {
                // (to, fuzzy=false, verbatim=true): curated casing applies as-is.
                dict_lookup
                    .entry(from.to_owned())
                    .or_insert((to, false, true));
            }
        }
        // Multi-word EXACT phrase recognition (capitalization: "lune voice" ->
        // "Lune Voice") from the bundled vocab, then user-dictionary multi-word
        // entries (re-located here since `scan()`'s exact multi-word matches
        // were never reachable through the single-word `dict_lookup` above —
        // this also fixes that dormant gap), overwriting on a position
        // collision so the user dictionary wins. The bundled vocab contributes
        // recognition/casing only; all FUZZY correction now comes solely from
        // the user dictionary via `dict_replacements`.
        let mut phrase_matches: HashMap<usize, PhraseMatch> = lexicon
            .vocab_phrase_matches(&words)
            .into_iter()
            .map(|m| (m.start, m))
            .collect();
        // Built-in always-on multi-word corrections (see ALWAYS_REPLACE), exact
        // and never gated; user-dictionary entries below still overwrite them.
        let builtin_phrases: Vec<crate::DictMatch> = ALWAYS_REPLACE
            .iter()
            .filter(|(from, _)| from.contains(' '))
            .map(|&(from, to)| crate::DictMatch {
                matched: from.to_owned(),
                replacement: to.to_owned(),
                fuzzy: false,
            })
            .collect();
        for m in locate_multi_word_dict_matches(&words, &builtin_phrases) {
            phrase_matches.insert(m.start, m);
        }
        for m in locate_multi_word_dict_matches(&words, dict_replacements) {
            phrase_matches.insert(m.start, m);
        }
        // Matches at *different* start indices can still overlap ("new york" at
        // 3 and "york times" at 4). The consumption loop below jumps past a
        // consumed window, so the later entry would be silently skipped.
        // Resolve it explicitly instead: earliest start wins, longest match
        // breaks a tie.
        prune_overlapping_phrase_matches(&mut phrase_matches);
        // Estimate output capacity: input length plus a small slack for
        // expansions (e.g. "im" → "I'm"). Slightly oversize beats reallocs.
        let approx_cap = text.len() + words.len() * 2 + 8;
        let mut corrected_text = String::with_capacity(approx_cap);
        let mut unresolved = Vec::new();

        let mut i = 0;
        while i < words.len() {
            if i > 0 {
                corrected_text.push(' ');
            }

            if let Some(pm) = phrase_matches.get(&i) {
                // Gate fuzzy phrase matches on the max risk across the
                // consumed window — the same `FUZZY_DICT_RISK_GATE` the
                // single-word fuzzy path uses, no new threshold invented.
                // Exact phrase matches (`fuzzy == false`, vocab re-casing or
                // a user-dictionary multi-word entry) are never gated, same
                // as the single-word exact/multi-word precedent.
                let window_risk = (i..i + pm.len)
                    .map(|p| risk_lookup.get(&p).copied().unwrap_or(0.0))
                    .fold(0.0_f32, f32::max);
                if !(pm.fuzzy && window_risk >= FUZZY_DICT_RISK_GATE) {
                    let softened = soften_forced_title_case(&pm.replacement);
                    let replacement = softened.as_deref().unwrap_or(&pm.replacement);
                    let first = words[i];
                    let last = words[i + pm.len - 1];
                    let prefix =
                        &first[..first.len() - first.trim_start_matches(is_edge_punct).len()];
                    let suffix = &last[last.trim_end_matches(is_edge_punct).len()..];
                    corrected_text.push_str(prefix);
                    corrected_text.push_str(replacement);
                    corrected_text.push_str(suffix);
                    i += pm.len;
                    continue;
                }
                // Gate blocked the fuzzy phrase — fall through and handle
                // just this one position with the normal per-word logic
                // below (don't consume the rest of the window).
            }

            let word = words[i];
            let risk_score = risk_lookup.get(&i).copied().unwrap_or(0.0);
            let mean_confidence = mean_confidence_lookup.get(&i).copied().unwrap_or(0.0);
            let (leading, spans) = split_internal_punctuation(word);

            if spans.len() <= 1 {
                // Common case: at most one alnum span — identical to a
                // single `process_single_word` call on the whole token.
                let stripped = strip_punctuation(word);
                let lexicon_info = lexicon.lookup(stripped);
                let processed = process_single_word(
                    word,
                    i,
                    stripped,
                    &lexicon_info,
                    risk_score,
                    flag_threshold,
                    &dict_lookup,
                    brand_casing,
                );
                corrected_text.push_str(&processed.corrected_word);
                if let Some(word_score) = processed.unresolved {
                    unresolved.push(word_score);
                }
                i += 1;
                continue;
            }

            // Multiple alnum spans glued by internal punctuation (e.g. an
            // ASR ellipsis with no surrounding whitespace: "is...Tayton.").
            // Correct each span independently and reassemble with the
            // original glue intact.
            //
            // Every span shares the WHOLE token's MIN-derived risk score as
            // its baseline — whisper's own word-confidence grouping reported
            // one value for this glued unit, so that is the primary honest
            // signal. It is corroborated (never independently overridden) by
            // the group's MEAN confidence — the mean-vs-min two-level check
            // (arXiv:2407.21414), applied at word-group
            // granularity. Taking the min of (risk, mean-derived risk) means
            // the mean can only LOWER the effective risk, and only down to
            // what the mean itself would justify — it confirms a borderline
            // min-risk is plausibly caused by one outlier token (e.g. the
            // glued punctuation itself), it never invents trust beyond the
            // mean's own evidence. Scoped to multi-span tokens only — that is
            // the one demonstrated failure mode; ordinary single-span words
            // keep today's exact min-only behavior, unchanged below.
            let mean_risk = 1.0 - mean_confidence;
            let effective_risk = risk_score.min(mean_risk);
            corrected_text.push_str(leading);
            for (span, glue) in &spans {
                let lexicon_info = lexicon.lookup(span);
                let processed = process_single_word(
                    span,
                    i,
                    span,
                    &lexicon_info,
                    effective_risk,
                    flag_threshold,
                    &dict_lookup,
                    brand_casing,
                );
                corrected_text.push_str(&processed.corrected_word);
                corrected_text.push_str(glue);
                if let Some(word_score) = processed.unresolved {
                    unresolved.push(word_score);
                }
            }
            i += 1;
        }

        let all_resolved = unresolved.is_empty();
        CorrectionResult {
            corrected_text,
            word_risk_positions,
            unresolved_words: unresolved,
            all_resolved,
            dropped_risk_entries,
        }
    }
}

/// Splits `word` into its alphanumeric spans and the punctuation glue
/// around/between them, so each span can be looked up/corrected
/// independently while the glue is reproduced byte-for-byte on reassembly.
/// A token with no internal split point (the common case) yields exactly
/// one span, equivalent to plain `strip_punctuation`.
///
/// Needed because ASR output sometimes glues two real words together with no
/// surrounding whitespace (e.g. a trailing-thought ellipsis: `"is...Tayton."`
/// is one `split_whitespace` token) — `strip_punctuation` only trims a
/// token's edges, so an internal punctuation run like that hides a known
/// word from every lookup.
///
/// A single apostrophe or hyphen strictly between two alphanumeric
/// characters is treated as part of the span, NOT a split point —
/// contractions ("don't"), possessives ("Sarah's"), and hyphenated
/// compounds ("well-known") must not be torn apart. Any other punctuation
/// (including a lone period/comma, or any multi-character run) splits.
pub(crate) fn split_internal_punctuation(word: &str) -> (&str, Vec<(&str, &str)>) {
    let chars: Vec<(usize, char)> = word.char_indices().collect();
    let n = chars.len();
    let byte_len = word.len();

    let is_glue_char = |idx: usize, c: char| -> bool {
        if c.is_alphanumeric() {
            return true;
        }
        // U+2019 alongside the ASCII apostrophe: whisper and styling LLMs
        // routinely emit the typographic form, and without it "don’t" split
        // into "don"/"t" and each half was looked up as its own word.
        (c == '\'' || c == '\u{2019}' || c == '-')
            && idx > 0
            && idx + 1 < n
            && chars[idx - 1].1.is_alphanumeric()
            && chars[idx + 1].1.is_alphanumeric()
    };
    let byte_at = |chars: &[(usize, char)], idx: usize| -> usize {
        chars.get(idx).map_or(byte_len, |&(b, _)| b)
    };

    let mut i = 0;
    while i < n && !is_glue_char(i, chars[i].1) {
        i += 1;
    }
    let leading = &word[..byte_at(&chars, i)];

    let mut spans = Vec::new();
    while i < n {
        let span_start = chars[i].0;
        while i < n && is_glue_char(i, chars[i].1) {
            i += 1;
        }
        let span_end = byte_at(&chars, i);
        let glue_start = span_end;
        while i < n && !is_glue_char(i, chars[i].1) {
            i += 1;
        }
        let glue_end = byte_at(&chars, i);
        spans.push((&word[span_start..span_end], &word[glue_start..glue_end]));
    }
    (leading, spans)
}

/// Trim edge punctuation so the bare word can be used as a dictionary/vocab
/// lookup key. Deliberately Unicode-wide, not `is_ascii_punctuation`: trimming
/// stopped at the first non-ASCII char, so a typographic-quoted `“github”`
/// never matched a `github` entry and passed through uncorrected.
pub(crate) fn strip_punctuation(word: &str) -> &str {
    let s = word.trim_start_matches(is_edge_punct);
    s.trim_end_matches(is_edge_punct)
}

/// Non-ASCII characters that wrap or terminate a word in transcript text —
/// typographic quotes, guillemets, dashes and ellipsis. Whisper emits these
/// routinely for quoted or read-aloud speech.
const UNICODE_EDGE_PUNCT: &[char] = &[
    '\u{2018}', '\u{2019}', // ‘ ’
    '\u{201c}', '\u{201d}', // “ ”
    '\u{201a}', '\u{201e}', // ‚ „
    '\u{ab}', '\u{bb}', // « »
    '\u{2039}', '\u{203a}', // ‹ ›
    '\u{2013}', '\u{2014}', '\u{2015}', // – — ―
    '\u{2026}', // …
    '\u{00bf}', '\u{00a1}', // ¿ ¡
];

/// Edge punctuation for lookup-key purposes. Shared by `strip_punctuation` and
/// every prefix/suffix reconstruction so what gets trimmed is always what gets
/// put back — an ASCII-only predicate left curly quotes attached to the key,
/// and `github` never matched `“github”`.
///
/// Deliberately an explicit list rather than "not alphanumeric": `char`'s
/// alphanumeric test excludes combining marks, so the general predicate would
/// trim the vowel signs off Devanagari, Thai and Arabic words and truncate
/// their lookup keys.
fn is_edge_punct(c: char) -> bool {
    c.is_ascii_punctuation() || UNICODE_EDGE_PUNCT.contains(&c)
}

/// Aligned positions plus a count of risk entries that could not be matched
/// to any remaining text token. `mean_confidence` is a second, independent
/// alignment of the same positions (kept separate from `positions` so the
/// public `CorrectionResult::word_risk_positions` field — risk only — stays
/// unchanged for existing external readers).
struct AlignedRisks {
    positions: Vec<(usize, f32)>,
    mean_confidence: Vec<(usize, f32)>,
    dropped: usize,
}

/// Re-locates each multi-word `DictMatch` (from `DictionaryManager::scan`'s
/// exact multi-word matching) to its start position in `words`, so the
/// corrector's loop can treat it exactly like a vocab phrase match. Needed
/// because `scan()` validates and returns the matched phrase as plain text —
/// without this, multi-word user-dictionary entries were structurally
/// unreachable: `process_single_word`'s `dict_lookup` is keyed by a single
/// stripped word, so a `matched` value containing whitespace could never be
/// found there. Mirrors `scan()`'s own window-matching rule (case-insensitive,
/// per-word, edge punctuation stripped) so the re-derived position is
/// guaranteed consistent with how the match was originally found upstream.
fn locate_multi_word_dict_matches(
    words: &[&str],
    dict_replacements: &[crate::DictMatch],
) -> Vec<PhraseMatch> {
    let mut out = Vec::new();
    for m in dict_replacements {
        let phrase_words: Vec<&str> = m.matched.split_whitespace().collect();
        let n = phrase_words.len();
        if n < 2 || words.len() < n {
            continue;
        }
        // Every non-overlapping occurrence, not just the first: a phrase
        // repeated in one utterance ("loon voice … loon voice") must be
        // corrected at each position, matching `vocab_phrase_matches`.
        let mut start = 0;
        while start + n <= words.len() {
            let hit = words[start..start + n]
                .iter()
                .zip(&phrase_words)
                .all(|(w, pw)| crate::eq_ci(strip_punctuation(w), pw));
            if hit {
                out.push(PhraseMatch {
                    start,
                    len: n,
                    replacement: m.replacement.clone(),
                    fuzzy: m.fuzzy,
                });
                start += n;
            } else {
                start += 1;
            }
        }
    }
    out
}

/// Drop any phrase match whose word window overlaps an earlier-starting
/// surviving match, so overlap resolution is an explicit rule rather than a
/// side effect of which start index the consumption loop happens to land on.
/// Earliest start wins — the map is keyed by start, so starts are unique and
/// no tie-break is possible or needed. This mirrors the consumption loop,
/// which scans left to right and jumps past each window it applies.
fn prune_overlapping_phrase_matches(phrase_matches: &mut HashMap<usize, PhraseMatch>) {
    if phrase_matches.len() < 2 {
        return;
    }
    let mut starts: Vec<usize> = phrase_matches.keys().copied().collect();
    starts.sort_unstable();

    let mut consumed_until = 0usize;
    for start in starts {
        let len = phrase_matches[&start].len;
        if start < consumed_until {
            phrase_matches.remove(&start);
        } else {
            consumed_until = start + len;
        }
    }
}

fn align_word_risk_positions<S: AsRef<str>>(
    words: &[&str],
    word_risk_scores: &[(S, f32, f32)],
) -> AlignedRisks {
    let mut out = Vec::new();
    let mut means = Vec::new();
    let mut dropped = 0usize;
    let mut cursor = 0usize;

    for (risk_word, risk, mean_confidence) in word_risk_scores {
        let risk_word = strip_punctuation(risk_word.as_ref());
        if risk_word.is_empty() {
            continue;
        }

        if let Some(offset) = words[cursor..]
            .iter()
            .position(|word| strip_punctuation(word).eq_ignore_ascii_case(risk_word))
        {
            let idx = cursor + offset;
            out.push((idx, *risk));
            means.push((idx, *mean_confidence));
            cursor = (idx + 1).min(words.len());
        } else {
            dropped += 1;
        }
    }

    AlignedRisks {
        positions: out,
        mean_confidence: means,
        dropped,
    }
}

fn restore_punctuation(original: &str, corrected: &str) -> String {
    let prefix = &original[..original.len() - original.trim_start_matches(is_edge_punct).len()];
    let suffix_start = original.trim_end_matches(is_edge_punct).len();
    let suffix = &original[suffix_start..];
    format!("{prefix}{corrected}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Typed empty risk slice for calls that exercise correction without any
    /// ASR risk scores (lets `correct`'s generic `S` parameter infer).
    const NO_RISKS: &[(&str, f32, f32)] = &[];

    #[test]
    fn soften_targets_title_cased_common_words_only() {
        assert_eq!(soften_forced_title_case("Dev"), Some("dev".into()));
        assert_eq!(soften_forced_title_case("Server"), Some("server".into()));
        // Proper nouns that are not dictionary words keep their capital.
        assert_eq!(soften_forced_title_case("Kubernetes"), None);
        assert_eq!(soften_forced_title_case("Roblox"), None);
        // Brand casing (interior uppercase) is never softened.
        assert_eq!(soften_forced_title_case("GitHub"), None);
        assert_eq!(soften_forced_title_case("NASA"), None);
        assert_eq!(soften_forced_title_case("iPhone"), None);
        // "Lune Voice" passes through via its interior uppercase 'V';
        // lowercase replacements carry no forced capital to begin with.
        assert_eq!(soften_forced_title_case("Lune Voice"), None);
        assert_eq!(soften_forced_title_case("dev"), None);
        // Accented capitals are detected as title case, but the ASCII-only
        // word list never recognizes their lowercase — verbatim is the safe
        // outcome for likely proper nouns.
        assert_eq!(soften_forced_title_case("Élan"), None);
    }

    #[test]
    fn dict_replacement_does_not_force_mid_sentence_capital() {
        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "dev".to_string(),
            replacement: "Dev".to_string(),
            fuzzy: false,
        }];
        let result = c.correct("restart the dev box", NO_RISKS, &dict, "en");
        assert_eq!(result.corrected_text, "restart the dev box");
    }

    #[test]
    fn dict_replacement_keeps_brand_casing() {
        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "github".to_string(),
            replacement: "GitHub".to_string(),
            fuzzy: false,
        }];
        let result = c.correct("i use github daily", NO_RISKS, &dict, "en");
        assert_eq!(result.corrected_text, "i use GitHub daily");
    }

    #[test]
    fn known_word_not_flagged_passes_through() {
        let c = LexicalGate::new();
        let result = c.correct("the quick brown fox", NO_RISKS, &[], "en");
        assert_eq!(result.corrected_text, "the quick brown fox");
    }

    #[test]
    fn pruned_generic_phrases_are_not_force_capitalized() {
        // Generic Title-Case ML jargon ("Context Window", "System Prompt",
        // "Vector Search", "Greedy Decoding", ...) was removed from the bundled
        // vocab: exact multi-word phrase re-casing is applied UNCONDITIONALLY
        // (only fuzzy matches are risk-gated), so those common noun phrases got
        // force-capitalized in ordinary speech. They must pass through exactly
        // as spoken now — re-adding any to data/vocab/ should fail here.
        let c = LexicalGate::new();
        for text in [
            "the context window is too small",
            "check the system prompt first",
            "we ran a vector search over the notes",
            "greedy decoding was noticeably faster",
        ] {
            let result = c.correct(text, NO_RISKS, &[], "en");
            assert_eq!(result.corrected_text, text, "must not re-case: {text}");
        }
    }

    #[test]
    fn internal_punctuation_splits_and_corrects_each_span() {
        // Whisper sometimes glues a trailing-thought ellipsis directly onto
        // the next word with no whitespace: "is...Clayten." is ONE
        // split_whitespace token. The internal-punctuation split must expose
        // the "Clayten" span so a user-dictionary fuzzy entry can correct it,
        // ellipsis and trailing period preserved byte-for-byte. Low risk -> the
        // fuzzy entry applies. (Uses a non-builtin word so the split/gate is
        // tested independently of ALWAYS_REPLACE.)
        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "clayten".into(),
            replacement: "Clayton".into(),
            fuzzy: true,
        }];
        let scores = vec![("is...Clayten.", 0.10, 0.90)];
        let result = c.correct("hello my name is...Clayten.", &scores, &dict, "en");
        assert_eq!(result.corrected_text, "hello my name is...Clayton.");
    }

    #[test]
    fn internal_punctuation_high_risk_span_not_corrected() {
        // Same glued token, but at high risk — the fuzzy gate must still block
        // the user-dictionary fuzzy entry, exactly as for an ungated word.
        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "clayten".into(),
            replacement: "Clayton".into(),
            fuzzy: true,
        }];
        let scores = vec![("is...Clayten.", 0.90, 0.10)];
        let result = c.correct("hello my name is...Clayten.", &scores, &dict, "en");
        assert_eq!(result.corrected_text, "hello my name is...Clayten.");
    }

    #[test]
    fn internal_punctuation_corroborated_by_mean_confidence_is_corrected() {
        // Same glued token, MIN-derived risk is high (0.90, would block on
        // its own — see the test above) but the group's MEAN confidence is
        // also high (0.90 -> mean-derived risk 0.10). That corroborates that
        // one outlier token (plausibly the ellipsis itself) dragged the MIN
        // down while the rest of the group is fine, so the corroborated
        // (lower) risk clears the fuzzy gate and the correction applies.
        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "clayten".into(),
            replacement: "Clayton".into(),
            fuzzy: true,
        }];
        let scores = vec![("is...Clayten.", 0.90, 0.90)];
        let result = c.correct("hello my name is...Clayten.", &scores, &dict, "en");
        assert_eq!(result.corrected_text, "hello my name is...Clayton.");
    }

    #[test]
    fn always_replace_applies_unconditionally_and_yields_to_user_dict() {
        let c = LexicalGate::new();
        // Multi-word built-in mishear -> canonical, with no risk scores at all.
        let phrase = c.correct("Launch Loon Voice now", NO_RISKS, &[], "en");
        assert!(
            phrase.corrected_text.contains("Lune Voice"),
            "built-in phrase should apply: {}",
            phrase.corrected_text
        );
        // Standalone "loon" is a real English word (a bird) and is NOT rewritten
        // — only the unambiguous "loon voice" phrase maps to the product.
        let bird = c.correct("i saw a loon", NO_RISKS, &[], "en");
        assert_eq!(bird.corrected_text, "i saw a loon");
        // Correctly-spoken form is re-cased to canonical too, verbatim (not
        // lowercased by soften even though "lune" is a dictionary word).
        let recase = c.correct("i love lune", NO_RISKS, &[], "en");
        assert_eq!(recase.corrected_text, "i love Lune");
        // Single-word built-in applies even at HIGH risk (exact, never gated).
        let scores = vec![("Tayton", 0.95, 0.05)];
        let single = c.correct("hello Tayton", &scores, &[], "en");
        assert_eq!(single.corrected_text, "hello Dayton");
        // A user-dictionary entry for the same word overrides the built-in.
        let dict = vec![crate::DictMatch {
            matched: "tayton".into(),
            replacement: "Clayton".into(),
            fuzzy: false,
        }];
        let overridden = c.correct("hello Tayton", NO_RISKS, &dict, "en");
        assert_eq!(overridden.corrected_text, "hello Clayton");
    }

    #[test]
    fn multi_word_entries_apply_to_every_occurrence() {
        let c = LexicalGate::new();
        // Built-in multi-word phrase repeated in one utterance.
        let builtin = c.correct("loon voice hears loon voice", NO_RISKS, &[], "en");
        assert_eq!(builtin.corrected_text, "Lune Voice hears Lune Voice");
        // Same for a user-dictionary multi-word entry.
        let dict = vec![crate::DictMatch {
            matched: "new york".into(),
            replacement: "New York".into(),
            fuzzy: false,
        }];
        let user = c.correct("new york to new york", NO_RISKS, &dict, "en");
        assert_eq!(user.corrected_text, "New York to New York");
    }

    #[test]
    fn contractions_are_not_split_on_internal_apostrophe() {
        let c = LexicalGate::new();
        let result = c.correct("i don't know what she's doing", NO_RISKS, &[], "en");
        assert_eq!(result.corrected_text, "i don't know what she's doing");
    }

    #[test]
    fn hyphenated_compounds_are_not_split() {
        let c = LexicalGate::new();
        let result = c.correct("a well-known co-worker", NO_RISKS, &[], "en");
        assert_eq!(result.corrected_text, "a well-known co-worker");
    }

    #[test]
    fn word_risk_positions_survive_heuristic() {
        let c = LexicalGate::new();
        let scores = vec![("teh", 0.80, 0.20)];
        let result = c.correct("teh quick", &scores, &[], "en");
        assert_eq!(result.word_risk_positions, vec![(0, 0.80)]);
    }

    #[test]
    fn word_risk_positions_handle_duplicate_words_by_order() {
        let c = LexicalGate::new();
        let scores = vec![("the", 0.10, 0.90), ("the", 0.85, 0.15)];
        let result = c.correct("the build fixed the deployment", &scores, &[], "en");
        assert_eq!(result.word_risk_positions, vec![(0, 0.10), (3, 0.85)]);
    }

    #[test]
    fn proper_noun_skipped_when_not_flagged() {
        let c = LexicalGate::new();
        let result = c.correct("London is great", NO_RISKS, &[], "en");
        assert!(result.corrected_text.contains("London"));
    }

    #[test]
    fn acronym_skipped_when_not_flagged() {
        let c = LexicalGate::new();
        let result = c.correct("the NASA program", NO_RISKS, &[], "en");
        assert!(result.corrected_text.contains("NASA"));
    }

    /// U+2019 is what whisper and styling LLMs usually emit; without it in the
    /// glue set, "don’t" split into "don"/"t" and each half was looked up
    /// independently.
    #[test]
    fn typographic_apostrophe_does_not_split_contractions() {
        let c = LexicalGate::new();
        let result = c.correct(
            "i don\u{2019}t know what she\u{2019}s doing",
            NO_RISKS,
            &[],
            "en",
        );
        assert_eq!(
            result.corrected_text,
            "i don\u{2019}t know what she\u{2019}s doing"
        );
    }

    /// Trimming stopped at the first non-ASCII char, so a curly-quoted token
    /// never matched its dictionary/vocab entry.
    #[test]
    fn typographic_quotes_are_stripped_before_lookup() {
        assert_eq!(strip_punctuation("\u{201c}github\u{201d}"), "github");
        assert_eq!(strip_punctuation("\u{ab}lune\u{bb}"), "lune");
        // Unchanged for the ASCII cases it already handled.
        assert_eq!(strip_punctuation("\"github\"."), "github");
        // Combining marks are part of the word, not edge punctuation.
        assert_eq!(strip_punctuation("\u{915}\u{93e}"), "\u{915}\u{93e}");
        assert_eq!(strip_punctuation("caf\u{e9}."), "caf\u{e9}");

        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "github".into(),
            replacement: "GitHub".into(),
            fuzzy: false,
        }];
        let result = c.correct("visit \u{201c}github\u{201d} now", NO_RISKS, &dict, "en");
        assert_eq!(result.corrected_text, "visit \u{201c}GitHub\u{201d} now");
    }

    /// Two dictionary phrases can match at different, overlapping start
    /// indices. The consumption loop jumps past a consumed window, so overlap
    /// has to be resolved up front rather than left to which index it lands on.
    #[test]
    fn overlapping_phrase_matches_resolve_to_the_earliest() {
        let c = LexicalGate::new();
        let dict = vec![
            crate::DictMatch {
                matched: "new york".into(),
                replacement: "New York".into(),
                fuzzy: false,
            },
            crate::DictMatch {
                matched: "york times".into(),
                replacement: "NYT".into(),
                fuzzy: false,
            },
        ];
        let result = c.correct("i read the new york times today", NO_RISKS, &dict, "en");
        assert_eq!(result.corrected_text, "i read the New York times today");
    }

    #[test]
    fn dict_replacement_takes_priority() {
        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "helo".into(),
            replacement: "hello".into(),
            fuzzy: false,
        }];
        let result = c.correct("helo world", NO_RISKS, &dict, "en");
        assert_eq!(result.corrected_text, "hello world");
    }

    #[test]
    fn fuzzy_dict_match_skipped_on_low_confidence_word() {
        let c = LexicalGate::new();
        // A garbled, low-confidence onset (risk 0.8 ≥ gate) coincidentally
        // fuzzy-matches a short dict term — it must NOT be rewritten.
        let risks = vec![("intate", 0.8, 0.0)];
        let dict = vec![crate::DictMatch {
            matched: "intate".into(),
            replacement: "INT8".into(),
            fuzzy: true,
        }];
        let result = c.correct("intate the buffer", &risks, &dict, "en");
        assert!(
            !result.corrected_text.contains("INT8"),
            "fuzzy match on a low-confidence word must be skipped: {}",
            result.corrected_text
        );
    }

    #[test]
    fn fuzzy_dict_match_applied_on_confident_word() {
        let c = LexicalGate::new();
        // A confident mishear (risk 0.1 < gate) fuzzy-matches the dict term and
        // IS corrected — the dictionary's core purpose is preserved.
        let risks = vec![("kubernets", 0.1, 0.0)];
        let dict = vec![crate::DictMatch {
            matched: "kubernets".into(),
            replacement: "Kubernetes".into(),
            fuzzy: true,
        }];
        let result = c.correct("deploy to kubernets now", &risks, &dict, "en");
        assert!(
            result.corrected_text.to_lowercase().contains("kubernetes"),
            "fuzzy match on a confident word must apply: {}",
            result.corrected_text
        );
    }

    #[test]
    fn exact_dict_match_applies_even_on_low_confidence() {
        let c = LexicalGate::new();
        // Exact (and multi-word) matches are never gated, regardless of risk.
        let risks = vec![("helo", 0.9, 0.0)];
        let dict = vec![crate::DictMatch {
            matched: "helo".into(),
            replacement: "hello".into(),
            fuzzy: false,
        }];
        let result = c.correct("helo world", &risks, &dict, "en");
        assert!(
            result.corrected_text.contains("hello"),
            "exact match must apply regardless of confidence: {}",
            result.corrected_text
        );
    }

    #[test]
    fn unsupported_lang_bypasses_corrector() {
        let c = LexicalGate::new();
        let result = c.correct("jambo dunia", NO_RISKS, &[], "swahili");
        assert_eq!(result.corrected_text, "jambo dunia");
    }

    #[test]
    fn french_known_words_pass_through() {
        let c = LexicalGate::new();
        let result = c.correct("bonjour le monde", NO_RISKS, &[], "fr");
        assert_eq!(result.corrected_text, "bonjour le monde");
    }

    #[test]
    fn unresolved_words_reported() {
        let c = LexicalGate::new();
        let risks = vec![("xyzqqq".to_string(), 0.9, 0.0)];
        let result = c.correct("the xyzqqq fox", &risks, &[], "en");
        assert!(!result.unresolved_words.is_empty());
        assert!(result.unresolved_words.iter().any(|ws| ws.word == "xyzqqq"));
    }

    #[test]
    fn empty_input() {
        let c = LexicalGate::new();
        let result = c.correct("", NO_RISKS, &[], "en");
        assert_eq!(result.corrected_text, "");
    }

    #[test]
    fn single_char_word_not_corrected() {
        let c = LexicalGate::new();
        let result = c.correct("I a", NO_RISKS, &[], "en");
        assert_eq!(result.corrected_text, "I a");
    }

    #[test]
    fn flagged_curated_vocab_passes_through_without_correction() {
        let c = LexicalGate::new();
        let risks = vec![("WebSocket".to_string(), 0.95, 0.0)];
        let result = c.correct("WebSocket failed", &risks, &[], "en");
        assert_eq!(result.corrected_text, "WebSocket failed");
    }

    #[test]
    fn dev_vocabulary_word_passes_through_at_high_risk() {
        let c = LexicalGate::new();
        // "hotkey" is curated dev vocabulary (WordSource::Vocab): trusted and
        // preserved verbatim even above the risk ceiling, never sent to the LLM.
        let risks = vec![("hotkey", 0.95, 0.0)];
        let result = c.correct("press the hotkey", &risks, &[], "en");
        assert!(
            result.unresolved_words.is_empty(),
            "curated dev term should not be flagged: {:?}",
            result.unresolved_words
        );
        assert_eq!(result.corrected_text, "press the hotkey");
    }

    #[test]
    fn wordnik_word_preserved_under_risk_ceiling() {
        let c = LexicalGate::new();
        // "unmounted" is valid English (Wordnik tier, freq 0). At moderate
        // risk it must pass through verbatim, not be flagged for the LLM.
        let risks = vec![("unmounted", 0.60, 0.0)];
        let result = c.correct("the unmounted drive", &risks, &[], "en");
        assert!(
            result.unresolved_words.is_empty(),
            "valid dictionary word should not be flagged: {:?}",
            result.unresolved_words
        );
        assert_eq!(result.corrected_text, "the unmounted drive");
    }

    #[test]
    fn wordnik_word_flagged_above_ceiling() {
        let c = LexicalGate::new();
        // Above CANONICAL_OVERRIDE_RISK_CEIL even a valid word is surfaced.
        let risks = vec![("unmounted", 0.90, 0.0)];
        let result = c.correct("the unmounted drive", &risks, &[], "en");
        assert!(
            result
                .unresolved_words
                .iter()
                .any(|ws| ws.word == "unmounted"),
            "very-high-risk word should still reach the LLM"
        );
    }

    #[test]
    fn sub_gate_risk_does_not_fire_llm() {
        // A risk just under the gate must neither flag the word nor trigger
        // a needs_llm decision (the session uses HWCE_FLAG_THRESHOLD).
        let c = LexicalGate::new();
        let just_under = HWCE_FLAG_THRESHOLD - 0.05;
        let risks = vec![("the".to_string(), just_under, 0.0)];
        let result = c.correct("the quick brown fox", &risks, &[], "en");
        assert!(result.unresolved_words.is_empty());
        assert!(!risks.iter().any(|(_, r, _)| *r >= HWCE_FLAG_THRESHOLD));
    }

    #[test]
    fn custom_risk_threshold_controls_unresolved_words() {
        let c = LexicalGate::new();
        let risks = vec![("xyzqqq".to_string(), 0.35, 0.0)];
        let default_result = c.correct("the xyzqqq fox", &risks, &[], "en");
        assert!(default_result.unresolved_words.is_empty());

        let custom_result =
            c.correct_with_risk_threshold("the xyzqqq fox", &risks, &[], "en", 0.30, true);
        assert!(
            custom_result
                .unresolved_words
                .iter()
                .any(|ws| ws.word == "xyzqqq")
        );
    }

    #[test]
    fn camel_case_leading_lowercase_preserved() {
        // "iPhone" has only one uppercase char — under the old (≥2 upper)
        // predicate it would not have been recognized as a proper-shaped
        // token and could be marked unresolved.
        let c = LexicalGate::new();
        let risks = vec![("iPhone".to_string(), 0.7, 0.0)];
        let result = c.correct("the iPhone is new", &risks, &[], "en");
        assert!(result.corrected_text.contains("iPhone"));
        assert!(
            !result.unresolved_words.iter().any(|w| w.word == "iPhone"),
            "iPhone should not be flagged as unresolved at moderate risk"
        );
    }

    #[test]
    fn dropped_risks_counted() {
        // Risk word that doesn't appear in the text should be reported.
        let c = LexicalGate::new();
        let risks = vec![
            ("the".to_string(), 0.1, 0.0),
            ("nonexistent".to_string(), 0.5, 0.0),
        ];
        let result = c.correct("the quick fox", &risks, &[], "en");
        assert_eq!(result.dropped_risk_entries, 1);
    }

    #[test]
    fn no_dropped_risks_on_full_match() {
        let c = LexicalGate::new();
        let risks = vec![("the".to_string(), 0.1, 0.0), ("fox".to_string(), 0.2, 0.0)];
        let result = c.correct("the quick fox", &risks, &[], "en");
        assert_eq!(result.dropped_risk_entries, 0);
    }

    #[test]
    fn flagged_sentence_initial_capital_preserved() {
        let c = LexicalGate::new();
        // High risk on "All" used to force-lowercase it via the canonical
        // override because freq-dict canonicals fall back to lowercase.
        let risks = vec![("All".to_string(), 0.85, 0.0)];
        let result = c.correct("All I'm doing is trash work.", &risks, &[], "en");
        assert!(
            result.corrected_text.starts_with("All "),
            "expected leading 'All' preserved, got {:?}",
            result.corrected_text
        );
    }

    #[test]
    fn title_case_vocab_entry_does_not_capitalize_common_word() {
        let c = LexicalGate::new();
        // "Calibration" appears title-cased in harvested vocab data; flagged
        // mid-sentence occurrences must not inherit that casing.
        let risks = vec![("calibration".to_string(), 0.55, 0.0)];
        let result = c.correct("the calibration step failed", &risks, &[], "en");
        assert_eq!(result.corrected_text, "the calibration step failed");
    }

    #[test]
    fn high_caprate_proper_noun_keeps_canonical_capital() {
        let c = LexicalGate::new();
        // "Amazon" is a title-cased vocab entry with a high capitalization rate
        // (~0.85). This exercises the `Some(p)` p_cap branch (p >= keep ->
        // not common), so the canonical capital is applied where the old
        // dictionary-membership fallback would have dropped it.
        let risks = vec![("amazon".to_string(), 0.5, 0.0)];
        let result = c.correct("i deployed it on amazon today", &risks, &[], "en");
        assert_eq!(result.corrected_text, "i deployed it on Amazon today");
    }

    #[test]
    fn vocab_word_gets_canonical_casing() {
        let c = LexicalGate::new();
        let risks = vec![("github".to_string(), 0.5, 0.0)];
        let result = c.correct("push to github", &risks, &[], "en");
        assert_eq!(result.corrected_text, "push to GitHub");
    }

    #[test]
    fn brand_casing_toggle_controls_canonical_casing() {
        let c = LexicalGate::new();
        let risks = vec![("github".to_string(), 0.5, 0.0)];
        // Off: the input casing is preserved.
        assert_eq!(
            c.correct_with_risk_threshold(
                "push to github",
                &risks,
                &[],
                "en",
                HWCE_FLAG_THRESHOLD,
                false
            )
            .corrected_text,
            "push to github"
        );
        // On: canonical brand casing is applied.
        assert_eq!(
            c.correct_with_risk_threshold(
                "push to github",
                &risks,
                &[],
                "en",
                HWCE_FLAG_THRESHOLD,
                true
            )
            .corrected_text,
            "push to GitHub"
        );
    }

    #[test]
    fn has_dict_checks() {
        assert!(has_dict("en"));
        assert!(has_dict("eng"));
        assert!(has_dict("English"));
        assert!(has_dict(""));
        assert!(has_dict("fr"));
        assert!(has_dict("de"));
        assert!(has_dict("Japanese"));
        assert!(has_dict("nl"));
        assert!(has_dict("Dutch"));
        assert!(!has_dict("swahili"));
    }

    #[test]
    fn restore_punctuation_edges() {
        assert_eq!(restore_punctuation("\"hello\"", "hi"), "\"hi\"");
        assert_eq!(restore_punctuation("hello.", "hi"), "hi.");
        assert_eq!(restore_punctuation("hello", "hi"), "hi");
    }

    #[test]
    fn punctuation_preserved_with_dict() {
        let c = LexicalGate::new();
        let dict = vec![crate::DictMatch {
            matched: "teh".into(),
            replacement: "the".into(),
            fuzzy: false,
        }];
        let result = c.correct("teh, quick", NO_RISKS, &dict, "en");
        assert!(result.corrected_text.contains("the,"));
    }

    #[test]
    fn qd_002_no_false_positives() {
        let c = LexicalGate::new();
        let text = "The memory leak was traced back to the event listener on the WebSocket connection. It wasn't being cleaned up when the component unmounted. I pushed a fix to the dev branch and it looks stable now.";
        let result = c.correct(text, NO_RISKS, &[], "en");
        assert!(
            result.unresolved_words.is_empty(),
            "unexpected unresolved: {:?}",
            result.unresolved_words
        );
    }
}
