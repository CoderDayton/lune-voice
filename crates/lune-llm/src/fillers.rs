//! Regex-based filler word detection and removal for ASR transcripts.
//!
//! Handles elongated forms ("Uhmmmm", "euhhhhh", "mmmm") and both bare
//! and `[?..?]`-wrapped fillers. Language-specific patterns for EN, ES, FR.

use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// ─── Filler Patterns Per Language ────────────────────────────────────────────

struct LangFillers {
    /// Matches `[?filler?]`-wrapped fillers inline in text (self-delimited, no false positives).
    marked_only: Regex,
    /// Anchored regex for whole-token matching: `^(?:p1|p2|...)$`.
    /// Applied per-token after whitespace split — structurally prevents substring matches.
    bare_anchored: Regex,
}

fn build_patterns(raw_patterns: &[&str]) -> LangFillers {
    // Match both annotation formats:
    //   `[?filler?]`         — legacy uncertainty marker
    //   `{filler|alternate}` — runner-up marker (Parakeet TDT alternates)
    //
    // The runner-up branch matches when the *guess* slot is a filler,
    // regardless of what the alternate slot contains. A filler with a
    // plausible runner-up is still a filler and should be deleted.
    let marked_parts: Vec<String> = raw_patterns
        .iter()
        .flat_map(|p| {
            [
                format!(r"\[\?(?:{p})\?\]"),
                format!(r"\{{(?:{p})\|[^}}]*\}}"),
            ]
        })
        .collect();

    let marked_combined = format!("(?i)(?:{})", marked_parts.join("|"));
    // Anchored: only matches if the ENTIRE token is a filler (no \b needed)
    let bare_combined = format!("(?i)^(?:{})$", raw_patterns.join("|"));

    LangFillers {
        marked_only: Regex::new(&marked_combined).expect("invalid marked filler regex"),
        bare_anchored: Regex::new(&bare_combined).expect("invalid bare filler regex"),
    }
}

/// Strip leading and trailing punctuation, returning the alphabetic core.
/// e.g. `"um,"` → `"um"`, `"...uh..."` → `"uh"`, `","` → `""`.
fn strip_punct(token: &str) -> &str {
    token.trim_matches(|c: char| !c.is_alphanumeric())
}

static EN_PATTERNS: &[&str] = &[
    r"u+h*m+", // um, umm, ummmm, uhm, Uhmmmm
    r"u+h+",   // uh, uhh, uhhhhh
    r"h+m+",   // hm, hmm, hmmm
    r"e+r+m*", // er, err, erm (British)
    r"a+h+",   // ah, ahh
];

static ES_PATTERNS: &[&str] = &[
    r"e+h+",     // eh, ehh, eehh
    r"e+m+",     // em, emm
    r"m+",       // m, mm, mmm
    r"a+h+",     // ah, ahh
    r"e+s+t+e+", // este (elongatable)
    r"p+u+e+s+", // pues (discourse filler)
];

static FR_PATTERNS: &[&str] = &[
    r"e+u+h*",   // eu, euh, euuh, euhhhhh
    r"h+e+i+n+", // hein, heiin
    r"b+a+h*",   // bah, baah
    r"b+e+n+",   // ben
    r"h+m+",     // hm, hmm
    r"e+h+",     // eh, ehh
];

static FILLERS: LazyLock<HashMap<&'static str, LangFillers>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("en", build_patterns(EN_PATTERNS));
    m.insert("es", build_patterns(ES_PATTERNS));
    m.insert("fr", build_patterns(FR_PATTERNS));
    m
});

// ─── Post-Removal Cleanup Regexes ─────────────────────────────────────────────

static ORPHAN_PUNCT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*,\s*,").unwrap());
static LEADING_PUNCT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*[,،]\s*").unwrap());
static MULTI_SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"  +").unwrap());
static SPACE_BEFORE_PUNCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+([,.!?;:])").unwrap());

fn clean_after_removal(text: &str) -> String {
    let s = ORPHAN_PUNCT.replace_all(text, ",");
    let s = LEADING_PUNCT.replace_all(&s, "");
    let s = MULTI_SPACE.replace_all(&s, " ");
    let s = SPACE_BEFORE_PUNCT.replace_all(&s, "$1");
    s.trim().to_string()
}

// ─── Public API ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum FillerError {
    UnsupportedLanguage(String),
}

impl std::fmt::Display for FillerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FillerError::UnsupportedLanguage(lang) => {
                write!(f, "unsupported language '{lang}'; supported: en, es, fr")
            }
        }
    }
}

impl std::error::Error for FillerError {}

/// Remove all filler words (bare and `[?..?]`-wrapped) from transcript.
/// Handles elongated forms: "Uhmmmm", "euhhhhh", "mmmm" etc.
///
/// Two-pass approach:
/// 1. Remove `[?filler?]` markers inline (self-delimited, always safe).
/// 2. Tokenize on whitespace, remove tokens whose alphabetic core matches
///    a filler pattern (`^pattern$`). This structurally prevents substring
///    matches inside real words like "summer", "hummer", "errand".
pub fn remove_fillers(text: &str, lang: &str) -> Result<String, FillerError> {
    let fillers = FILLERS
        .get(lang)
        .ok_or_else(|| FillerError::UnsupportedLanguage(lang.to_string()))?;

    // Pass 1: remove [?filler?] markers inline (self-delimited)
    let after_marked = fillers.marked_only.replace_all(text, "");

    // Pass 2: tokenize and remove bare filler tokens
    let result: Vec<&str> = after_marked
        .split_whitespace()
        .filter(|token| {
            let core = strip_punct(token);
            core.is_empty() || !fillers.bare_anchored.is_match(core)
        })
        .collect();

    Ok(clean_after_removal(&result.join(" ")))
}

/// Only removes `[?..?]`-wrapped fillers, leaves bare words untouched.
pub fn remove_fillers_marked_only(text: &str, lang: &str) -> Result<String, FillerError> {
    let fillers = FILLERS
        .get(lang)
        .ok_or_else(|| FillerError::UnsupportedLanguage(lang.to_string()))?;

    let cleaned = fillers.marked_only.replace_all(text, "");
    Ok(clean_after_removal(&cleaned))
}

/// Detect if text contains any fillers for the given language.
pub fn has_fillers(text: &str, lang: &str) -> Result<bool, FillerError> {
    let fillers = FILLERS
        .get(lang)
        .ok_or_else(|| FillerError::UnsupportedLanguage(lang.to_string()))?;

    // Check [?filler?] markers
    if fillers.marked_only.is_match(text) {
        return Ok(true);
    }

    // Check bare tokens
    let has_bare = text.split_whitespace().any(|token| {
        let core = strip_punct(token);
        !core.is_empty() && fillers.bare_anchored.is_match(core)
    });

    Ok(has_bare)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_bare_elongated() {
        let r = remove_fillers("Uhmmmm their results were um great", "en").unwrap();
        assert_eq!(r, "their results were great");
    }

    #[test]
    fn en_marked_elongated() {
        let r = remove_fillers("[?Ummmm?] the [?uh?] meeting is at noon", "en").unwrap();
        assert_eq!(r, "the meeting is at noon");
    }

    #[test]
    fn en_hmm() {
        let r = remove_fillers("it was hmmm kind of unclear", "en").unwrap();
        assert_eq!(r, "it was kind of unclear");
    }

    #[test]
    fn en_no_false_positive() {
        // "summer" must not be altered
        let r = remove_fillers("summer is great", "en").unwrap();
        assert_eq!(r, "summer is great");
    }

    #[test]
    fn es_mixed() {
        let r = remove_fillers("este mmmmm no sé cómo decirlo", "es").unwrap();
        assert_eq!(r, "no sé cómo decirlo");
    }

    #[test]
    fn fr_elongated() {
        let r = remove_fillers("euhhhhh je ne suis pas sûr", "fr").unwrap();
        assert_eq!(r, "je ne suis pas sûr");
    }

    #[test]
    fn fr_marked_only_leaves_bare() {
        let r = remove_fillers_marked_only("euh c'est [?euh?] compliqué", "fr").unwrap();
        assert_eq!(r, "euh c'est compliqué");
    }

    #[test]
    fn unsupported_lang() {
        assert!(remove_fillers("text", "de").is_err());
    }

    #[test]
    fn has_fillers_detects() {
        assert!(has_fillers("um hello", "en").unwrap());
        assert!(!has_fillers("hello world", "en").unwrap());
    }

    #[test]
    fn cleanup_orphan_punct() {
        let r = remove_fillers("hello, um, world", "en").unwrap();
        assert_eq!(r, "hello, world");
    }

    #[test]
    fn cleanup_leading_filler() {
        let r = remove_fillers("um, hello world", "en").unwrap();
        assert_eq!(r, "hello world");
    }

    #[test]
    fn en_er_erm() {
        let r = remove_fillers("I think er we should erm go", "en").unwrap();
        assert_eq!(r, "I think we should go");
    }

    #[test]
    fn en_no_false_positive_hummer() {
        let r = remove_fillers("that hummer was loud", "en").unwrap();
        assert_eq!(r, "that hummer was loud");
    }

    #[test]
    fn en_no_false_positive_errand() {
        let r = remove_fillers("I need to run an errand", "en").unwrap();
        assert_eq!(r, "I need to run an errand");
    }

    #[test]
    fn en_no_false_positive_premium() {
        let r = remove_fillers("the premium plan costs more", "en").unwrap();
        assert_eq!(r, "the premium plan costs more");
    }

    #[test]
    fn en_mixed_marked_and_bare() {
        let r = remove_fillers("[?um?] so the summer um festival was ahh amazing", "en").unwrap();
        assert_eq!(r, "so the summer festival was amazing");
    }

    // ── Runner-up `{filler|alt}` marker removal ─────────────────────────

    #[test]
    fn en_runner_up_filler_stripped() {
        // Parakeet TDT may emit `{um|and}` when the filler is uncertain and
        // the runner-up token happens to be a real word. The marker must
        // still be deleted — a filler with an alternate is still a filler.
        let r = remove_fillers("{um|and} the meeting starts", "en").unwrap();
        assert_eq!(r, "the meeting starts");
    }

    #[test]
    fn en_runner_up_filler_stripped_marked_only() {
        let r = remove_fillers_marked_only("this {uh|a} sounds {um|and} right", "en").unwrap();
        assert_eq!(r, "this sounds right");
    }

    #[test]
    fn en_runner_up_real_word_preserved() {
        // `{summer|summit}` has a real-word guess, not a filler — do NOT strip.
        let r = remove_fillers("the {summer|summit} festival", "en").unwrap();
        assert_eq!(r, "the {summer|summit} festival");
    }

    #[test]
    fn en_runner_up_filler_elongated() {
        let r = remove_fillers("{Uhmmmm|and} the results", "en").unwrap();
        assert_eq!(r, "the results");
    }

    #[test]
    fn es_runner_up_filler() {
        let r = remove_fillers("{eh|y} no sé", "es").unwrap();
        assert_eq!(r, "no sé");
    }

    #[test]
    fn fr_runner_up_filler() {
        let r = remove_fillers("{euh|et} je ne sais pas", "fr").unwrap();
        assert_eq!(r, "je ne sais pas");
    }
}
