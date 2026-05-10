//! Language-tag normalization shared across the pipeline.
//!
//! The runtime mixes BCP-47 codes (`en`, `en-US`) and the English language
//! names emitted by Qwen3-ASR's language head (`English`, `Dutch`). Anything
//! that surfaces the language to a user prompt, log, or event should run it
//! through [`display_name`] so the rest of the code sees a single canonical
//! form.

/// Convert a language hint to its English display name (e.g. `"English"`).
///
/// Accepts BCP-47 codes (`"en"`, `"en-US"`, `"en_GB"`) and the full English
/// names emitted by Qwen3-ASR. Returns `None` for empty input, `"auto"`, or
/// any tag we don't have a label for, so callers can drop the language tag
/// rather than emit ungrammatical text like "the following en transcript".
pub fn display_name(raw: &str) -> Option<&'static str> {
    let lang = raw.trim();
    if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
        return None;
    }
    let primary = lang
        .split(['-', '_'])
        .next()
        .unwrap_or(lang)
        .to_ascii_lowercase();
    Some(match primary.as_str() {
        "en" | "english" => "English",
        "fr" | "french" => "French",
        "de" | "german" => "German",
        "es" | "spanish" => "Spanish",
        "it" | "italian" => "Italian",
        "pt" | "portuguese" => "Portuguese",
        "nl" | "dutch" => "Dutch",
        "ja" | "japanese" => "Japanese",
        "zh" | "cmn" | "yue" | "chinese" | "mandarin" => "Chinese",
        "ko" | "korean" => "Korean",
        "ru" | "russian" => "Russian",
        "pl" | "polish" => "Polish",
        "ar" | "arabic" => "Arabic",
        "hi" | "hindi" => "Hindi",
        "vi" | "vietnamese" => "Vietnamese",
        "th" | "thai" => "Thai",
        "id" | "indonesian" => "Indonesian",
        "tr" | "turkish" => "Turkish",
        _ => return None,
    })
}

/// Same as [`display_name`] but returns the original input verbatim when no
/// label is known. Useful at boundaries where we'd rather propagate the raw
/// tag than discard it (logs, telemetry, fallback paths).
pub fn display_name_or_raw(raw: &str) -> &str {
    match display_name(raw) {
        Some(name) => name,
        None => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_bcp47_codes_to_english_names() {
        assert_eq!(display_name("en"), Some("English"));
        assert_eq!(display_name("en-US"), Some("English"));
        assert_eq!(display_name("en_GB"), Some("English"));
        assert_eq!(display_name("de"), Some("German"));
        assert_eq!(display_name("ja"), Some("Japanese"));
    }

    #[test]
    fn passes_through_full_english_names() {
        assert_eq!(display_name("English"), Some("English"));
        assert_eq!(display_name("ENGLISH"), Some("English"));
        assert_eq!(display_name("Dutch"), Some("Dutch"));
    }

    #[test]
    fn drops_auto_and_unknown() {
        assert_eq!(display_name(""), None);
        assert_eq!(display_name("auto"), None);
        assert_eq!(display_name("AUTO"), None);
        assert_eq!(display_name("klingon"), None);
    }

    #[test]
    fn or_raw_falls_back() {
        assert_eq!(display_name_or_raw("en"), "English");
        assert_eq!(display_name_or_raw("klingon"), "klingon");
        assert_eq!(display_name_or_raw("auto"), "auto");
    }
}
