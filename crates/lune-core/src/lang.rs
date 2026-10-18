// SPDX-License-Identifier: GPL-3.0-or-later

//! Language-tag normalization shared across the pipeline.
//!
//! The runtime mixes BCP-47 codes (`en`, `en-US`) with full English language
//! names (`English`, `Dutch`). Anything that surfaces the language to a user
//! prompt, log, or event should run it through [`display_name`] so the rest of
//! the code sees a single canonical form.

/// Convert a language hint to its English display name (e.g. `"English"`).
///
/// Accepts BCP-47 codes (`"en"`, `"en-US"`, `"en_GB"`) and full English
/// language names. Returns `None` for empty input, `"auto"`, or any tag we
/// don't have a label for, so callers can drop the language tag rather than
/// emit ungrammatical text like "the following en transcript".
pub fn display_name(raw: &str) -> Option<&'static str> {
    let lang = raw.trim();
    if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
        return None;
    }
    let primary = lang.split(['-', '_']).next().unwrap_or(lang);
    // `(alias, display name)` pairs matched case-insensitively, so the primary
    // subtag never has to be lowercased into a heap `String`. One table is the
    // canonical mapping for both BCP-47 codes and English language names.
    const TABLE: &[(&str, &str)] = &[
        ("en", "English"),
        ("english", "English"),
        ("fr", "French"),
        ("french", "French"),
        ("de", "German"),
        ("german", "German"),
        ("es", "Spanish"),
        ("spanish", "Spanish"),
        ("it", "Italian"),
        ("italian", "Italian"),
        ("pt", "Portuguese"),
        ("portuguese", "Portuguese"),
        ("nl", "Dutch"),
        ("dutch", "Dutch"),
        ("ja", "Japanese"),
        ("japanese", "Japanese"),
        ("zh", "Chinese"),
        ("cmn", "Chinese"),
        ("yue", "Chinese"),
        ("chinese", "Chinese"),
        ("mandarin", "Chinese"),
        ("ko", "Korean"),
        ("korean", "Korean"),
        ("ru", "Russian"),
        ("russian", "Russian"),
        ("pl", "Polish"),
        ("polish", "Polish"),
        ("ar", "Arabic"),
        ("arabic", "Arabic"),
        ("hi", "Hindi"),
        ("hindi", "Hindi"),
        ("vi", "Vietnamese"),
        ("vietnamese", "Vietnamese"),
        ("th", "Thai"),
        ("thai", "Thai"),
        ("id", "Indonesian"),
        ("indonesian", "Indonesian"),
        ("tr", "Turkish"),
        ("turkish", "Turkish"),
        // The rest of the 25 European languages Parakeet covers (and Whisper
        // with it), so a dictation in any of them is formatted as itself
        // rather than falling back to English rules.
        ("bg", "Bulgarian"),
        ("bulgarian", "Bulgarian"),
        ("cs", "Czech"),
        ("czech", "Czech"),
        ("da", "Danish"),
        ("danish", "Danish"),
        ("el", "Greek"),
        ("greek", "Greek"),
        ("et", "Estonian"),
        ("estonian", "Estonian"),
        ("fi", "Finnish"),
        ("finnish", "Finnish"),
        ("hr", "Croatian"),
        ("croatian", "Croatian"),
        ("hu", "Hungarian"),
        ("hungarian", "Hungarian"),
        ("lt", "Lithuanian"),
        ("lithuanian", "Lithuanian"),
        ("lv", "Latvian"),
        ("latvian", "Latvian"),
        ("mt", "Maltese"),
        ("maltese", "Maltese"),
        ("ro", "Romanian"),
        ("romanian", "Romanian"),
        ("sk", "Slovak"),
        ("slovak", "Slovak"),
        ("sl", "Slovenian"),
        ("slovenian", "Slovenian"),
        ("sv", "Swedish"),
        ("swedish", "Swedish"),
        ("uk", "Ukrainian"),
        ("ukrainian", "Ukrainian"),
    ];
    TABLE
        .iter()
        .find(|(alias, _)| primary.eq_ignore_ascii_case(alias))
        .map(|(_, name)| *name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every language a local engine can be set to has a name, so none of
    /// them is formatted with English rules for want of one.
    #[test]
    fn every_parakeet_language_has_a_display_name() {
        for code in crate::config::ParakeetModel::default().languages() {
            assert!(display_name(code).is_some(), "{code} has no display name");
        }
        assert_eq!(display_name("cs"), Some("Czech"));
        assert_eq!(display_name("uk-UA"), Some("Ukrainian"));
    }

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
}
