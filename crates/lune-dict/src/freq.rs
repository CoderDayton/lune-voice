use std::collections::HashMap;
use std::sync::LazyLock;

pub struct FreqDict {
    freq: HashMap<Box<str>, u32>,
}

impl FreqDict {
    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                freq: HashMap::new(),
            };
        }
        let (entries, _): (Vec<(String, u32)>, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())
                .expect("decode freq bin");

        let mut freq = HashMap::with_capacity(entries.len());
        for (word, f) in entries {
            freq.insert(word.into_boxed_str(), f);
        }

        Self { freq }
    }

    #[inline]
    pub fn get(&self, word: &str) -> Option<u32> {
        self.freq.get(word).copied()
    }

    #[inline]
    pub fn contains(&self, word: &str) -> bool {
        self.freq.contains_key(word)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.freq.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.freq.is_empty()
    }
}

macro_rules! freq_dict {
    ($name:ident, $file:literal) => {
        static $name: LazyLock<FreqDict> =
            LazyLock::new(|| FreqDict::from_bytes(include_bytes!($file)));
    };
}

freq_dict!(EN_FREQ, "../data/en_freq.bin");
freq_dict!(DE_FREQ, "../data/de_freq.bin");
freq_dict!(ES_FREQ, "../data/es_freq.bin");
freq_dict!(FR_FREQ, "../data/fr_freq.bin");
freq_dict!(IT_FREQ, "../data/it_freq.bin");
freq_dict!(JA_FREQ, "../data/ja_freq.bin");
freq_dict!(KO_FREQ, "../data/ko_freq.bin");
freq_dict!(NL_FREQ, "../data/nl_freq.bin");
freq_dict!(PT_FREQ, "../data/pt_freq.bin");
freq_dict!(RU_FREQ, "../data/ru_freq.bin");
freq_dict!(ZH_FREQ, "../data/zh_freq.bin");

pub fn dict_for(lang: &str) -> &'static FreqDict {
    let bytes = lang.as_bytes();
    match bytes.first().map(|b| b.to_ascii_lowercase()) {
        Some(b'd') if bytes.get(1) == Some(&b'u') => &NL_FREQ, // "dutch"
        Some(b'd') => &DE_FREQ,
        Some(b'g') if lang.len() > 2 => &DE_FREQ, // "german"
        Some(b'n') if bytes.get(1) == Some(&b'l') => &NL_FREQ, // "nl"
        Some(b's') if bytes.get(1) == Some(&b'p') => &ES_FREQ, // "spanish"
        Some(b'f') => &FR_FREQ,
        Some(b'i') => &IT_FREQ,
        Some(b'j') => &JA_FREQ,
        Some(b'k') => &KO_FREQ,
        Some(b'p') if bytes.get(1) == Some(&b't') => &PT_FREQ,
        Some(b'p') if lang.len() > 2 => &PT_FREQ, // "portuguese"
        Some(b'r') => &RU_FREQ,
        Some(b'z') => &ZH_FREQ,
        Some(b'c') if lang.len() > 2 => &ZH_FREQ, // "chinese"
        Some(b'e') if bytes.get(1) == Some(&b's') => &ES_FREQ,
        _ => &EN_FREQ,
    }
}

pub fn is_english(lang: &str) -> bool {
    lang.is_empty()
        || lang.starts_with("en")
        || lang.eq_ignore_ascii_case("english")
        || lang.starts_with("En")
}

pub const SUPPORTED: &[&str] = &[
    "en", "de", "es", "fr", "it", "ja", "ko", "nl", "pt", "ru", "zh",
];

pub fn has_dict(lang: &str) -> bool {
    if lang.is_empty() {
        return true;
    }
    let l = lang.to_ascii_lowercase();
    SUPPORTED.iter().any(|&s| l.starts_with(s))
        || matches!(
            l.as_str(),
            "english"
                | "german"
                | "spanish"
                | "french"
                | "italian"
                | "japanese"
                | "korean"
                | "dutch"
                | "portuguese"
                | "russian"
                | "chinese"
        )
}

pub fn warm() {
    let _ = EN_FREQ.len();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_for_routes_dutch_to_nl_not_german() {
        // "dutch" shares its first byte with "de"/"german"; ensure it is not
        // misrouted to the German dictionary.
        assert!(std::ptr::eq(dict_for("nl"), dict_for("Dutch")));
        assert!(std::ptr::eq(dict_for("Dutch"), &*NL_FREQ));
        assert!(!std::ptr::eq(dict_for("Dutch"), dict_for("de")));
        assert!(std::ptr::eq(dict_for("de"), &*DE_FREQ));
    }

    #[test]
    fn dutch_dictionary_loads_common_words() {
        // Sanity-check the bundled nl_freq.bin actually decoded.
        let nl = dict_for("nl");
        assert!(nl.len() > 10_000, "nl dict too small: {}", nl.len());
    }
}
