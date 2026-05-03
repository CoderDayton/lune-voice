use super::gms_tables::GmsLanguage;

impl GmsLanguage {
    pub const fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::French => "fr",
            Self::German => "de",
            Self::Spanish => "es",
            Self::Italian => "it",
            Self::Chinese => "zh",
            Self::Japanese => "ja",
            Self::Korean => "ko",
            Self::Portuguese => "pt",
            Self::Russian => "ru",
        }
    }

    pub const fn qwen_prompt_hint(self) -> &'static str {
        match self {
            Self::English => "Transcribe the speech in English.",
            Self::French => "Transcribe the speech in French.",
            Self::German => "Transcribe the speech in German.",
            Self::Spanish => "Transcribe the speech in Spanish.",
            Self::Italian => "Transcribe the speech in Italian.",
            Self::Chinese => "Transcribe the speech in Chinese.",
            Self::Japanese => "Transcribe the speech in Japanese.",
            Self::Korean => "Transcribe the speech in Korean.",
            Self::Portuguese => "Transcribe the speech in Portuguese.",
            Self::Russian => "Transcribe the speech in Russian.",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetectedAsrLanguage {
    pub gms: GmsLanguage,
    pub gms_code: &'static str,
    pub qwen_code: &'static str,
    pub qwen_prompt_hint: &'static str,
}

impl DetectedAsrLanguage {
    pub const fn from_gms(gms: GmsLanguage) -> Self {
        Self {
            gms,
            gms_code: gms.code(),
            qwen_code: gms.code(),
            qwen_prompt_hint: gms.qwen_prompt_hint(),
        }
    }
}

pub fn detect_asr_language(text: &str) -> DetectedAsrLanguage {
    DetectedAsrLanguage::from_gms(detect_gms_language(text))
}

pub fn detect_gms_language(text: &str) -> GmsLanguage {
    let mut stats = LangStats::default();
    for token in text.split_whitespace() {
        let token = normalize_token(token);
        if token.is_empty() {
            continue;
        }
        stats.observe_token(&token);
    }
    stats.finish()
}

pub fn normalize_gms_language_code(code: &str) -> Option<&'static str> {
    match code.trim().to_ascii_lowercase().as_str() {
        "en" | "eng" | "english" => Some("en"),
        "fr" | "fra" | "fre" | "french" => Some("fr"),
        "de" | "deu" | "ger" | "german" => Some("de"),
        "es" | "spa" | "spanish" => Some("es"),
        "it" | "ita" | "italian" => Some("it"),
        "zh" | "zho" | "chi" | "chinese" | "cn" => Some("zh"),
        "ja" | "jpn" | "japanese" => Some("ja"),
        "ko" | "kor" | "korean" => Some("ko"),
        "pt" | "por" | "portuguese" => Some("pt"),
        "ru" | "rus" | "russian" => Some("ru"),
        _ => None,
    }
}

#[derive(Debug, Default)]
struct LangStats {
    latin_letters: usize,
    cyrillic_letters: usize,
    han: usize,
    hiragana: usize,
    katakana: usize,
    hangul: usize,

    en: i32,
    fr: i32,
    de: i32,
    es: i32,
    it: i32,
    pt: i32,
    ru: i32,
    ja: i32,
    ko: i32,
}

impl LangStats {
    fn observe_token(&mut self, token: &str) {
        for c in token.chars() {
            let cp = c as u32;
            match cp {
                0x0041..=0x024F => self.latin_letters += 1,
                0x0400..=0x052F => {
                    self.cyrillic_letters += 1;
                    self.ru += 1;
                }
                0x4E00..=0x9FFF | 0x3400..=0x4DBF => {
                    self.han += 1;
                }
                0x3040..=0x309F => {
                    self.hiragana += 1;
                    self.ja += 5;
                }
                0x30A0..=0x30FF => {
                    self.katakana += 1;
                    self.ja += 4;
                }
                0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => {
                    self.hangul += 1;
                    self.ko += 5;
                }
                _ => {}
            }
        }
        self.observe_latin_token(token);
    }

    fn observe_latin_token(&mut self, token: &str) {
        if token.is_empty() {
            return;
        }

        for c in token.chars() {
            match c {
                'ç' | 'œ' => self.fr += 3,
                'â' | 'æ' | 'ê' | 'ë' | 'î' | 'ï' | 'ô' | 'û' | 'ÿ' => self.fr += 2,
                'è' | 'ù' => {
                    self.fr += 2;
                    self.it += 1;
                }
                'à' => {
                    self.fr += 2;
                    self.it += 2;
                }
                'é' => {
                    self.fr += 2;
                    self.es += 1;
                    self.pt += 1;
                }
                'ä' | 'ö' | 'ß' => self.de += 4,
                'ü' => {
                    self.de += 3;
                    self.es += 1;
                }
                'ñ' => self.es += 5,
                'á' | 'í' | 'ó' | 'ú' => {
                    self.es += 2;
                    self.pt += 2;
                }
                'ì' | 'ò' => self.it += 3,
                'ã' | 'õ' => self.pt += 5,
                _ => {}
            }
        }

        match token {
            "the" | "and" | "that" | "this" | "you" | "your" | "with" | "for" | "from" | "have"
            | "has" | "was" | "were" | "are" | "is" | "not" | "but" | "they" | "their"
            | "there" | "what" | "when" | "where" | "which" | "would" | "could" | "should"
            | "people" | "information" => self.en += 4,

            "le" | "la" | "les" | "un" | "une" | "des" | "du" | "je" | "tu" | "il" | "elle"
            | "nous" | "vous" | "ils" | "elles" | "est" | "sont" | "avec" | "pour" | "dans"
            | "que" | "qui" | "pas" | "plus" | "sur" => {
                self.fr += 4;
            }

            "der" | "die" | "das" | "den" | "dem" | "ein" | "eine" | "und" | "oder" | "ich"
            | "er" | "wir" | "ihr" | "ist" | "sind" | "nicht" | "mit" | "auf" | "aus" | "von"
            | "zu" | "dass" | "werden" => {
                self.de += 4;
            }

            "el" | "los" | "las" | "unos" | "unas" | "yo" | "nosotros" | "ustedes" | "son"
            | "con" | "para" | "por" | "quien" | "donde" | "como" | "pero" | "no" => self.es += 4,

            "lo" | "gli" | "uno" | "io" | "lui" | "lei" | "noi" | "voi" | "sono" | "per"
            | "che" | "chi" | "non" | "ma" => {
                self.it += 4;
            }

            "os" | "as" | "uma" | "uns" | "umas" | "ou" | "eu" | "ele" | "ela" | "vocês"
            | "são" | "quem" | "onde" | "não" | "mais" | "mas" => self.pt += 4,

            "и" | "в" | "во" | "не" | "что" | "он" | "на" | "я" | "с" | "со" | "как" | "а"
            | "то" | "все" | "она" | "так" | "его" | "но" | "да" | "ты" | "к" | "у" | "же"
            | "вы" | "за" | "бы" | "по" | "только" => {
                self.ru += 5;
            }

            _ => {}
        }

        if token.ends_with("tion") || token.ends_with("ing") || token.ends_with("ly") {
            self.en += 2;
        }
        if token.ends_with("ment") || token.ends_with("eux") || token.ends_with("ique") {
            self.fr += 2;
        }
        if token.ends_with("ung") || token.ends_with("keit") || token.ends_with("chen") {
            self.de += 3;
        }
        if token.ends_with("ción") || token.ends_with("dad") || token.ends_with("mente") {
            self.es += 3;
        }
        if token.ends_with("zione") || token.ends_with("ità") {
            self.it += 3;
        }
        if token.ends_with("ção") || token.ends_with("ções") {
            self.pt += 3;
        }
    }

    fn finish(&self) -> GmsLanguage {
        let total_cjk = self.han + self.hiragana + self.katakana + self.hangul;
        let total_chars = self.latin_letters + self.cyrillic_letters + total_cjk;

        if total_chars == 0 {
            return GmsLanguage::English;
        }

        if self.hangul > 0 && self.hangul * 2 >= total_cjk {
            return GmsLanguage::Korean;
        }
        if (self.hiragana + self.katakana) > 0 && (self.hiragana + self.katakana) * 2 >= total_cjk {
            return GmsLanguage::Japanese;
        }
        if self.han > 0 && self.han * 2 >= total_chars {
            return GmsLanguage::Chinese;
        }
        if self.cyrillic_letters > 0 && self.cyrillic_letters * 2 >= total_chars {
            return GmsLanguage::Russian;
        }

        let scores = [
            (self.en, GmsLanguage::English),
            (self.fr, GmsLanguage::French),
            (self.de, GmsLanguage::German),
            (self.es, GmsLanguage::Spanish),
            (self.it, GmsLanguage::Italian),
            (self.pt, GmsLanguage::Portuguese),
        ];

        let (best_score, best_lang) = scores.iter().copied().max_by_key(|(s, _)| *s).unwrap();

        if best_score < 2 {
            return GmsLanguage::English;
        }

        let second_best = scores
            .iter()
            .copied()
            .filter(|(_, l)| *l != best_lang)
            .map(|(s, _)| s)
            .max()
            .unwrap_or(0);

        if best_score - second_best >= 2 {
            best_lang
        } else {
            GmsLanguage::English
        }
    }
}

fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|c| c.is_alphabetic() || *c == '\'' || *c == '\u{2019}')
        .flat_map(|c| c.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_english() {
        let lang = detect_gms_language("The quick brown fox jumps over the lazy dog");
        assert_eq!(lang, GmsLanguage::English);
    }

    #[test]
    fn detect_french() {
        let lang = detect_gms_language("Je suis allé dans les rues de Paris avec mon ami");
        assert_eq!(lang, GmsLanguage::French);
    }

    #[test]
    fn detect_german() {
        let lang = detect_gms_language(
            "Ich bin mit dem Zug nach Berlin gefahren und habe die Stadt gesehen",
        );
        assert_eq!(lang, GmsLanguage::German);
    }

    #[test]
    fn detect_spanish() {
        let lang = detect_gms_language("Yo no sé dónde están los libros para el examen");
        assert_eq!(lang, GmsLanguage::Spanish);
    }

    #[test]
    fn detect_italian() {
        let lang = detect_gms_language("Io non sono andato per le strade di Roma con lui");
        assert_eq!(lang, GmsLanguage::Italian);
    }

    #[test]
    fn detect_portuguese() {
        let lang = detect_gms_language("Eu não sei onde estão os livros para a prova");
        assert_eq!(lang, GmsLanguage::Portuguese);
    }

    #[test]
    fn detect_russian() {
        let lang = detect_gms_language("Я не знаю что он сказал но я с ним согласен");
        assert_eq!(lang, GmsLanguage::Russian);
    }

    #[test]
    fn detect_chinese() {
        let lang = detect_gms_language("我今天去了北京的长城非常壮观");
        assert_eq!(lang, GmsLanguage::Chinese);
    }

    #[test]
    fn detect_japanese() {
        let lang = detect_gms_language("きょうは東京に行きました");
        assert_eq!(lang, GmsLanguage::Japanese);
    }

    #[test]
    fn detect_korean() {
        let lang = detect_gms_language("오늘 서울에 갔습니다");
        assert_eq!(lang, GmsLanguage::Korean);
    }

    #[test]
    fn empty_defaults_english() {
        assert_eq!(detect_gms_language(""), GmsLanguage::English);
    }

    #[test]
    fn ambiguous_defaults_english() {
        assert_eq!(detect_gms_language("hello"), GmsLanguage::English);
    }

    #[test]
    fn accent_detection_french() {
        let lang = detect_gms_language("façade résumé naïve");
        assert_eq!(lang, GmsLanguage::French);
    }

    #[test]
    fn accent_detection_german() {
        let lang = detect_gms_language("Straße Ärger Öffnung über");
        assert_eq!(lang, GmsLanguage::German);
    }

    #[test]
    fn normalize_code() {
        assert_eq!(normalize_gms_language_code("en"), Some("en"));
        assert_eq!(normalize_gms_language_code("ENG"), Some("en"));
        assert_eq!(normalize_gms_language_code("French"), Some("fr"));
        assert_eq!(normalize_gms_language_code("xxx"), None);
    }

    #[test]
    fn detected_asr_language_roundtrip() {
        let det = detect_asr_language("Bonjour je suis français avec les amis");
        assert_eq!(det.gms, GmsLanguage::French);
        assert_eq!(det.gms_code, "fr");
        assert_eq!(det.qwen_code, "fr");
    }

    #[test]
    fn normalize_token_strips_punctuation() {
        assert_eq!(normalize_token("Hello,"), "hello");
        assert_eq!(normalize_token("don't"), "don't");
        assert_eq!(normalize_token("123abc"), "abc");
    }
}
