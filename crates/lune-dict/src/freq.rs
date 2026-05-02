use std::collections::HashMap;
use std::sync::LazyLock;

const MAX_WORD_LEN: usize = 32;

pub struct FreqDict {
    freq: HashMap<Box<str>, u32>,
    by_len: [Vec<u32>; MAX_WORD_LEN],
    words: Vec<Box<str>>,
}

impl FreqDict {
    fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                freq: HashMap::new(),
                by_len: std::array::from_fn(|_| Vec::new()),
                words: Vec::new(),
            };
        }
        let (entries, _): (Vec<(String, u32)>, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())
                .expect("decode freq bin");

        let mut freq = HashMap::with_capacity(entries.len());
        let mut words = Vec::with_capacity(entries.len());
        let mut by_len: [Vec<u32>; MAX_WORD_LEN] = std::array::from_fn(|_| Vec::new());

        for (word, f) in entries {
            let len = word.len().min(MAX_WORD_LEN - 1);
            let idx = words.len() as u32;
            let boxed: Box<str> = word.into_boxed_str();
            freq.insert(boxed.clone(), f);
            words.push(boxed);
            by_len[len].push(idx);
        }

        Self {
            freq,
            by_len,
            words,
        }
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

    pub fn candidates_iter(&self, lo: usize, hi: usize) -> CandidateIter<'_> {
        CandidateIter {
            dict: self,
            hi,
            bucket: lo,
            pos: 0,
        }
    }
}

pub struct CandidateIter<'a> {
    dict: &'a FreqDict,
    hi: usize,
    bucket: usize,
    pos: usize,
}

impl<'a> Iterator for CandidateIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.bucket >= self.hi || self.bucket >= MAX_WORD_LEN {
                return None;
            }
            let bucket = &self.dict.by_len[self.bucket];
            if self.pos < bucket.len() {
                let idx = bucket[self.pos] as usize;
                self.pos += 1;
                return Some(&self.dict.words[idx]);
            }
            self.bucket += 1;
            self.pos = 0;
        }
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
freq_dict!(PT_FREQ, "../data/pt_freq.bin");
freq_dict!(RU_FREQ, "../data/ru_freq.bin");
freq_dict!(ZH_FREQ, "../data/zh_freq.bin");

#[inline]
pub fn dict_for(lang: &str) -> &'static FreqDict {
    let bytes = lang.as_bytes();
    match bytes.first().map(|b| b.to_ascii_lowercase()) {
        Some(b'd') => &DE_FREQ,
        Some(b'g') if lang.len() > 2 => &DE_FREQ, // "german"
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

pub const SUPPORTED: &[&str] = &["en", "de", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"];

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
                | "portuguese"
                | "russian"
                | "chinese"
        )
}

pub fn warm() {
    let _ = EN_FREQ.len();
}
