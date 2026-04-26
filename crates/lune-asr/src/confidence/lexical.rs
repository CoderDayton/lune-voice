//! Graphotactic Markov Surprisal (GMS) for ASR lexical plausibility.
//!
//! Drop-in lexical anomaly prior for ASR confidence scoring.
//!
//! This does not prove that a word is correct. It estimates whether the token
//! is graphotactically plausible versus ASR-garbled, looped, repeated,
//! script-mixed, digit-intruded, or run-length anomalous.
//!
//! Model:
//!   1. Boundary-aware Markov chain over coarse orthographic classes.
//!   2. Log-space product-of-experts fusion.
//!   3. Entropy/balance prior for degenerate class distributions.
//!   4. Run-length anomaly penalty.
//!   5. Approximate bigram diversity penalty with fixed bitsets.
//!   6. Script-mixing and digit-intrusion penalties.
//!   7. Segment-level repetition penalty.
//!
//! Public API intentionally unchanged:
//!   - word_confidence(word: &str) -> f32
//!   - segment_lexical_confidence(text: &str) -> f32

const CLASS_COUNT: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum CharClass {
    Vowel = 0,
    Stop = 1,
    Fricative = 2,
    Nasal = 3,
    Liquid = 4,
    Glide = 5,
    Rare = 6,
    Digit = 7,
    Separator = 8,
    Neutral = 9,
}

impl CharClass {
    #[inline]
    fn from_char(c: char) -> Self {
        let c = c.to_lowercase().next().unwrap_or(c);

        match c {
            'a' | 'e' | 'i' | 'o' | 'u' | 'y' => Self::Vowel,
            'b' | 'c' | 'd' | 'g' | 'k' | 'p' | 't' => Self::Stop,
            'f' | 'h' | 's' | 'v' | 'z' => Self::Fricative,
            'm' | 'n' => Self::Nasal,
            'l' | 'r' => Self::Liquid,
            'w' | 'j' => Self::Glide,
            'q' | 'x' => Self::Rare,
            '0'..='9' => Self::Digit,

            '\'' | '’' | '`' | '-' | '_' | '.' => Self::Separator,

            // Latin-extended vowels.
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'æ' | 'è' | 'é' | 'ê' | 'ë' | 'ì' | 'í' | 'î'
            | 'ï' | 'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ù' | 'ú' | 'û' | 'ü' | 'ý' | 'ÿ' | 'œ' => {
                Self::Vowel
            }

            'ñ' => Self::Nasal,
            'ç' | 'ß' | 'þ' | 'ð' => Self::Fricative,

            // Cyrillic vowels.
            'а' | 'е' | 'ё' | 'и' | 'о' | 'у' | 'ы' | 'э' | 'ю' | 'я' => Self::Vowel,
            // Cyrillic stops / affricates.
            'б' | 'г' | 'д' | 'к' | 'п' | 'т' | 'ц' => Self::Stop,
            // Cyrillic fricatives / affricates.
            'в' | 'ж' | 'з' | 'с' | 'ф' | 'х' | 'ш' | 'щ' | 'ч' => Self::Fricative,
            // Cyrillic nasals.
            'м' | 'н' => Self::Nasal,
            // Cyrillic liquids.
            'л' | 'р' => Self::Liquid,
            // Cyrillic glide / signs.
            'й' => Self::Glide,
            'ъ' | 'ь' => Self::Vowel,

            _ => {
                let cp = c as u32;

                if is_cjk_or_hangul(cp) || c.is_alphabetic() {
                    Self::Neutral
                } else {
                    Self::Separator
                }
            }
        }
    }
}

use super::gms_tables::{self, GmsTables};

fn default_tables() -> &'static GmsTables {
    gms_tables::tables_for_code("en").unwrap_or(&gms_tables::GMS_TABLES[0])
}

fn tables_for(lang: &str) -> &'static GmsTables {
    gms_tables::tables_for_code(lang).unwrap_or_else(|| default_tables())
}

pub fn detect_lang(text: &str) -> &'static str {
    super::lang_detect::detect_gms_language(text).code()
}

const MARKOV_POWER: f32 = 0.27;

const SCRIPT_LATIN: u8 = 1 << 0;
const SCRIPT_CYRILLIC: u8 = 1 << 1;
const SCRIPT_CJK: u8 = 1 << 2;
const SCRIPT_OTHER_ALPHA: u8 = 1 << 3;

const ENGLISH_CLUSTER_BONUS_CAP: f32 = 1.28;
const ENGLISH_CLUSTER_BONUS_STEP: f32 = 0.08;

const ENGLISH_CLUSTERS: &[&[u8]] = &[
    b"str", b"scr", b"spr", b"thr", b"tr", b"br", b"cr", b"dr", b"fr", b"gr", b"pr", b"pl", b"bl",
    b"cl", b"fl", b"gl", b"sl", b"sp", b"st", b"sk", b"sh", b"ch", b"th", b"ck", b"ng", b"tion",
    b"sion", b"qu", b"ough",
];

use super::common_words::COMMON_WORDS;

#[derive(Debug, Clone)]
struct WordStats {
    len: usize,
    alphabetic: usize,
    latin: usize,
    cyrillic: usize,
    cjk: usize,
    other_alpha: usize,
    digits: usize,
    separators: usize,
    latin_vowels: usize,
    rare: usize,

    max_run: usize,
    current_run: usize,

    bigrams: usize,
    unique_bigrams: usize,
    bigram_seen_lo: u64,
    bigram_seen_hi: u64,

    class_counts: [usize; CLASS_COUNT],
    script_mask: u8,

    log_prob_sum: f32,
    log_prob_count: usize,

    prev_char: Option<char>,
    prev_class: Option<CharClass>,
    first_class: Option<CharClass>,
    last_class: Option<CharClass>,
}

impl Default for WordStats {
    fn default() -> Self {
        Self {
            len: 0,
            alphabetic: 0,
            latin: 0,
            cyrillic: 0,
            cjk: 0,
            other_alpha: 0,
            digits: 0,
            separators: 0,
            latin_vowels: 0,
            rare: 0,

            max_run: 1,
            current_run: 0,

            bigrams: 0,
            unique_bigrams: 0,
            bigram_seen_lo: 0,
            bigram_seen_hi: 0,

            class_counts: [0; CLASS_COUNT],
            script_mask: 0,

            log_prob_sum: 0.0,
            log_prob_count: 0,

            prev_char: None,
            prev_class: None,
            first_class: None,
            last_class: None,
        }
    }
}

impl WordStats {
    fn observe(&mut self, c: char, tables: &GmsTables) {
        let class = CharClass::from_char(c);
        let script = script_bit(c);

        if self.len == 0 {
            self.first_class = Some(class);
            self.log_prob_sum += tables.start[class as usize].ln();
            self.log_prob_count += 1;
            self.current_run = 1;
        } else if let Some(prev_class) = self.prev_class {
            self.log_prob_sum += tables.trans[prev_class as usize][class as usize].ln();
            self.log_prob_count += 1;
        }

        if let Some(prev) = self.prev_char {
            if same_char_relaxed(prev, c) {
                self.current_run += 1;
            } else {
                self.current_run = 1;
            }

            self.max_run = self.max_run.max(self.current_run);

            let idx = hash_pair_chars(prev, c);
            if bitset_insert_128(&mut self.bigram_seen_lo, &mut self.bigram_seen_hi, idx) {
                self.unique_bigrams += 1;
            }
            self.bigrams += 1;
        }

        self.len += 1;
        self.last_class = Some(class);
        self.prev_class = Some(class);
        self.prev_char = Some(c);
        self.class_counts[class as usize] += 1;

        match script {
            SCRIPT_LATIN => {
                self.latin += 1;
                self.alphabetic += 1;
                self.script_mask |= SCRIPT_LATIN;
            }
            SCRIPT_CYRILLIC => {
                self.cyrillic += 1;
                self.alphabetic += 1;
                self.script_mask |= SCRIPT_CYRILLIC;
            }
            SCRIPT_CJK => {
                self.cjk += 1;
                self.alphabetic += 1;
                self.script_mask |= SCRIPT_CJK;
            }
            SCRIPT_OTHER_ALPHA => {
                self.other_alpha += 1;
                self.alphabetic += 1;
                self.script_mask |= SCRIPT_OTHER_ALPHA;
            }
            _ => {}
        }

        match class {
            CharClass::Digit => self.digits += 1,
            CharClass::Separator => self.separators += 1,
            CharClass::Rare => self.rare += 1,
            CharClass::Vowel if script == SCRIPT_LATIN => self.latin_vowels += 1,
            _ => {}
        }
    }

    fn finish_markov(&mut self, tables: &GmsTables) {
        if let Some(last) = self.last_class {
            self.log_prob_sum += tables.end[last as usize].ln();
            self.log_prob_count += 1;
        }
    }

    #[inline]
    fn markov_score(&self) -> f32 {
        if self.log_prob_count == 0 {
            return 0.0;
        }

        let geo_mean = (self.log_prob_sum / self.log_prob_count as f32).exp();
        geo_mean.powf(MARKOV_POWER).clamp(0.05, 0.99)
    }
}

/// Per-word lexical plausibility factor.
/// Returns [0.05, 0.99], except empty input returns 0.0.
///
/// Higher means the word is graphotactically plausible. This is not a
/// probability that ASR transcribed the word correctly.
pub fn word_confidence(word: &str) -> f32 {
    word_confidence_with_tables(word, default_tables())
}

pub fn word_confidence_for_lang(word: &str, lang: &str) -> f32 {
    word_confidence_with_tables(word, tables_for(lang))
}

fn word_confidence_with_tables(word: &str, tables: &GmsTables) -> f32 {
    let base = strip_contraction(word);
    let word = if base.is_empty() { word } else { base };

    let mut stats = WordStats::default();

    for c in word.chars() {
        stats.observe(c, tables);
    }

    if stats.len == 0 {
        return 0.0;
    }

    if stats.len == 1 {
        return single_char_score(word.chars().next().unwrap());
    }

    stats.finish_markov(tables);

    if stats.alphabetic == 0 {
        return non_alpha_score(&stats);
    }

    let markov = stats.markov_score();
    let run_factor = run_length_factor(stats.max_run);
    let diversity_factor = bigram_diversity_factor(stats.len, stats.bigrams, stats.unique_bigrams);
    let length_factor = length_factor(stats.len);
    let vowel_factor = vowel_balance_factor(&stats);
    let no_vowel_factor = no_vowel_factor(&stats);
    let intrusion_factor = intrusion_factor(&stats);
    let script_factor = script_mixing_factor(stats.script_mask);
    let rare_factor = rare_letter_factor(&stats);
    let entropy_factor = class_entropy_factor(&stats);
    let cluster_factor = if stats.latin > 0 {
        english_cluster_factor(word)
    } else {
        1.0
    };

    let mut score = markov
        * run_factor
        * diversity_factor
        * length_factor
        * vowel_factor
        * no_vowel_factor
        * intrusion_factor
        * script_factor
        * rare_factor
        * entropy_factor
        * cluster_factor;

    // Non-Latin scripts should be mostly neutral unless they show obvious
    // anomaly patterns. The English graphotactic matrix is not allowed to
    // strongly judge CJK/Cyrillic-only tokens.
    if stats.latin == 0 && (stats.cyrillic + stats.cjk + stats.other_alpha) > 0 {
        score = score.max(0.52) * run_factor * diversity_factor * script_factor;
    }

    if is_common_ascii_word(word) {
        score = score.max(0.68);
    }

    score.clamp(0.05, 0.99)
}

/// Segment-level lexical plausibility with repeated-word and loop detection.
pub fn segment_lexical_confidence(text: &str) -> f32 {
    segment_lexical_confidence_with_tables(text, default_tables())
}

pub fn segment_lexical_confidence_for_lang(text: &str, lang: &str) -> f32 {
    segment_lexical_confidence_with_tables(text, tables_for(lang))
}

fn segment_lexical_confidence_with_tables(text: &str, tables: &GmsTables) -> f32 {
    let mut count = 0usize;
    let mut sum = 0.0_f32;

    let mut seen_words = [0_u64; 4];
    let mut unique_words = 0usize;

    let mut seen_bigrams = [0_u64; 4];
    let mut bigram_count = 0usize;
    let mut repeated_bigram_hits = 0usize;

    let mut prev_hash: Option<u64> = None;
    let mut repeated_run = 1usize;
    let mut max_repeated_run = 1usize;

    for raw in text.split_whitespace() {
        let word = raw.trim_matches(is_outer_token_punct);
        if word.is_empty() {
            continue;
        }

        count += 1;
        sum += word_confidence_with_tables(word, tables);

        let hash = hash_str_normalized(word);
        let word_idx = (hash & 0xff) as u8;
        if bitset_insert_256(&mut seen_words, word_idx) {
            unique_words += 1;
        }

        if let Some(prev) = prev_hash {
            if prev == hash {
                repeated_run += 1;
                max_repeated_run = max_repeated_run.max(repeated_run);
            } else {
                repeated_run = 1;
            }

            let pair_hash = mix_u64_pair(prev, hash);
            let pair_idx = (pair_hash & 0xff) as u8;
            if !bitset_insert_256(&mut seen_bigrams, pair_idx) {
                repeated_bigram_hits += 1;
            }
            bigram_count += 1;
        }

        prev_hash = Some(hash);
    }

    if count == 0 {
        return 0.0;
    }

    let mean = sum / count as f32;
    let unique_ratio = unique_words as f32 / count as f32;

    let unique_factor = if count >= 4 && unique_ratio < 0.55 {
        (0.45 + unique_ratio).clamp(0.35, 1.0)
    } else if count >= 6 && unique_ratio < 0.75 {
        (0.80 + 0.25 * unique_ratio).clamp(0.70, 1.0)
    } else {
        1.0
    };

    let run_factor = if max_repeated_run >= 3 {
        (-0.50 * (max_repeated_run - 2).pow(2) as f32)
            .exp()
            .max(0.10)
    } else {
        1.0
    };

    let repeated_bigram_factor = if bigram_count >= 4 {
        let repeat_ratio = repeated_bigram_hits as f32 / bigram_count as f32;
        if repeat_ratio > 0.25 {
            (1.0 - repeat_ratio).clamp(0.30, 1.0)
        } else {
            1.0
        }
    } else {
        1.0
    };

    (mean * unique_factor * run_factor * repeated_bigram_factor).clamp(0.05, 0.99)
}

#[inline]
fn single_char_score(c: char) -> f32 {
    let lc = c.to_ascii_lowercase();

    if lc == 'a' || lc == 'i' {
        0.90
    } else if c.is_ascii_digit() {
        0.72
    } else if script_bit(c) == SCRIPT_LATIN {
        0.58
    } else if c.is_alphabetic() {
        0.60
    } else {
        0.35
    }
}

#[inline]
fn non_alpha_score(stats: &WordStats) -> f32 {
    if stats.digits == stats.len {
        0.70
    } else if stats.separators == stats.len {
        0.20
    } else {
        0.35
    }
}

#[inline]
fn run_length_factor(max_run: usize) -> f32 {
    if max_run <= 2 {
        1.0
    } else {
        let excess = (max_run - 2) as f32;
        (-0.38 * excess * excess).exp().max(0.04)
    }
}

#[inline]
fn bigram_diversity_factor(len: usize, bigrams: usize, unique_bigrams: usize) -> f32 {
    if len < 6 || bigrams == 0 {
        return 1.0;
    }

    let ratio = unique_bigrams as f32 / bigrams as f32;

    if ratio < 0.50 {
        (0.12 + 1.60 * ratio).clamp(0.18, 1.0)
    } else {
        1.0
    }
}

#[inline]
fn length_factor(len: usize) -> f32 {
    if len <= 24 {
        1.0
    } else {
        (-0.045 * (len - 24) as f32).exp().max(0.40)
    }
}

#[inline]
fn vowel_balance_factor(stats: &WordStats) -> f32 {
    if stats.latin < 4 {
        return 1.0;
    }

    let ratio = stats.latin_vowels as f32 / stats.latin as f32;
    let z = (ratio - 0.40) / 0.38;

    (-0.5 * z * z).exp().max(0.88)
}

#[inline]
fn no_vowel_factor(stats: &WordStats) -> f32 {
    if stats.latin >= 5 && stats.latin_vowels == 0 {
        0.38
    } else if stats.latin >= 9 && stats.latin_vowels <= 1 {
        0.90
    } else {
        1.0
    }
}

#[inline]
fn intrusion_factor(stats: &WordStats) -> f32 {
    let mut factor = 1.0_f32;

    if stats.digits > 0 && stats.alphabetic > 0 {
        factor *= (-0.40 * stats.digits as f32).exp().max(0.25);
    }

    if stats.separators > 1 {
        factor *= (-0.25 * (stats.separators - 1) as f32).exp().max(0.55);
    }

    factor
}

#[inline]
fn script_mixing_factor(script_mask: u8) -> f32 {
    if script_mask.count_ones() > 1 {
        0.35
    } else {
        1.0
    }
}

#[inline]
fn rare_letter_factor(stats: &WordStats) -> f32 {
    if stats.latin == 0 {
        return 1.0;
    }

    let rare_ratio = stats.rare as f32 / stats.latin as f32;

    if rare_ratio <= 0.22 {
        1.0
    } else {
        (-2.60 * (rare_ratio - 0.22)).exp().max(0.45)
    }
}

fn class_entropy_factor(stats: &WordStats) -> f32 {
    if stats.len < 6 {
        return 1.0;
    }

    let mut entropy = 0.0_f32;
    let n = stats.len as f32;

    for &count in &stats.class_counts {
        if count == 0 {
            continue;
        }

        let p = count as f32 / n;
        entropy -= p * p.ln();
    }

    let max_entropy = (stats.len.min(CLASS_COUNT) as f32).ln();
    if max_entropy <= f32::EPSILON {
        return 1.0;
    }

    let normalized = entropy / max_entropy;
    (0.55 + 0.45 * normalized).clamp(0.55, 1.0)
}

fn english_cluster_factor(word: &str) -> f32 {
    let mut hits = 0usize;

    for needle in ENGLISH_CLUSTERS {
        if ascii_contains_case_insensitive(word, needle) {
            hits += 1;
        }
    }

    (1.0 + ENGLISH_CLUSTER_BONUS_STEP * hits as f32).min(ENGLISH_CLUSTER_BONUS_CAP)
}

fn ascii_contains_case_insensitive(haystack: &str, needle: &[u8]) -> bool {
    let bytes = haystack.as_bytes();

    if needle.is_empty() {
        return true;
    }

    if bytes.len() < needle.len() {
        return false;
    }

    bytes.windows(needle.len()).any(|window| {
        window
            .iter()
            .zip(needle.iter())
            .all(|(&a, &b)| a.eq_ignore_ascii_case(&b))
    })
}

fn is_common_ascii_word(word: &str) -> bool {
    COMMON_WORDS
        .iter()
        .any(|candidate| word.eq_ignore_ascii_case(candidate))
}

#[inline]
fn script_bit(c: char) -> u8 {
    let cp = c as u32;

    if (0x0041..=0x024F).contains(&cp) {
        SCRIPT_LATIN
    } else if (0x0400..=0x052F).contains(&cp) {
        SCRIPT_CYRILLIC
    } else if is_cjk_or_hangul(cp) {
        SCRIPT_CJK
    } else if c.is_alphabetic() {
        SCRIPT_OTHER_ALPHA
    } else {
        0
    }
}

fn strip_contraction(word: &str) -> &str {
    for sep in ["'", "'", "'"] {
        if let Some(pos) = word.find(sep) {
            let suffix = &word[pos + sep.len()..];
            let s = suffix.to_ascii_lowercase();
            if matches!(
                s.as_str(),
                "m" | "re" | "ll" | "ve" | "d" | "s" | "t" | "nt"
            ) {
                return &word[..pos];
            }
        }
    }
    word
}

#[inline]
fn is_cjk_or_hangul(cp: u32) -> bool {
    (0x3000..=0x303F).contains(&cp)
        || (0x3040..=0x30FF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x4E00..=0x9FFF).contains(&cp)
        || (0xAC00..=0xD7AF).contains(&cp)
}

#[inline]
fn same_char_relaxed(a: char, b: char) -> bool {
    a.eq_ignore_ascii_case(&b)
}

#[inline]
fn hash_pair_chars(a: char, b: char) -> u8 {
    let mut x = (a as u32 as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (b as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);

    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;

    (x & 0x7F) as u8
}

fn hash_str_normalized(s: &str) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325_u64;

    for c in s.chars() {
        let c = c.to_ascii_lowercase() as u32 as u64;
        h ^= c;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }

    h
}

#[inline]
fn mix_u64_pair(a: u64, b: u64) -> u64 {
    let mut x = a.rotate_left(17) ^ b.wrapping_mul(0x9E37_79B1_85EB_CA87);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 29;
    x
}

#[inline]
fn bitset_insert_128(lo: &mut u64, hi: &mut u64, idx: u8) -> bool {
    if idx < 64 {
        let mask = 1_u64 << u32::from(idx);
        let is_new = (*lo & mask) == 0;
        *lo |= mask;
        is_new
    } else {
        let mask = 1_u64 << u32::from(idx - 64);
        let is_new = (*hi & mask) == 0;
        *hi |= mask;
        is_new
    }
}

#[inline]
fn bitset_insert_256(bits: &mut [u64; 4], idx: u8) -> bool {
    let bucket = (idx / 64) as usize;
    let offset = idx % 64;
    let mask = 1_u64 << u32::from(offset);
    let is_new = (bits[bucket] & mask) == 0;
    bits[bucket] |= mask;
    is_new
}

#[inline]
fn is_outer_token_punct(c: char) -> bool {
    c.is_ascii_punctuation() || matches!(c, '“' | '”' | '‘' | '’' | '…' | '«' | '»' | '‹' | '›')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_english() {
        for w in [
            "hello",
            "world",
            "transcription",
            "the",
            "beautiful",
            "string",
        ] {
            assert!(word_confidence(w) > 0.45, "{w}={}", word_confidence(w));
        }
    }

    #[test]
    fn stutter_hallucination() {
        assert!(word_confidence("aaaaa") < 0.3);
        assert!(word_confidence("bbbbb") < 0.3);
    }

    #[test]
    fn garbled_consonants() {
        assert!(
            word_confidence("bcdfghjk") < 0.4,
            "bcdfghjk={}",
            word_confidence("bcdfghjk")
        );
        assert!(word_confidence("xzqxzq") < 0.3);
        assert!(word_confidence("gxbzk") < 0.4);
    }

    #[test]
    fn digit_intrusion() {
        assert!(word_confidence("test123") < 0.6);
        assert!(word_confidence("12345") > 0.3);
    }

    #[test]
    fn excessive_length() {
        let long = "a".repeat(30);
        assert!(word_confidence(&long) < 0.2);
    }

    #[test]
    fn cjk_neutral() {
        assert!(word_confidence("你好") > 0.4);
        assert!(word_confidence("東京") > 0.4);
    }

    #[test]
    fn mixed_script() {
        assert!(word_confidence("приветhello") < 0.6);
        assert!(word_confidence("hello你好") < 0.6);
    }

    #[test]
    fn russian_neutral_positive() {
        assert!(
            word_confidence("привет") > 0.5,
            "привет={}",
            word_confidence("привет")
        );
        assert!(
            word_confidence("здравствуйте") > 0.4,
            "здравствуйте={}",
            word_confidence("здравствуйте")
        );
        assert!(
            word_confidence("москва") > 0.5,
            "москва={}",
            word_confidence("москва")
        );
    }

    #[test]
    fn french_german_spanish() {
        assert!(
            word_confidence("résumé") > 0.5,
            "résumé={}",
            word_confidence("résumé")
        );
        assert!(
            word_confidence("über") > 0.5,
            "über={}",
            word_confidence("über")
        );
        assert!(
            word_confidence("señor") > 0.5,
            "señor={}",
            word_confidence("señor")
        );
        assert!(
            word_confidence("café") > 0.5,
            "café={}",
            word_confidence("café")
        );
        assert!(
            word_confidence("straße") > 0.5,
            "straße={}",
            word_confidence("straße")
        );
    }

    #[test]
    fn repeated_words_penalty() {
        let s = segment_lexical_confidence("the the the the the");
        assert!(s < 0.5, "repeated={s}");
    }

    #[test]
    fn repeated_bigram_phrase_penalty() {
        let s = segment_lexical_confidence("go home go home go home go home");
        assert!(s < 0.8, "looped phrase={s}");
    }

    #[test]
    fn clean_segment() {
        let s = segment_lexical_confidence("the quick brown fox jumps");
        assert!(s > 0.5, "clean_seg={s}");
    }

    #[test]
    fn oscillation_pattern() {
        assert!(
            word_confidence("ababababab") < 0.5,
            "osc={}",
            word_confidence("ababababab")
        );
        assert!(
            word_confidence("cdcdcdcdcd") < 0.5,
            "osc={}",
            word_confidence("cdcdcdcdcd")
        );
    }

    #[test]
    fn empty() {
        assert_eq!(word_confidence(""), 0.0);
        assert_eq!(segment_lexical_confidence(""), 0.0);
    }

    #[test]
    fn markov_discriminates() {
        let natural = word_confidence("string");
        let garbled = word_confidence("gxbzk");
        assert!(
            natural > garbled * 2.0,
            "natural={natural} garbled={garbled}"
        );
    }

    #[test]
    fn common_words_high() {
        for w in [
            "I",
            "a",
            "is",
            "are",
            "have",
            "they",
            "which",
            "people",
            "information",
        ] {
            assert!(word_confidence(w) > 0.6, "{w}={}", word_confidence(w));
        }
    }

    #[test]
    fn hard_english_clusters_are_not_overpunished() {
        for w in [
            "strength", "through", "straight", "stretch", "script", "spring",
        ] {
            assert!(word_confidence(w) > 0.5, "{w}={}", word_confidence(w));
        }
    }

    #[test]
    fn value_sweep() {
        let good = [
            "the",
            "and",
            "that",
            "have",
            "with",
            "this",
            "will",
            "your",
            "from",
            "they",
            "been",
            "said",
            "each",
            "which",
            "their",
            "would",
            "about",
            "hello",
            "world",
            "transcription",
            "beautiful",
            "string",
            "people",
            "information",
            "understanding",
            "extraordinary",
            "communication",
            "book",
            "cool",
            "need",
            "feel",
            "well",
            "still",
            "across",
            "balloon",
            "strength",
            "through",
            "straight",
            "stretch",
            "script",
            "spring",
            "résumé",
            "café",
            "château",
            "être",
            "français",
            "garçon",
            "über",
            "straße",
            "mädchen",
            "größe",
            "schön",
            "würde",
            "señor",
            "niño",
            "año",
            "también",
            "después",
            "più",
            "perché",
            "città",
            "però",
            "não",
            "ação",
            "ração",
            "привет",
            "москва",
            "работа",
            "спасибо",
            "пожалуйста",
            "здравствуйте",
        ];

        let bad = [
            "aaaaa",
            "bbbbb",
            "eeeee",
            "bcdfghjk",
            "xzqxzq",
            "gxbzk",
            "kptkpt",
            "zxcvbn",
            "ababababab",
            "cdcdcdcdcd",
            "qqqxxx",
            "jjjkkk",
            "zzzzz",
            "приветhello",
            "hello你好",
            "xqzjkw",
            "bdfgkpt",
        ];

        let sep = "=".repeat(60);
        println!("\n{sep}");
        println!("  GOOD WORDS");
        println!("{sep}");

        let mut good_min = 1.0_f32;
        let mut good_sum = 0.0_f32;

        for w in &good {
            let s = word_confidence(w);
            good_min = good_min.min(s);
            good_sum += s;
            let flag = if s <= 0.5 { " *** LOW" } else { "" };
            println!("  {w:<25} {s:.4}{flag}");
        }

        let good_mean = good_sum / good.len() as f32;

        println!("\n{sep}");
        println!("  BAD WORDS");
        println!("{sep}");

        let mut bad_max = 0.0_f32;
        let mut bad_sum = 0.0_f32;

        for w in &bad {
            let s = word_confidence(w);
            bad_max = bad_max.max(s);
            bad_sum += s;
            let flag = if s >= 0.5 { " *** HIGH" } else { "" };
            println!("  {w:<25} {s:.4}{flag}");
        }

        let bad_mean = bad_sum / bad.len() as f32;
        let margin = good_min - bad_max;

        println!("\n{sep}");
        println!("  GOOD: min={good_min:.4}  mean={good_mean:.4}");
        println!(" BAD:  max={bad_max:.4}  mean={bad_mean:.4}");
        println!("  MARGIN: {margin:.4}");
        println!("{sep}");

        assert!(good_min > 0.45, "good_min too low: {good_min}");
        assert!(bad_max < 0.55, "bad_max too high: {bad_max}");
    }
}
