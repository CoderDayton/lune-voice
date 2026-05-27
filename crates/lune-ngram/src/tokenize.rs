//! Tokenization, acoustic shape hashing, and string similarity helpers.

use strsim::{jaro_winkler, normalized_damerau_levenshtein};

use crate::{END, START};

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect()
}

pub fn acoustic_shape(text: &str) -> String {
    let mut out = String::new();
    let mut last = '\0';

    for ch in text.to_ascii_lowercase().chars() {
        let class = match ch {
            'a' | 'e' | 'i' | 'o' | 'u' | 'y' => 'A',
            'b' | 'p' => 'P',
            'd' | 't' => 'T',
            'g' | 'k' | 'q' => 'K',
            'c' | 's' | 'z' => 'S',
            'f' | 'v' => 'F',
            'm' | 'n' => 'N',
            'r' | 'l' => 'R',
            'j' => 'J',
            'h' | 'w' => 'H',
            'x' => 'X',
            '0'..='9' => 'D',
            _ => continue,
        };

        if class != last {
            out.push(class);
            last = class;
        }
    }

    out
}

pub(crate) fn normalize_word_or_boundary(word: &str) -> String {
    match word {
        START => START.to_owned(),
        END => END.to_owned(),
        _ => word
            .to_ascii_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect(),
    }
}

pub(crate) fn last_or_start(prefix: &[String]) -> &str {
    prefix.last().map(String::as_str).unwrap_or(START)
}

pub(crate) fn last_two_or_start(prefix: &[String]) -> [String; 2] {
    match prefix.len() {
        0 => [START.to_owned(), START.to_owned()],
        1 => [START.to_owned(), prefix[0].clone()],
        len => [prefix[len - 2].clone(), prefix[len - 1].clone()],
    }
}

pub(crate) fn minmax_normalize(values: &[f64]) -> Vec<f64> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.is_finite() {
        return vec![0.0; values.len()];
    }
    let span = max - min;

    values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                0.0
            } else if span <= f64::EPSILON {
                0.5
            } else {
                ((value - min) / span).clamp(0.0, 1.0)
            }
        })
        .collect()
}

pub(crate) fn normalize_similarity_text(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

pub(crate) fn lexical_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let jaro = jaro_winkler(a, b);
    let damerau = normalized_damerau_levenshtein(a, b);
    (0.60 * jaro + 0.40 * damerau).clamp(0.0, 1.0)
}
