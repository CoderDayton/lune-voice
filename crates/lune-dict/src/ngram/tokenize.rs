//! Tokenization, acoustic shape hashing, and string similarity helpers.

use super::{END, START};

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
