use std::collections::HashMap;
use std::sync::LazyLock;

// ── AlphabetMap ──────────────────────────────────────────────────────────

struct AlphabetMap {
    costs: HashMap<(char, char), f32>,
}

impl AlphabetMap {
    fn build() -> Self {
        let mut costs = HashMap::new();
        let mut insert = |a: char, b: char, cost: f32| {
            let key_ab = (a, b);
            let key_ba = (b, a);
            let e = costs.entry(key_ab).or_insert(cost);
            if cost < *e {
                *e = cost;
            }
            let e = costs.entry(key_ba).or_insert(cost);
            if cost < *e {
                *e = cost;
            }
        };

        // Axis 1: QWERTY proximity
        for &(a, b, c) in QWERTY_PAIRS {
            insert(a, b, c);
        }
        // Axis 2: Phonetic equivalences
        for &(a, b, c) in PHONETIC_PAIRS {
            insert(a, b, c);
        }
        // Axis 3: Visual lookalikes
        for &(a, b, c) in VISUAL_PAIRS {
            insert(a, b, c);
        }
        // Axis 4: Unicode/multilingual
        for &(from, to, c) in UNICODE_MAPPINGS {
            insert(from, to, c);
        }

        Self { costs }
    }

    fn sub_cost(&self, a: char, b: char) -> f32 {
        if a == b {
            return 0.0;
        }
        if a.eq_ignore_ascii_case(&b) {
            return 0.05;
        }
        self.costs
            .get(&(a.to_ascii_lowercase(), b.to_ascii_lowercase()))
            .copied()
            .unwrap_or(1.0)
    }
}

static ALPHABET: LazyLock<AlphabetMap> = LazyLock::new(AlphabetMap::build);

const QWERTY_PAIRS: &[(char, char, f32)] = &[
    ('q', 'w', 0.35),
    ('w', 'e', 0.35),
    ('e', 'r', 0.35),
    ('r', 't', 0.35),
    ('t', 'y', 0.35),
    ('y', 'u', 0.35),
    ('u', 'i', 0.35),
    ('i', 'o', 0.35),
    ('o', 'p', 0.35),
    ('a', 's', 0.35),
    ('s', 'd', 0.35),
    ('d', 'f', 0.35),
    ('f', 'g', 0.35),
    ('g', 'h', 0.35),
    ('h', 'j', 0.35),
    ('j', 'k', 0.35),
    ('k', 'l', 0.35),
    ('z', 'x', 0.35),
    ('x', 'c', 0.35),
    ('c', 'v', 0.35),
    ('v', 'b', 0.35),
    ('b', 'n', 0.35),
    ('n', 'm', 0.35),
    // vertical
    ('q', 'a', 0.40),
    ('w', 's', 0.40),
    ('e', 'd', 0.40),
    ('r', 'f', 0.40),
    ('t', 'g', 0.40),
    ('y', 'h', 0.40),
    ('u', 'j', 0.40),
    ('i', 'k', 0.40),
    ('o', 'l', 0.40),
    ('a', 'z', 0.40),
    ('s', 'x', 0.40),
    ('d', 'c', 0.40),
    ('f', 'v', 0.40),
    ('g', 'b', 0.40),
    ('h', 'n', 0.40),
    ('j', 'm', 0.40),
    // diagonal
    ('q', 's', 0.45),
    ('w', 'd', 0.45),
    ('e', 'f', 0.45),
    ('r', 'g', 0.45),
    ('t', 'h', 0.45),
    ('y', 'j', 0.45),
    ('u', 'k', 0.45),
    ('i', 'l', 0.45),
    ('w', 'a', 0.45),
    ('e', 's', 0.45),
    ('r', 'd', 0.45),
    ('t', 'f', 0.45),
    ('y', 'g', 0.45),
    ('u', 'h', 0.45),
    ('i', 'j', 0.45),
    ('o', 'k', 0.45),
    ('p', 'l', 0.45),
    ('s', 'z', 0.45),
    ('d', 'x', 0.45),
    ('f', 'c', 0.45),
    ('g', 'v', 0.45),
    ('h', 'b', 0.45),
    ('j', 'n', 0.45),
    ('k', 'm', 0.45),
];

const PHONETIC_PAIRS: &[(char, char, f32)] = &[
    ('c', 'k', 0.25),
    ('c', 's', 0.30),
    ('f', 'v', 0.35),
    ('z', 's', 0.30),
    ('g', 'j', 0.35),
    ('b', 'p', 0.40),
    ('d', 't', 0.40),
    ('m', 'n', 0.35),
    ('v', 'w', 0.40),
    ('i', 'y', 0.35),
    ('s', 'z', 0.30),
    ('x', 'z', 0.40),
    ('f', 'p', 0.45),
    ('g', 'k', 0.35),
    ('b', 'v', 0.40),
    ('c', 'z', 0.45),
    ('d', 'b', 0.45),
    ('l', 'r', 0.40),
    ('n', 'g', 0.45),
    ('a', 'e', 0.50),
    ('e', 'i', 0.50),
    ('o', 'u', 0.50),
    ('a', 'o', 0.50),
    ('i', 'e', 0.50),
];

const VISUAL_PAIRS: &[(char, char, f32)] = &[
    ('i', 'l', 0.40),
    ('l', '1', 0.40),
    ('o', '0', 0.40),
    ('i', '1', 0.45),
    ('s', '5', 0.45),
    ('b', '6', 0.50),
    ('g', '9', 0.50),
    ('z', '2', 0.50),
];

const UNICODE_MAPPINGS: &[(char, char, f32)] = &[
    // Latin diacritics → base
    ('à', 'a', 0.20),
    ('á', 'a', 0.20),
    ('â', 'a', 0.20),
    ('ã', 'a', 0.20),
    ('ä', 'a', 0.20),
    ('å', 'a', 0.20),
    ('è', 'e', 0.20),
    ('é', 'e', 0.20),
    ('ê', 'e', 0.20),
    ('ë', 'e', 0.20),
    ('ì', 'i', 0.20),
    ('í', 'i', 0.20),
    ('î', 'i', 0.20),
    ('ï', 'i', 0.20),
    ('ò', 'o', 0.20),
    ('ó', 'o', 0.20),
    ('ô', 'o', 0.20),
    ('õ', 'o', 0.20),
    ('ö', 'o', 0.20),
    ('ù', 'u', 0.20),
    ('ú', 'u', 0.20),
    ('û', 'u', 0.20),
    ('ü', 'u', 0.20),
    ('ñ', 'n', 0.20),
    ('ç', 'c', 0.20),
    ('ÿ', 'y', 0.20),
    ('ð', 'd', 0.20),
    ('ø', 'o', 0.20),
    ('æ', 'a', 0.20),
    ('ß', 's', 0.20),
    // Cyrillic visual confusables
    ('а', 'a', 0.15),
    ('е', 'e', 0.15),
    ('о', 'o', 0.15),
    ('р', 'p', 0.15),
    ('с', 'c', 0.15),
    ('у', 'y', 0.15),
    ('х', 'x', 0.15),
    ('і', 'i', 0.15),
    ('ј', 'j', 0.15),
    ('ѕ', 's', 0.15),
    // Greek phonetic/visual
    ('α', 'a', 0.25),
    ('β', 'b', 0.30),
    ('γ', 'g', 0.35),
    ('δ', 'd', 0.35),
    ('ε', 'e', 0.25),
    ('ζ', 'z', 0.35),
    ('η', 'n', 0.40),
    ('θ', 't', 0.40),
    ('ι', 'i', 0.25),
    ('κ', 'k', 0.30),
    ('λ', 'l', 0.35),
    ('μ', 'm', 0.30),
    ('ν', 'n', 0.30),
    ('ξ', 'x', 0.40),
    ('ο', 'o', 0.25),
    ('π', 'p', 0.30),
    ('ρ', 'r', 0.30),
    ('σ', 's', 0.30),
    ('τ', 't', 0.30),
    ('υ', 'u', 0.30),
    ('φ', 'f', 0.35),
    ('χ', 'x', 0.35),
    ('ψ', 'p', 0.40),
    ('ω', 'w', 0.35),
    // Fullwidth ASCII
    ('ａ', 'a', 0.10),
    ('ｂ', 'b', 0.10),
    ('ｃ', 'c', 0.10),
    ('ｄ', 'd', 0.10),
    ('ｅ', 'e', 0.10),
    ('ｆ', 'f', 0.10),
    ('ｇ', 'g', 0.10),
    ('ｈ', 'h', 0.10),
    ('ｉ', 'i', 0.10),
    ('ｊ', 'j', 0.10),
    ('ｋ', 'k', 0.10),
    ('ｌ', 'l', 0.10),
    ('ｍ', 'm', 0.10),
    ('ｎ', 'n', 0.10),
    ('ｏ', 'o', 0.10),
    ('ｐ', 'p', 0.10),
    ('ｑ', 'q', 0.10),
    ('ｒ', 'r', 0.10),
    ('ｓ', 's', 0.10),
    ('ｔ', 't', 0.10),
    ('ｕ', 'u', 0.10),
    ('ｖ', 'v', 0.10),
    ('ｗ', 'w', 0.10),
    ('ｘ', 'x', 0.10),
    ('ｙ', 'y', 0.10),
    ('ｚ', 'z', 0.10),
];

// ── PhoneticNormalizer ───────────────────────────────────────────────────

static PHONETIC_RULES: LazyLock<Vec<(&'static str, &'static str)>> = LazyLock::new(|| {
    vec![
        ("ph", "f"),
        ("ck", "k"),
        ("kn", "n"),
        ("tion", "shn"),
        ("sion", "shn"),
        ("wr", "r"),
        ("wh", "w"),
        ("gh", ""),
        ("ght", "t"),
        ("dge", "j"),
        ("tch", "ch"),
        ("qu", "kw"),
        ("sch", "sk"),
        ("sc", "s"),
        ("gn", "n"),
    ]
});

pub fn phonetic_key(word: &str) -> String {
    let mut s = word.to_ascii_lowercase();
    for &(from, to) in PHONETIC_RULES.iter() {
        s = s.replace(from, to);
    }
    // collapse doubles
    let mut out = String::with_capacity(s.len());
    let mut prev = '\0';
    for c in s.chars() {
        if c != prev {
            out.push(c);
            prev = c;
        }
    }
    out
}

// ── WED (Weighted Edit Distance) ────────────────────────────────────────

const DEL_COST: f32 = 0.80;
const INS_COST: f32 = 0.80;
const TRANS_COST: f32 = 0.25;

pub fn wed(a: &str, b: &str) -> f32 {
    wed_inner(a, b, f32::INFINITY).unwrap_or(f32::INFINITY)
}

pub fn wed_with_limit(a: &str, b: &str, max_score: f32) -> Option<f32> {
    wed_inner(a, b, max_score)
}

fn wed_inner(a: &str, b: &str, max_score: f32) -> Option<f32> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    let raw_limit = max_score + 0.35;

    if (m.abs_diff(n) as f32) * DEL_COST > raw_limit {
        return None;
    }

    let mut prev_prev = vec![0.0f32; n + 1];
    let mut prev = vec![0.0f32; n + 1];
    let mut curr = vec![0.0f32; n + 1];

    for (j, slot) in prev.iter_mut().enumerate().take(n + 1) {
        *slot = j as f32 * INS_COST;
    }

    for i in 1..=m {
        curr[0] = i as f32 * DEL_COST;
        let mut row_min = curr[0];
        for j in 1..=n {
            let sub = prev[j - 1] + ALPHABET.sub_cost(a_chars[i - 1], b_chars[j - 1]);
            let del = prev[j] + DEL_COST;
            let ins = curr[j - 1] + INS_COST;
            let mut best = sub.min(del).min(ins);

            if i >= 2
                && j >= 2
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                best = best.min(prev_prev[j - 2] + TRANS_COST);
            }

            curr[j] = best;
            row_min = row_min.min(best);
        }
        if row_min > raw_limit {
            return None;
        }
        prev_prev.copy_from_slice(&prev);
        std::mem::swap(&mut prev, &mut curr);
    }

    let raw = prev[n];
    let pk_a = phonetic_key(a);
    let pk_b = phonetic_key(b);
    let score = if pk_a == pk_b {
        (raw - 0.35).max(0.0)
    } else {
        raw
    };
    if score <= max_score {
        Some(score)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wed_teh_the_vs_ten() {
        let d1 = wed("teh", "the");
        let d2 = wed("teh", "ten");
        assert!(
            d1 < d2,
            "teh→the ({d1}) should be cheaper than teh→ten ({d2})"
        );
    }

    #[test]
    fn limited_wed_matches_full_score_for_close_candidates() {
        for (a, b) in [
            ("teh", "the"),
            ("brwon", "brown"),
            ("hllo", "hello"),
            ("fone", "phone"),
            ("conection", "connection"),
        ] {
            assert_eq!(wed_with_limit(a, b, 2.5).unwrap(), wed(a, b));
        }
    }

    #[test]
    fn limited_wed_rejects_distant_candidates() {
        assert!(wed_with_limit("teh", "algorithm", 2.5).is_none());
    }

    #[test]
    fn limited_wed_rejects_length_gap_before_dp() {
        assert!(wed_with_limit("a", "encyclopedia", 2.5).is_none());
    }

    #[test]
    fn phonetic_key_works() {
        assert_eq!(phonetic_key("phone"), phonetic_key("fone"));
        assert_eq!(phonetic_key("knight"), phonetic_key("nit"));
    }
}
