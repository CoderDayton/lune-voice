use super::entropy;
use super::scoring::{TokenInput, TokenSpan};

#[derive(Debug, Clone, Copy)]
pub struct WordConfidenceFeatures {
    pub mean_token_entropy: f32,
    pub min_token_entropy: f32,
    pub mean_top1_top2_margin: f32,
    pub word_length: u8,
    pub has_rare_casing: bool,
    pub is_proper_noun_candidate: bool,
    pub nearby_entropy_spike: bool,
    pub has_repetition: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenFeatures {
    pub entropy: f64,
    pub tsallis: f64,
    pub margin: f64,
}

const ENTROPY_SPIKE_THRESHOLD: f64 = 0.35;

pub(crate) fn extract_token_features(tokens: &[TokenInput]) -> Vec<TokenFeatures> {
    // softmax is the dominant cost per token; compute once and feed the
    // resulting probability slice to all three measures instead of paying
    // for three independent Vec<f64> allocations and three exp passes.
    tokens
        .iter()
        .map(|t| {
            let probs = entropy::softmax(&t.top_k_logits);
            TokenFeatures {
                entropy: entropy::shannon_entropy_from_probs(&probs),
                tsallis: entropy::tsallis_confidence_from_probs(&probs),
                margin: entropy::top_margin_from_probs(&probs),
            }
        })
        .collect()
}

pub(crate) fn extract_word_features(
    word: &str,
    span: TokenSpan,
    tokens: &[TokenInput],
    all_token_features: &[TokenFeatures],
) -> WordConfidenceFeatures {
    let tok_feats = &all_token_features[span.start..span.end];
    let n = tok_feats.len().max(1) as f64;

    let mean_token_entropy = (tok_feats.iter().map(|t| t.entropy).sum::<f64>() / n) as f32;

    // Min Tsallis confidence across tokens in the word.
    // Tsallis is 1=confident, 0=uncertain; invert so min confidence → max entropy risk.
    let min_tsallis = tok_feats
        .iter()
        .map(|t| t.tsallis)
        .fold(f64::INFINITY, f64::min);
    let min_token_entropy = (1.0 - min_tsallis).clamp(0.0, 1.0) as f32;

    let mean_top1_top2_margin = (tok_feats.iter().map(|t| t.margin).sum::<f64>() / n) as f32;

    let word_length = word.len().min(255) as u8;
    let has_rare_casing = detect_rare_casing(word);
    let is_proper_noun_candidate = detect_proper_noun(word);
    let nearby_entropy_spike = detect_nearby_spike(span, all_token_features);
    let has_repetition = detect_repetition(span, tokens);

    WordConfidenceFeatures {
        mean_token_entropy,
        min_token_entropy,
        mean_top1_top2_margin,
        word_length,
        has_rare_casing,
        is_proper_noun_candidate,
        nearby_entropy_spike,
        has_repetition,
    }
}

fn detect_rare_casing(word: &str) -> bool {
    if word.len() <= 1 {
        return false;
    }
    let has_upper = word.chars().any(|c| c.is_uppercase());
    let has_lower = word.chars().any(|c| c.is_lowercase());
    let has_digit = word.chars().any(|c| c.is_ascii_digit());
    let interior_upper = word.chars().skip(1).any(|c| c.is_uppercase()) && has_lower;
    has_digit
        || interior_upper
        || (has_upper && has_lower && word.chars().next().is_some_and(|c| c.is_lowercase()))
}

fn detect_proper_noun(word: &str) -> bool {
    let mut chars = word.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    first.is_uppercase() && chars.all(|c| c.is_lowercase())
}

fn detect_nearby_spike(span: TokenSpan, all_features: &[TokenFeatures]) -> bool {
    let check_before = span.start.saturating_sub(2)..span.start;
    let check_after = span.end..(span.end + 2).min(all_features.len());

    for i in check_before.chain(check_after) {
        if all_features[i].entropy > ENTROPY_SPIKE_THRESHOLD {
            return true;
        }
    }
    false
}

fn detect_repetition(span: TokenSpan, tokens: &[TokenInput]) -> bool {
    let slice = &tokens[span.start..span.end];
    if slice.len() < 2 {
        return false;
    }
    slice.windows(2).any(|w| w[0].token_id == w[1].token_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rare_casing_detection() {
        assert!(!detect_rare_casing("hello"));
        assert!(!detect_rare_casing("Hello"));
        assert!(detect_rare_casing("hEllo"));
        assert!(detect_rare_casing("hello2"));
        assert!(!detect_rare_casing("I"));
        assert!(!detect_rare_casing(""));
    }

    #[test]
    fn proper_noun_detection() {
        assert!(detect_proper_noun("London"));
        assert!(!detect_proper_noun("london"));
        assert!(!detect_proper_noun("LONDON"));
        assert!(!detect_proper_noun(""));
    }

    #[test]
    fn repetition_detection() {
        let tokens = vec![
            TokenInput {
                token_id: 1,
                confidence: 0.9,
                top_k_logits: vec![],
            },
            TokenInput {
                token_id: 1,
                confidence: 0.9,
                top_k_logits: vec![],
            },
        ];
        let span = TokenSpan { start: 0, end: 2 };
        assert!(detect_repetition(span, &tokens));

        let tokens2 = vec![
            TokenInput {
                token_id: 1,
                confidence: 0.9,
                top_k_logits: vec![],
            },
            TokenInput {
                token_id: 2,
                confidence: 0.9,
                top_k_logits: vec![],
            },
        ];
        assert!(!detect_repetition(span, &tokens2));
    }
}
