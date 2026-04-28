//! HERC: Herfindahl-Echo Recursive Confidence
//!
//! Word-level ASR confidence from full-vocab softmax + top-k entropy.
//!
//!   fix_score = σ(scale · (midpoint - r_word + β · ces_word))
//!
//! Sigmoid mapping creates a sharp cutoff: words with confidence above
//! midpoint get low fix_scores, below get high. CES shifts the threshold
//! for "torn" distributions (moderate confidence + high entropy).

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub struct HercConfig {
    pub scale: f64,
    pub midpoint: f64,
    pub ces_beta: f64,
}

impl Default for HercConfig {
    fn default() -> Self {
        Self {
            scale: 10.0,
            midpoint: 0.40,
            ces_beta: 1.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

impl TokenSpan {
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone)]
pub struct WordConfidence {
    pub word: String,
    pub confidence: f64,
    pub fix_score: f64,
    pub reliability: f64,
    pub ces: f64,
}

#[derive(Debug, Clone)]
pub struct TokenInput {
    pub confidence: f32,
    pub top_k_logits: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct TokenEvidence {
    reliability: f64,
    ces: f64,
}

pub fn estimate(
    words: &[String],
    tokens: &[TokenInput],
    token_spans: &[TokenSpan],
    config: HercConfig,
) -> Vec<WordConfidence> {
    if words.is_empty() || tokens.is_empty() {
        return Vec::new();
    }

    let evidence: Vec<TokenEvidence> = tokens
        .iter()
        .map(|tok| {
            let r = (tok.confidence as f64).clamp(0.0, 1.0);
            let h_norm = normalized_entropy(&tok.top_k_logits);
            let ces = h_norm * r * (1.0 - r);
            TokenEvidence {
                reliability: r,
                ces,
            }
        })
        .collect();

    let mut output = Vec::with_capacity(words.len());

    for (wi, word) in words.iter().enumerate() {
        let span = token_spans[wi];
        let toks = &evidence[span.start..span.end];
        if toks.is_empty() {
            output.push(WordConfidence {
                word: word.clone(),
                confidence: 0.5,
                fix_score: 0.5,
                reliability: 0.0,
                ces: 0.0,
            });
            continue;
        }

        let r_word = geometric_mean(toks.iter().map(|e| e.reliability));
        let ces_word = toks.iter().map(|e| e.ces).fold(0.0f64, f64::max);

        let x = config.scale * (config.midpoint - r_word + config.ces_beta * ces_word);
        let fix_score = sigmoid(x);
        let confidence = (1.0 - fix_score).clamp(0.01, 0.99);

        output.push(WordConfidence {
            word: word.clone(),
            confidence,
            fix_score,
            reliability: r_word,
            ces: ces_word,
        });
    }

    output
}

fn normalized_entropy(logits: &[f64]) -> f64 {
    if logits.len() <= 1 {
        return 0.0;
    }
    let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return 0.0;
    }
    let h: f64 = exps
        .iter()
        .map(|&e| {
            let p = e / sum;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();
    (h / (logits.len() as f64).ln()).clamp(0.0, 1.0)
}

fn geometric_mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum_log = 0.0f64;
    let mut count = 0usize;
    for v in values {
        sum_log += (v + EPS).ln();
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    (sum_log / count as f64).exp()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(conf: f32, logits: Vec<f64>) -> TokenInput {
        TokenInput {
            confidence: conf,
            top_k_logits: logits,
        }
    }

    #[test]
    fn high_confidence_low_fix() {
        let words = vec!["hello".into()];
        let tokens = vec![tok(0.95, vec![10.0, -5.0, -5.0, -5.0])];
        let spans = vec![TokenSpan::new(0, 1)];
        let r = estimate(&words, &tokens, &spans, HercConfig::default());
        assert!(r[0].fix_score < 0.01, "fix={}", r[0].fix_score);
    }

    #[test]
    fn low_confidence_high_fix() {
        let words = vec!["garble".into()];
        let tokens = vec![tok(0.05, vec![1.0, 0.9, 0.8, 0.7])];
        let spans = vec![TokenSpan::new(0, 1)];
        let r = estimate(&words, &tokens, &spans, HercConfig::default());
        assert!(r[0].fix_score > 0.95, "fix={}", r[0].fix_score);
    }

    #[test]
    fn moderate_confidence_not_flagged() {
        let words = vec!["place".into()];
        let tokens = vec![tok(0.60, vec![5.0, 2.0, -1.0, -3.0])];
        let spans = vec![TokenSpan::new(0, 1)];
        let r = estimate(&words, &tokens, &spans, HercConfig::default());
        assert!(r[0].fix_score < 0.20, "fix={}", r[0].fix_score);
    }

    #[test]
    fn torn_model_gets_flagged() {
        let words = vec!["torn".into()];
        let tokens = vec![tok(0.45, vec![2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3])];
        let spans = vec![TokenSpan::new(0, 1)];
        let r = estimate(&words, &tokens, &spans, HercConfig::default());
        assert!(r[0].fix_score > 0.60, "fix={}", r[0].fix_score);
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(estimate(&[], &[], &[], HercConfig::default()).is_empty());
    }
}
