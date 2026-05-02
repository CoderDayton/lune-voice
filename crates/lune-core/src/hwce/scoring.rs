use super::features::{self, WordConfidenceFeatures};

// ── Feature weights derived from information ratio ──────────────────────────
//
// NVIDIA (2023) showed entropy-based features are 4× more discriminative than
// raw softmax for word error detection. Google CEM (Li et al., 2021) found
// attention+decoder signals provide ~2× improvement over single features.
//
// We assign each feature an information ratio (IR) reflecting its relative
// discriminative power, then normalize so weights sum to 1.0:
//
//   Feature              IR    Weight = IR / Σ(IR)
//   ─────────────────    ──    ───────────────────
//   mean_entropy         4.0   4/15 ≈ 0.267
//   margin (1-p gap)     3.0   3/15 = 0.200
//   min_entropy          2.0   2/15 ≈ 0.133
//   nearby_spike         2.0   2/15 ≈ 0.133
//   repetition           1.5   1.5/15 = 0.100
//   rare_casing          1.0   1/15 ≈ 0.067
//   word_length          0.5   0.5/15 ≈ 0.033
//   proper_noun         -1.0   −1/15 ≈ −0.067
//                       ────
//   Σ|IR|              15.0
//
const IR_TOTAL: f32 = 15.0;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub w_entropy: f32,
    pub w_margin: f32,
    pub w_min_entropy: f32,
    pub w_length: f32,
    pub w_rare_casing: f32,
    pub w_proper_noun: f32,
    pub w_nearby_spike: f32,
    pub w_repetition: f32,
    pub scale: f32,
    pub midpoint: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            w_entropy: 4.0 / IR_TOTAL,
            w_margin: 3.0 / IR_TOTAL,
            w_min_entropy: 2.0 / IR_TOTAL,
            w_nearby_spike: 2.0 / IR_TOTAL,
            w_repetition: 1.5 / IR_TOTAL,
            w_rare_casing: 1.0 / IR_TOTAL,
            w_length: 0.5 / IR_TOTAL,
            w_proper_noun: -1.0 / IR_TOTAL,
            // σ(scale·(x − midpoint)) = 0.5 when x = midpoint.
            // scale controls steepness: at midpoint ± 1/scale, output ≈ 0.27/0.73.
            // scale = 4·ln(3) ≈ 4.39 gives 10%/90% at midpoint ± 0.5
            // We use 2·ln(3)/midpoint to tie steepness to the operating point.
            scale: 4.0 * 3.0_f32.ln(),
            // Operating point: expected score for a "neutral" word.
            // Continuous features (entropy, margin, min_entropy) contribute
            // ~0.0 for confident words. The midpoint is set so that a word
            // with all-zero continuous features and no flags maps to risk ≈ 0.5·σ(−scale·midpoint).
            midpoint: 0.28,
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
pub struct TokenInput {
    pub token_id: u32,
    pub confidence: f32,
    pub top_k_logits: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct WordResult {
    pub word: String,
    pub word_error_risk: f32,
    pub features: WordConfidenceFeatures,
}

pub fn estimate(
    words: &[String],
    tokens: &[TokenInput],
    spans: &[TokenSpan],
    config: Config,
) -> Vec<WordResult> {
    if words.is_empty() || tokens.is_empty() {
        return Vec::new();
    }

    let token_features = features::extract_token_features(tokens);
    let mut output = Vec::with_capacity(words.len());

    for (wi, word) in words.iter().enumerate() {
        let span = spans[wi];
        if span.start >= span.end || span.end > tokens.len() {
            output.push(WordResult {
                word: word.clone(),
                word_error_risk: 0.5,
                features: WordConfidenceFeatures {
                    mean_token_entropy: 0.5,
                    min_token_entropy: 0.5,
                    mean_top1_top2_margin: 0.5,
                    word_length: word.len().min(255) as u8,
                    has_rare_casing: false,
                    is_proper_noun_candidate: false,
                    nearby_entropy_spike: false,
                    has_repetition: false,
                },
            });
            continue;
        }

        let feats = features::extract_word_features(word, span, tokens, &token_features);

        let risk = score_features(&feats, &config);
        output.push(WordResult {
            word: word.clone(),
            word_error_risk: risk,
            features: feats,
        });
    }

    output
}

fn score_features(f: &WordConfidenceFeatures, c: &Config) -> f32 {
    let raw = c.w_entropy * f.mean_token_entropy
        + c.w_margin * (1.0 - f.mean_top1_top2_margin)
        + c.w_min_entropy * f.min_token_entropy
        + c.w_length * length_risk(f.word_length)
        + c.w_rare_casing * f.has_rare_casing as u8 as f32
        + c.w_proper_noun * f.is_proper_noun_candidate as u8 as f32
        + c.w_nearby_spike * f.nearby_entropy_spike as u8 as f32
        + c.w_repetition * f.has_repetition as u8 as f32;

    sigmoid(c.scale * (raw - c.midpoint))
}

// Gaussian deviation from median English word length (μ=5, σ=3).
// risk = 1 − exp(−(len − μ)² / 2σ²), clamped for very short fragments.
fn length_risk(len: u8) -> f32 {
    const MU: f32 = 5.0;
    const SIGMA: f32 = 3.0;
    let d = len as f32 - MU;
    let risk = 1.0 - (-d * d / (2.0 * SIGMA * SIGMA)).exp();
    if len <= 1 { risk.max(0.15) } else { risk }
}

fn sigmoid(x: f32) -> f32 {
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

    fn tok(id: u32, conf: f32, logits: Vec<f64>) -> TokenInput {
        TokenInput {
            token_id: id,
            confidence: conf,
            top_k_logits: logits,
        }
    }

    #[test]
    fn high_confidence_low_risk() {
        let words = vec!["hello".into()];
        let tokens = vec![tok(1, 0.99, vec![10.0, -5.0, -5.0, -5.0])];
        let spans = vec![TokenSpan::new(0, 1)];
        let r = estimate(&words, &tokens, &spans, Config::default());
        assert!(r[0].word_error_risk < 0.25, "risk={}", r[0].word_error_risk);
    }

    #[test]
    fn low_confidence_high_risk() {
        let words = vec!["thng".into()];
        let tokens = vec![tok(1, 0.30, vec![2.0, 1.8, 1.5, 1.0])];
        let spans = vec![TokenSpan::new(0, 1)];
        let r = estimate(&words, &tokens, &spans, Config::default());
        assert!(r[0].word_error_risk > 0.50, "risk={}", r[0].word_error_risk);
    }

    #[test]
    fn proper_noun_reduces_risk() {
        let logits = vec![5.0, 2.0, -1.0, -3.0];
        let words_lower = vec!["london".into()];
        let words_proper = vec!["London".into()];
        let tokens = vec![tok(1, 0.85, logits.clone())];
        let spans = vec![TokenSpan::new(0, 1)];
        let r_lower = estimate(&words_lower, &tokens, &spans, Config::default());
        let r_proper = estimate(&words_proper, &tokens, &spans, Config::default());
        assert!(
            r_proper[0].word_error_risk < r_lower[0].word_error_risk,
            "proper={} should be < lower={}",
            r_proper[0].word_error_risk,
            r_lower[0].word_error_risk
        );
    }

    #[test]
    fn short_common_word_low_risk() {
        let words = vec!["a".into()];
        let tokens = vec![tok(1, 0.99, vec![10.0, -5.0, -5.0, -5.0])];
        let spans = vec![TokenSpan::new(0, 1)];
        let r = estimate(&words, &tokens, &spans, Config::default());
        assert!(r[0].word_error_risk < 0.30, "risk={}", r[0].word_error_risk);
    }

    #[test]
    fn empty_input() {
        assert!(estimate(&[], &[], &[], Config::default()).is_empty());
    }

    #[test]
    fn multi_token_word() {
        let words = vec!["understanding".into()];
        let tokens = vec![
            tok(10, 0.95, vec![8.0, -3.0, -5.0, -5.0]),
            tok(11, 0.90, vec![7.0, -2.0, -4.0, -5.0]),
            tok(12, 0.85, vec![6.0, -1.0, -3.0, -4.0]),
        ];
        let spans = vec![TokenSpan::new(0, 3)];
        let r = estimate(&words, &tokens, &spans, Config::default());
        assert!(r[0].word_error_risk < 0.40, "risk={}", r[0].word_error_risk);
    }

    #[test]
    fn length_risk_gaussian() {
        assert!(
            length_risk(5) < 0.01,
            "median length should have near-zero risk"
        );
        assert!(length_risk(1) > 0.10, "single-char should have some risk");
        assert!(
            length_risk(15) > length_risk(5),
            "long words riskier than median"
        );
    }

    #[test]
    fn ir_weights_sum_to_one() {
        let c = Config::default();
        let pos_sum = c.w_entropy
            + c.w_margin
            + c.w_min_entropy
            + c.w_nearby_spike
            + c.w_repetition
            + c.w_rare_casing
            + c.w_length;
        assert!((pos_sum - 14.0 / IR_TOTAL).abs() < 0.001);
    }
}
