const TSALLIS_ALPHA: f64 = 1.0 / 3.0;

pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = probs.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

/// Normalized Shannon entropy from precomputed softmax probabilities.
pub(crate) fn shannon_entropy_from_probs(probs: &[f64]) -> f64 {
    let k = probs.len();
    if k <= 1 {
        return 0.0;
    }
    let h: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    (h / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Tsallis confidence from precomputed softmax probabilities.
pub(crate) fn tsallis_confidence_from_probs(probs: &[f64]) -> f64 {
    let k = probs.len();
    if k <= 1 {
        return 1.0;
    }

    let sum_p_alpha: f64 = probs.iter().map(|&p| p.powf(TSALLIS_ALPHA)).sum();
    let k_term = (k as f64).powf(1.0 - TSALLIS_ALPHA);
    let scale = 1.0 / (1.0 - TSALLIS_ALPHA);

    let raw = scale * (k_term - sum_p_alpha);
    let max_raw = scale * (k_term - 1.0);

    if max_raw <= 0.0 {
        return 1.0;
    }

    let numer = raw.exp() - 1.0;
    let denom = max_raw.exp() - 1.0;
    if denom <= 0.0 {
        return 1.0;
    }
    (numer / denom).clamp(0.0, 1.0)
}

/// Top-1 vs top-2 margin from precomputed softmax probabilities.
pub(crate) fn top_margin_from_probs(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let mut top1 = f64::NEG_INFINITY;
    let mut top2 = f64::NEG_INFINITY;
    for &p in probs {
        if p > top1 {
            top2 = top1;
            top1 = p;
        } else if p > top2 {
            top2 = p;
        }
    }
    (top1 - top2).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaked_distribution_high_confidence() {
        let p = softmax(&[10.0, -5.0, -5.0, -5.0]);
        assert!(tsallis_confidence_from_probs(&p) > 0.95);
        assert!(shannon_entropy_from_probs(&p) < 0.05);
        assert!(top_margin_from_probs(&p) > 0.95);
    }

    #[test]
    fn uniform_distribution_low_confidence() {
        let p = softmax(&[0.0, 0.0, 0.0, 0.0]);
        assert!(tsallis_confidence_from_probs(&p) < 0.05);
        assert!(shannon_entropy_from_probs(&p) > 0.95);
        assert!(top_margin_from_probs(&p) < 0.05);
    }

    #[test]
    fn moderate_distribution() {
        let p = softmax(&[3.0, 1.0, -1.0, -3.0]);
        let tc = tsallis_confidence_from_probs(&p);
        let se = shannon_entropy_from_probs(&p);
        assert!(tc > 0.1 && tc < 0.95, "tsallis={tc}");
        assert!(se > 0.1 && se < 0.7, "shannon={se}");
    }

    #[test]
    fn empty_logits() {
        let p = softmax(&[]);
        assert_eq!(shannon_entropy_from_probs(&p), 0.0);
        assert_eq!(tsallis_confidence_from_probs(&p), 1.0);
        assert_eq!(top_margin_from_probs(&p), 1.0);
    }
}
