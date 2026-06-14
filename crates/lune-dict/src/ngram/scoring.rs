//! N-gram interpolation weights and scoring configuration.

use serde::{Deserialize, Serialize};

use super::AnyError;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InterpolationWeights {
    pub unigram: f64,
    pub bigram: f64,
    pub trigram: f64,
}

impl InterpolationWeights {
    pub fn new(unigram: f64, bigram: f64, trigram: f64) -> Result<Self, AnyError> {
        if unigram < 0.0 || bigram < 0.0 || trigram < 0.0 {
            return Err("interpolation weights must be non-negative".into());
        }
        let total = unigram + bigram + trigram;
        if total <= f64::EPSILON {
            return Err("at least one interpolation weight must be positive".into());
        }
        Ok(Self {
            unigram: unigram / total,
            bigram: bigram / total,
            trigram: trigram / total,
        })
    }

    pub fn transcript_default() -> Self {
        Self {
            unigram: 0.10,
            bigram: 0.30,
            trigram: 0.60,
        }
    }

    pub fn bigram_default() -> Self {
        Self {
            unigram: 0.20,
            bigram: 0.80,
            trigram: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoringConfig {
    pub max_order: usize,
    pub add_k: f64,
    pub weights: InterpolationWeights,
}

impl ScoringConfig {
    pub fn trigram_default() -> Self {
        Self {
            max_order: 3,
            add_k: 0.05,
            weights: InterpolationWeights::transcript_default(),
        }
    }

    pub fn bigram_default() -> Self {
        Self {
            max_order: 2,
            add_k: 0.05,
            weights: InterpolationWeights::bigram_default(),
        }
    }

    pub fn validate(self) -> Result<Self, AnyError> {
        if !(1..=3).contains(&self.max_order) {
            return Err("this scorer supports max_order in 1..=3".into());
        }
        if self.add_k <= 0.0 {
            return Err("add_k must be > 0 for unseen-event robustness".into());
        }
        Ok(self)
    }
}
