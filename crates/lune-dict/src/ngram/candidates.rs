//! Candidate score types for n-gram scoring and ASR reranking.

use serde::{Deserialize, Serialize};

use super::AnyError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateScore {
    pub candidate: String,
    pub tokens: Vec<String>,
    pub log_prob: f64,
    pub avg_log_prob: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrCandidateInput {
    pub text: String,
    /// Optional ASR model score. Higher is better. Log-probability, confidence, or margin are all fine
    /// because the reranker min-max normalizes scores within each candidate set.
    pub acoustic_score: Option<f64>,
}

impl AsrCandidateInput {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            acoustic_score: None,
        }
    }

    pub fn with_acoustic_score(text: impl Into<String>, acoustic_score: f64) -> Self {
        Self {
            text: text.into(),
            acoustic_score: Some(acoustic_score),
        }
    }
}

impl From<&str> for AsrCandidateInput {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for AsrCandidateInput {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AsrRerankWeights {
    pub ngram: f64,
    pub acoustic: f64,
    pub edit: f64,
    pub phonetic: f64,
}

impl AsrRerankWeights {
    pub fn transcript_default() -> Self {
        Self {
            ngram: 0.70,
            acoustic: 0.10,
            edit: 0.10,
            phonetic: 0.10,
        }
    }

    pub fn ngram_heavy() -> Self {
        Self {
            ngram: 0.85,
            acoustic: 0.05,
            edit: 0.05,
            phonetic: 0.05,
        }
    }

    pub fn validate(self) -> Result<Self, AnyError> {
        if self.ngram < 0.0 || self.acoustic < 0.0 || self.edit < 0.0 || self.phonetic < 0.0 {
            return Err("ASR rerank weights must be non-negative".into());
        }
        if self.ngram + self.acoustic + self.edit + self.phonetic <= f64::EPSILON {
            return Err("at least one ASR rerank weight must be positive".into());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrCandidateScore {
    pub candidate: String,
    pub tokens: Vec<String>,
    pub final_score: f64,
    pub ngram_log_prob: f64,
    pub ngram_avg_log_prob: f64,
    pub normalized_ngram: f64,
    pub acoustic_score: Option<f64>,
    pub normalized_acoustic: Option<f64>,
    pub edit_similarity: f64,
    pub phonetic_similarity: f64,
    pub candidate_phonetic_key: String,
}
