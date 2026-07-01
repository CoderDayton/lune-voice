//! Candidate score type for n-gram scoring.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateScore {
    pub candidate: String,
    pub tokens: Vec<String>,
    pub log_prob: f64,
    pub avg_log_prob: f64,
}
