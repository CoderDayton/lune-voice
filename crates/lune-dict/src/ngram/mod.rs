//! N-gram language model and ASR candidate reranker.
//!
//! Split across focused submodules; this module re-exports the public surface
//! so callers use `lune_dict::ngram::NGramModel`,
//! `lune_dict::ngram::ScoringConfig`, etc.

use std::error::Error;

pub type AnyError = Box<dyn Error + Send + Sync + 'static>;

pub const START: &str = "<s>";
pub const END: &str = "</s>";
pub const MODEL_FORMAT_VERSION: u16 = 1;

pub mod candidates;
pub mod model;
pub mod scoring;
pub mod tokenize;

#[cfg(feature = "training")]
pub mod training;

// CorpusConfig and CorpusFetch are used by training but also exposed for callers
// who want to inspect what was fetched. Keep them in core behind a feature gate
// since they're only meaningful alongside training.
#[cfg(feature = "training")]
pub use training::{CorpusConfig, CorpusFetch, fetch_wordnik_corpus};

pub use candidates::{AsrCandidateInput, AsrCandidateScore, AsrRerankWeights, CandidateScore};
pub use model::NGramModel;
pub use scoring::{InterpolationWeights, ScoringConfig};
pub use tokenize::{acoustic_shape, tokenize};
