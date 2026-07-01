//! N-gram language model: training, scoring, persistence.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use super::candidates::CandidateScore;
use super::scoring::{InterpolationWeights, ScoringConfig};
use super::tokenize::{last_or_start, last_two_or_start, normalize_word_or_boundary, tokenize};
#[cfg(feature = "training")]
use super::training::{CorpusConfig, CorpusFetch, fetch_wordnik_corpus};
use super::{AnyError, END, MODEL_FORMAT_VERSION, START};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NGramModel {
    config: ScoringConfig,

    unigram_counts: HashMap<String, usize>,
    bigram_counts: HashMap<[String; 2], usize>,
    trigram_counts: HashMap<[String; 3], usize>,

    bigram_context_totals: HashMap<String, usize>,
    trigram_context_totals: HashMap<[String; 2], usize>,

    total_unigrams: usize,
    vocab: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelSnapshot {
    format_version: u16,
    model: NGramModel,
}

impl NGramModel {
    pub fn new(config: ScoringConfig) -> Result<Self, AnyError> {
        let config = config.validate()?;
        let mut vocab = HashSet::new();
        vocab.insert(END.to_owned());

        Ok(Self {
            config,
            unigram_counts: HashMap::new(),
            bigram_counts: HashMap::new(),
            trigram_counts: HashMap::new(),
            bigram_context_totals: HashMap::new(),
            trigram_context_totals: HashMap::new(),
            total_unigrams: 0,
            vocab,
        })
    }

    pub fn config(&self) -> ScoringConfig {
        self.config
    }

    /// Build a model from Wordnik example sentences. Requires the `training` feature.
    #[cfg(feature = "training")]
    pub async fn from_wordnik_seed_words<I, S>(
        api_key: &str,
        seed_words: I,
        config: ScoringConfig,
        corpus_config: CorpusConfig,
    ) -> Result<(Self, CorpusFetch), AnyError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut model = Self::new(config)?;
        let corpus = fetch_wordnik_corpus(api_key, seed_words, corpus_config).await?;

        if corpus.sentences.is_empty() {
            return Err("Wordnik returned no usable example sentences".into());
        }

        model.train_sentences(&corpus.sentences);
        Ok((model, corpus))
    }

    pub fn train_sentences<I, S>(&mut self, sentences: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for sentence in sentences {
            self.train_sentence(sentence.as_ref());
        }
    }

    pub fn train_sentence(&mut self, sentence: &str) {
        let mut tokens = tokenize(sentence);
        if tokens.is_empty() {
            return;
        }

        for token in &tokens {
            self.vocab.insert(token.clone());
        }
        // `END` is inserted once in `new()`; the sequence below counts it.

        let mut sequence = Vec::with_capacity(tokens.len() + 3);
        sequence.push(START.to_owned());
        sequence.push(START.to_owned());
        sequence.append(&mut tokens);
        sequence.push(END.to_owned());

        for token in sequence.iter().skip(2) {
            *self.unigram_counts.entry(token.clone()).or_default() += 1;
            self.total_unigrams += 1;
        }

        for window in sequence.windows(2) {
            let prev = window[0].clone();
            let next = window[1].clone();
            *self.bigram_counts.entry([prev.clone(), next]).or_default() += 1;
            *self.bigram_context_totals.entry(prev).or_default() += 1;
        }

        for window in sequence.windows(3) {
            let context = [window[0].clone(), window[1].clone()];
            let trigram = [window[0].clone(), window[1].clone(), window[2].clone()];
            *self.trigram_counts.entry(trigram).or_default() += 1;
            *self.trigram_context_totals.entry(context).or_default() += 1;
        }
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocab.len().max(1)
    }

    // ─── Persistence ──────────────────────────────────────────────────────────

    pub fn to_bincode_bytes(&self) -> Result<Vec<u8>, AnyError> {
        let snapshot = ModelSnapshot {
            format_version: MODEL_FORMAT_VERSION,
            model: self.clone(),
        };
        Ok(bincode::serde::encode_to_vec(
            &snapshot,
            bincode::config::standard(),
        )?)
    }

    pub fn from_bincode_bytes(bytes: &[u8]) -> Result<Self, AnyError> {
        let (snapshot, bytes_read): (ModelSnapshot, usize) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;

        if bytes_read != bytes.len() {
            return Err("trailing bytes in model snapshot".into());
        }
        if snapshot.format_version != MODEL_FORMAT_VERSION {
            return Err(format!(
                "unsupported model snapshot version: expected {}, got {}",
                MODEL_FORMAT_VERSION, snapshot.format_version
            )
            .into());
        }

        Ok(snapshot.model)
    }

    pub fn save_bincode(&self, path: impl AsRef<Path>) -> Result<(), AnyError> {
        fs::write(path, self.to_bincode_bytes()?)?;
        Ok(())
    }

    pub fn load_bincode(path: impl AsRef<Path>) -> Result<Self, AnyError> {
        let bytes = fs::read(path)?;
        Self::from_bincode_bytes(&bytes)
    }

    /// Load the pre-trained model bundled into the binary via `include_bytes!`.
    ///
    /// The model file must exist at `crates/lune-dict/data/lune_ngram_model.bin`
    /// at compile time. Run the `train_model` binary to produce it first.
    pub fn load_bundled() -> Result<Self, AnyError> {
        let bytes = include_bytes!("../../data/lune_ngram_model.bin");
        Self::from_bincode_bytes(bytes)
    }

    // ─── Scoring ──────────────────────────────────────────────────────────────

    pub fn interpolated_prob(&self, prefix: &[String], next: &str) -> f64 {
        let next = normalize_word_or_boundary(next);
        if next.is_empty() {
            return 0.0;
        }

        let uni = self.unigram_prob(&next);
        let bi = if self.config.max_order >= 2 {
            self.bigram_prob(last_or_start(prefix), &next)
        } else {
            0.0
        };
        let tri = if self.config.max_order >= 3 {
            let [w1, w2] = last_two_or_start(prefix);
            self.trigram_prob(&w1, &w2, &next)
        } else {
            0.0
        };

        let w = self.effective_weights();
        (w.unigram * uni + w.bigram * bi + w.trigram * tri).max(f64::MIN_POSITIVE)
    }

    pub fn score_candidate(&self, prefix: &str, candidate: &str) -> CandidateScore {
        let mut context = tokenize(prefix);
        let tokens = tokenize(candidate);

        if tokens.is_empty() {
            return CandidateScore {
                candidate: candidate.to_owned(),
                tokens,
                log_prob: f64::NEG_INFINITY,
                avg_log_prob: f64::NEG_INFINITY,
            };
        }

        let mut log_prob = 0.0;
        for token in &tokens {
            let p = self.interpolated_prob(&context, token);
            log_prob += p.ln();
            context.push(token.clone());
        }

        CandidateScore {
            candidate: candidate.to_owned(),
            avg_log_prob: log_prob / tokens.len() as f64,
            tokens,
            log_prob,
        }
    }

    pub fn rank_candidates<I, S>(&self, prefix: &str, candidates: I) -> Vec<CandidateScore>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut scored: Vec<_> = candidates
            .into_iter()
            .map(|candidate| self.score_candidate(prefix, candidate.as_ref()))
            .collect();

        scored.sort_by(|a, b| b.avg_log_prob.total_cmp(&a.avg_log_prob));
        scored
    }

    pub fn best_candidate<I, S>(&self, prefix: &str, candidates: I) -> Option<CandidateScore>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.rank_candidates(prefix, candidates).into_iter().next()
    }

    pub fn score_phrase_avg_logprob(&self, phrase: &str) -> f64 {
        self.score_candidate("", phrase).avg_log_prob
    }

    /// How many times the 2-word context formed by the last two tokens of
    /// `prefix` was observed during training (trigram context total).
    ///
    /// Callers gate context-dependent decisions on this: a context with little
    /// support carries no evidence beyond unigram/bigram frequency, so trusting
    /// the trigram there would merely echo global word frequency. `prefix` is
    /// expected to be already-tokenized (see [`tokenize`](super::tokenize)).
    pub fn context_support(&self, prefix: &[String]) -> usize {
        let [w1, w2] = last_two_or_start(prefix);
        self.trigram_context_totals
            .get(&[w1, w2])
            .copied()
            .unwrap_or(0)
    }

    /// Average log-probability of `candidate` under the UNIGRAM model only
    /// (no context). Compared against the full contextual score, the gap
    /// isolates how much the *context* — rather than raw word frequency —
    /// favors a candidate (a PMI-style signal). Returns `-inf` for empty input.
    pub fn unigram_avg_logprob(&self, candidate: &str) -> f64 {
        let tokens = tokenize(candidate);
        if tokens.is_empty() {
            return f64::NEG_INFINITY;
        }
        let lp: f64 = tokens
            .iter()
            .map(|t| self.unigram_prob(&normalize_word_or_boundary(t)).ln())
            .sum();
        lp / tokens.len() as f64
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    fn smoothed_prob(&self, count: usize, context_total: usize) -> f64 {
        let count_f = count as f64;
        let denominator = context_total as f64 + self.config.add_k * self.vocabulary_size() as f64;
        (count_f + self.config.add_k) / denominator.max(f64::MIN_POSITIVE)
    }

    fn unigram_prob(&self, next: &str) -> f64 {
        let count = self.unigram_counts.get(next).copied().unwrap_or(0);
        self.smoothed_prob(count, self.total_unigrams)
    }

    fn bigram_prob(&self, prev: &str, next: &str) -> f64 {
        let count = self
            .bigram_counts
            .get(&[prev.to_owned(), next.to_owned()])
            .copied()
            .unwrap_or(0);
        let context_total = self.bigram_context_totals.get(prev).copied().unwrap_or(0);
        self.smoothed_prob(count, context_total)
    }

    fn trigram_prob(&self, w1: &str, w2: &str, next: &str) -> f64 {
        let count = self
            .trigram_counts
            .get(&[w1.to_owned(), w2.to_owned(), next.to_owned()])
            .copied()
            .unwrap_or(0);
        let context = [w1.to_owned(), w2.to_owned()];
        let context_total = self
            .trigram_context_totals
            .get(&context)
            .copied()
            .unwrap_or(0);
        self.smoothed_prob(count, context_total)
    }

    fn effective_weights(&self) -> InterpolationWeights {
        match self.config.max_order {
            1 => InterpolationWeights {
                unigram: 1.0,
                bigram: 0.0,
                trigram: 0.0,
            },
            2 => {
                let total = self.config.weights.unigram + self.config.weights.bigram;
                InterpolationWeights {
                    unigram: self.config.weights.unigram / total,
                    bigram: self.config.weights.bigram / total,
                    trigram: 0.0,
                }
            }
            _ => self.config.weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained_test_model() -> NGramModel {
        let mut model = NGramModel::new(ScoringConfig::trigram_default()).unwrap();
        model.train_sentences([
            "play some roblox with friends",
            "play some roblox after school",
            "play some roblox today",
            "play some games today",
            "the black car stopped",
            "black coffee tastes bitter",
            "paint it black",
        ]);
        model
    }

    #[test]
    fn ranks_domain_candidate_above_common_but_wrong_candidate() {
        let model = trained_test_model();
        let ranked = model.rank_candidates("play some", ["black", "roblox"]);
        assert_eq!(ranked[0].candidate, "roblox");
    }

    #[test]
    fn supports_multi_token_candidates_without_length_bias() {
        let model = trained_test_model();
        let ranked = model.rank_candidates("play", ["some roblox", "some black"]);
        assert_eq!(ranked[0].candidate, "some roblox");
    }

    #[test]
    fn unseen_candidates_still_have_finite_score() {
        let model = trained_test_model();
        let score = model.score_candidate("play some", "minecraft");
        assert!(score.avg_log_prob.is_finite());
    }

    #[test]
    fn model_roundtrips_through_bincode() {
        let model = trained_test_model();
        let bytes = model.to_bincode_bytes().unwrap();
        let restored = NGramModel::from_bincode_bytes(&bytes).unwrap();

        let before = model.rank_candidates("play some", ["black", "roblox"]);
        let after = restored.rank_candidates("play some", ["black", "roblox"]);
        assert_eq!(before[0].candidate, after[0].candidate);
    }

    #[test]
    fn load_bundled_model() {
        let model = NGramModel::load_bundled().unwrap();
        assert!(model.vocabulary_size() > 100);
        let prob = model.interpolated_prob(&["the".to_string()], "quick");
        assert!(prob > 0.0);
    }

    #[test]
    fn unigram_avg_logprob_ranks_frequent_above_rare() {
        let model = trained_test_model(); // "black" appears often, "minecraft" never
        let frequent = model.unigram_avg_logprob("black");
        let rare = model.unigram_avg_logprob("minecraft");
        assert!(frequent.is_finite() && rare.is_finite());
        assert!(
            frequent > rare,
            "frequent word must have higher unigram logprob"
        );
        assert_eq!(model.unigram_avg_logprob(""), f64::NEG_INFINITY);
    }
}
