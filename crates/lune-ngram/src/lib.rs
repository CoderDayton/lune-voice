use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::Path;
use strsim::{jaro_winkler, normalized_damerau_levenshtein};

pub type AnyError = Box<dyn Error + Send + Sync + 'static>;

pub const START: &str = "<s>";
pub const END: &str = "</s>";
pub const MODEL_FORMAT_VERSION: u16 = 1;

#[cfg(feature = "training")]
pub mod training;

// CorpusConfig and CorpusFetch are used by training but also exposed for callers
// who want to inspect what was fetched. Keep them in core behind a feature gate
// since they're only meaningful alongside training.
#[cfg(feature = "training")]
pub use training::{CorpusConfig, CorpusFetch, fetch_wordnik_corpus};

// ─── Interpolation weights ────────────────────────────────────────────────────

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

// ─── Scoring config ───────────────────────────────────────────────────────────

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

// ─── Score types ──────────────────────────────────────────────────────────────

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

// ─── Model ────────────────────────────────────────────────────────────────────

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
        self.vocab.insert(END.to_owned());

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
    /// The model file must exist at `crates/lune-ngram/data/lune_ngram_model.bin`
    /// at compile time. Run the training binary to produce it first.
    pub fn load_bundled() -> Result<Self, AnyError> {
        let bytes = include_bytes!("../data/lune_ngram_model.bin");
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

    pub fn rerank_asr_candidates<I, C>(
        &self,
        prefix: &str,
        observed_slot: &str,
        candidates: I,
        weights: AsrRerankWeights,
    ) -> Result<Vec<AsrCandidateScore>, AnyError>
    where
        I: IntoIterator<Item = C>,
        C: Into<AsrCandidateInput>,
    {
        let weights = weights.validate()?;
        let inputs: Vec<AsrCandidateInput> = candidates.into_iter().map(Into::into).collect();
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let ngram_scores: Vec<CandidateScore> = inputs
            .iter()
            .map(|candidate| self.score_candidate(prefix, &candidate.text))
            .collect();
        let raw_ngram: Vec<f64> = ngram_scores
            .iter()
            .map(|score| score.avg_log_prob)
            .collect();
        let normalized_ngram = minmax_normalize(&raw_ngram);

        let acoustic_available = inputs
            .iter()
            .any(|candidate| candidate.acoustic_score.is_some());
        let acoustic_values: Vec<f64> = inputs
            .iter()
            .map(|candidate| candidate.acoustic_score.unwrap_or(f64::NEG_INFINITY))
            .collect();
        let normalized_acoustic = if acoustic_available {
            Some(minmax_normalize(&acoustic_values))
        } else {
            None
        };

        let observed_norm = normalize_similarity_text(observed_slot);
        let observed_key = acoustic_shape(observed_slot);

        let active_acoustic_weight = if acoustic_available {
            weights.acoustic
        } else {
            0.0
        };
        let total_weight = weights.ngram + active_acoustic_weight + weights.edit + weights.phonetic;

        let mut scores = Vec::with_capacity(inputs.len());
        for (idx, input) in inputs.into_iter().enumerate() {
            let candidate_norm = normalize_similarity_text(&input.text);
            let candidate_key = acoustic_shape(&input.text);
            let edit_similarity = lexical_similarity(&observed_norm, &candidate_norm);
            let phonetic_similarity = lexical_similarity(&observed_key, &candidate_key);
            let acoustic_component = normalized_acoustic
                .as_ref()
                .map(|scores| scores[idx])
                .unwrap_or(0.0);

            let final_score = ((weights.ngram * normalized_ngram[idx])
                + (active_acoustic_weight * acoustic_component)
                + (weights.edit * edit_similarity)
                + (weights.phonetic * phonetic_similarity))
                / total_weight.max(f64::MIN_POSITIVE);

            scores.push(AsrCandidateScore {
                candidate: input.text,
                tokens: ngram_scores[idx].tokens.clone(),
                final_score,
                ngram_log_prob: ngram_scores[idx].log_prob,
                ngram_avg_log_prob: ngram_scores[idx].avg_log_prob,
                normalized_ngram: normalized_ngram[idx],
                acoustic_score: input.acoustic_score,
                normalized_acoustic: normalized_acoustic.as_ref().map(|scores| scores[idx]),
                edit_similarity,
                phonetic_similarity,
                candidate_phonetic_key: candidate_key,
            });
        }

        scores.sort_by(|a, b| b.final_score.total_cmp(&a.final_score));
        Ok(scores)
    }

    pub fn best_asr_candidate<I, C>(
        &self,
        prefix: &str,
        observed_slot: &str,
        candidates: I,
        weights: AsrRerankWeights,
    ) -> Result<Option<AsrCandidateScore>, AnyError>
    where
        I: IntoIterator<Item = C>,
        C: Into<AsrCandidateInput>,
    {
        Ok(self
            .rerank_asr_candidates(prefix, observed_slot, candidates, weights)?
            .into_iter()
            .next())
    }

    pub fn score_phrase_avg_logprob(&self, phrase: &str) -> f64 {
        self.score_candidate("", phrase).avg_log_prob
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

// ─── Free functions ───────────────────────────────────────────────────────────

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect()
}

pub fn acoustic_shape(text: &str) -> String {
    let mut out = String::new();
    let mut last = '\0';

    for ch in text.to_ascii_lowercase().chars() {
        let class = match ch {
            'a' | 'e' | 'i' | 'o' | 'u' | 'y' => 'A',
            'b' | 'p' => 'P',
            'd' | 't' => 'T',
            'g' | 'k' | 'q' => 'K',
            'c' | 's' | 'z' => 'S',
            'f' | 'v' => 'F',
            'm' | 'n' => 'N',
            'r' | 'l' => 'R',
            'j' => 'J',
            'h' | 'w' => 'H',
            'x' => 'X',
            '0'..='9' => 'D',
            _ => continue,
        };

        if class != last {
            out.push(class);
            last = class;
        }
    }

    out
}

fn normalize_word_or_boundary(word: &str) -> String {
    match word {
        START => START.to_owned(),
        END => END.to_owned(),
        _ => word
            .to_ascii_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect(),
    }
}

fn last_or_start(prefix: &[String]) -> &str {
    prefix.last().map(String::as_str).unwrap_or(START)
}

fn last_two_or_start(prefix: &[String]) -> [String; 2] {
    match prefix.len() {
        0 => [START.to_owned(), START.to_owned()],
        1 => [START.to_owned(), prefix[0].clone()],
        len => [prefix[len - 2].clone(), prefix[len - 1].clone()],
    }
}

fn minmax_normalize(values: &[f64]) -> Vec<f64> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.is_finite() {
        return vec![0.0; values.len()];
    }
    let span = max - min;

    values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                0.0
            } else if span <= f64::EPSILON {
                0.5
            } else {
                ((value - min) / span).clamp(0.0, 1.0)
            }
        })
        .collect()
}

fn normalize_similarity_text(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn lexical_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let jaro = jaro_winkler(a, b);
    let damerau = normalized_damerau_levenshtein(a, b);
    (0.60 * jaro + 0.40 * damerau).clamp(0.0, 1.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

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
    fn asr_reranker_blends_ngram_and_similarity_features() {
        let model = trained_test_model();
        let ranked = model
            .rerank_asr_candidates(
                "play some",
                "black",
                ["black", "roblox"],
                AsrRerankWeights::transcript_default(),
            )
            .unwrap();

        assert_eq!(ranked[0].candidate, "roblox");
        assert!(ranked[0].final_score >= ranked[1].final_score);
    }

    #[test]
    fn load_bundled_model() {
        let model = NGramModel::load_bundled().unwrap();
        assert!(model.vocabulary_size() > 100);
        let prob = model.interpolated_prob(&["the".to_string()], "quick");
        assert!(prob > 0.0);
    }

    #[test]
    fn acoustic_score_can_override_when_weighted_heavily() {
        let model = trained_test_model();
        let ranked = model
            .rerank_asr_candidates(
                "play some",
                "black",
                [
                    AsrCandidateInput::with_acoustic_score("black", 0.95),
                    AsrCandidateInput::with_acoustic_score("roblox", 0.10),
                ],
                AsrRerankWeights {
                    ngram: 0.10,
                    acoustic: 0.80,
                    edit: 0.05,
                    phonetic: 0.05,
                },
            )
            .unwrap();

        assert_eq!(ranked[0].candidate, "black");
    }
}
