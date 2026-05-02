use futures::{StreamExt, stream};
use reqwest::{Client, Url};
use serde_json::Value;
use std::collections::HashSet;
use urlencoding;
use wordnik_list::word_exists;

use crate::{AnyError, NGramModel, ScoringConfig, tokenize};

// ─── Config types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CorpusConfig {
    pub examples_per_word: usize,
    pub concurrency: usize,
    pub validate_seed_words: bool,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            examples_per_word: 25,
            concurrency: 8,
            validate_seed_words: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CorpusFetch {
    pub sentences: Vec<String>,
    pub skipped_seeds: Vec<String>,
    pub fetch_errors: Vec<(String, String)>,
}

// ─── Wordnik client ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WordnikClient {
    http: Client,
    api_key: String,
    base_url: String,
}

impl WordnikClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.wordnik.com/v4".to_owned(),
        }
    }

    pub async fn fetch_examples(&self, word: &str, limit: usize) -> Result<Vec<String>, AnyError> {
        let encoded_word = urlencoding::encode(word);
        let mut url = Url::parse(&format!(
            "{}/word.json/{}/examples",
            self.base_url, encoded_word
        ))?;

        url.query_pairs_mut()
            .append_pair("includeDuplicates", "false")
            .append_pair("useCanonical", "true")
            .append_pair("skip", "0")
            .append_pair("limit", &limit.to_string())
            .append_pair("api_key", &self.api_key);

        let resp: Value = self
            .http
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let examples = resp["examples"]
            .as_array()
            .map(|arr: &Vec<Value>| {
                arr.iter()
                    .filter_map(|ex| ex["text"].as_str().map(ToOwned::to_owned))
                    .collect()
            })
            .unwrap_or_default();

        Ok(examples)
    }
}

// ─── Corpus fetch ─────────────────────────────────────────────────────────────

pub async fn fetch_wordnik_corpus<I, S>(
    api_key: &str,
    seed_words: I,
    config: CorpusConfig,
) -> Result<CorpusFetch, AnyError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let client = WordnikClient::new(api_key);
    let (valid_seeds, skipped_seeds) = normalize_seed_words(seed_words, config.validate_seed_words);

    if valid_seeds.is_empty() {
        return Err("no valid seed words after preprocessing/validation".into());
    }

    let concurrency = config.concurrency.max(1);
    let limit = config.examples_per_word;

    let mut corpus = CorpusFetch {
        skipped_seeds,
        ..CorpusFetch::default()
    };

    let mut requests = stream::iter(valid_seeds.into_iter().map(|word| {
        let client = client.clone();
        async move {
            let result = client.fetch_examples(&word, limit).await;
            (word, result)
        }
    }))
    .buffer_unordered(concurrency);

    while let Some((word, result)) = requests.next().await {
        match result {
            Ok(sentences) => corpus.sentences.extend(sentences),
            Err(err) => corpus.fetch_errors.push((word, err.to_string())),
        }
    }

    Ok(corpus)
}

// ─── NGramModel training helper (Wordnik) ─────────────────────────────────────

impl NGramModel {
    /// Convenience: build and train a model from Wordnik in one call.
    pub async fn from_wordnik<I, S>(
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
}

// ─── Private helpers ──────────────────────────────────────────────────────────

fn normalize_seed_words<I, S>(
    seed_words: I,
    validate_with_wordnik_list: bool,
) -> (Vec<String>, Vec<String>)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut seen = HashSet::new();
    let mut valid = Vec::new();
    let mut skipped = Vec::new();

    for raw in seed_words {
        let raw = raw.as_ref();
        let tokens = tokenize(raw);

        if tokens.len() != 1 {
            skipped.push(raw.to_owned());
            continue;
        }

        let word = tokens[0].clone();
        if validate_with_wordnik_list && !word_exists(&word) {
            skipped.push(word);
            continue;
        }

        if seen.insert(word.clone()) {
            valid.push(word);
        }
    }

    (valid, skipped)
}
