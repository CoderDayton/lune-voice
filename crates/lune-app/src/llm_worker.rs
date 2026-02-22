//! Async LLM correction worker.
//!
//! Runs on a dedicated channel, processing low-confidence utterances asynchronously.
//! The main session loop delivers text immediately and doesn't block on LLM results.

use std::sync::Arc;

use tokio::sync::mpsc;

use lune_core::config::StylePreset;
use lune_core::dict::DictionaryEntry;
use lune_core::llm::{PostProcessInput, TextPostProcessor};

/// Request sent to the LLM worker.
#[derive(Debug)]
pub struct LlmRequest {
    pub utterance_id: u64,
    pub text: String,
    pub context: Option<String>,
    pub style_preset: StylePreset,
    pub dictionary: Vec<DictionaryEntry>,
}

/// Response from the LLM worker.
#[derive(Debug)]
pub struct LlmResponse {
    pub utterance_id: u64,
    pub original_text: String,
    pub corrected_text: String,
    /// `false` if the LLM output failed the length sanity check.
    pub accepted: bool,
}

/// Spawn the LLM correction worker.
///
/// Returns `(sender, receiver)` — send `LlmRequest`s, receive `LlmResponse`s.
pub fn spawn_llm_worker(
    post_processor: Arc<dyn TextPostProcessor>,
    max_length_ratio: f32,
    min_length_ratio: f32,
) -> (mpsc::Sender<LlmRequest>, mpsc::Receiver<LlmResponse>) {
    let (req_tx, mut req_rx) = mpsc::channel::<LlmRequest>(32);
    let (resp_tx, resp_rx) = mpsc::channel::<LlmResponse>(32);

    tokio::task::spawn_local(async move {
        while let Some(first) = req_rx.recv().await {
            // Drain all queued requests into a batch (CognitivePipe pattern).
            let mut batch = vec![first];
            while let Ok(extra) = req_rx.try_recv() {
                batch.push(extra);
            }

            // Combine text from all batched requests — compute total capacity upfront.
            let total_len: usize =
                batch.iter().map(|r| r.text.len()).sum::<usize>() + batch.len().saturating_sub(1); // spaces between segments
            let mut combined_text = String::with_capacity(total_len);
            for (i, r) in batch.iter().enumerate() {
                if i > 0 {
                    combined_text.push(' ');
                }
                combined_text.push_str(&r.text);
            }
            // combined_original is identical to combined_text at this point.
            // Clone once here; move combined_text into PostProcessInput to avoid a second clone.
            let combined_original = combined_text.clone();

            // Use context from the last request, max utterance_id.
            let last = batch.last().unwrap();
            let max_utterance_id = batch.iter().map(|r| r.utterance_id).max().unwrap();

            let input = PostProcessInput {
                raw_text: combined_text, // moved — no extra clone
                prev_context: last.context.clone(),
                style_preset: last.style_preset.clone(),
                app_category: None,
                dictionary: last.dictionary.clone(),
            };

            if batch.len() > 1 {
                tracing::debug!(
                    count = batch.len(),
                    combined_words = input.raw_text.split_whitespace().count(),
                    "LLM batch: combined {} requests",
                    batch.len(),
                );
            }

            let (corrected, accepted) = match post_processor.transform(&input).await {
                Ok(output) => {
                    let ratio = if input.raw_text.is_empty() {
                        f32::MAX
                    } else {
                        output.cleaned_text.len() as f32 / input.raw_text.len() as f32
                    };
                    let ok = ratio >= min_length_ratio && ratio <= max_length_ratio;
                    if !ok {
                        tracing::debug!(
                            utterance_id = max_utterance_id,
                            ratio,
                            "LLM correction rejected (length sanity)"
                        );
                    }
                    (output.cleaned_text, ok)
                }
                Err(e) => {
                    #[cold]
                    fn warn_failed(e: &dyn std::fmt::Display) {
                        tracing::warn!(error = %e, "LLM correction failed");
                    }
                    warn_failed(&e);
                    // combined_text was moved into input; clone original for corrected.
                    (combined_original.clone(), false)
                }
            };

            let resp = LlmResponse {
                utterance_id: max_utterance_id,
                original_text: combined_original,
                corrected_text: corrected,
                accepted,
            };

            if resp_tx.send(resp).await.is_err() {
                tracing::debug!("LLM response channel closed, stopping worker");
                break;
            }
        }

        tracing::debug!("LLM worker stopped");
    });

    (req_tx, resp_rx)
}
