//! Async LLM correction worker.
//!
//! All finalized utterances are held until the LLM processes them — no raw text
//! is delivered to the user. Short bursts return quickly; longer paragraphs take
//! proportionally more time. The worker batches queued requests into a single call.

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
    // Bump channel depths: bursty ASR can spike; 64 avoids back-pressure drops.
    let (req_tx, mut req_rx) = mpsc::channel::<LlmRequest>(64);
    let (resp_tx, resp_rx) = mpsc::channel::<LlmResponse>(64);

    tokio::task::spawn_local(async move {
        // Reuse the batch vec across loop iterations to avoid per-loop allocations.
        let mut batch: Vec<LlmRequest> = Vec::with_capacity(8);

        while let Some(first) = req_rx.recv().await {
            batch.clear();
            batch.push(first);

            // Drain all already-queued requests (CognitivePipe batching).
            // Use recv_many for a single lock acquisition over the channel internals.
            // recv_many appends up to `limit` items; 31 keeps total ≤ 32.
            let _ = req_rx.recv_many(&mut batch, 31).await;

            // --- build combined_text with a single allocation ---
            // Compute exact capacity: sum of text lengths + (n-1) spaces.
            let total_len: usize =
                batch.iter().map(|r| r.text.len()).sum::<usize>() + batch.len().saturating_sub(1);
            let mut combined_text = String::with_capacity(total_len);
            let mut max_utterance_id: u64 = 0;
            for (i, r) in batch.iter().enumerate() {
                if i > 0 {
                    combined_text.push(' ');
                }
                combined_text.push_str(&r.text);
                if r.utterance_id > max_utterance_id {
                    max_utterance_id = r.utterance_id;
                }
            }

            // Keep a byte-length snapshot before moving combined_text into the input;
            // avoids re-computing it inside the hot ratio check.
            let original_byte_len = combined_text.len();
            // Clone before moving combined_text — needed for original_text in the
            // response (always) and corrected_text when LLM correction fails.
            let combined_original = combined_text.clone();

            let last = batch.last().expect("batch non-empty");

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

            #[cold]
            #[inline(never)]
            fn warn_failed(e: &dyn std::fmt::Display) {
                tracing::warn!(error = %e, "LLM correction failed");
            }

            let resp = match post_processor.transform(&input).await {
                Ok(output) => {
                    let ratio = if original_byte_len == 0 {
                        f32::MAX
                    } else {
                        output.cleaned_text.len() as f32 / original_byte_len as f32
                    };
                    let accepted = ratio >= min_length_ratio && ratio <= max_length_ratio;
                    if !accepted {
                        tracing::debug!(
                            utterance_id = max_utterance_id,
                            ratio,
                            "LLM correction rejected (length sanity)"
                        );
                    }
                    LlmResponse {
                        utterance_id: max_utterance_id,
                        original_text: combined_original,
                        corrected_text: output.cleaned_text,
                        accepted,
                    }
                }
                Err(e) => {
                    warn_failed(&e);
                    LlmResponse {
                        utterance_id: max_utterance_id,
                        corrected_text: combined_original.clone(),
                        original_text: combined_original,
                        accepted: false,
                    }
                }
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
