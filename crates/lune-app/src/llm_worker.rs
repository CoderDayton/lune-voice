use std::sync::Arc;

use tokio::sync::mpsc;

use lune_core::config::StylePreset;
use lune_core::dict::DictionaryEntry;
use lune_core::llm::{PostProcessInput, TextPostProcessor};

/// Request sent to the LLM worker.
#[derive(Debug)]
pub struct LlmRequest {
    pub text: String,
    pub style_preset: Arc<StylePreset>,
    pub dictionary: Vec<DictionaryEntry>,
    pub context: Option<String>,
    /// Last ~10 words of already-committed text — intra-utterance continuation signal.
    pub prefix_context: Option<String>,
}

/// Completion signal from the LLM worker.
#[derive(Debug)]
pub struct LlmResponse {
    pub corrected_text: String,
    /// When true, this is the completion signal for the current batch.
    /// Streaming chunks have `is_final = false`.
    pub is_final: bool,
}

/// Maximum combined character length per batched LLM call.
/// LLM latency scales super-linearly with input length; keep batches bounded.
const BATCH_CHAR_BUDGET: usize = 1500;

/// Maximum utterances per batch.
const BATCH_MAX_REQUESTS: usize = 16;

/// Spawn the LLM correction worker.
///
/// Returns `(request_tx, response_rx)`:
/// - Send `LlmRequest`s on `request_tx`
/// - Receive `LlmResponse` on `response_rx` when generation completes
pub fn spawn_llm_worker(
    post_processor: Arc<dyn TextPostProcessor>,
) -> (mpsc::Sender<LlmRequest>, mpsc::Receiver<LlmResponse>) {
    let (req_tx, mut req_rx) = mpsc::channel::<LlmRequest>(64);
    // Response channel only needs shallow depth — one batch completes before the next starts.
    let (resp_tx, resp_rx) = mpsc::channel::<LlmResponse>(8);

    tokio::task::spawn_local(async move {
        let mut batch: Vec<LlmRequest> = Vec::with_capacity(BATCH_MAX_REQUESTS);
        // Holds an incompatible request that was popped but couldn't be batched.
        let mut overflow: Option<LlmRequest> = None;

        loop {
            let first = if let Some(req) = overflow.take() {
                req
            } else {
                match req_rx.recv().await {
                    Some(r) => r,
                    None => break,
                }
            };

            let mut combined_len = first.text.len();
            let first_preset_ptr = Arc::as_ptr(&first.style_preset);
            batch.clear();
            batch.push(first);

            // Yield once so more requests accumulate in the channel during bursts.
            tokio::task::yield_now().await;

            // Drain queued requests that share the same style preset and fit the char budget.
            // Budget check moved inside loop body to prevent overshoot: the old while-condition
            // `combined_len < BATCH_CHAR_BUDGET` allowed the next item to push well past the
            // limit because it was checked *before* adding, but the item was added unconditionally.
            while batch.len() < BATCH_MAX_REQUESTS {
                match req_rx.try_recv() {
                    Ok(r) => {
                        // Only batch requests with the same style preset; incompatible
                        // ones get deferred to the next iteration via overflow.
                        if Arc::as_ptr(&r.style_preset) != first_preset_ptr {
                            overflow = Some(r);
                            break;
                        }
                        let candidate_len = 1 + r.text.len(); // +1 for space separator
                        if combined_len + candidate_len > BATCH_CHAR_BUDGET {
                            // Would exceed budget — defer to next batch.
                            overflow = Some(r);
                            break;
                        }
                        combined_len += candidate_len;
                        batch.push(r);
                    }
                    Err(_) => break,
                }
            }

            // Build combined text; capacity known from incremental tracking.
            let mut combined_text = String::with_capacity(combined_len);
            for (i, r) in batch.iter().enumerate() {
                if i > 0 {
                    combined_text.push(' ');
                }
                combined_text.push_str(&r.text);
            }

            if batch.len() > 1 {
                tracing::debug!(
                    count = batch.len(),
                    chars = combined_len,
                    "LLM batch: combined {} requests",
                    batch.len(),
                );
            }

            // Track raw word count before the text is consumed by PostProcessInput.
            let raw_word_count = combined_text.split_whitespace().count();

            tracing::debug!(
                raw_word_count,
                raw_text = %combined_text,
                "LLM worker: sending to post-processor"
            );

            // Use first request's config — all batched requests share the same style preset.
            let style_preset = Arc::clone(&batch[0].style_preset);
            // Merge dictionaries from every batched request so per-utterance entries are not lost.
            let mut dictionary = Vec::new();
            for req in &mut batch {
                dictionary.append(&mut req.dictionary);
            }
            let input = PostProcessInput {
                raw_text: combined_text,
                style_preset,
                app_category: None,
                dictionary,
                context: batch[0].context.clone(),
                // Use the first request's prefix — combined text starts from its context.
                prefix_context: batch[0].prefix_context.clone(),
            };

            // Stream corrected text in word-boundary chunks as the LLM generates.
            // Live hallucination guard aborts generation if output exceeds 2x input + 5 words.
            let raw_for_log = input.raw_text.clone();
            let (chunk_tx, mut chunk_rx) = mpsc::channel::<String>(32);
            let pp = Arc::clone(&post_processor);
            let inference_task =
                tokio::task::spawn_local(
                    async move { pp.transform_streaming(&input, chunk_tx).await },
                );

            let hallu_limit = raw_word_count * 2 + 5;
            let mut cum_words = 0usize;
            let mut aborted = false;
            while let Some(chunk) = chunk_rx.recv().await {
                cum_words += chunk.split_whitespace().count();
                if cum_words > hallu_limit {
                    tracing::warn!(
                        raw_words = raw_word_count,
                        streamed_words = cum_words,
                        "streaming output exceeds hallucination threshold, aborting"
                    );
                    inference_task.abort();
                    aborted = true;
                    break;
                }
                if resp_tx
                    .send(LlmResponse {
                        corrected_text: chunk,
                        is_final: false,
                    })
                    .await
                    .is_err()
                {
                    inference_task.abort();
                    aborted = true;
                    break;
                }
            }

            // Await inference completion (may already be done or aborted).
            let full_text = match inference_task.await {
                Ok(Ok(output)) => output.cleaned_text,
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "LLM correction failed");
                    String::new()
                }
                Err(_) => String::new(), // JoinError from abort
            };

            tracing::debug!(
                output_len = full_text.len(),
                aborted,
                "LLM worker: post-processor complete"
            );

            eprintln!("\x1b[2m[RAW]\x1b[0m {raw_for_log}");
            if aborted {
                eprintln!("\x1b[33m[LLM]\x1b[0m (hallucination aborted)");
            } else if full_text.is_empty() {
                eprintln!("\x1b[33m[LLM]\x1b[0m (error)");
            } else if full_text == raw_for_log {
                eprintln!("\x1b[32m[LLM]\x1b[0m (unchanged)");
            } else {
                eprintln!("\x1b[32m[LLM]\x1b[0m {full_text}");
            }

            // Completion signal so the session knows this batch is done.
            if resp_tx
                .send(LlmResponse {
                    corrected_text: String::new(),
                    is_final: true,
                })
                .await
                .is_err()
            {
                tracing::debug!("LLM response channel closed, stopping worker");
                break;
            }
        }

        tracing::debug!("LLM worker stopped");
    });

    (req_tx, resp_rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use lune_core::Result as LlmResult;
    use lune_core::llm::PostProcessOutput;

    #[derive(Default)]
    struct RecordingPostProcessor {
        seen_dictionaries: Mutex<Vec<Vec<DictionaryEntry>>>,
    }

    #[async_trait::async_trait]
    impl TextPostProcessor for RecordingPostProcessor {
        async fn transform(&self, input: &PostProcessInput) -> LlmResult<PostProcessOutput> {
            self.seen_dictionaries
                .lock()
                .expect("lock seen_dictionaries")
                .push(input.dictionary.clone());
            Ok(PostProcessOutput {
                cleaned_text: input.raw_text.clone(),
                metadata: None,
            })
        }
    }

    fn dict_entry(phrase: &str, replacement: &str, priority: u8) -> DictionaryEntry {
        DictionaryEntry {
            phrase: phrase.into(),
            replacement: replacement.into(),
            priority,
        }
    }

    #[tokio::test]
    async fn batch_merges_dictionaries_from_all_requests() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let processor = Arc::new(RecordingPostProcessor::default());
                let (req_tx, mut resp_rx) = spawn_llm_worker(processor.clone());
                let style = Arc::new(StylePreset::general());

                req_tx
                    .send(LlmRequest {
                        text: "first".into(),
                        style_preset: Arc::clone(&style),
                        dictionary: vec![dict_entry("alpha", "Alpha", 10)],
                        context: None,
                        prefix_context: None,
                    })
                    .await
                    .expect("send first request");
                req_tx
                    .send(LlmRequest {
                        text: "second".into(),
                        style_preset: Arc::clone(&style),
                        dictionary: vec![dict_entry("beta", "Beta", 9)],
                        context: None,
                        prefix_context: None,
                    })
                    .await
                    .expect("send second request");

                let _ = resp_rx.recv().await.expect("receive LLM response");
                drop(req_tx);

                let seen = processor
                    .seen_dictionaries
                    .lock()
                    .expect("lock seen_dictionaries");
                let first_batch = seen.first().expect("batch recorded");
                assert_eq!(first_batch.len(), 2);
                assert!(first_batch.iter().any(|e| e.phrase == "alpha"));
                assert!(first_batch.iter().any(|e| e.phrase == "beta"));
            })
            .await;
    }
}
