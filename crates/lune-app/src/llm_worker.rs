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
}

/// Completion signal from the LLM worker.
#[derive(Debug)]
pub struct LlmResponse {
    pub corrected_text: String,
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
            };

            // Call transform() directly — streaming is disabled; full corrected text is
            // delivered atomically after the hallucination guard passes, preventing partial
            // garbage from reaching the keyboard and avoiding cursor-interleave issues.
            let corrected_text = match post_processor.transform(&input).await {
                Ok(output) => {
                    let out_words = output.cleaned_text.split_whitespace().count();
                    if out_words > raw_word_count * 2 + 5 {
                        tracing::warn!(
                            raw_words = raw_word_count,
                            llm_words = out_words,
                            "LLM output suspiciously long, discarding"
                        );
                        String::new()
                    } else {
                        output.cleaned_text
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LLM correction failed");
                    String::new()
                }
            };

            tracing::debug!(
                corrected_text = %corrected_text,
                "LLM worker: post-processor output"
            );

            // Always print raw vs corrected comparison to stderr for diagnostics.
            let raw = &input.raw_text;
            eprintln!("\x1b[2m[RAW]\x1b[0m {raw}");
            if corrected_text.is_empty() {
                eprintln!("\x1b[33m[LLM]\x1b[0m (discarded)");
            } else if corrected_text == *raw {
                eprintln!("\x1b[32m[LLM]\x1b[0m (unchanged)");
            } else {
                eprintln!("\x1b[32m[LLM]\x1b[0m {corrected_text}");
            }

            if resp_tx.send(LlmResponse { corrected_text }).await.is_err() {
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
                    })
                    .await
                    .expect("send first request");
                req_tx
                    .send(LlmRequest {
                        text: "second".into(),
                        style_preset: Arc::clone(&style),
                        dictionary: vec![dict_entry("beta", "Beta", 9)],
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
