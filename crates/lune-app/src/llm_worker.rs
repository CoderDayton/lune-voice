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

/// Streaming chunk emitted during LLM generation.
#[derive(Debug)]
pub struct LlmChunk {
    pub text: String,
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
/// Returns `(request_tx, response_rx, chunk_rx)`:
/// - Send `LlmRequest`s on `request_tx`
/// - Receive streaming `LlmChunk`s on `chunk_rx` as the LLM generates
/// - Receive `LlmResponse` on `response_rx` when generation completes
pub fn spawn_llm_worker(
    post_processor: Arc<dyn TextPostProcessor>,
) -> (
    mpsc::Sender<LlmRequest>,
    mpsc::Receiver<LlmResponse>,
    mpsc::Receiver<LlmChunk>,
) {
    let (req_tx, mut req_rx) = mpsc::channel::<LlmRequest>(64);
    // Response channel only needs shallow depth — one batch completes before the next starts.
    let (resp_tx, resp_rx) = mpsc::channel::<LlmResponse>(8);
    let (stream_tx, stream_rx) = mpsc::channel::<LlmChunk>(64);

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

            // Use first request's config — all batched requests share the same style preset.
            // Take ownership of the dictionary to avoid a clone; batch is cleared next iteration.
            let style_preset = Arc::clone(&batch[0].style_preset);
            let dictionary = std::mem::take(&mut batch[0].dictionary);

            let input = PostProcessInput {
                raw_text: combined_text,
                prev_context: None,
                style_preset,
                app_category: None,
                dictionary,
            };

            let (chunk_tx, mut chunk_rx) = mpsc::channel::<String>(32);
            let transform_fut = post_processor.transform_streaming(&input, chunk_tx);
            tokio::pin!(transform_fut);

            let transform_result = loop {
                tokio::select! {
                    biased;
                    result = &mut transform_fut => {
                        while let Ok(chunk) = chunk_rx.try_recv() {
                            let _ = stream_tx.send(LlmChunk { text: chunk }).await;
                        }
                        break result;
                    }
                    Some(chunk) = chunk_rx.recv() => {
                        let _ = stream_tx.send(LlmChunk { text: chunk }).await;
                    }
                }
            };

            let corrected_text = match transform_result {
                Ok(output) => output.cleaned_text,
                Err(e) => {
                    tracing::warn!(error = %e, "LLM correction failed");
                    String::new()
                }
            };

            if resp_tx.send(LlmResponse { corrected_text }).await.is_err() {
                tracing::debug!("LLM response channel closed, stopping worker");
                break;
            }
        }

        tracing::debug!("LLM worker stopped");
    });

    (req_tx, resp_rx, stream_rx)
}
