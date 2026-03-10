//! ORT-based Qwen ASR backend.
//!
//! Uses three ONNX Runtime sessions (encoder, embed_tokens, decoder) for
//! batch transcription. Audio accumulates via `push_audio`; transcription
//! happens in a single `transcribe_oneshot` call on `poll_final`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ort::session::Session;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig, WordConfidence};
use lune_core::error::{Error, Result};

use crate::lcs_merge::lcs_merge_words;
use crate::ort_forced_aligner::{FaConfig, OrtForcedAligner};
use crate::ort_qwen_stream;
use crate::ort_utils::build_ort_session;

// ── Configuration ───────────────────────────────────────────────────────────

/// Maximum audio the encoder will process in a single pass.
/// The ONNX model supports dynamic mel_frames up to 6000 (=60s).
/// The encoder uses windowed attention internally (8s windows) —
/// this is NOT a segmentation boundary. The decoder sees all
/// encoder outputs via cross-attention for global coherence.
const MAX_ENCODER_SAMPLES: usize = 60 * 16_000;

/// Decoder stays grounded within 8s (matches encoder's internal attention window).
const CHUNK_SAMPLES: usize = 8 * 16_000;
/// 2s overlap — at normal speech rate (~2.5 words/s) this yields ~5 shared
/// words, well above the LCS ≥ 2 minimum for reliable stitching.
const OVERLAP_SAMPLES: usize = 2 * 16_000;
/// Step = chunk - overlap = 6s.
const STEP_SAMPLES: usize = CHUNK_SAMPLES - OVERLAP_SAMPLES;
/// Only chunk audio longer than this (oneshot is fine for short clips).
const CHUNK_THRESHOLD: usize = 10 * 16_000;

/// Runtime configuration for the ORT Qwen ASR backend.
#[derive(Debug, Clone)]
pub struct OrtQwenAsrConfig {
    /// Directory containing qwen3-asr-encoder.onnx, qwen3-asr-embed-tokens.onnx, qwen3-asr-decoder.onnx, tokenizer.json.
    pub model_dir: PathBuf,
    /// Whether to use GPU (CUDA EP registered globally via init_ort_runtime).
    pub use_gpu: bool,
}

impl Default for OrtQwenAsrConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            use_gpu: false,
        }
    }
}

// ── Backend ─────────────────────────────────────────────────────────────────

/// ORT-based Qwen ASR backend using three ONNX sessions.
pub struct OrtQwenAsrBackend {
    encoder: Arc<Mutex<Session>>,
    embed_tokens: Arc<Mutex<Session>>,
    decoder: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    config: OrtQwenAsrConfig,
    /// Whether the decoder ONNX model has an `attention_mask` input.
    has_attention_mask: bool,
    /// Optional forced aligner for timestamp-based chunk merging.
    /// Attached after construction via `attach_aligner()`.
    aligner: OnceLock<Arc<OrtForcedAligner>>,
}

impl OrtQwenAsrBackend {
    pub fn new(config: OrtQwenAsrConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir));
        }

        let use_gpu = config.use_gpu;
        let dir = &config.model_dir;

        let encoder_path = dir.join("qwen3-asr-encoder.onnx");
        let embed_path = dir.join("qwen3-asr-embed-tokens.onnx");
        let decoder_path = dir.join("qwen3-asr-decoder.onnx");
        let tokenizer_path = dir.join("tokenizer.json");

        let mut encoder = build_ort_session(&encoder_path, use_gpu)?;
        let mut embed_tokens = build_ort_session(&embed_path, use_gpu)?;
        let mut decoder = build_ort_session(&decoder_path, use_gpu)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Asr(format!("tokenizer load: {e}")))?;

        // Detect whether decoder accepts attention_mask.
        let has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");

        // Warmup: prime CUDA kernels for all three sessions.
        warmup_asr(
            &mut encoder,
            &mut embed_tokens,
            &mut decoder,
            has_attention_mask,
        )?;

        tracing::info!(
            model_dir = %dir.display(),
            "initialized ORT Qwen ASR backend (batch mode)"
        );

        Ok(Self {
            encoder: Arc::new(Mutex::new(encoder)),
            embed_tokens: Arc::new(Mutex::new(embed_tokens)),
            decoder: Mutex::new(decoder),
            tokenizer,
            config,
            has_attention_mask,
            aligner: OnceLock::new(),
        })
    }
}

// ── Session state ───────────────────────────────────────────────────────────

struct SessionState {
    audio_buf: Vec<f32>,
    next_id: u64,
    sample_rate: u32,
    forced_language: Option<String>,
}

pub struct OrtQwenAsrSession {
    state: Mutex<SessionState>,
}

// ── Language mapping ────────────────────────────────────────────────────────

#[inline]
fn map_language_tag(language: &str) -> Option<&'static str> {
    let lower = language.trim().to_ascii_lowercase();
    let base = lower.split(['-', '_']).next().unwrap_or_default();
    match base {
        "zh" => Some("Chinese"),
        "en" => Some("English"),
        "yue" => Some("Cantonese"),
        "ar" => Some("Arabic"),
        "de" => Some("German"),
        "fr" => Some("French"),
        "es" => Some("Spanish"),
        "pt" => Some("Portuguese"),
        "id" => Some("Indonesian"),
        "it" => Some("Italian"),
        "ko" => Some("Korean"),
        "ru" => Some("Russian"),
        "th" => Some("Thai"),
        "vi" => Some("Vietnamese"),
        "ja" => Some("Japanese"),
        "tr" => Some("Turkish"),
        "hi" => Some("Hindi"),
        "ms" => Some("Malay"),
        "nl" => Some("Dutch"),
        "sv" => Some("Swedish"),
        "da" => Some("Danish"),
        "fi" => Some("Finnish"),
        "pl" => Some("Polish"),
        "cs" => Some("Czech"),
        "fil" | "tl" => Some("Filipino"),
        "fa" => Some("Persian"),
        "el" => Some("Greek"),
        "ro" => Some("Romanian"),
        "hu" => Some("Hungarian"),
        "mk" => Some("Macedonian"),
        _ => None,
    }
}

// ── Warmup ──────────────────────────────────────────────────────────────────

/// Run minimal dummy inference through each session to prime CUDA kernels.
#[cold]
fn warmup_asr(
    encoder: &mut Session,
    embed_tokens: &mut Session,
    decoder: &mut Session,
    has_attention_mask: bool,
) -> Result<()> {
    use ndarray::{Array2, Array3, ArrayD, IxDyn};
    use ort::value::{DynValue, Value};
    use ort_qwen_stream::{HEAD_DIM, HIDDEN_DIM, NUM_DECODER_LAYERS, NUM_KV_HEADS};

    // 1) Encoder: tiny mel [1, 128, 2] — model expects f16.
    let mel = Array3::from_elem((1, 128, 2), half::f16::ZERO);
    let mel_val: DynValue = Value::from_array(mel.into_dyn())
        .map_err(|e| Error::Asr(format!("warmup mel: {e}")))?
        .into();
    let _enc = encoder
        .run(vec![("mel".to_string(), mel_val)])
        .map_err(|e| Error::Asr(format!("warmup encoder: {e}")))?;

    // 2) Embed_tokens: single token
    let ids = Array2::from_shape_vec((1, 1), vec![0i64]).unwrap();
    let ids_val: DynValue = Value::from_array(ids)
        .map_err(|e| Error::Asr(format!("warmup ids: {e}")))?
        .into();
    let _emb = embed_tokens
        .run(vec![("input_ids".to_string(), ids_val)])
        .map_err(|e| Error::Asr(format!("warmup embed_tokens: {e}")))?;

    // 3) Decoder: single-token input_embeds with empty KV cache — model expects f16.
    let embeds = ArrayD::from_elem(IxDyn(&[1, 1, HIDDEN_DIM]), half::f16::ZERO);
    let pos = Array3::from_shape_vec((3, 1, 1), vec![0i64, 0, 0]).unwrap();
    let empty_kv = ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), half::f16::ZERO);

    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(4 + NUM_DECODER_LAYERS * 2);
    inputs.push((
        "inputs_embeds".into(),
        Value::from_array(embeds).unwrap().into(),
    ));
    inputs.push((
        "position_ids".into(),
        Value::from_array(pos).unwrap().into(),
    ));
    // attention_mask: [1, past_len + seq_len] = [1, 0 + 1] = [1, 1]
    // Only add if the model supports it (new export with attention_mask input).
    if has_attention_mask {
        let attn_mask = Array2::from_shape_vec((1, 1), vec![1i64]).unwrap();
        inputs.push((
            "attention_mask".into(),
            Value::from_array(attn_mask).unwrap().into(),
        ));
    }
    for layer in 0..NUM_DECODER_LAYERS {
        for kind in &["key", "value"] {
            inputs.push((
                format!("past_key_values.{layer}.{kind}"),
                Value::from_array(empty_kv.clone()).unwrap().into(),
            ));
        }
    }
    let _dec = decoder
        .run(inputs)
        .map_err(|e| Error::Asr(format!("warmup decoder: {e}")))?;

    tracing::debug!("ASR warmup complete");
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[cold]
#[inline(never)]
fn lock_err(context: &'static str, e: impl std::fmt::Display) -> Error {
    Error::Asr(format!("{context}: {e}"))
}

/// Average confidence from log-probs: exp(mean(log_probs)).
fn avg_confidence(log_probs: &[f32]) -> Option<f32> {
    if log_probs.is_empty() {
        return None;
    }
    Some((log_probs.iter().sum::<f32>() / log_probs.len() as f32).exp())
}

/// Minimum confidence from log-probs: exp(min(log_probs)).
fn min_confidence(log_probs: &[f32]) -> Option<f32> {
    log_probs
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .map(|lp| lp.exp())
}

// ── Per-word confidence ─────────────────────────────────────────────────────

/// Aggregate BPE subword tokens into per-word confidence values.
///
/// Word boundary = raw BPE token starts with Ġ (U+0120) or is the first token.
/// Per-word confidence = minimum exp(log_prob) across constituent subwords
/// (conservative: one uncertain subword flags the whole word).
fn tokens_to_word_confidences(
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[i64],
    log_probs: &[f32],
) -> Vec<WordConfidence> {
    debug_assert_eq!(token_ids.len(), log_probs.len());
    if token_ids.is_empty() {
        return vec![];
    }

    let mut words: Vec<WordConfidence> = Vec::new();
    let mut current_word = String::new();
    let mut current_min_prob = f32::INFINITY;

    for (i, (&tok_id, &lp)) in token_ids.iter().zip(log_probs.iter()).enumerate() {
        let raw_token = match tokenizer.id_to_token(tok_id as u32) {
            Some(t) => t,
            None => continue,
        };

        // Word boundary: first token or starts with Ġ (GPT-2 BPE word-initial marker).
        let is_boundary = i == 0 || raw_token.starts_with('Ġ');

        if is_boundary && !current_word.is_empty() {
            // Push previous word.
            words.push(WordConfidence {
                word: std::mem::take(&mut current_word),
                confidence: current_min_prob.exp(),
            });
            current_min_prob = f32::INFINITY;
        }

        // Decode the token to get clean text (strips Ġ prefix).
        let decoded = tokenizer.decode(&[tok_id as u32], true).unwrap_or_default();
        let trimmed = decoded.trim();
        if trimmed.is_empty() {
            continue;
        }

        current_word.push_str(trimmed);
        current_min_prob = current_min_prob.min(lp);
    }

    // Push final word.
    if !current_word.is_empty() {
        words.push(WordConfidence {
            word: current_word,
            confidence: current_min_prob.exp(),
        });
    }

    words
}

// ── Chunked inference ───────────────────────────────────────────────────────

/// A word with its confidence and timestamp (from FA or proportional estimate).
struct TimedWord {
    word: String,
    confidence: f32,
    /// Absolute start time in milliseconds from beginning of full audio.
    abs_start_ms: u32,
    /// Source chunk index (for overlap dedup: prefer more interior word).
    chunk_idx: usize,
    /// Source chunk's absolute start time in ms.
    chunk_start_ms: u32,
    /// Source chunk's absolute end time in ms.
    chunk_end_ms: u32,
}

impl OrtQwenAsrBackend {
    /// Attach a forced aligner for timestamp-based chunk merging.
    /// Called after construction when the FA model is available.
    pub fn attach_aligner(&self, fa: Arc<OrtForcedAligner>) {
        let _ = self.aligner.set(fa);
    }

    /// Split long audio into overlapping 8s chunks, decode each independently,
    /// and merge using FA timestamps (preferred) or LCS fallback.
    ///
    /// Locks encoder/embed_tokens/decoder per-chunk so FA can re-acquire
    /// the same Arc<Mutex> sessions between chunks.
    fn transcribe_chunked(
        &self,
        pcm: &[f32],
        forced_language: Option<&str>,
    ) -> Result<(String, Vec<WordConfidence>)> {
        let mut chunk_results: Vec<(usize, String, ort_qwen_stream::OneshotResult)> = Vec::new();
        // Accumulated token IDs from previous chunks — gives the decoder
        // linguistic context for continuation (what was already transcribed).
        let mut past_text_tokens: Vec<i64> = Vec::new();

        let mut offset = 0usize;
        while offset < pcm.len() {
            let end = (offset + CHUNK_SAMPLES).min(pcm.len());
            let chunk = &pcm[offset..end];

            let result = {
                let mut enc = self
                    .encoder
                    .lock()
                    .map_err(|e| lock_err("encoder lock", e))?;
                let mut emb = self
                    .embed_tokens
                    .lock()
                    .map_err(|e| lock_err("embed_tokens lock", e))?;
                let mut dec = self
                    .decoder
                    .lock()
                    .map_err(|e| lock_err("decoder lock", e))?;
                ort_qwen_stream::transcribe_oneshot(
                    &mut enc,
                    &mut emb,
                    &mut dec,
                    &self.tokenizer,
                    chunk,
                    forced_language,
                    self.has_attention_mask,
                    &past_text_tokens,
                )?
            }; // locks dropped here — FA can acquire them

            let text = result.text.trim().to_owned();
            if !text.is_empty() {
                tracing::debug!(
                    chunk = chunk_results.len(),
                    offset_s = offset as f32 / 16_000.0,
                    words = text.split_whitespace().count(),
                    "chunked ASR segment"
                );
                // Accumulate tokens for next chunk's past_text conditioning.
                past_text_tokens.extend_from_slice(&result.token_ids);
                chunk_results.push((offset, text, result));
            }

            if end == pcm.len() {
                break;
            }
            offset += STEP_SAMPLES;
        }

        if chunk_results.is_empty() {
            return Ok((String::new(), vec![]));
        }

        // Single chunk — no merge needed.
        if chunk_results.len() == 1 {
            let (_, text, result) = chunk_results.into_iter().next().unwrap();
            let wc =
                tokens_to_word_confidences(&self.tokenizer, &result.token_ids, &result.log_probs);
            return Ok((text, wc));
        }

        // Try FA-based timestamp merge, fall back to LCS.
        if let Some(fa) = self.aligner.get() {
            match self.merge_with_fa(fa, pcm, &chunk_results) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    tracing::warn!(error = %e, "FA timestamp merge failed, falling back to LCS");
                }
            }
        }

        // Fallback: LCS merge with per-chunk word confidences.
        self.merge_with_lcs(&chunk_results)
    }

    /// Merge chunks using FA timestamps: align each chunk, assign absolute
    /// timestamps, collect all words tagged by source chunk, then resolve
    /// overlaps by keeping the word more interior to its source chunk.
    fn merge_with_fa(
        &self,
        fa: &OrtForcedAligner,
        pcm: &[f32],
        chunks: &[(usize, String, ort_qwen_stream::OneshotResult)],
    ) -> Result<(String, Vec<WordConfidence>)> {
        let mut all_timed: Vec<TimedWord> = Vec::new();

        for (i, (offset, text, result)) in chunks.iter().enumerate() {
            let end = (*offset + CHUNK_SAMPLES).min(pcm.len());
            let chunk_pcm = &pcm[*offset..end];
            let chunk_start_ms = (*offset as u32) * 1000 / 16_000;
            let chunk_end_ms = (end as u32) * 1000 / 16_000;

            // Per-chunk word confidences from BPE tokens.
            let bpe_confs =
                tokens_to_word_confidences(&self.tokenizer, &result.token_ids, &result.log_probs);

            // FA align for timestamps.
            let timestamps = fa.align(chunk_pcm, text)?;

            // Greedy text-match scanner: walk BPE words forward to find
            // confidence for each FA word by text content (not position).
            // Handles FA/BPE word count mismatches from tokenization differences.
            let mut bpe_cursor = 0usize;
            for wt in &timestamps {
                let fa_word_lower = wt.text.to_ascii_lowercase();
                let confidence = match bpe_confs[bpe_cursor..]
                    .iter()
                    .position(|wc| wc.word.to_ascii_lowercase() == fa_word_lower)
                {
                    Some(rel_idx) => {
                        let matched = &bpe_confs[bpe_cursor + rel_idx];
                        bpe_cursor += rel_idx + 1; // consume up to and including match
                        matched.confidence
                    }
                    None => 0.5, // no BPE match — neutral confidence
                };

                all_timed.push(TimedWord {
                    word: wt.text.clone(),
                    confidence,
                    abs_start_ms: chunk_start_ms + wt.start_ms,
                    chunk_idx: i,
                    chunk_start_ms,
                    chunk_end_ms,
                });
            }
        }

        // Sort all words by timestamp.
        all_timed.sort_by_key(|tw| tw.abs_start_ms);

        // Resolve overlaps: when consecutive words from different chunks
        // cover the same region, keep the one more interior to its source chunk.
        // "Interiority" = min distance from either chunk edge.
        let mut resolved: Vec<TimedWord> = Vec::with_capacity(all_timed.len());
        let mut i = 0;
        while i < all_timed.len() {
            // Look ahead for words from a different chunk at similar timestamp.
            // "Similar" = within overlap duration (2s).
            let overlap_ms = (OVERLAP_SAMPLES as u32) * 1000 / 16_000;
            let mut group_end = i + 1;
            while group_end < all_timed.len()
                && all_timed[group_end]
                    .abs_start_ms
                    .saturating_sub(all_timed[i].abs_start_ms)
                    < overlap_ms
                && all_timed[group_end]
                    .word
                    .eq_ignore_ascii_case(&all_timed[i].word)
                && all_timed[group_end].chunk_idx != all_timed[i].chunk_idx
            {
                group_end += 1;
            }

            if group_end > i + 1 {
                // Multiple versions of the same word from different chunks.
                // Pick the one most interior to its source chunk.
                let best = all_timed[i..group_end]
                    .iter()
                    .max_by_key(|tw| {
                        let dist_from_start = tw.abs_start_ms.saturating_sub(tw.chunk_start_ms);
                        let dist_from_end = tw.chunk_end_ms.saturating_sub(tw.abs_start_ms);
                        dist_from_start.min(dist_from_end) // interiority
                    })
                    .unwrap();
                resolved.push(TimedWord {
                    word: best.word.clone(),
                    confidence: best.confidence,
                    abs_start_ms: best.abs_start_ms,
                    chunk_idx: best.chunk_idx,
                    chunk_start_ms: best.chunk_start_ms,
                    chunk_end_ms: best.chunk_end_ms,
                });
                i = group_end;
            } else {
                // No overlap conflict — keep as-is.
                let tw = &all_timed[i];
                resolved.push(TimedWord {
                    word: tw.word.clone(),
                    confidence: tw.confidence,
                    abs_start_ms: tw.abs_start_ms,
                    chunk_idx: tw.chunk_idx,
                    chunk_start_ms: tw.chunk_start_ms,
                    chunk_end_ms: tw.chunk_end_ms,
                });
                i += 1;
            }
        }

        // Safety net: adjacent dedup for any remaining artifacts.
        resolved.dedup_by(|b, a| a.word.eq_ignore_ascii_case(&b.word));

        let text = resolved
            .iter()
            .map(|tw| tw.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let word_confidences = resolved
            .into_iter()
            .map(|tw| WordConfidence {
                word: tw.word,
                confidence: tw.confidence,
            })
            .collect();

        Ok((text, word_confidences))
    }

    /// Fallback merge using LCS word-overlap stitching with per-chunk
    /// word confidences carried through.
    fn merge_with_lcs(
        &self,
        chunks: &[(usize, String, ort_qwen_stream::OneshotResult)],
    ) -> Result<(String, Vec<WordConfidence>)> {
        let mut merged_words: Vec<String> = Vec::new();
        let mut merged_confs: Vec<f32> = Vec::new();

        for (_, _, result) in chunks {
            let text = result.text.trim();
            if text.is_empty() {
                continue;
            }
            let chunk_words: Vec<String> = text.split_whitespace().map(String::from).collect();
            let chunk_wc =
                tokens_to_word_confidences(&self.tokenizer, &result.token_ids, &result.log_probs);

            // Build confidence vector aligned to chunk_words.
            let chunk_confs: Vec<f32> = chunk_words
                .iter()
                .enumerate()
                .map(|(j, _)| chunk_wc.get(j).map_or(0.5, |wc| wc.confidence))
                .collect();

            if merged_words.is_empty() {
                merged_words = chunk_words;
                merged_confs = chunk_confs;
            } else {
                let new_merged = lcs_merge_words(&merged_words, &chunk_words);
                // Rebuild confidence: words from merged keep their conf,
                // new words from chunk_words get chunk confidence.
                let mut new_confs = Vec::with_capacity(new_merged.len());
                let mut mi = 0usize; // index into old merged_words
                let mut ci = 0usize; // index into chunk_words
                for w in &new_merged {
                    if mi < merged_words.len() && w.eq_ignore_ascii_case(&merged_words[mi]) {
                        new_confs.push(merged_confs[mi]);
                        mi += 1;
                    } else if ci < chunk_words.len() && w.eq_ignore_ascii_case(&chunk_words[ci]) {
                        new_confs.push(chunk_confs[ci]);
                        ci += 1;
                    } else {
                        new_confs.push(0.5); // unmatched — neutral confidence
                    }
                }
                merged_words = new_merged;
                merged_confs = new_confs;
            }
        }

        // Adjacent dedup.
        let mut deduped_words: Vec<String> = Vec::with_capacity(merged_words.len());
        let mut deduped_confs: Vec<f32> = Vec::with_capacity(merged_confs.len());
        for (i, w) in merged_words.iter().enumerate() {
            if i == 0 || !w.eq_ignore_ascii_case(&merged_words[i - 1]) {
                deduped_words.push(w.clone());
                deduped_confs.push(merged_confs[i]);
            }
        }

        let text = deduped_words.join(" ");
        let word_confidences = deduped_words
            .into_iter()
            .zip(deduped_confs)
            .map(|(word, confidence)| WordConfidence { word, confidence })
            .collect();

        Ok((text, word_confidences))
    }
}

// ── AsrBackend implementation ───────────────────────────────────────────────

#[async_trait::async_trait]
impl AsrBackend for OrtQwenAsrBackend {
    type SessionHandle = OrtQwenAsrSession;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        if config.sample_rate != 16_000 {
            return Err(Error::Asr(format!(
                "ORT Qwen ASR expects 16000 Hz mono PCM, got {} Hz",
                config.sample_rate
            )));
        }

        let forced_language = match config
            .language
            .as_deref()
            .map(str::trim)
            .filter(|language| !language.is_empty())
        {
            Some(language) => {
                let mapped = map_language_tag(language).ok_or_else(|| {
                    Error::Asr(format!(
                        "ORT Qwen ASR does not support forced language hint `{language}`"
                    ))
                })?;
                // Skip prompt for English — model defaults to English and the prompt
                // causes hallucination after context reset (decoder outputs the prompt
                // text when there's no past-text context to anchor generation).
                if mapped == "English" {
                    None
                } else {
                    Some(format!("Transcribe the {mapped} audio into text."))
                }
            }
            None => None,
        };

        Ok(OrtQwenAsrSession {
            state: Mutex::new(SessionState {
                audio_buf: Vec::with_capacity(16_000 * 8),
                next_id: 1,
                sample_rate: config.sample_rate,
                forced_language,
            }),
        })
    }

    #[inline]
    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;
        state.audio_buf.extend_from_slice(pcm);
        // Cap at 60s (MAX_ENCODER_SAMPLES). Trim oldest audio from front.
        if state.audio_buf.len() > MAX_ENCODER_SAMPLES {
            let trim = state.audio_buf.len() - MAX_ENCODER_SAMPLES;
            state.audio_buf.drain(..trim);
        }
        Ok(())
    }

    async fn poll_results(&self, _session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        // Batch mode: accumulate audio, defer all transcription to poll_final.
        Ok(vec![])
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;

        if state.audio_buf.is_empty() {
            return Ok(vec![]);
        }

        let id = state.next_id;
        state.next_id = id.saturating_add(1);
        let total = state.audio_buf.len();
        let sample_rate = state.sample_rate;

        let (text, word_confidences, confidence, min_token_confidence) = if total > CHUNK_THRESHOLD
        {
            // Chunked path: locks per-chunk internally, returns merged word confidences.
            let (text, wc) =
                self.transcribe_chunked(&state.audio_buf, state.forced_language.as_deref())?;
            // Derive segment-level confidence from per-word confidences.
            let conf = if wc.is_empty() {
                None
            } else {
                let sum: f32 = wc.iter().map(|w| w.confidence).sum();
                Some(sum / wc.len() as f32)
            };
            let min_conf = wc
                .iter()
                .map(|w| w.confidence)
                .min_by(|a, b| a.partial_cmp(b).unwrap());
            (text, wc, conf, min_conf)
        } else {
            // Short audio: single-pass oneshot.
            let result = {
                let mut enc = self
                    .encoder
                    .lock()
                    .map_err(|e| lock_err("encoder lock", e))?;
                let mut emb = self
                    .embed_tokens
                    .lock()
                    .map_err(|e| lock_err("embed_tokens lock", e))?;
                let mut dec = self
                    .decoder
                    .lock()
                    .map_err(|e| lock_err("decoder lock", e))?;
                ort_qwen_stream::transcribe_oneshot(
                    &mut enc,
                    &mut emb,
                    &mut dec,
                    &self.tokenizer,
                    &state.audio_buf,
                    state.forced_language.as_deref(),
                    self.has_attention_mask,
                    &[],
                )?
            };
            let wc =
                tokens_to_word_confidences(&self.tokenizer, &result.token_ids, &result.log_probs);
            let conf = avg_confidence(&result.log_probs);
            let min_conf = min_confidence(&result.log_probs);
            (result.text, wc, conf, min_conf)
        };

        let text = text.trim().to_owned();
        if text.is_empty() {
            return Ok(vec![]);
        }

        let end_ms = (total as u64).saturating_mul(1000) / sample_rate as u64;

        Ok(vec![AsrSegment {
            id,
            confidence,
            min_token_confidence,
            word_confidences,
            is_final: true,
            start_ms: Some(0),
            end_ms: Some(end_ms),
            text,
        }])
    }

    async fn end_session(&self, _session: Self::SessionHandle) -> Result<()> {
        Ok(())
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;
        state.audio_buf.clear();
        Ok(())
    }

    fn segment_capacity_hint(&self) -> usize {
        1
    }
}

impl OrtQwenAsrBackend {
    /// Build an `OrtForcedAligner` using a wav2vec2 CTC model.
    ///
    /// `fa_model_dir` must contain:
    ///   - `model.onnx` (+ `model.onnx.data`) — wav2vec2 CTC ONNX model
    ///   - `fa_config.json` — labels, blank_id, stride_ms, etc.
    pub fn make_forced_aligner(&self, fa_model_dir: &Path) -> Result<OrtForcedAligner> {
        let model_path = fa_model_dir.join("model.onnx");
        let config_path = fa_model_dir.join("fa_config.json");

        let fa_model = build_ort_session(&model_path, self.config.use_gpu)?;

        let cfg_bytes = std::fs::read(&config_path)
            .map_err(|e| Error::Asr(format!("fa_config.json read: {e}")))?;
        let fa_config: FaConfig = serde_json::from_slice(&cfg_bytes)
            .map_err(|e| Error::Asr(format!("fa_config.json parse: {e}")))?;

        OrtForcedAligner::new(fa_model, fa_config)
    }
}
