//! Streaming state machine for the ORT Qwen ASR backend.
//!
//! Ports the streaming algorithm from the `qwen-asr` crate to ONNX Runtime.
//! The encoder, embed_tokens, and decoder are ORT sessions passed by reference;
//! this module owns only the streaming state and mel computation.

use std::sync::OnceLock;

use ndarray::{Array2, Array3, ArrayD, IxDyn, s};
use ort::session::Session;
use ort::value::{DynValue, Value};
use rustfft::{FftPlanner, num_complex::Complex};

use lune_core::error::{Error, Result};

// ── Architecture constants (Qwen3-ASR-0.6B) ─────────────────────────────────

pub(crate) const NUM_DECODER_LAYERS: usize = 28;
pub(crate) const NUM_KV_HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const HIDDEN_DIM: usize = 1024;

// ── Mel spectrogram constants ────────────────────────────────────────────────

const MEL_BINS: usize = 128;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const N_FFT: usize = 400;
const N_FREQ: usize = N_FFT / 2 + 1; // 201
const SAMPLE_RATE: usize = 16_000;

// ── Streaming thresholds ─────────────────────────────────────────────────────

const DEGEN_MAX_PERIOD: usize = 6;
const DEGEN_MIN_REPEATS: usize = 4;
const STALE_CHUNKS: i32 = 4;
const RESET_INTERVAL_CHUNKS: i32 = 45;
const RESET_CARRY_TOKENS: usize = 24;

/// Maximum audio window passed to the encoder (30 seconds). Caps encoder
/// cost to O(1) per chunk regardless of total session length.
const MAX_ENCODER_SAMPLES: usize = 30 * SAMPLE_RATE;

// ── Special tokens (Qwen3 vocabulary) ────────────────────────────────────────

const TOKEN_IM_START: i64 = 151644;
const TOKEN_IM_END: i64 = 151645;
const TOKEN_AUDIO_START: i64 = 151669;
const TOKEN_AUDIO_END: i64 = 151670;
const TOKEN_ASR_TEXT: i64 = 151704;

/// Prefix tokens before the audio: `<|im_start|>system\n`
const PREFIX_HEAD: &[i64] = &[TOKEN_IM_START, 8948, 198];
/// Suffix after prefix head / before audio: `<|im_end|>\n<|im_start|>user\n<audio>`
const PREFIX_TAIL: &[i64] = &[
    TOKEN_IM_END,
    198,
    TOKEN_IM_START,
    872,
    198,
    TOKEN_AUDIO_START,
];
/// After audio: `</audio><|im_end|>\n<|im_start|>assistant\n`
const SUFFIX_BASE: &[i64] = &[
    TOKEN_AUDIO_END,
    TOKEN_IM_END,
    198,
    TOKEN_IM_START,
    77091,
    198,
];

// ── Mel spectrogram computation (Qwen ASR specific) ─────────────────────────

// Slaney mel scale constants (same as mel.rs).
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.068_751_777_420_949_12;

static QWEN_MEL_FB: OnceLock<Array2<f32>> = OnceLock::new();
static QWEN_HANN: OnceLock<Vec<f32>> = OnceLock::new();

#[inline(always)]
fn hz_to_mel(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

#[inline(always)]
fn mel_to_hz(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

fn qwen_mel_filterbank() -> &'static Array2<f32> {
    QWEN_MEL_FB.get_or_init(|| {
        let freq_bins = N_FREQ;
        let mut fb = Array2::<f32>::zeros((MEL_BINS, freq_bins));
        let fmax = SAMPLE_RATE as f64 / 2.0;
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(fmax);
        let mel_range = mel_max - mel_min;
        let n_points = MEL_BINS + 2;
        let mel_points: Vec<f64> = (0..n_points)
            .map(|i| mel_to_hz(mel_min + mel_range * i as f64 / (MEL_BINS + 1) as f64))
            .collect();
        let fft_freqs: Vec<f64> = (0..freq_bins)
            .map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
            .collect();
        let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();
        let fb_slice = fb.as_slice_mut().expect("fb contiguous");
        for i in 0..MEL_BINS {
            let enorm = (2.0 / (mel_points[i + 2] - mel_points[i])) as f32;
            let lo = mel_points[i];
            let hi = mel_points[i + 2];
            let di0 = fdiff[i];
            let di1 = fdiff[i + 1];
            for (k, &freq) in fft_freqs.iter().enumerate() {
                let lower = (freq - lo) / di0;
                let upper = (hi - freq) / di1;
                fb_slice[i * freq_bins + k] = lower.min(upper).max(0.0) as f32 * enorm;
            }
        }
        fb
    })
}

fn qwen_hann() -> &'static [f32] {
    QWEN_HANN.get_or_init(|| {
        let scale = 2.0 * std::f32::consts::PI / (WIN_LENGTH as f32 - 1.0);
        (0..WIN_LENGTH)
            .map(|i| 0.5 - 0.5 * (scale * i as f32).cos())
            .collect()
    })
}

/// Compute Qwen-style mel spectrogram: reflection pad, STFT, mel filterbank,
/// log10 + clamp + normalize. Returns `[1, 128, T]`.
pub(crate) fn compute_qwen_mel(audio: &[f32]) -> Array3<f32> {
    if audio.is_empty() {
        return Array3::<f32>::zeros((1, MEL_BINS, 0));
    }

    let pad = N_FFT / 2; // 200
    let padded_len = pad + audio.len() + pad;
    let mut padded = vec![0.0f32; padded_len];

    // Reflection padding.
    for i in 0..pad {
        padded[pad - 1 - i] = audio[(i + 1).min(audio.len() - 1)];
    }
    padded[pad..pad + audio.len()].copy_from_slice(audio);
    for i in 0..pad {
        let src_idx = audio.len().saturating_sub(2).saturating_sub(i);
        padded[pad + audio.len() + i] = audio[src_idx];
    }

    // STFT with Hann window.
    let window = qwen_hann();
    let num_frames = (padded_len - N_FFT) / HOP_LENGTH + 1;
    let mut power_spec = Array2::<f32>::zeros((N_FREQ, num_frames));

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let zero = Complex::new(0.0f32, 0.0f32);
    let mut frame = vec![zero; N_FFT];
    let mut scratch = vec![zero; fft.get_inplace_scratch_len()];

    let spec_slice = power_spec.as_slice_mut().expect("spec contiguous");
    for f in 0..num_frames {
        let start = f * HOP_LENGTH;
        frame.fill(zero);
        let copy_len = WIN_LENGTH.min(padded_len - start);
        for (c, (&s, &w)) in frame[..copy_len].iter_mut().zip(
            padded[start..start + copy_len]
                .iter()
                .zip(window[..copy_len].iter()),
        ) {
            c.re = s * w;
        }
        fft.process_with_scratch(&mut frame, &mut scratch);
        for k in 0..N_FREQ {
            spec_slice[k * num_frames + f] = frame[k].norm_sqr();
        }
    }

    // Mel filterbank dot product.
    let mel_fb = qwen_mel_filterbank();
    let mut mel = mel_fb.dot(&power_spec); // [128, num_frames]

    // Log10 + clamp + normalize (Qwen-specific).
    let mel_data = mel.as_slice_mut().expect("mel contiguous");
    let mut global_max = f32::NEG_INFINITY;
    for v in mel_data.iter_mut() {
        *v = (*v).max(1e-10).log10();
        if *v > global_max {
            global_max = *v;
        }
    }
    let clamp_min = global_max - 8.0;
    for v in mel_data.iter_mut() {
        *v = ((*v).max(clamp_min) + 4.0) / 4.0;
    }

    // Reshape to [1, 128, T].
    let n_frames = mel.shape()[1];
    let (raw, _offset) = mel.into_raw_vec_and_offset();
    // mel is [128, T] row-major. Array3 [1, 128, T] has same layout.
    Array3::from_shape_vec((1, MEL_BINS, n_frames), raw).expect("mel reshape to [1, 128, T]")
}

// ── Streaming state ─────────────────────────────────────────────────────────

/// Persistent streaming state between chunks.
pub(crate) struct OrtStreamState {
    /// Raw decoded tokens from all chunks (subject to rollback).
    pub raw_tokens: Vec<i64>,
    /// Stable committed tokens (monotonically increasing).
    pub stable_text_tokens: Vec<i64>,
    /// Accumulated result text from stable emissions.
    pub result: String,

    /// Previous prefill embeddings for LCP reuse (flat f32, row-major).
    prev_prefill_embeds: Vec<f32>,
    prev_prefill_len: usize,

    /// Previous tail token snapshot for stale detection.
    prev_tail_snapshot: Vec<i64>,
    stale_count: i32,

    /// Audio cursor: how many samples have been consumed.
    pub audio_cursor: usize,
    /// Chunk counter for periodic re-anchoring.
    chunk_idx: i32,

    /// KV cache from previous chunk (per-layer tensors).
    kv_cache: Vec<DynValue>,
    kv_cache_seq_len: usize,

    /// Cached encoder features from the most recent encoding pass.
    enc_cache: Option<ArrayD<f32>>,
    /// Audio sample position covered by `enc_cache`.
    enc_cache_audio_end: usize,
}

impl OrtStreamState {
    pub fn new() -> Self {
        Self {
            raw_tokens: Vec::new(),
            stable_text_tokens: Vec::new(),
            result: String::new(),
            prev_prefill_embeds: Vec::new(),
            prev_prefill_len: 0,
            prev_tail_snapshot: Vec::new(),
            stale_count: 0,
            audio_cursor: 0,
            chunk_idx: 0,
            kv_cache: Vec::new(),
            kv_cache_seq_len: 0,
            enc_cache: None,
            enc_cache_audio_end: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    #[inline]
    pub fn audio_cursor(&self) -> usize {
        self.audio_cursor
    }
}

// ── Streaming configuration ─────────────────────────────────────────────────

pub(crate) struct StreamConfig {
    pub chunk_samples: usize,
    pub rollback: usize,
    pub unfixed_chunks: i32,
    pub max_new_tokens: usize,
    pub past_text_conditioning: bool,
}

// ── Main streaming function ─────────────────────────────────────────────────

/// Push audio and run one streaming decode cycle.
///
/// Returns the delta text (newly committed tokens decoded to string).
pub(crate) fn stream_push_audio(
    encoder: &mut Session,
    embed_tokens: &mut Session,
    decoder: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    audio_buf: &[f32],
    state: &mut OrtStreamState,
    cfg: &StreamConfig,
    finalize: bool,
) -> Result<String> {
    let total_samples = audio_buf.len();
    if total_samples == 0 && !finalize {
        return Ok(String::new());
    }

    let prev_stable_len = state.stable_text_tokens.len();

    // Process chunks until we've consumed enough audio.
    loop {
        let cursor = state.audio_cursor;
        let remaining = total_samples.saturating_sub(cursor);

        // Need at least one full chunk, or finalize with remaining audio.
        if remaining < cfg.chunk_samples && !finalize {
            break;
        }
        if remaining == 0 && finalize && state.chunk_idx > 0 {
            break;
        }

        // Determine audio slice for this chunk.
        let chunk_end = if remaining >= cfg.chunk_samples {
            cursor + cfg.chunk_samples
        } else {
            total_samples // finalize: use all remaining
        };

        // Encode with caching: sliding window cap + lazy skip.
        let need_reencode = match state.enc_cache {
            Some(_) => chunk_end.saturating_sub(state.enc_cache_audio_end) >= cfg.chunk_samples / 2,
            None => true,
        };

        let (enc_features, enc_seq_len) = if need_reencode {
            let encode_start = chunk_end.saturating_sub(MAX_ENCODER_SAMPLES);
            let mel = compute_qwen_mel(&audio_buf[encode_start..chunk_end]);
            let features = run_encoder(encoder, mel)?;
            let seq_len = features.shape()[1];
            state.enc_cache = Some(features.clone());
            state.enc_cache_audio_end = chunk_end;
            (features, seq_len)
        } else {
            let cache = state.enc_cache.as_ref().unwrap();
            (cache.clone(), cache.shape()[1])
        };

        // Build input_embeds: prefix + encoder_features + suffix.
        let (input_embeds, prefill_len) = build_input_embeds(
            embed_tokens,
            &enc_features,
            enc_seq_len,
            &state.raw_tokens,
            &state.stable_text_tokens,
            cfg.past_text_conditioning,
        )?;

        // LCP reuse: compare with previous prefill.
        let reused = compute_lcp_reuse(
            &state.prev_prefill_embeds,
            state.prev_prefill_len,
            &input_embeds,
            prefill_len,
        );

        // Truncate KV cache to reused prefix length.
        let kv_cache = if reused > 0 && !state.kv_cache.is_empty() {
            truncate_kv_cache(&state.kv_cache, state.kv_cache_seq_len, reused)?
        } else {
            empty_kv_cache()
        };

        // Decoder prefill for delta embeddings.
        let delta_start = reused;
        let delta_len = prefill_len - reused;
        let (mut current_kv, last_logits) = if delta_len > 0 {
            let delta_embeds = input_embeds.slice(s![.., delta_start..prefill_len, ..]);
            let delta_arr = delta_embeds.to_owned().into_dyn();
            decoder_prefill(decoder, delta_arr, kv_cache, reused)?
        } else {
            // Full reuse — no prefill needed. Extract logits from cache isn't
            // possible, so we do a minimal single-token re-prefill of the last token.
            let last_embed = input_embeds.slice(s![.., prefill_len - 1..prefill_len, ..]);
            let kv_trunc = if reused > 1 {
                truncate_kv_cache(&state.kv_cache, state.kv_cache_seq_len, reused - 1)?
            } else {
                empty_kv_cache()
            };
            decoder_prefill(
                decoder,
                last_embed.to_owned().into_dyn(),
                kv_trunc,
                reused.saturating_sub(1),
            )?
        };

        // Save prefill embeddings for next chunk's LCP comparison.
        let embeds_data = input_embeds.as_slice().unwrap_or(&[]);
        state.prev_prefill_embeds = embeds_data[..prefill_len * HIDDEN_DIM].to_vec();
        state.prev_prefill_len = prefill_len;

        // Autoregressive decode: generate up to max_new_tokens.
        let chunk_tokens = autoregressive_decode(
            decoder,
            embed_tokens,
            tokenizer,
            &mut current_kv,
            last_logits,
            cfg.max_new_tokens,
        )?;

        let kv_seq_after_decode = reused.max(delta_start + delta_len) + chunk_tokens.len();

        // Speech-end detection: empty decode → commit all rollback tokens.
        let is_speech_end = chunk_tokens.is_empty() && state.chunk_idx >= cfg.unfixed_chunks;

        // Append decoded tokens to raw_tokens.
        if !chunk_tokens.is_empty() {
            // Ensure raw_tokens starts with ASR_TEXT if this is first output.
            if state.raw_tokens.is_empty() {
                state.raw_tokens.push(TOKEN_ASR_TEXT);
            }
            state.raw_tokens.extend_from_slice(&chunk_tokens);
        }

        // Degeneracy detection.
        let (best_reps, _) = tail_repeat_blocks(&state.raw_tokens, DEGEN_MAX_PERIOD);
        if state.raw_tokens == state.prev_tail_snapshot {
            state.stale_count += 1;
        } else {
            state.stale_count = 0;
        }
        state.prev_tail_snapshot = state.raw_tokens.clone();

        let is_degen = state.stale_count >= STALE_CHUNKS || best_reps >= DEGEN_MIN_REPEATS;

        // Periodic re-anchor.
        let is_reanchor = state.chunk_idx > 0 && state.chunk_idx % RESET_INTERVAL_CHUNKS == 0;

        if is_degen || is_reanchor {
            // Reset: carry last RESET_CARRY_TOKENS stable tokens.
            let carry_start = state
                .stable_text_tokens
                .len()
                .saturating_sub(RESET_CARRY_TOKENS);
            state.raw_tokens.clear();
            if carry_start < state.stable_text_tokens.len() {
                state.raw_tokens.push(TOKEN_ASR_TEXT);
                state
                    .raw_tokens
                    .extend_from_slice(&state.stable_text_tokens[carry_start..]);
            }
            state.prev_prefill_embeds.clear();
            state.prev_prefill_len = 0;
            state.kv_cache.clear();
            state.kv_cache_seq_len = 0;
            state.enc_cache = None;
            state.enc_cache_audio_end = 0;
        } else {
            state.kv_cache = current_kv;
            state.kv_cache_seq_len = kv_seq_after_decode;
        }

        // Rollback-buffered token emission.
        let emit_all = is_speech_end || finalize;
        let frontier = if emit_all {
            state.raw_tokens.len()
        } else {
            state.raw_tokens.len().saturating_sub(cfg.rollback)
        };

        // Emit tokens beyond current stable frontier.
        if frontier > state.stable_text_tokens.len() {
            // Find text tokens only (skip ASR_TEXT marker).
            let new_tokens: Vec<i64> = state.raw_tokens[state.stable_text_tokens.len()..frontier]
                .iter()
                .copied()
                .filter(|&t| t != TOKEN_ASR_TEXT)
                .collect();
            state.stable_text_tokens.extend_from_slice(&new_tokens);
        }

        state.audio_cursor = chunk_end;
        state.chunk_idx += 1;

        if is_speech_end || (remaining < cfg.chunk_samples * 2 && !finalize) {
            break;
        }
    }

    // Decode newly emitted stable tokens to text.
    let new_stable = &state.stable_text_tokens[prev_stable_len..];
    if new_stable.is_empty() {
        return Ok(String::new());
    }

    let token_ids: Vec<u32> = new_stable.iter().map(|&t| t as u32).collect();
    let delta = tokenizer
        .decode(&token_ids, true)
        .map_err(|e| Error::Asr(format!("token decode: {e}")))?;

    state.result.push_str(&delta);
    Ok(delta)
}

// ── Encoder ─────────────────────────────────────────────────────────────────

fn run_encoder(encoder: &mut Session, mel: Array3<f32>) -> Result<ArrayD<f32>> {
    let mel_dyn = mel.into_dyn();
    let mel_val: DynValue = Value::from_array(mel_dyn)
        .map_err(|e| Error::Asr(format!("mel tensor: {e}")))?
        .into();
    let inputs = vec![("mel".to_string(), mel_val)];

    let outputs = encoder
        .run(inputs)
        .map_err(|e| Error::Asr(format!("encoder: {e}")))?;

    let features: ndarray::ArrayViewD<f32> = outputs[0]
        .try_extract_array()
        .map_err(|e| Error::Asr(format!("encoder features: {e}")))?;

    Ok(features.to_owned())
}

// ── Input embeddings ────────────────────────────────────────────────────────

/// Build the full input_embeds sequence for the decoder.
///
/// Returns `(input_embeds [1, total_len, HIDDEN_DIM], prefill_len)`.
fn build_input_embeds(
    embed_tokens: &mut Session,
    enc_features: &ArrayD<f32>,
    enc_seq_len: usize,
    _raw_tokens: &[i64],
    stable_tokens: &[i64],
    past_text_conditioning: bool,
) -> Result<(ArrayD<f32>, usize)> {
    // Collect all token IDs that need embedding.
    let mut pre_tokens: Vec<i64> = Vec::with_capacity(32);
    pre_tokens.extend_from_slice(PREFIX_HEAD);
    pre_tokens.extend_from_slice(PREFIX_TAIL);

    let mut post_tokens: Vec<i64> = Vec::with_capacity(32);
    post_tokens.extend_from_slice(SUFFIX_BASE);
    post_tokens.push(TOKEN_ASR_TEXT);

    // Add past text tokens for conditioning.
    if past_text_conditioning && !stable_tokens.is_empty() {
        post_tokens.extend_from_slice(stable_tokens);
    }

    // Get embeddings for pre and post tokens.
    let pre_embeds = token_embeddings(embed_tokens, &pre_tokens)?;
    let post_embeds = token_embeddings(embed_tokens, &post_tokens)?;

    let pre_len = pre_tokens.len();
    let post_len = post_tokens.len();
    let total_len = pre_len + enc_seq_len + post_len;

    // Concatenate: pre_embeds + enc_features + post_embeds → [1, total_len, HIDDEN_DIM]
    let mut combined = vec![0.0f32; total_len * HIDDEN_DIM];

    // Copy pre_embeds [1, pre_len, HIDDEN_DIM].
    let pre_data = pre_embeds.as_slice().unwrap_or(&[]);
    combined[..pre_len * HIDDEN_DIM].copy_from_slice(&pre_data[..pre_len * HIDDEN_DIM]);

    // Copy encoder features [1, enc_seq_len, HIDDEN_DIM].
    let enc_data = enc_features.as_slice().unwrap_or(&[]);
    let enc_offset = pre_len * HIDDEN_DIM;
    combined[enc_offset..enc_offset + enc_seq_len * HIDDEN_DIM]
        .copy_from_slice(&enc_data[..enc_seq_len * HIDDEN_DIM]);

    // Copy post_embeds [1, post_len, HIDDEN_DIM].
    let post_data = post_embeds.as_slice().unwrap_or(&[]);
    let post_offset = (pre_len + enc_seq_len) * HIDDEN_DIM;
    combined[post_offset..post_offset + post_len * HIDDEN_DIM]
        .copy_from_slice(&post_data[..post_len * HIDDEN_DIM]);

    let result = ArrayD::from_shape_vec(IxDyn(&[1, total_len, HIDDEN_DIM]), combined)
        .map_err(|e| Error::Asr(format!("input_embeds shape: {e}")))?;

    Ok((result, total_len))
}

/// Get token embeddings from the embed_tokens ONNX session.
fn token_embeddings(embed_tokens: &mut Session, tokens: &[i64]) -> Result<ArrayD<f32>> {
    let ids = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())
        .map_err(|e| Error::Asr(format!("token ids array: {e}")))?;
    let ids_val: DynValue = Value::from_array(ids)
        .map_err(|e| Error::Asr(format!("token ids tensor: {e}")))?
        .into();
    let inputs = vec![("input_ids".to_string(), ids_val)];

    let outputs = embed_tokens
        .run(inputs)
        .map_err(|e| Error::Asr(format!("embed_tokens: {e}")))?;

    let embeds: ndarray::ArrayViewD<f32> = outputs[0]
        .try_extract_array()
        .map_err(|e| Error::Asr(format!("token embeds: {e}")))?;

    Ok(embeds.to_owned())
}

// ── LCP reuse ───────────────────────────────────────────────────────────────

/// Compare previous and current prefill embeddings row-by-row.
/// Returns the number of reusable prefix rows.
fn compute_lcp_reuse(
    prev_embeds: &[f32],
    prev_len: usize,
    curr_embeds: &ArrayD<f32>,
    curr_len: usize,
) -> usize {
    if prev_embeds.is_empty() || prev_len == 0 {
        return 0;
    }
    let curr_data = match curr_embeds.as_slice() {
        Some(d) => d,
        None => return 0,
    };
    let cmp_len = prev_len.min(curr_len);
    let mut reused = 0;
    while reused < cmp_len {
        let a = &prev_embeds[reused * HIDDEN_DIM..(reused + 1) * HIDDEN_DIM];
        let b = &curr_data[reused * HIDDEN_DIM..(reused + 1) * HIDDEN_DIM];
        if a != b {
            break;
        }
        reused += 1;
    }
    reused
}

// ── KV cache management ─────────────────────────────────────────────────────

/// Create empty KV cache tensors (seq_len=0) for all layers.
fn empty_kv_cache() -> Vec<DynValue> {
    let mut kv = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
    for _ in 0..NUM_DECODER_LAYERS * 2 {
        let arr = ArrayD::<f32>::zeros(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]));
        kv.push(Value::from_array(arr).expect("empty kv tensor").into());
    }
    kv
}

/// Truncate KV cache tensors to `target_len` along the sequence dimension.
fn truncate_kv_cache(
    kv_cache: &[DynValue],
    current_len: usize,
    target_len: usize,
) -> Result<Vec<DynValue>> {
    if target_len >= current_len {
        // Need to clone — can't borrow DynValue.
        return clone_kv_cache(kv_cache);
    }
    let mut result = Vec::with_capacity(kv_cache.len());
    for (i, kv) in kv_cache.iter().enumerate() {
        let (shape, data) = kv
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Asr(format!("kv extract[{i}]: {e}")))?;
        let dims: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        // Shape: [1, num_kv_heads, seq_len, head_dim]
        let full = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|e| Error::Asr(format!("kv array[{i}]: {e}")))?;
        let sliced = full.slice(s![.., .., ..target_len, ..]).to_owned();
        result.push(
            Value::from_array(sliced)
                .map_err(|e| Error::Asr(format!("kv value[{i}]: {e}")))?
                .into(),
        );
    }
    Ok(result)
}

/// Clone KV cache by extracting and recreating tensors.
fn clone_kv_cache(kv_cache: &[DynValue]) -> Result<Vec<DynValue>> {
    let mut result = Vec::with_capacity(kv_cache.len());
    for (i, kv) in kv_cache.iter().enumerate() {
        let (shape, data) = kv
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Asr(format!("kv clone[{i}]: {e}")))?;
        let dims: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|e| Error::Asr(format!("kv clone array[{i}]: {e}")))?;
        result.push(
            Value::from_array(arr)
                .map_err(|e| Error::Asr(format!("kv clone value[{i}]: {e}")))?
                .into(),
        );
    }
    Ok(result)
}

// ── Decoder prefill ─────────────────────────────────────────────────────────

/// Run decoder prefill: feed input_embeds with KV cache, return updated KV + logits.
fn decoder_prefill(
    decoder: &mut Session,
    input_embeds: ArrayD<f32>,
    kv_cache: Vec<DynValue>,
    past_seq_len: usize,
) -> Result<(Vec<DynValue>, Vec<f32>)> {
    let seq_len = input_embeds.shape()[1];
    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_DECODER_LAYERS * 2);

    inputs.push((
        "inputs_embeds".to_string(),
        Value::from_array(input_embeds)
            .map_err(|e| Error::Asr(format!("prefill embeds: {e}")))?
            .into(),
    ));

    // mRoPE position_ids: [3, 1, seq_len] — same sequential positions for all 3 dims.
    let positions: Vec<i64> = (past_seq_len..past_seq_len + seq_len)
        .map(|p| p as i64)
        .collect();
    let mut pos_data = Vec::with_capacity(3 * seq_len);
    for _ in 0..3 {
        pos_data.extend_from_slice(&positions);
    }
    let pos_ids = ndarray::Array3::<i64>::from_shape_vec((3, 1, seq_len), pos_data)
        .map_err(|e| Error::Asr(format!("position_ids: {e}")))?;
    inputs.push((
        "position_ids".to_string(),
        Value::from_array(pos_ids)
            .map_err(|e| Error::Asr(format!("position_ids value: {e}")))?
            .into(),
    ));

    for (i, kv) in kv_cache.into_iter().enumerate() {
        let layer = i / 2;
        let kind = if i % 2 == 0 { "key" } else { "value" };
        inputs.push((format!("past_key_values.{layer}.{kind}"), kv));
    }

    let outputs = decoder
        .run(inputs)
        .map_err(|e| Error::Asr(format!("decoder prefill: {e}")))?;

    // Extract logits — last position only.
    let logits_view: ndarray::ArrayViewD<f32> = outputs[0]
        .try_extract_array()
        .map_err(|e| Error::Asr(format!("prefill logits: {e}")))?;
    let seq_len = logits_view.shape()[1];
    let last_logits = logits_view.slice(s![0, seq_len - 1, ..]).to_vec();

    // Extract updated KV cache.
    let mut new_kv = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
    for i in 1..outputs.len() {
        let (shape, data) = outputs[i]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Asr(format!("prefill kv[{i}]: {e}")))?;
        let dims: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|e| Error::Asr(format!("prefill kv arr[{i}]: {e}")))?;
        new_kv.push(
            Value::from_array(arr)
                .map_err(|e| Error::Asr(format!("prefill kv val[{i}]: {e}")))?
                .into(),
        );
    }

    Ok((new_kv, last_logits))
}

// ── Autoregressive decode ───────────────────────────────────────────────────

/// Run autoregressive decode steps. Returns generated token IDs.
fn autoregressive_decode(
    decoder: &mut Session,
    embed_tokens: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    kv_cache: &mut Vec<DynValue>,
    initial_logits: Vec<f32>,
    max_tokens: usize,
) -> Result<Vec<i64>> {
    let eos_id = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(151643) as i64;

    let mut tokens = Vec::with_capacity(max_tokens);
    let mut current_logits = initial_logits;

    for _ in 0..max_tokens {
        // Greedy decode (temperature=0 for ASR).
        let next_token = argmax(&current_logits);

        if next_token == eos_id || next_token == TOKEN_IM_END {
            break;
        }

        tokens.push(next_token);

        // Get embedding for next token.
        let tok_embeds = token_embeddings(embed_tokens, &[next_token])?;

        // Determine current position from existing KV cache length.
        let cur_pos = if kv_cache.is_empty() {
            0usize
        } else {
            let (shape, _) = kv_cache[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Asr(format!("kv pos extract: {e}")))?;
            shape.as_ref()[2] as usize
        };

        // Single-token decode step.
        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_DECODER_LAYERS * 2);

        inputs.push((
            "inputs_embeds".to_string(),
            Value::from_array(tok_embeds)
                .map_err(|e| Error::Asr(format!("decode embed: {e}")))?
                .into(),
        ));

        // mRoPE position_ids for single token: [3, 1, 1]
        let pos_val = cur_pos as i64;
        let step_pos =
            ndarray::Array3::<i64>::from_shape_vec((3, 1, 1), vec![pos_val, pos_val, pos_val])
                .map_err(|e| Error::Asr(format!("step position_ids: {e}")))?;
        inputs.push((
            "position_ids".to_string(),
            Value::from_array(step_pos)
                .map_err(|e| Error::Asr(format!("step position_ids value: {e}")))?
                .into(),
        ));

        for (i, kv) in std::mem::take(kv_cache).into_iter().enumerate() {
            let layer = i / 2;
            let kind = if i % 2 == 0 { "key" } else { "value" };
            inputs.push((format!("past_key_values.{layer}.{kind}"), kv));
        }

        let step_outputs = decoder
            .run(inputs)
            .map_err(|e| Error::Asr(format!("decode step: {e}")))?;

        // Extract logits.
        let logits_view: ndarray::ArrayViewD<f32> = step_outputs[0]
            .try_extract_array()
            .map_err(|e| Error::Asr(format!("step logits: {e}")))?;
        current_logits = logits_view.slice(s![0, 0, ..]).to_vec();

        // Extract updated KV cache.
        for i in 1..step_outputs.len() {
            let (shape, data) = step_outputs[i]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Asr(format!("step kv[{i}]: {e}")))?;
            let dims: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
            let arr = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
                .map_err(|e| Error::Asr(format!("step kv arr[{i}]: {e}")))?;
            kv_cache.push(
                Value::from_array(arr)
                    .map_err(|e| Error::Asr(format!("step kv val[{i}]: {e}")))?
                    .into(),
            );
        }
    }

    Ok(tokens)
}

#[inline]
fn argmax(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i64)
        .unwrap_or(0)
}

// ── Degeneracy detection ────────────────────────────────────────────────────

/// Detect repeating token patterns at the tail of the sequence.
/// Returns `(best_repetitions, best_period)`.
fn tail_repeat_blocks(tokens: &[i64], max_period: usize) -> (usize, usize) {
    let n = tokens.len();
    if n < 2 {
        return (0, 0);
    }
    let period_cap = max_period.min(n / 2);
    let mut best_reps = 0;
    let mut best_period = 0;

    for p in 1..=period_cap {
        let tail = &tokens[n - p..];
        let mut reps = 1usize;
        loop {
            let start = n.checked_sub((reps + 1) * p);
            let Some(start) = start else { break };
            if &tokens[start..start + p] == tail {
                reps += 1;
            } else {
                break;
            }
        }
        if reps > best_reps {
            best_reps = reps;
            best_period = p;
        }
    }

    (best_reps, best_period)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_empty_audio() {
        let mel = compute_qwen_mel(&[]);
        assert_eq!(mel.shape(), &[1, 128, 0]);
    }

    #[test]
    fn mel_one_second() {
        let audio = vec![0.0f32; 16_000];
        let mel = compute_qwen_mel(&audio);
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 128);
        assert!(mel.shape()[2] > 0);
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tail_repeat_no_repeat() {
        let tokens = vec![1, 2, 3, 4, 5];
        let (reps, _) = tail_repeat_blocks(&tokens, 6);
        // reps=1 means single occurrence (no actual repetition).
        assert!(reps < DEGEN_MIN_REPEATS);
    }

    #[test]
    fn tail_repeat_period_2() {
        let tokens = vec![0, 1, 2, 1, 2, 1, 2];
        let (reps, period) = tail_repeat_blocks(&tokens, 6);
        assert_eq!(period, 2);
        assert_eq!(reps, 3);
    }

    #[test]
    fn lcp_reuse_identical() {
        let data = vec![1.0f32; 3 * HIDDEN_DIM];
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 3, HIDDEN_DIM]), data.clone()).unwrap();
        assert_eq!(compute_lcp_reuse(&data, 3, &arr, 3), 3);
    }

    #[test]
    fn lcp_reuse_partial() {
        let prev = vec![1.0f32; 3 * HIDDEN_DIM];
        let mut curr_data = prev.clone();
        // Change last row.
        curr_data[2 * HIDDEN_DIM] = 99.0;
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 3, HIDDEN_DIM]), curr_data).unwrap();
        assert_eq!(compute_lcp_reuse(&prev, 3, &arr, 3), 2);
    }

    #[test]
    fn empty_kv_cache_shape() {
        let kv = empty_kv_cache();
        assert_eq!(kv.len(), NUM_DECODER_LAYERS * 2);
    }

    #[test]
    fn sliding_window_mel_bounded() {
        // Audio longer than MAX_ENCODER_SAMPLES should only encode the tail.
        let long_audio = vec![0.0f32; MAX_ENCODER_SAMPLES + 16_000];
        let windowed = compute_qwen_mel(&long_audio[16_000..]);
        let full = compute_qwen_mel(&long_audio);
        // Windowed mel should have fewer frames than full.
        assert!(windowed.shape()[2] < full.shape()[2]);
        // Windowed mel should match the last MAX_ENCODER_SAMPLES worth of frames.
        let expected_mel = compute_qwen_mel(&long_audio[16_000..]);
        assert_eq!(windowed.shape(), expected_mel.shape());
    }

    #[test]
    fn enc_cache_cleared_on_reset() {
        let mut state = OrtStreamState::new();
        let dummy = ArrayD::<f32>::zeros(IxDyn(&[1, 10, HIDDEN_DIM]));
        state.enc_cache = Some(dummy);
        state.enc_cache_audio_end = 8000;
        state.reset();
        assert!(state.enc_cache.is_none());
        assert_eq!(state.enc_cache_audio_end, 0);
    }
}
