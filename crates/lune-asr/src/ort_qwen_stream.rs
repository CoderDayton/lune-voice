//! Streaming state machine for the ORT Qwen ASR backend.
//!
//! Ports the streaming algorithm from the `qwen-asr` crate to ONNX Runtime.
//! The encoder, embed_tokens, and decoder are ORT sessions passed by reference;
//! this module owns only the streaming state and mel computation.

use std::cell::RefCell;
use std::sync::OnceLock;

use ndarray::{Array2, Array3, ArrayD, IxDyn, s};
use ort::session::Session;
use ort::value::{DynValue, Value};
use rustfft::{Fft, FftPlanner, num_complex::Complex};

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

// Max repetition period to detect. Phrases like "and she was two or three princesses"
// tokenize to ~9 tokens, so we need period detection beyond 6.
const DEGEN_MAX_PERIOD: usize = 16;
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
/// Cached FFT plan for N_FFT=400. Planning is amortized across all mel calls.
/// Arc<dyn Fft<f32>>: Send + Sync — safe for OnceLock.
static QWEN_FFT: OnceLock<std::sync::Arc<dyn Fft<f32> + Send + Sync>> = OnceLock::new();
/// Pre-computed ORT input/output key strings for KV cache tensors.
/// Eliminates ~2700 format! calls per streaming chunk (56 keys × 48 decode steps).
static KV_NAMES: OnceLock<KvNames> = OnceLock::new();

struct KvNames {
    /// "past_key_values.{layer}.{kind}" for layer in 0..NUM_DECODER_LAYERS, kind in [key, value]
    input: Vec<String>,
    /// "present_key_values.{layer}.{kind}"
    output: Vec<String>,
}

fn kv_names() -> &'static KvNames {
    KV_NAMES.get_or_init(|| {
        let mut input = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
        let mut output = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
        for layer in 0..NUM_DECODER_LAYERS {
            for kind in ["key", "value"] {
                input.push(format!("past_key_values.{layer}.{kind}"));
                output.push(format!("present_key_values.{layer}.{kind}"));
            }
        }
        KvNames { input, output }
    })
}

/// Thread-local scratch buffers for mel spectrogram computation.
/// Avoids 3 heap allocations per mel call (~220 KB total for 1s chunks).
struct MelScratch {
    padded: Vec<f32>,
    frame: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

thread_local! {
    static MEL_SCRATCH: RefCell<MelScratch> = RefCell::new(MelScratch {
        padded: Vec::new(),
        frame: vec![Complex::new(0.0f32, 0.0f32); N_FFT],
        fft_scratch: Vec::new(), // sized on first use after FFT plan available
    });
}

fn qwen_fft() -> &'static std::sync::Arc<dyn Fft<f32> + Send + Sync> {
    QWEN_FFT.get_or_init(|| {
        let mut planner = FftPlanner::<f32>::new();
        planner.plan_fft_forward(N_FFT)
    })
}

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
///
/// Uses thread-local scratch buffers (padded, frame, fft_scratch) and a
/// process-global cached FFT plan to eliminate ~220 KB of heap allocations per call.
pub fn compute_qwen_mel(audio: &[f32]) -> Array3<f32> {
    if audio.is_empty() {
        return Array3::<f32>::zeros((1, MEL_BINS, 0));
    }

    let fft = qwen_fft();
    let window = qwen_hann();

    let pad = N_FFT / 2; // 200
    let padded_len = pad + audio.len() + pad;
    let num_frames = (padded_len - N_FFT) / HOP_LENGTH + 1;
    let scratch_len = fft.get_inplace_scratch_len();
    let zero = Complex::new(0.0f32, 0.0f32);

    let mut power_spec = Array2::<f32>::zeros((N_FREQ, num_frames));
    let spec_slice = power_spec.as_slice_mut().expect("spec contiguous");

    MEL_SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();

        // Grow scratch buffers without reallocating if already large enough.
        s.padded.resize(padded_len, 0.0f32);
        s.padded.fill(0.0);
        s.fft_scratch.resize(scratch_len, zero);

        // Reflection padding (uses only s.padded).
        for i in 0..pad {
            s.padded[pad - 1 - i] = audio[(i + 1).min(audio.len() - 1)];
        }
        s.padded[pad..pad + audio.len()].copy_from_slice(audio);
        for i in 0..pad {
            let src_idx = audio.len().saturating_sub(2).saturating_sub(i);
            s.padded[pad + audio.len() + i] = audio[src_idx];
        }

        // Destructure to get disjoint mutable borrows of separate fields.
        // The borrow checker cannot see through DerefMut on RefMut<T> to prove
        // s.padded, s.frame, and s.fft_scratch are non-overlapping without this.
        let MelScratch {
            ref padded,
            ref mut frame,
            ref mut fft_scratch,
        } = *s;

        // STFT with Hann window — reuse frame buffer (always N_FFT elements).
        for f in 0..num_frames {
            let start = f * HOP_LENGTH;
            frame.fill(zero);
            let copy_len = WIN_LENGTH.min(padded_len - start);
            for (c, (&samp, &w)) in frame[..copy_len].iter_mut().zip(
                padded[start..start + copy_len]
                    .iter()
                    .zip(window[..copy_len].iter()),
            ) {
                c.re = samp * w;
            }
            fft.process_with_scratch(frame, fft_scratch);
            for k in 0..N_FREQ {
                spec_slice[k * num_frames + f] = frame[k].norm_sqr();
            }
        }
    });

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
    /// Position in `raw_tokens` up to which tokens have been emitted as stable.
    /// Advances with each emission; resets to `raw_tokens.len()` on degen/reanchor
    /// to skip past re-populated carry tokens.
    raw_cursor: usize,
    /// Stable committed tokens (monotonically increasing).
    pub stable_text_tokens: Vec<i64>,
    /// Accumulated result text from stable emissions.
    pub result: String,

    /// Per-token log-probabilities (parallel to `raw_tokens`).
    raw_log_probs: Vec<f32>,
    /// Per-token log-probabilities for stable tokens (parallel to `stable_text_tokens`).
    stable_log_probs: Vec<f32>,

    /// Previous prefill embeddings for LCP reuse (flat f16, row-major).
    prev_prefill_embeds: Vec<half::f16>,
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

    /// Cached encoder features from the most recent encoding pass (f16).
    enc_cache: Option<ArrayD<half::f16>>,
    /// Audio sample position covered by `enc_cache`.
    enc_cache_audio_end: usize,
}

impl OrtStreamState {
    pub fn new() -> Self {
        Self {
            raw_tokens: Vec::new(),
            raw_cursor: 0,
            stable_text_tokens: Vec::new(),
            result: String::new(),
            raw_log_probs: Vec::new(),
            stable_log_probs: Vec::new(),
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

    /// Geometric mean of token-level probabilities for the stable tokens
    /// emitted since `prev_stable_len`. Returns `None` if no new tokens.
    pub fn confidence_since(&self, prev_stable_len: usize) -> Option<f32> {
        let new_lps = &self.stable_log_probs[prev_stable_len..];
        if new_lps.is_empty() {
            return None;
        }
        let mean_log_prob = new_lps.iter().sum::<f32>() / new_lps.len() as f32;
        Some(mean_log_prob.exp())
    }

    /// Minimum single-token confidence (exp of min log-prob) for the stable
    /// tokens emitted since `prev_stable_len`. Returns `None` if no new tokens.
    pub fn min_confidence_since(&self, prev_stable_len: usize) -> Option<f32> {
        self.stable_log_probs[prev_stable_len..]
            .iter()
            .copied()
            .reduce(f32::min)
            .map(f32::exp)
    }
}

// ── Streaming configuration ─────────────────────────────────────────────────

pub(crate) struct StreamConfig {
    pub chunk_samples: usize,
    pub rollback: usize,
    pub unfixed_chunks: i32,
    pub max_new_tokens: usize,
    pub past_text_conditioning: bool,
    /// Maximum audio samples passed to the encoder per chunk.
    /// Shorter windows keep the encoder feature sequence short, which helps
    /// when the model was fine-tuned on clips shorter than the full session.
    /// `None` falls back to `MAX_ENCODER_SAMPLES` (30s).
    pub max_encoder_samples: Option<usize>,
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
            let enc_limit = cfg.max_encoder_samples.unwrap_or(MAX_ENCODER_SAMPLES);
            let encode_start = chunk_end.saturating_sub(enc_limit);
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

        // Take ownership of KV cache — avoids clone when full reuse is possible.
        let old_kv = std::mem::take(&mut state.kv_cache);
        let old_kv_seq = state.kv_cache_seq_len;
        state.kv_cache_seq_len = 0;

        let kv_cache = if reused > 0 && !old_kv.is_empty() {
            if reused >= old_kv_seq {
                // Full reuse — zero-copy ownership transfer, no extraction.
                old_kv
            } else {
                // Partial reuse — truncate (CPU round-trip unavoidable without IoBinding).
                truncate_kv_cache(old_kv, old_kv_seq, reused)?
            }
        } else {
            drop(old_kv); // Free old GPU tensors.
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
                // kv_cache is already owned; truncate it for the last-token re-prefill.
                truncate_kv_cache(kv_cache, old_kv_seq, reused - 1)?
            } else {
                drop(kv_cache);
                empty_kv_cache()
            };
            decoder_prefill(
                decoder,
                last_embed.to_owned().into_dyn(),
                kv_trunc,
                reused.saturating_sub(1),
            )?
        };

        // Save prefill embeddings (f16) for next chunk's LCP comparison.
        let embeds_data = input_embeds.as_slice().unwrap_or(&[]);
        state.prev_prefill_embeds = embeds_data[..prefill_len * HIDDEN_DIM].to_vec();
        state.prev_prefill_len = prefill_len;

        // Autoregressive decode: generate up to max_new_tokens.
        let prefill_end_pos = reused.max(delta_start + delta_len);
        let (chunk_tokens, chunk_log_probs) = autoregressive_decode(
            decoder,
            embed_tokens,
            tokenizer,
            &mut current_kv,
            last_logits,
            cfg.max_new_tokens,
            prefill_end_pos,
        )?;

        let kv_seq_after_decode = prefill_end_pos + chunk_tokens.len();

        // Speech-end detection: empty decode → commit all rollback tokens.
        let is_speech_end = chunk_tokens.is_empty() && state.chunk_idx >= cfg.unfixed_chunks;

        // Replace rollback zone with fresh decode.
        // Truncating to raw_cursor discards the previous rollback zone before
        // appending new tokens, preventing raw_tokens from accumulating duplicates
        // across chunks (rollback tokens re-decoded from the stable boundary).
        state.raw_tokens.truncate(state.raw_cursor);
        state.raw_log_probs.truncate(state.raw_cursor);
        if !chunk_tokens.is_empty() {
            state.raw_tokens.extend_from_slice(&chunk_tokens);
            state.raw_log_probs.extend_from_slice(&chunk_log_probs);
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
            // Reset: carry last RESET_CARRY_TOKENS stable tokens into raw_tokens
            // so the decoder sees prior context. Set raw_cursor past the carry tokens
            // so they are not re-emitted as new stable output.
            let carry_start = state
                .stable_text_tokens
                .len()
                .saturating_sub(RESET_CARRY_TOKENS);
            state.raw_tokens.clear();
            state.raw_log_probs.clear();
            if carry_start < state.stable_text_tokens.len() {
                state
                    .raw_tokens
                    .extend_from_slice(&state.stable_text_tokens[carry_start..]);
                state
                    .raw_log_probs
                    .extend_from_slice(&state.stable_log_probs[carry_start..]);
            }
            // Advance cursor past carry tokens — they are already in stable_text_tokens.
            state.raw_cursor = state.raw_tokens.len();
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

        // Emit tokens beyond current raw_cursor.
        if frontier > state.raw_cursor {
            let range = state.raw_cursor..frontier;
            for (i, &tok) in state.raw_tokens[range.clone()].iter().enumerate() {
                state.stable_text_tokens.push(tok);
                state
                    .stable_log_probs
                    .push(state.raw_log_probs[range.start + i]);
            }
            state.raw_cursor = frontier;
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

fn run_encoder(encoder: &mut Session, mel: Array3<f32>) -> Result<ArrayD<half::f16>> {
    // Model expects f16 mel input; convert at boundary.
    let mel_f16 = mel.mapv(half::f16::from_f32).into_dyn();
    let mel_val: DynValue = Value::from_array(mel_f16)
        .map_err(|e| Error::Asr(format!("mel tensor: {e}")))?
        .into();
    let inputs = vec![("mel".to_string(), mel_val)];

    let outputs = encoder
        .run(inputs)
        .map_err(|e| Error::Asr(format!("encoder: {e}")))?;

    // Keep encoder output in f16 — no dtype conversion.
    let features: ndarray::ArrayViewD<half::f16> = outputs[0]
        .try_extract_array()
        .map_err(|e| Error::Asr(format!("encoder features: {e}")))?;

    Ok(features.to_owned())
}

// ── Input embeddings ────────────────────────────────────────────────────────

/// Build the full input_embeds sequence for the decoder.
///
/// All operations stay in f16 — no dtype conversion needed.
/// Returns `(input_embeds [1, total_len, HIDDEN_DIM], prefill_len)`.
fn build_input_embeds(
    embed_tokens: &mut Session,
    enc_features: &ArrayD<half::f16>,
    enc_seq_len: usize,
    _raw_tokens: &[i64],
    stable_tokens: &[i64],
    past_text_conditioning: bool,
) -> Result<(ArrayD<half::f16>, usize)> {
    // Collect all token IDs that need embedding.
    let mut pre_tokens: Vec<i64> = Vec::with_capacity(32);
    pre_tokens.extend_from_slice(PREFIX_HEAD);
    pre_tokens.extend_from_slice(PREFIX_TAIL);

    let mut post_tokens: Vec<i64> = Vec::with_capacity(32);
    post_tokens.extend_from_slice(SUFFIX_BASE);
    post_tokens.push(TOKEN_ASR_TEXT);

    // Add past text tokens for conditioning.
    // Full stable history prevents the decoder from re-transcribing already-emitted
    // content — without it, the decoder repeats earlier audio on each chunk.
    if past_text_conditioning && !stable_tokens.is_empty() {
        post_tokens.extend_from_slice(stable_tokens);
    }

    // Get embeddings for pre and post tokens (already f16).
    let pre_embeds = token_embeddings(embed_tokens, &pre_tokens)?;
    let post_embeds = token_embeddings(embed_tokens, &post_tokens)?;

    let pre_len = pre_tokens.len();
    let post_len = post_tokens.len();
    let total_len = pre_len + enc_seq_len + post_len;

    // Concatenate in f16: pre_embeds + enc_features + post_embeds → [1, total_len, HIDDEN_DIM]
    let mut combined = vec![half::f16::ZERO; total_len * HIDDEN_DIM];

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
/// Returns f16 directly — no dtype conversion.
fn token_embeddings(embed_tokens: &mut Session, tokens: &[i64]) -> Result<ArrayD<half::f16>> {
    let ids = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())
        .map_err(|e| Error::Asr(format!("token ids array: {e}")))?;
    let ids_val: DynValue = Value::from_array(ids)
        .map_err(|e| Error::Asr(format!("token ids tensor: {e}")))?
        .into();
    let inputs = vec![("input_ids".to_string(), ids_val)];

    let outputs = embed_tokens
        .run(inputs)
        .map_err(|e| Error::Asr(format!("embed_tokens: {e}")))?;

    let embeds: ndarray::ArrayViewD<half::f16> = outputs[0]
        .try_extract_array()
        .map_err(|e| Error::Asr(format!("token embeds: {e}")))?;

    Ok(embeds.to_owned())
}

// ── LCP reuse ───────────────────────────────────────────────────────────────

/// Compare previous and current prefill embeddings row-by-row (f16).
/// Returns the number of reusable prefix rows.
fn compute_lcp_reuse(
    prev_embeds: &[half::f16],
    prev_len: usize,
    curr_embeds: &ArrayD<half::f16>,
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
        let arr = ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), half::f16::ZERO);
        kv.push(Value::from_array(arr).expect("empty kv tensor").into());
    }
    kv
}

/// Truncate owned KV cache tensors to `target_len` along the sequence dimension.
///
/// Takes ownership of the input Vec so old DynValues are dropped immediately
/// after extraction — halves peak GPU memory vs borrowing + cloning.
/// The CPU round-trip per tensor is unavoidable without IoBinding.
fn truncate_kv_cache(
    kv_cache: Vec<DynValue>,
    current_len: usize,
    target_len: usize,
) -> Result<Vec<DynValue>> {
    if target_len >= current_len {
        // No truncation needed — return as-is (zero-copy).
        return Ok(kv_cache);
    }
    let mut result = Vec::with_capacity(kv_cache.len());
    for (i, kv) in kv_cache.into_iter().enumerate() {
        let (shape, data) = kv
            .try_extract_tensor::<half::f16>()
            .map_err(|e| Error::Asr(format!("kv extract[{i}]: {e}")))?;
        let dims: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        // Shape: [1, num_kv_heads, seq_len, head_dim]
        let full = ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
            .map_err(|e| Error::Asr(format!("kv array[{i}]: {e}")))?;
        let sliced = full.slice(s![.., .., ..target_len, ..]).to_owned();
        // Old DynValue `kv` dropped here — frees GPU memory before new tensor.
        result.push(
            Value::from_array(sliced)
                .map_err(|e| Error::Asr(format!("kv value[{i}]: {e}")))?
                .into(),
        );
    }
    Ok(result)
}

// ── Decoder prefill ─────────────────────────────────────────────────────────

/// Run decoder prefill: feed input_embeds (f16) with KV cache, return updated KV + logits.
fn decoder_prefill(
    decoder: &mut Session,
    input_embeds: ArrayD<half::f16>,
    kv_cache: Vec<DynValue>,
    past_seq_len: usize,
) -> Result<(Vec<DynValue>, Vec<f32>)> {
    let seq_len = input_embeds.shape()[1];
    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_DECODER_LAYERS * 2);

    // input_embeds already f16 — no conversion needed.
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

    let names = kv_names();
    for (i, kv) in kv_cache.into_iter().enumerate() {
        inputs.push((names.input[i].clone(), kv));
    }

    let mut outputs = decoder
        .run(inputs)
        .map_err(|e| Error::Asr(format!("decoder prefill: {e}")))?;

    // Extract logits (f16) — last position only, convert to f32.
    let logits_view: ndarray::ArrayViewD<half::f16> = outputs[0]
        .try_extract_array()
        .map_err(|e| Error::Asr(format!("prefill logits: {e}")))?;
    let seq_len = logits_view.shape()[1];
    let last_logits: Vec<f32> = logits_view
        .slice(s![0, seq_len - 1, ..])
        .iter()
        .map(|v| v.to_f32())
        .collect();

    // Take ownership of KV cache outputs — zero-copy GPU passthrough.
    let mut new_kv = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
    for (i, name) in names.output.iter().enumerate() {
        new_kv.push(
            outputs
                .remove(name.as_str())
                .ok_or_else(|| Error::Asr(format!("missing output[{i}]: {name}")))?,
        );
    }

    Ok((new_kv, last_logits))
}

// ── Autoregressive decode ───────────────────────────────────────────────────

/// Run autoregressive decode steps. Returns `(token_ids, log_probs)`.
fn autoregressive_decode(
    decoder: &mut Session,
    embed_tokens: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    kv_cache: &mut Vec<DynValue>,
    initial_logits: Vec<f32>,
    max_tokens: usize,
    initial_pos: usize,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let eos_id = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(151643) as i64;

    let mut tokens = Vec::with_capacity(max_tokens);
    let mut log_probs = Vec::with_capacity(max_tokens);
    let mut current_logits = initial_logits;
    let mut cur_pos = initial_pos;

    for _ in 0..max_tokens {
        let (next_token, log_prob) = argmax_with_log_prob(&current_logits);

        if next_token == eos_id || next_token == TOKEN_IM_END {
            break;
        }

        tokens.push(next_token);
        log_probs.push(log_prob);

        // Get embedding for next token (already f16).
        let tok_embeds = token_embeddings(embed_tokens, &[next_token])?;

        // Single-token decode step.
        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(3 + NUM_DECODER_LAYERS * 2);

        // tok_embeds already f16 — no conversion needed.
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

        let names = kv_names();
        for (i, kv) in std::mem::take(kv_cache).into_iter().enumerate() {
            inputs.push((names.input[i].clone(), kv));
        }

        let mut step_outputs = decoder
            .run(inputs)
            .map_err(|e| Error::Asr(format!("decode step: {e}")))?;

        // Extract logits (f16), convert to f32.
        let logits_view: ndarray::ArrayViewD<half::f16> = step_outputs[0]
            .try_extract_array()
            .map_err(|e| Error::Asr(format!("step logits: {e}")))?;
        current_logits = logits_view
            .slice(s![0, 0, ..])
            .iter()
            .map(|v| v.to_f32())
            .collect();

        // Take ownership of KV outputs — zero-copy GPU passthrough.
        for (i, name) in names.output.iter().enumerate() {
            kv_cache.push(
                step_outputs
                    .remove(name.as_str())
                    .ok_or_else(|| Error::Asr(format!("missing output[{i}]: {name}")))?,
            );
        }

        cur_pos += 1;
    }

    Ok((tokens, log_probs))
}

/// Greedy argmax with log-probability of the chosen token.
///
/// Computes `log_softmax(logits)[argmax]` using the log-sum-exp trick.
/// Since the chosen token is the maximum, this simplifies to
/// `-ln(sum(exp(logits - max)))` — no extra pass needed.
#[inline]
fn argmax_with_log_prob(logits: &[f32]) -> (i64, f32) {
    let (max_idx, &max_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    // logsumexp = max + ln(sum(exp(logits - max)))
    // log_prob  = max - logsumexp = -ln(sum(exp(logits - max)))
    let sum_exp: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
    let log_prob = -(sum_exp.ln());

    (max_idx as i64, log_prob)
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

    // ── Test helpers ─────────────────────────────────────────────────────────

    fn workspace_root() -> std::path::PathBuf {
        // CARGO_MANIFEST_DIR = crates/lune-asr
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn model_dir() -> std::path::PathBuf {
        workspace_root().join("models/qwen3-asr-0.6b-onnx")
    }

    fn ensure_ort() {
        static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        INIT.get_or_init(|| {
            let _ = ort::init().commit();
        });
    }

    fn load_wav_f32(path: &std::path::Path) -> Vec<f32> {
        let reader = hound::WavReader::open(path).expect("open wav");
        let spec = reader.spec();
        assert_eq!(spec.channels, 1, "expected mono");
        assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz");
        match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .map(|s| s.expect("read sample"))
                .collect(),
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.expect("read sample") as f32 / max)
                    .collect()
            }
        }
    }

    fn tone_440hz(secs: f32) -> Vec<f32> {
        let n = (16_000.0 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.3)
            .collect()
    }

    // ── Encoder tests ────────────────────────────────────────────────────────

    /// Encoder output shape: [1, S, 1024]. Values must be non-trivially non-zero.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx"]
    fn encoder_output_shape_and_values() {
        ensure_ort();
        let dir = model_dir();
        if !dir.exists() {
            eprintln!("[skip] model dir not found");
            return;
        }

        let mut encoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-encoder.onnx"), false)
                .expect("load encoder");

        // 1 second of 440 Hz tone
        let audio = tone_440hz(1.0);
        let mel = compute_qwen_mel(&audio);
        eprintln!("[encoder] mel shape: {:?}", mel.shape());

        let features = run_encoder(&mut encoder, mel).expect("encoder forward");
        let shape = features.shape().to_vec();
        eprintln!("[encoder] features shape: {shape:?}");

        assert_eq!(shape[0], 1, "batch=1");
        assert_eq!(shape[2], HIDDEN_DIM, "hidden_dim=1024");
        assert!(shape[1] > 0, "enc_seq_len > 0");

        let total = features.len();
        let non_zero = features.iter().filter(|&&v| v != half::f16::ZERO).count();
        let max_abs = features
            .iter()
            .map(|v| v.to_f32().abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[encoder] non-zero={non_zero}/{total} ({:.1}%), max_abs={max_abs:.4}",
            100.0 * non_zero as f32 / total as f32
        );
        assert!(non_zero > total / 10, "encoder output mostly zeros");
        assert!(max_abs > 0.0, "encoder output all zero");
    }

    // ── Decoder tests ────────────────────────────────────────────────────────

    /// Single decoder step: what is the first token produced from a 1s audio clip?
    ///
    /// This is the key diagnostic for "zero segments" — if the first token is
    /// EOS/im_end, the autoregressive_decode loop terminates immediately.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx"]
    fn decoder_first_token_diagnostic() {
        ensure_ort();
        let dir = model_dir();
        if !dir.exists() {
            eprintln!("[skip] model dir not found");
            return;
        }

        let mut encoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-encoder.onnx"), false)
                .expect("load encoder");
        let mut embed_tokens =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-embed-tokens.onnx"), false)
                .expect("load embed_tokens");
        let mut decoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-decoder.onnx"), false)
                .expect("load decoder");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer");

        // 440 Hz tone — not real speech but deterministic
        let audio = tone_440hz(1.0);
        let mel = compute_qwen_mel(&audio);
        let features = run_encoder(&mut encoder, mel).expect("encoder forward");
        let enc_seq_len = features.shape()[1];

        let (input_embeds, prefill_len) =
            build_input_embeds(&mut embed_tokens, &features, enc_seq_len, &[], &[], false)
                .expect("build_input_embeds");
        eprintln!("[decoder] enc_seq_len={enc_seq_len}, prefill_len={prefill_len}");

        let kv = empty_kv_cache();
        let (mut new_kv, last_logits) =
            decoder_prefill(&mut decoder, input_embeds, kv, 0).expect("decoder_prefill");

        let (first_tok, lp) = argmax_with_log_prob(&last_logits);
        let first_str = tokenizer
            .decode(&[first_tok as u32], false)
            .unwrap_or_default();
        eprintln!(
            "[decoder] first_token={first_tok} ({first_str:?}) log_prob={lp:.4} prob={:.4}",
            lp.exp()
        );

        // Top-10 tokens
        let mut ranked: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[decoder] top-10 after prefill:");
        for (idx, val) in ranked.iter().take(10) {
            let s = tokenizer.decode(&[*idx as u32], false).unwrap_or_default();
            eprintln!("  idx={idx:6} logit={val:8.3} ({s:?})");
        }

        // Check whether we immediately hit EOS
        let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645) as i64;
        let eot_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(151643) as i64;

        if first_tok == eos_id || first_tok == eot_id || first_tok == TOKEN_IM_END {
            eprintln!("[decoder] ⚠ FIRST TOKEN IS EOS — autoregressive_decode will return empty!");
        }

        // Full autoregressive decode (up to 32 tokens)
        let (tokens, log_probs) = autoregressive_decode(
            &mut decoder,
            &mut embed_tokens,
            &tokenizer,
            &mut new_kv,
            last_logits,
            32,
            prefill_len,
        )
        .expect("autoregressive_decode");

        eprintln!("[decoder] generated {} tokens: {tokens:?}", tokens.len());
        if !tokens.is_empty() {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let text = tokenizer.decode(&ids, true).unwrap_or_default();
            eprintln!("[decoder] decoded: {text:?}");
            eprintln!(
                "[decoder] mean log_prob={:.4} (confidence={:.4})",
                log_probs.iter().sum::<f32>() / log_probs.len() as f32,
                (log_probs.iter().sum::<f32>() / log_probs.len() as f32).exp()
            );
        } else {
            eprintln!("[decoder] ⚠ ZERO TOKENS GENERATED — first token was EOS");
        }
    }

    /// Same as above but with real LibriSpeech audio.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx and test_fixtures/librispeech_clean.wav"]
    fn decoder_first_token_real_speech() {
        ensure_ort();
        let dir = model_dir();
        let wav = workspace_root().join("test_fixtures/librispeech_clean.wav");
        if !dir.exists() || !wav.exists() {
            eprintln!("[skip] model or wav not found");
            return;
        }

        let mut encoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-encoder.onnx"), false)
                .expect("load encoder");
        let mut embed_tokens =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-embed-tokens.onnx"), false)
                .expect("load embed_tokens");
        let mut decoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-decoder.onnx"), false)
                .expect("load decoder");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer");

        // First 0.5s of real speech
        let audio = load_wav_f32(&wav);
        let chunk = &audio[..8_000.min(audio.len())];
        eprintln!("[real_speech] using {} samples", chunk.len());

        let mel = compute_qwen_mel(chunk);
        let features = run_encoder(&mut encoder, mel).expect("encoder forward");
        let enc_seq_len = features.shape()[1];
        eprintln!("[real_speech] enc_seq_len={enc_seq_len}");

        let (input_embeds, prefill_len) =
            build_input_embeds(&mut embed_tokens, &features, enc_seq_len, &[], &[], false)
                .expect("build_input_embeds");

        let kv = empty_kv_cache();
        let (mut new_kv, last_logits) =
            decoder_prefill(&mut decoder, input_embeds, kv, 0).expect("decoder_prefill");

        let (first_tok, lp) = argmax_with_log_prob(&last_logits);
        let first_str = tokenizer
            .decode(&[first_tok as u32], false)
            .unwrap_or_default();
        eprintln!(
            "[real_speech] first_token={first_tok} ({first_str:?}) prob={:.4}",
            lp.exp()
        );

        let mut ranked: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[real_speech] top-10 after prefill:");
        for (idx, val) in ranked.iter().take(10) {
            let s = tokenizer.decode(&[*idx as u32], false).unwrap_or_default();
            eprintln!("  idx={idx:6} logit={val:8.3} ({s:?})");
        }

        let (tokens, _) = autoregressive_decode(
            &mut decoder,
            &mut embed_tokens,
            &tokenizer,
            &mut new_kv,
            last_logits,
            64,
            prefill_len,
        )
        .expect("autoregressive_decode");

        if tokens.is_empty() {
            eprintln!("[real_speech] ⚠ ZERO TOKENS — first generated token was EOS");
        } else {
            let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let text = tokenizer.decode(&ids, true).unwrap_or_default();
            eprintln!("[real_speech] decoded {} tokens: {text:?}", tokens.len());
        }
    }

    // ── Stream state machine tests ───────────────────────────────────────────

    /// Verify audio features affect decoder output.
    ///
    /// Runs decoder with real encoder features vs all-zero features.
    /// If logits are IDENTICAL → decoder is ignoring audio (position_ids or
    /// attention bug). If logits differ → audio is being used correctly.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx"]
    fn decoder_audio_features_affect_logits() {
        ensure_ort();
        let dir = model_dir();
        if !dir.exists() {
            eprintln!("[skip] model dir not found");
            return;
        }

        let mut encoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-encoder.onnx"), false)
                .expect("load encoder");
        let mut embed_tokens =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-embed-tokens.onnx"), false)
                .expect("load embed_tokens");
        let mut decoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-decoder.onnx"), false)
                .expect("load decoder");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer");

        let audio = tone_440hz(1.0);
        let mel = compute_qwen_mel(&audio);
        let real_features = run_encoder(&mut encoder, mel).expect("encoder forward");
        let enc_seq_len = real_features.shape()[1];

        // Case A: real encoder features
        let (emb_a, _plen_a) = build_input_embeds(
            &mut embed_tokens,
            &real_features,
            enc_seq_len,
            &[],
            &[],
            false,
        )
        .expect("build_input_embeds A");
        let kv_a = empty_kv_cache();
        let (_, logits_a) = decoder_prefill(&mut decoder, emb_a, kv_a, 0).expect("prefill A");

        // Case B: all-zero encoder features (same shape)
        let zero_features =
            ArrayD::from_elem(IxDyn(&[1, enc_seq_len, HIDDEN_DIM]), half::f16::ZERO);
        let (emb_b, _plen_b) = build_input_embeds(
            &mut embed_tokens,
            &zero_features,
            enc_seq_len,
            &[],
            &[],
            false,
        )
        .expect("build_input_embeds B");
        let kv_b = empty_kv_cache();
        let (_, logits_b) = decoder_prefill(&mut decoder, emb_b, kv_b, 0).expect("prefill B");

        // Compare top-1 and L∞ distance
        let (tok_a, lp_a) = argmax_with_log_prob(&logits_a);
        let (tok_b, lp_b) = argmax_with_log_prob(&logits_b);
        let str_a = tokenizer.decode(&[tok_a as u32], false).unwrap_or_default();
        let str_b = tokenizer.decode(&[tok_b as u32], false).unwrap_or_default();

        let max_diff: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_diff: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / logits_a.len() as f32;

        eprintln!("[audio_effect] enc_seq_len={enc_seq_len}");
        eprintln!(
            "[audio_effect] A (real):  tok={tok_a} ({str_a:?}) prob={:.4}",
            lp_a.exp()
        );
        eprintln!(
            "[audio_effect] B (zeros): tok={tok_b} ({str_b:?}) prob={:.4}",
            lp_b.exp()
        );
        eprintln!("[audio_effect] logit diff: L∞={max_diff:.4} mean={mean_diff:.6}");

        if tok_a == tok_b && max_diff < 1.0 {
            eprintln!(
                "[audio_effect] ⚠ AUDIO FEATURES HAVE NO EFFECT — decoder ignores encoder output!"
            );
            eprintln!(
                "[audio_effect] Root cause: likely incorrect position_ids or missing attention to audio tokens."
            );
        } else {
            eprintln!("[audio_effect] ✓ Audio features DO affect logits (expected behavior).");
        }
    }

    /// Full streaming trace over LibriSpeech audio.
    ///
    /// Prints per-chunk: cursor, new raw tokens, new stable tokens, text delta.
    /// Use this to diagnose where tokens are generated/suppressed.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx and test_fixtures/librispeech_clean.wav"]
    fn stream_full_librispeech_trace() {
        ensure_ort();
        let dir = model_dir();
        let wav = workspace_root().join("test_fixtures/librispeech_clean.wav");
        if !dir.exists() || !wav.exists() {
            eprintln!("[skip] model or wav not found");
            return;
        }

        let mut encoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-encoder.onnx"), false)
                .expect("load encoder");
        let mut embed_tokens =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-embed-tokens.onnx"), false)
                .expect("load embed_tokens");
        let mut decoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-decoder.onnx"), false)
                .expect("load decoder");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer");

        let audio = load_wav_f32(&wav);
        eprintln!(
            "[stream] audio: {} samples ({:.1}s)",
            audio.len(),
            audio.len() as f32 / 16_000.0
        );

        // Match current OrtQwenAsrConfig::default() parameters.
        let chunk_sec = 1.0f32;
        let chunk_size = (chunk_sec * 16_000.0) as usize;
        // 8s encoder window — matches OrtQwenAsrConfig::default().max_encoder_sec=8.0
        let max_encoder_samples: Option<usize> = Some(8 * 16_000);
        let cfg = StreamConfig {
            chunk_samples: chunk_size,
            rollback: 3,
            unfixed_chunks: 2,
            max_new_tokens: 48,
            past_text_conditioning: true,
            max_encoder_samples,
        };

        let mut state = OrtStreamState::new();

        for i in 0..audio.len().div_ceil(chunk_size) {
            let buf_end = ((i + 1) * chunk_size).min(audio.len());
            let buf = &audio[..buf_end];

            let prev_raw = state.raw_tokens.len();
            let prev_stable = state.stable_text_tokens.len();

            let text = stream_push_audio(
                &mut encoder,
                &mut embed_tokens,
                &mut decoder,
                &tokenizer,
                buf,
                &mut state,
                &cfg,
                false,
            )
            .expect("stream_push_audio");

            let new_raw = state.raw_tokens.len() as isize - prev_raw as isize;
            let new_stable = state.stable_text_tokens.len() as isize - prev_stable as isize;
            let cursor_sec = state.audio_cursor as f32 / 16_000.0;

            // Always print first few chunks; after that only when interesting
            if i < 6 || new_raw != 0 || !text.is_empty() {
                eprintln!(
                    "[chunk {:03}] cursor={:.1}s raw_Δ={:+3} stable_Δ={:+3} stale={} raw_len={} cursor={} text={:?}",
                    i,
                    cursor_sec,
                    new_raw,
                    new_stable,
                    state.stale_count,
                    state.raw_tokens.len(),
                    state.raw_cursor,
                    text
                );
                // Decode current raw_tokens for debugging stalls
                if !state.raw_tokens.is_empty() {
                    let ids: Vec<u32> = state.raw_tokens.iter().map(|&t| t as u32).collect();
                    let raw_text = tokenizer.decode(&ids, true).unwrap_or_default();
                    eprintln!("          raw_tokens decoded: {raw_text:?}");
                }
            }
        }

        eprintln!("[stream] final raw_tokens: {}", state.raw_tokens.len());
        eprintln!(
            "[stream] final stable_tokens: {}",
            state.stable_text_tokens.len()
        );
        eprintln!("[stream] accumulated result: {:?}", state.result);

        if !state.stable_text_tokens.is_empty() {
            let ids: Vec<u32> = state.stable_text_tokens.iter().map(|&t| t as u32).collect();
            let text = tokenizer.decode(&ids, true).unwrap_or_default();
            eprintln!("[stream] stable text decoded: {text:?}");
        } else {
            eprintln!("[stream] ⚠ ZERO STABLE TOKENS — no text was committed");
        }
    }

    /// Print model input/output names and shapes for all three sessions.
    ///
    /// Use this to verify ONNX export correctness — wrong names cause silent failures
    /// where ORT silently uses zero tensors instead of erroring.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx"]
    fn model_io_names() {
        ensure_ort();
        let dir = model_dir();
        if !dir.exists() {
            eprintln!("[skip] model dir not found");
            return;
        }

        for name in &[
            "qwen3-asr-encoder.onnx",
            "qwen3-asr-embed-tokens.onnx",
            "qwen3-asr-decoder.onnx",
        ] {
            let session =
                crate::ort_utils::build_ort_session(&dir.join(name), false).expect("load session");
            eprintln!("\n=== {name} ===");
            eprintln!("  INPUTS:");
            for inp in session.inputs().iter() {
                eprintln!("    {:?}  dtype={:?}", inp.name(), inp.dtype());
            }
            eprintln!("  OUTPUTS:");
            for out in session.outputs().iter() {
                eprintln!("    {:?}  dtype={:?}", out.name(), out.dtype());
            }
        }
    }

    /// Verify that TOKEN_ASR_TEXT (151704) exists in the tokenizer vocabulary.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx"]
    fn tokenizer_special_tokens() {
        ensure_ort();
        let dir = model_dir();
        if !dir.exists() {
            return;
        }
        let tok =
            tokenizers::Tokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer");

        let eos = tok.token_to_id("<|im_end|>");
        let eot = tok.token_to_id("<|endoftext|>");
        let asr_text = tok
            .token_to_id("<|ASR_TEXT|>")
            .or_else(|| tok.token_to_id("<asr_text>"));
        let audio_start = tok
            .token_to_id("<|AUDIO|>")
            .or_else(|| tok.token_to_id("<audio>"));

        eprintln!("[tok] <|im_end|>     = {eos:?}  (want 151645)");
        eprintln!("[tok] <|endoftext|>  = {eot:?}  (want 151643)");
        eprintln!("[tok] ASR_TEXT token = {asr_text:?} (want 151704)");
        eprintln!("[tok] audio start    = {audio_start:?} (want 151669)");

        // Decode the hardcoded token IDs to verify they're correct
        for (name, id) in [
            ("TOKEN_IM_START", TOKEN_IM_START),
            ("TOKEN_IM_END", TOKEN_IM_END),
            ("TOKEN_AUDIO_START", TOKEN_AUDIO_START),
            ("TOKEN_AUDIO_END", TOKEN_AUDIO_END),
            ("TOKEN_ASR_TEXT", TOKEN_ASR_TEXT),
        ] {
            let decoded = tok.decode(&[id as u32], false).unwrap_or_default();
            eprintln!("[tok] {name} ({id}) → {decoded:?}");
        }
    }

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
        let data = vec![half::f16::from_f32(1.0); 3 * HIDDEN_DIM];
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 3, HIDDEN_DIM]), data.clone()).unwrap();
        assert_eq!(compute_lcp_reuse(&data, 3, &arr, 3), 3);
    }

    #[test]
    fn lcp_reuse_partial() {
        let prev = vec![half::f16::from_f32(1.0); 3 * HIDDEN_DIM];
        let mut curr_data = prev.clone();
        // Change last row.
        curr_data[2 * HIDDEN_DIM] = half::f16::from_f32(99.0);
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
        let dummy = ArrayD::from_elem(IxDyn(&[1, 10, HIDDEN_DIM]), half::f16::ZERO);
        state.enc_cache = Some(dummy);
        state.enc_cache_audio_end = 8000;
        state.reset();
        assert!(state.enc_cache.is_none());
        assert_eq!(state.enc_cache_audio_end, 0);
    }

    #[test]
    fn argmax_log_prob_uniform() {
        // Uniform logits: each token has equal probability 1/N.
        let n = 100;
        let logits = vec![0.0f32; n];
        let (idx, lp) = argmax_with_log_prob(&logits);
        assert!((0..n as i64).contains(&idx));
        let expected_lp = -(n as f32).ln();
        assert!(
            (lp - expected_lp).abs() < 1e-5,
            "got {lp}, expected {expected_lp}"
        );
    }

    #[test]
    fn argmax_log_prob_peaked() {
        // One dominant logit — probability should be close to 1.0.
        let mut logits = vec![-10.0f32; 1000];
        logits[42] = 50.0;
        let (idx, lp) = argmax_with_log_prob(&logits);
        assert_eq!(idx, 42);
        // log_prob should be very close to 0 (prob ≈ 1.0).
        assert!(lp > -0.01, "peaked log_prob {lp} should be near 0");
        assert!(lp <= 0.0, "log_prob {lp} should be non-positive");
    }

    #[test]
    fn argmax_log_prob_range() {
        // Random-ish logits — log_prob should always be in (-inf, 0].
        let logits: Vec<f32> = (0..500).map(|i| (i as f32 * 0.73).sin() * 5.0).collect();
        let (idx, lp) = argmax_with_log_prob(&logits);
        assert!((0..500).contains(&(idx as usize)));
        assert!(lp <= 0.0, "log_prob {lp} must be non-positive");
        assert!(lp.is_finite());
    }

    #[test]
    fn confidence_since_empty() {
        let state = OrtStreamState::new();
        assert_eq!(state.confidence_since(0), None);
    }

    #[test]
    fn confidence_since_computes_geometric_mean() {
        let mut state = OrtStreamState::new();
        // Simulate 3 stable tokens with known log-probs.
        state.stable_log_probs = vec![(-0.1f32), (-0.2), (-0.3)];
        state.stable_text_tokens = vec![1, 2, 3];

        // From position 0: geometric mean of all three.
        let conf = state.confidence_since(0).unwrap();
        let expected = ((-0.1 - 0.2 - 0.3) / 3.0f32).exp();
        assert!(
            (conf - expected).abs() < 1e-6,
            "got {conf}, expected {expected}"
        );

        // From position 2: only the last token.
        let conf2 = state.confidence_since(2).unwrap();
        let expected2 = (-0.3f32).exp();
        assert!((conf2 - expected2).abs() < 1e-6);

        // From position 3: no new tokens.
        assert_eq!(state.confidence_since(3), None);
    }

    #[test]
    fn min_confidence_since_returns_worst_token() {
        let mut state = OrtStreamState::new();
        state.stable_log_probs = vec![-0.1f32, -0.8, -0.2];
        state.stable_text_tokens = vec![1, 2, 3];

        // From position 0: minimum is the worst token (log_prob = -0.8).
        let min_conf = state.min_confidence_since(0).unwrap();
        let expected = (-0.8f32).exp();
        assert!(
            (min_conf - expected).abs() < 1e-6,
            "got {min_conf}, expected {expected}"
        );

        // From position 2: only last token.
        let min2 = state.min_confidence_since(2).unwrap();
        assert!((min2 - (-0.2f32).exp()).abs() < 1e-6);

        // From position 3: no new tokens.
        assert_eq!(state.min_confidence_since(3), None);
    }

    /// Process the entire LibriSpeech WAV in one shot — no chunking.
    ///
    /// Encodes all audio at once, builds full input_embeds (no stable tokens),
    /// runs a single prefill then autoregressive decode up to 200 tokens.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx and test_fixtures/librispeech_clean.wav"]
    fn batch_full_audio() {
        ensure_ort();
        let dir = model_dir();
        let wav = workspace_root().join("test_fixtures/librispeech_clean.wav");
        if !dir.exists() || !wav.exists() {
            eprintln!("[skip] model or wav not found");
            return;
        }

        let mut encoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-encoder.onnx"), false)
                .expect("load encoder");
        let mut embed_tokens =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-embed-tokens.onnx"), false)
                .expect("load embed_tokens");
        let mut decoder =
            crate::ort_utils::build_ort_session(&dir.join("qwen3-asr-decoder.onnx"), false)
                .expect("load decoder");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&dir.join("tokenizer.json")).expect("load tokenizer");

        let audio = load_wav_f32(&wav);
        eprintln!(
            "[batch] audio: {} samples ({:.1}s)",
            audio.len(),
            audio.len() as f32 / 16_000.0
        );

        // Step 1: compute mel for full audio at once (no sliding window).
        let mel = compute_qwen_mel(&audio);
        eprintln!("[batch] mel shape: {:?}", mel.shape());

        // Step 2: encode.
        let enc_features = run_encoder(&mut encoder, mel).expect("run_encoder");
        let enc_seq_len = enc_features.shape()[1]; // encoder output length (not mel frames)
        eprintln!(
            "[batch] enc_features shape: {:?}, enc_seq_len={enc_seq_len}",
            enc_features.shape()
        );

        // Step 3: build input_embeds — first pass, no stable tokens.
        let (input_embeds, prefill_len) = build_input_embeds(
            &mut embed_tokens,
            &enc_features,
            enc_seq_len,
            &[],
            &[],
            false,
        )
        .expect("build_input_embeds");
        eprintln!("[batch] prefill_len: {prefill_len}");

        // Step 4: prefill — past_seq_len=0, empty KV cache.
        let kv_cache = empty_kv_cache();
        let (mut kv_cache, last_logits) =
            decoder_prefill(&mut decoder, input_embeds, kv_cache, 0).expect("decoder_prefill");
        eprintln!("[batch] prefill done, kv_cache len: {}", kv_cache.len());

        // Step 5: autoregressive decode up to 200 tokens.
        let (token_ids, log_probs) = autoregressive_decode(
            &mut decoder,
            &mut embed_tokens,
            &tokenizer,
            &mut kv_cache,
            last_logits,
            200,
            prefill_len,
        )
        .expect("autoregressive_decode");
        eprintln!("[batch] decoded {} tokens", token_ids.len());

        let avg_conf = if log_probs.is_empty() {
            0.0f32
        } else {
            (log_probs.iter().sum::<f32>() / log_probs.len() as f32).exp()
        };
        eprintln!("[batch] avg token confidence: {avg_conf:.3}");

        // Step 6: decode token IDs to text.
        let ids_u32: Vec<u32> = token_ids.iter().map(|&t| t as u32).collect();
        let text = tokenizer.decode(&ids_u32, true).unwrap_or_default();
        eprintln!("[batch] full transcript: {text:?}");
    }
}
