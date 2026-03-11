//! Batch ASR transcription for the ORT Qwen ASR backend.
//!
//! Provides mel spectrogram computation, encoder/decoder infrastructure, and
//! the `transcribe_oneshot` entry point. The encoder, embed_tokens, and decoder
//! are ORT sessions passed by reference; this module owns only mel computation.

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

// ── Special tokens (Qwen3 vocabulary) ────────────────────────────────────────

const TOKEN_IM_START: i64 = 151644;
const TOKEN_IM_END: i64 = 151645;
pub(crate) const TOKEN_AUDIO_START: i64 = 151669;
pub(crate) const TOKEN_AUDIO_END: i64 = 151670;

/// Prefix tokens before the audio:
/// `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<audio_start>`
const PREFIX_HEAD: &[i64] = &[
    TOKEN_IM_START,
    8948,
    198,
    TOKEN_IM_END,
    198,
    TOKEN_IM_START,
    872,
    198,
];
/// Start of the audio placeholder span.
const PREFIX_AUDIO: &[i64] = &[TOKEN_AUDIO_START];
/// Suffix after the audio placeholder span:
/// `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
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

    // Match WhisperFeatureExtractor: drop the final STFT frame.
    let n_frames = mel.shape()[1].saturating_sub(1);
    let mel = mel.slice(s![.., ..n_frames]).to_owned();
    let expected = MEL_BINS * n_frames;
    let (raw, offset) = mel.into_raw_vec_and_offset();
    let raw = match offset {
        None | Some(0) if raw.len() == expected => raw,
        Some(start) if start + expected <= raw.len() => raw[start..start + expected].to_vec(),
        _ => panic!("mel reshape to [1, 128, T]: incompatible shapes"),
    };
    // mel is [128, T] row-major. Array3 [1, 128, T] has same layout.
    Array3::from_shape_vec((1, MEL_BINS, n_frames), raw).expect("mel reshape to [1, 128, T]")
}

// ── Encoder ─────────────────────────────────────────────────────────────────

pub(crate) fn run_encoder(encoder: &mut Session, mel: Array3<f32>) -> Result<ArrayD<half::f16>> {
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
    tokenizer: &tokenizers::Tokenizer,
    enc_features: &ArrayD<half::f16>,
    enc_seq_len: usize,
    forced_language: Option<&str>,
) -> Result<(ArrayD<half::f16>, usize)> {
    // Collect all token IDs that need embedding.
    // Match the official chat template:
    // <|im_start|>system\n{context}<|im_end|>\n
    // <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
    // <|im_start|>assistant\n
    // Optional forced-language prefix:
    // language English<asr_text>
    let mut pre_tokens: Vec<i64> = Vec::with_capacity(48);
    pre_tokens.extend_from_slice(PREFIX_HEAD);
    pre_tokens.extend_from_slice(PREFIX_AUDIO);

    let mut post_tokens: Vec<i64> = Vec::with_capacity(48);
    post_tokens.extend_from_slice(SUFFIX_BASE);
    if let Some(language) = forced_language {
        // Upstream text-only forcing appends this exact assistant prefix
        // after the generation prompt.
        post_tokens.extend(tokenize_prompt(
            tokenizer,
            &format!("language {language}<asr_text>"),
        )?);
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

fn tokenize_prompt(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<i64>> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| Error::Asr(format!("forced language prompt encode: {e}")))?;
    Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
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

// ── KV cache ────────────────────────────────────────────────────────────────

/// Create empty KV cache tensors (seq_len=0) for all layers.
fn empty_kv_cache() -> Vec<DynValue> {
    let mut kv = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
    for _ in 0..NUM_DECODER_LAYERS * 2 {
        let arr = ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), half::f16::ZERO);
        kv.push(Value::from_array(arr).expect("empty kv tensor").into());
    }
    kv
}

// ── Decoder prefill ─────────────────────────────────────────────────────────

/// Run decoder prefill: feed input_embeds (f16) with KV cache, return updated KV + logits.
fn decoder_prefill(
    decoder: &mut Session,
    input_embeds: ArrayD<half::f16>,
    kv_cache: Vec<DynValue>,
    past_seq_len: usize,
    attention_mask: Option<ndarray::Array2<i64>>,
) -> Result<(Vec<DynValue>, Vec<f32>)> {
    let seq_len = input_embeds.shape()[1];
    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(4 + NUM_DECODER_LAYERS * 2);

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

    // attention_mask: [B, past_seq_len + seq_len] — all-ones for B=1, padding mask for B>1.
    // Optional: old models without attention_mask input still work.
    if let Some(mask) = attention_mask {
        inputs.push((
            "attention_mask".to_string(),
            Value::from_array(mask)
                .map_err(|e| Error::Asr(format!("attention_mask: {e}")))?
                .into(),
        ));
    }

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
    use_attention_mask: bool,
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
        let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(4 + NUM_DECODER_LAYERS * 2);

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

        // attention_mask: all-ones [1, past + 1] for B=1 decode step.
        if use_attention_mask {
            let step_mask = ndarray::Array2::<i64>::ones((1, cur_pos + 1));
            inputs.push((
                "attention_mask".to_string(),
                Value::from_array(step_mask)
                    .map_err(|e| Error::Asr(format!("step attention_mask: {e}")))?
                    .into(),
            ));
        }
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

// ── One-shot transcription (no streaming state) ─────────────────────────────

/// Result of a single-shot transcription of one audio chunk.
pub(crate) struct OneshotResult {
    pub text: String,
    pub token_ids: Vec<i64>,
    pub log_probs: Vec<f32>,
}

/// Transcribe a single audio chunk in one shot (no streaming state).
///
/// Encodes the audio, builds ChatML input embeddings, runs prefill + autoregressive
/// decode, and returns the text with per-token confidence data.
///
/// This is the building block for overlap-stitched chunked transcription.
#[allow(clippy::too_many_arguments)]
pub(crate) fn transcribe_oneshot(
    encoder: &mut Session,
    embed_tokens: &mut Session,
    decoder: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    pcm: &[f32],
    forced_language: Option<&str>,
    has_attention_mask: bool,
) -> Result<OneshotResult> {
    if pcm.is_empty() {
        return Ok(OneshotResult {
            text: String::new(),
            token_ids: vec![],
            log_probs: vec![],
        });
    }

    // 1. Encode audio → features
    let mel = compute_qwen_mel(pcm);
    let features = run_encoder(encoder, mel)?;
    let enc_seq_len = features.shape()[1];

    // 2. Build input_embeds for the upstream chat template. When the language
    // is forced, Qwen expects the assistant prefix `language X<asr_text>`.
    let (input_embeds, prefill_len) = build_input_embeds(
        embed_tokens,
        tokenizer,
        &features,
        enc_seq_len,
        forced_language,
    )?;

    // 3. Prefill
    let kv_cache = empty_kv_cache();
    let mask = if has_attention_mask {
        Some(ndarray::Array2::<i64>::ones((1, prefill_len)))
    } else {
        None
    };
    let (mut kv, last_logits) = decoder_prefill(decoder, input_embeds, kv_cache, 0, mask)?;

    // 4. Autoregressive decode
    // Model supports up to 1200s audio; session hard-caps at 60s.
    // 512 tokens is sufficient for 60s of speech (~8 tokens/sec BPE).
    // The decoder emits EOS when done — this is just a safety ceiling.
    let (token_ids, log_probs) = autoregressive_decode(
        decoder,
        embed_tokens,
        tokenizer,
        &mut kv,
        last_logits,
        512,
        prefill_len,
        has_attention_mask,
    )?;

    // 5. Filter out special/prompt tokens before decoding to text.
    // The decoder sometimes emits ChatML structure tokens as content
    // (e.g. im_start, im_end, audio markers, asr_text).
    let special_ids: &[i64] = &[
        TOKEN_IM_START,
        TOKEN_IM_END,
        TOKEN_AUDIO_START,
        TOKEN_AUDIO_END,
    ];
    let filtered_ids: Vec<i64> = token_ids
        .iter()
        .copied()
        .filter(|t| !special_ids.contains(t))
        .collect();
    let filtered_lps: Vec<f32> = token_ids
        .iter()
        .zip(log_probs.iter())
        .filter(|(t, _)| !special_ids.contains(t))
        .map(|(_, &lp)| lp)
        .collect();

    let ids_u32: Vec<u32> = filtered_ids.iter().map(|&t| t as u32).collect();
    let mut raw_text = tokenizer.decode(&ids_u32, true).unwrap_or_default();

    // Strip any leaked prompt fragments (e.g. "language English", "Transcribe the")
    for pattern in &["<|im_end|>", "<|im_start|>"] {
        raw_text = raw_text.replace(pattern, "");
    }
    let raw_text = raw_text.trim().to_owned();
    let text = if forced_language.is_some() {
        raw_text.clone()
    } else if let Some((_, text)) = raw_text.split_once("<asr_text>") {
        text.trim().to_owned()
    } else {
        raw_text.clone()
    };

    Ok(OneshotResult {
        text,
        token_ids: filtered_ids,
        log_probs: filtered_lps,
    })
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
        let _has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        // 440 Hz tone — not real speech but deterministic
        let audio = tone_440hz(1.0);
        let mel = compute_qwen_mel(&audio);
        let features = run_encoder(&mut encoder, mel).expect("encoder forward");
        let enc_seq_len = features.shape()[1];

        let (input_embeds, prefill_len) =
            build_input_embeds(&mut embed_tokens, &tokenizer, &features, enc_seq_len, None)
                .expect("build_input_embeds");
        eprintln!("[decoder] enc_seq_len={enc_seq_len}, prefill_len={prefill_len}");

        let kv = empty_kv_cache();
        let mask = ndarray::Array2::<i64>::ones((1, prefill_len));
        let (mut new_kv, last_logits) =
            decoder_prefill(&mut decoder, input_embeds, kv, 0, Some(mask))
                .expect("decoder_prefill");

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
            true,
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
        let _has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        // First 0.5s of real speech
        let audio = load_wav_f32(&wav);
        let chunk = &audio[..8_000.min(audio.len())];
        eprintln!("[real_speech] using {} samples", chunk.len());

        let mel = compute_qwen_mel(chunk);
        let features = run_encoder(&mut encoder, mel).expect("encoder forward");
        let enc_seq_len = features.shape()[1];
        eprintln!("[real_speech] enc_seq_len={enc_seq_len}");

        let (input_embeds, prefill_len) =
            build_input_embeds(&mut embed_tokens, &tokenizer, &features, enc_seq_len, None)
                .expect("build_input_embeds");

        let kv = empty_kv_cache();
        let mask = ndarray::Array2::<i64>::ones((1, prefill_len));
        let (mut new_kv, last_logits) =
            decoder_prefill(&mut decoder, input_embeds, kv, 0, Some(mask))
                .expect("decoder_prefill");

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
            true,
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
        let _has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        let audio = tone_440hz(1.0);
        let mel = compute_qwen_mel(&audio);
        let real_features = run_encoder(&mut encoder, mel).expect("encoder forward");
        let enc_seq_len = real_features.shape()[1];

        // Case A: real encoder features
        let (emb_a, _plen_a) = build_input_embeds(
            &mut embed_tokens,
            &tokenizer,
            &real_features,
            enc_seq_len,
            None,
        )
        .expect("build_input_embeds A");
        let kv_a = empty_kv_cache();
        let mask_a = ndarray::Array2::<i64>::ones((1, emb_a.shape()[1]));
        let (_, logits_a) =
            decoder_prefill(&mut decoder, emb_a, kv_a, 0, Some(mask_a)).expect("prefill A");

        // Case B: all-zero encoder features (same shape)
        let zero_features =
            ArrayD::from_elem(IxDyn(&[1, enc_seq_len, HIDDEN_DIM]), half::f16::ZERO);
        let (emb_b, _plen_b) = build_input_embeds(
            &mut embed_tokens,
            &tokenizer,
            &zero_features,
            enc_seq_len,
            None,
        )
        .expect("build_input_embeds B");
        let kv_b = empty_kv_cache();
        let mask_b = ndarray::Array2::<i64>::ones((1, emb_b.shape()[1]));
        let (_, logits_b) =
            decoder_prefill(&mut decoder, emb_b, kv_b, 0, Some(mask_b)).expect("prefill B");

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
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

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
            ("TOKEN_ASR_TEXT", 151704),
        ] {
            let decoded = tok.decode(&[id as u32], false).unwrap_or_default();
            eprintln!("[tok] {name} ({id}) → {decoded:?}");
        }
    }

    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx"]
    fn tokenizer_chat_template_segments() {
        ensure_ort();
        let dir = model_dir();
        if !dir.exists() {
            return;
        }
        let tok =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        for text in [
            "<|im_start|>system\n",
            "<|im_end|>\n<|im_start|>user\n<|audio_start|>",
            "<|audio_end|><|im_end|>\n<|im_start|>assistant\n",
            "language English<asr_text>",
        ] {
            let enc = tok.encode(text, false).expect("encode template segment");
            eprintln!("[tmpl] text={text:?}");
            eprintln!("[tmpl] ids={:?}", enc.get_ids());
            eprintln!("[tmpl] toks={:?}", enc.get_tokens());
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
    fn empty_kv_cache_shape() {
        let kv = empty_kv_cache();
        assert_eq!(kv.len(), NUM_DECODER_LAYERS * 2);
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
        let _has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

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
            &tokenizer,
            &enc_features,
            enc_seq_len,
            None,
        )
        .expect("build_input_embeds");
        eprintln!("[batch] prefill_len: {prefill_len}");

        // Step 4: prefill — past_seq_len=0, empty KV cache.
        let kv_cache = empty_kv_cache();
        let mask = ndarray::Array2::<i64>::ones((1, prefill_len));
        let (mut kv_cache, last_logits) =
            decoder_prefill(&mut decoder, input_embeds, kv_cache, 0, Some(mask))
                .expect("decoder_prefill");
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
            true,
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

    #[test]
    #[ignore = "requires test_fixtures/librispeech_clean.wav"]
    fn qwen_mel_stats_librispeech() {
        let wav = workspace_root().join("test_fixtures/librispeech_clean.wav");
        if !wav.exists() {
            eprintln!("[skip] wav not found");
            return;
        }

        let audio = load_wav_f32(&wav);
        let mel = compute_qwen_mel(&audio);
        let data = mel.as_slice().expect("mel contiguous");
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = data.iter().copied().sum::<f32>() / data.len() as f32;
        eprintln!("[mel] shape: {:?}", mel.shape());
        eprintln!("[mel] min/max/mean: {min} {max} {mean}");
        eprintln!("[mel] first row first 8: {:?}", &data[..8.min(data.len())]);
        let t = mel.shape()[2];
        let mut col0 = Vec::new();
        for bin in 0..8.min(mel.shape()[1]) {
            col0.push(mel[[0, bin, 0]]);
        }
        eprintln!("[mel] col0 first 8 bins: {:?}", col0);
        eprintln!("[mel] frames: {t}");
    }

    /// Same as `batch_full_audio`, but with the official forced-language
    /// prefix used by the app path: `language English<asr_text>`.
    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx and test_fixtures/librispeech_clean.wav"]
    fn batch_full_audio_forced_english() {
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
        let has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        let audio = load_wav_f32(&wav);
        let result = transcribe_oneshot(
            &mut encoder,
            &mut embed_tokens,
            &mut decoder,
            &tokenizer,
            &audio,
            Some("English"),
            has_attention_mask,
        )
        .expect("transcribe_oneshot");

        eprintln!("[batch forced] transcript: {:?}", result.text);
        eprintln!("[batch forced] tokens: {}", result.token_ids.len());
    }

    fn resize_encoder_features_nearest(
        features: &ArrayD<half::f16>,
        target_len: usize,
    ) -> ArrayD<half::f16> {
        let src = features.as_slice().expect("encoder features contiguous");
        let src_len = features.shape()[1];
        let hidden = features.shape()[2];
        let mut out = vec![half::f16::ZERO; target_len * hidden];
        for dst_idx in 0..target_len {
            let src_idx = dst_idx * src_len / target_len;
            let src_row = &src[src_idx * hidden..(src_idx + 1) * hidden];
            let dst_row = &mut out[dst_idx * hidden..(dst_idx + 1) * hidden];
            dst_row.copy_from_slice(src_row);
        }
        ArrayD::from_shape_vec(IxDyn(&[1, target_len, hidden]), out).expect("resized features")
    }

    #[test]
    #[ignore = "requires models/qwen3-asr-0.6b-onnx and test_fixtures/librispeech_clean.wav"]
    fn batch_full_audio_forced_english_resized_to_303() {
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
        let has_attention_mask = decoder
            .inputs()
            .iter()
            .any(|inp| inp.name() == "attention_mask");
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        let audio = load_wav_f32(&wav);
        let mel = compute_qwen_mel(&audio);
        let features = run_encoder(&mut encoder, mel).expect("run_encoder");
        let resized = resize_encoder_features_nearest(&features, 303);
        let (input_embeds, prefill_len) = build_input_embeds(
            &mut embed_tokens,
            &tokenizer,
            &resized,
            303,
            Some("English"),
        )
        .expect("build_input_embeds");

        let kv_cache = empty_kv_cache();
        let mask = if has_attention_mask {
            Some(ndarray::Array2::<i64>::ones((1, prefill_len)))
        } else {
            None
        };
        let (mut kv_cache, last_logits) =
            decoder_prefill(&mut decoder, input_embeds, kv_cache, 0, mask)
                .expect("decoder_prefill");
        let (token_ids, _) = autoregressive_decode(
            &mut decoder,
            &mut embed_tokens,
            &tokenizer,
            &mut kv_cache,
            last_logits,
            200,
            prefill_len,
            has_attention_mask,
        )
        .expect("autoregressive_decode");

        let ids_u32: Vec<u32> = token_ids.iter().map(|&t| t as u32).collect();
        let text = tokenizer.decode(&ids_u32, true).unwrap_or_default();
        eprintln!("[batch forced 303] transcript: {text:?}");
        eprintln!("[batch forced 303] tokens: {}", token_ids.len());
    }
}
