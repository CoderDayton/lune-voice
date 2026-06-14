//! Qwen3-ASR 0.6B ONNX backend — encoder-decoder with autoregressive greedy decoding.
//!
//! Model: custom ONNX export via `scripts/export_onnx.py` (encoder + embed_tokens + decoder).
//! Mel: Whisper-compatible 128-bin log-mel at 16 kHz (see `whisper_mel.rs`).
//! Decoding: greedy argmax with per-token softmax confidence.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use half::f16;
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{DynValue, TensorRef, Value};
use tokenizers::Tokenizer;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::error::{Error, Result};

use crate::mel::whisper;
use crate::ort_utils::{OrtContext, OrtModelLoader, make_shrink_run_options};

// ── Model constants ─────────────────────────────────────────────────────────

const VOCAB_SIZE: usize = 151_936;
const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
/// Decode budget. Sized from a tokens/sec simulation over the real fixture
/// transcripts (peak ~3.3 tok/s English, ~4.3 tok/s multilingual) against the
/// encoder's 30 s trained geometry: ~60 s of dense speech × 2× margin ≈ 512.
/// Generous enough that no realistic single segment is truncated, yet bounded
/// so a degenerate run terminates (the repeat guard is the inner backstop).
const MAX_NEW_TOKENS: usize = 512;

const EOS_IDS: [i64; 2] = [151_643, 151_645];
const AUDIO_PAD_ID: i64 = 151_676;
/// `<asr_text>` special token.
const ASR_TEXT_ID: i64 = 151_704;

// Special token IDs for chat template construction.
const IM_START: i64 = 151_644; // <|im_start|>
const IM_END: i64 = 151_645; // <|im_end|>
const NEWLINE: i64 = 198; // \n
const AUDIO_START: i64 = 151_669; // <|audio_start|>
const AUDIO_END: i64 = 151_670; // <|audio_end|>

/// Base ASR instruction, matching the reference deployment
/// (`scripts/test_asr_reference.py`). An empty system prompt measurably
/// increases hallucination and translation drift on noisy audio.
const SYSTEM_PROMPT: &str = "You are an ASR engine. Transcribe spoken audio exactly in the spoken language. Do not translate. Do not add words not clearly present.";

/// Maximum audio buffer (5 min at 16 kHz).
const MAX_WINDOW_SAMPLES: usize = 300 * whisper::SAMPLE_RATE;

/// Target utterance RMS for loudness normalization (≈ −22 dBFS). The mel
/// front-end's final scale is absolute, so quiet input lands outside the
/// model's training distribution — accuracy becomes volume-dependent.
const NORM_TARGET_RMS: f32 = 0.08;
/// Boost ceiling: past this we are amplifying room tone, not speech.
const NORM_MAX_GAIN: f32 = 10.0;

/// Utterance-level loudness normalization: a single scalar gain per
/// utterance (boost-only, peak-limited) so quiet dictation reaches the
/// level the model was trained on. One scalar — unlike a running AGC — adds
/// no time-varying artifacts the encoder never saw in training. Loud input
/// passes through untouched; near-silence is left alone rather than blown
/// up into noise.
fn normalize_loudness(pcm: &[f32]) -> std::borrow::Cow<'_, [f32]> {
    use std::borrow::Cow;
    if pcm.is_empty() {
        return Cow::Borrowed(pcm);
    }
    let mut sum_sq = 0.0f64;
    let mut peak = 0.0f32;
    for &x in pcm {
        if x.is_finite() {
            sum_sq += f64::from(x) * f64::from(x);
            peak = peak.max(x.abs());
        }
    }
    let rms = (sum_sq / pcm.len() as f64).sqrt() as f32;
    if rms <= 1.0e-6 || peak <= 0.0 {
        return Cow::Borrowed(pcm); // effectively silence — nothing to gain
    }
    let gain = (NORM_TARGET_RMS / rms)
        .clamp(1.0, NORM_MAX_GAIN)
        .min(0.985 / peak)
        .max(1.0);
    if gain <= 1.01 {
        return Cow::Borrowed(pcm);
    }
    tracing::debug!(gain, rms, peak, "utterance loudness normalized");
    Cow::Owned(pcm.iter().map(|&x| (x * gain).clamp(-1.0, 1.0)).collect())
}

// ── Types ───────────────────────────────────────────────────────────────────

pub struct Qwen3AsrConfig {
    pub model_dir: PathBuf,
    pub use_gpu: bool,
    /// True when the user explicitly chose CUDA — registers the EP with
    /// `error_on_failure()` so the session fails to load instead of silently
    /// running on CPU.
    pub cuda_required: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen3AsrProfile {
    pub audio_ms: usize,
    pub total_ms: f64,
    pub mel_ms: f64,
    pub encoder_ms: f64,
    pub prompt_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_steps: usize,
    pub decoder_io_binding: bool,
}

struct Qwen3Models {
    encoder: Session,
    embed_tokens: Session,
    decoder: Session,
}

pub struct Qwen3AsrBackend {
    models: Mutex<Qwen3Models>,
    mel: whisper::WhisperMelF16,
    tokenizer: Tokenizer,
    use_gpu: bool,
    decoder_io_binding: bool,
    last_profile: Mutex<Option<Qwen3AsrProfile>>,
    /// Tokenized boost phrases (both bare and leading-space tokenizations)
    /// plus the additive logit score. Consumed per decode step by
    /// `phrase_boost_tokens` — prefix-constrained, not a flat token set.
    boost_seqs: Mutex<(Vec<Vec<i64>>, f32)>,
    /// Per-session context phrases injected into the system prompt.
    /// e.g. ["Toledans", "Rodolfo"] → "Preferred terms: Toledans, Rodolfo"
    context_phrases: Mutex<Vec<String>>,
}

pub struct Qwen3Session {
    pending: Vec<f32>,
    total_samples: u64,
    sample_rate: u32,
    next_id: u64,
    /// Language to force on the model prompt, in the form Qwen3-ASR expects
    /// ("English", "Chinese", ...). `None` means auto-detect.
    forced_language: Option<String>,
}

// ── Constructor ─────────────────────────────────────────────────────────────

impl Qwen3AsrBackend {
    pub fn new(config: Qwen3AsrConfig) -> Result<Self> {
        Self::new_with_decoder_io_binding(config, true)
    }

    pub fn new_with_decoder_io_binding(
        config: Qwen3AsrConfig,
        decoder_io_binding: bool,
    ) -> Result<Self> {
        let loader = OrtModelLoader::new(
            config.model_dir.clone(),
            config.use_gpu,
            config.cuda_required,
        );

        let encoder = loader.load_first(&["qwen3-asr-encoder.onnx"], "qwen3-asr-encoder")?;
        let embed_tokens =
            loader.load_first(&["qwen3-asr-embed-tokens.onnx"], "qwen3-asr-embed-tokens")?;
        let decoder = loader.load_first(&["qwen3-asr-decoder.onnx"], "qwen3-asr-decoder")?;

        let tok_path = config.model_dir.join("tokenizer.json");
        // Tokenizer's error type isn't `'static`, so it can't satisfy the
        // generic source bound used by `.ctx()`. Format inline instead.
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::asr(format!("load tokenizer: {e}")))?;

        tracing::info!(dir = %config.model_dir.display(), "Qwen3-ASR backend loaded");

        Ok(Self {
            models: Mutex::new(Qwen3Models {
                encoder,
                embed_tokens,
                decoder,
            }),
            mel: whisper::WhisperMelF16::new_qwen(),
            tokenizer,
            use_gpu: config.use_gpu,
            decoder_io_binding,
            last_profile: Mutex::new(None),
            boost_seqs: Mutex::new((Vec::new(), 0.0)),
            context_phrases: Mutex::new(Vec::new()),
        })
    }

    pub fn last_profile(&self) -> Option<Qwen3AsrProfile> {
        self.last_profile.lock().ok().and_then(|p| *p)
    }

    /// Set word boost phrases — dual-layer: system prompt context + logit bias.
    /// Phrases are injected into the system prompt as preferred terms AND
    /// tokenized for prefix-constrained additive logit bias on decode.
    ///
    /// Each phrase is tokenized twice — bare and with a leading space — because
    /// BPE assigns different ids to word-initial and mid-sentence forms.
    pub fn set_boost_phrases(&self, phrases: &[String], score: f32) {
        // Layer 1: context phrases for system prompt injection
        *self
            .context_phrases
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = phrases.to_vec();

        // Layer 2: tokenized sequences for prefix-constrained logit bias
        let mut seqs: Vec<Vec<i64>> = Vec::with_capacity(phrases.len() * 2);
        for phrase in phrases {
            for variant in [phrase.clone(), format!(" {phrase}")] {
                if let Ok(enc) = self.tokenizer.encode(variant.as_str(), false) {
                    let ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
                    if !ids.is_empty() && !seqs.contains(&ids) {
                        seqs.push(ids);
                    }
                }
            }
        }
        if !seqs.is_empty() {
            tracing::debug!(
                phrases = phrases.len(),
                seqs = seqs.len(),
                "boost phrases set"
            );
        }
        *self.boost_seqs.lock().unwrap_or_else(|e| e.into_inner()) = (seqs, score);
    }

    /// Build prompt token IDs: prefix (before audio) and suffix (after audio).
    ///
    /// System prompt = base invariants + optional context phrases.
    /// Language handling follows official Qwen3-ASR:
    ///   - `language` = Some("English") → suffix includes `language English` (skip LID)
    ///   - `language` = None → model auto-detects, generates `language X<asr_text>...`
    fn build_prompt_ids(&self, language: Option<&str>) -> (Vec<i64>, Vec<i64>) {
        let ctx = self
            .context_phrases
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Build system prompt text — always include base prompt.
        let sys_text = if ctx.is_empty() {
            SYSTEM_PROMPT.to_string()
        } else {
            format!("{}\nPreferred terms: {}", SYSTEM_PROMPT, ctx.join(", "))
        };

        // Tokenize system prompt body
        let sys_ids: Vec<i64> = self
            .tokenizer
            .encode(sys_text.as_str(), false)
            .map(|enc| enc.get_ids().iter().map(|&id| id as i64).collect())
            .unwrap_or_default();

        // <|im_start|>system\n{sys_text}<|im_end|>\n<|im_start|>user\n<|audio_start|>
        let mut prefix = Vec::with_capacity(6 + sys_ids.len());
        prefix.push(IM_START);
        prefix.push(8948); // "system"
        prefix.push(NEWLINE);
        prefix.extend_from_slice(&sys_ids);
        prefix.push(IM_END);
        prefix.push(NEWLINE);
        prefix.push(IM_START);
        prefix.push(872); // "user"
        prefix.push(NEWLINE);
        prefix.push(AUDIO_START);

        // <|audio_end|><|im_end|>\n<|im_start|>assistant\n[language X]<asr_text>
        // When language is forced, include `language X` before <asr_text> so the
        // model skips language detection. Otherwise model auto-detects.
        let mut suffix = vec![AUDIO_END, IM_END, NEWLINE, IM_START, 77091, NEWLINE];
        // 77091 = "assistant"

        if let Some(lang) = language {
            let lang_text = format!("language {lang}");
            if let Ok(enc) = self.tokenizer.encode(lang_text.as_str(), false) {
                suffix.extend(enc.get_ids().iter().map(|&id| id as i64));
            }
        }

        (prefix, suffix)
    }
}

// ── AsrBackend impl ─────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl AsrBackend for Qwen3AsrBackend {
    type SessionHandle = Mutex<Qwen3Session>;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        let forced_language = config.language.as_deref().and_then(bcp47_to_qwen3_language);
        let session = Qwen3Session {
            pending: Vec::with_capacity(MAX_WINDOW_SAMPLES),
            total_samples: 0,
            sample_rate: config.sample_rate,
            next_id: 1,
            forced_language: forced_language.clone(),
        };
        tracing::info!(
            sample_rate = config.sample_rate,
            language = ?forced_language,
            "Qwen3 ASR session started"
        );
        Ok(Mutex::new(session))
    }

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut sess = session.lock().map_err(|e| lock_err("session", e))?;
        sess.total_samples += pcm.len() as u64;
        sess.pending.extend_from_slice(pcm);
        if sess.pending.len() > MAX_WINDOW_SAMPLES {
            let trim = sess.pending.len() - MAX_WINDOW_SAMPLES;
            sess.pending.drain(..trim);
        }
        Ok(())
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let (pcm, ts_ms, id, language) = {
            let mut sess = session.lock().map_err(|e| lock_err("session", e))?;
            if sess.pending.is_empty() {
                return Ok(vec![]);
            }
            let ts = sess.total_samples * 1000 / sess.sample_rate as u64;
            let id = sess.next_id;
            sess.next_id += 1;
            let pcm = std::mem::take(&mut sess.pending);
            let lang = sess.forced_language.clone();
            (pcm, ts, id, lang)
        };

        if pcm.is_empty() {
            return Ok(vec![]);
        }

        self.transcribe_pcm(&pcm, ts_ms, id, language.as_deref())
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let mut sess = session.lock().map_err(|e| lock_err("session", e))?;
        sess.pending.clear();
        Ok(())
    }

    async fn end_session(&self, _session: Self::SessionHandle) -> Result<()> {
        Ok(())
    }
}

// ── Inference ───────────────────────────────────────────────────────────────
/// Extract FP16 tensor to owned FP16 ArrayD (no conversion).
fn extract_f16_array(value: &ort::value::Value) -> Result<ArrayD<f16>> {
    let (shape, data) = value.try_extract_tensor::<f16>().ctx("extract f16")?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec()).ctx("array reshape")
}

const MABLE_TOP_K: usize = 16;

/// Fused argmax + softmax confidence + top-k logit extraction on f16 logits tensor.
/// Returns (token_id, softmax_confidence, top_k_logits_as_f64).
fn argmax_on_logits(
    value: &ort::value::Value,
    vocab_size: usize,
    boost: &HashMap<i64, f32>,
    banned: &HashSet<i64>,
) -> Result<(i64, f32, Vec<f64>)> {
    let (shape, data) = value.try_extract_tensor::<f16>().ctx("extract logits")?;
    // Last position's logits: last `vocab_size` elements.
    let total = shape.iter().map(|&d| d as usize).product::<usize>();
    let start = total.saturating_sub(vocab_size);
    let logits = &data[start..];

    if logits.is_empty() {
        return Err(Error::asr("empty logits tensor"));
    }
    if logits.len() != vocab_size {
        return Err(Error::asr(format!(
            "logits slice {} != vocab {vocab_size}",
            logits.len()
        )));
    }

    // Top-k min-heap: (value, index), sorted ascending so heap[0] is the min.
    let mut heap = [(f32::NEG_INFINITY, 0usize); MABLE_TOP_K];
    let mut heap_min = f32::NEG_INFINITY;
    let mut heap_min_pos = 0usize;

    if boost.is_empty() && banned.is_empty() {
        // Single-pass: argmax + top-k + LSE fused (one f16→f32 conversion per element).
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            let f = v.to_f32();
            if f > max_val {
                sum_exp *= (max_val - f).exp();
                max_val = f;
                max_idx = i;
            }
            sum_exp += (f - max_val).exp();
            if f > heap_min {
                heap[heap_min_pos] = (f, i);
                heap_min = f32::INFINITY;
                for (j, &(val, _)) in heap.iter().enumerate() {
                    if val < heap_min {
                        heap_min = val;
                        heap_min_pos = j;
                    }
                }
            }
        }
        let top_k = heap.iter().map(|(v, _)| *v as f64).collect();
        if !sum_exp.is_normal() {
            return Ok((max_idx as i64, 0.0, top_k));
        }
        let lse = sum_exp.ln() + max_val;
        Ok((max_idx as i64, (max_val - lse).exp(), top_k))
    } else {
        // Constrained single-pass: argmax over boosted + no-repeat-masked
        // values, while the LSE and top-k stay on the raw distribution. The
        // reported confidence is the raw softmax probability of the *chosen*
        // token (`chosen_raw`), not of the raw argmax — boosting or banning can
        // divert the winner, and the confidence must follow the emitted token.
        let mut max_idx = 0usize;
        let mut max_adj = f32::NEG_INFINITY;
        let mut chosen_raw = f32::NEG_INFINITY;
        let mut raw_max = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            let raw = v.to_f32();
            let mut adj = raw + boost.get(&(i as i64)).copied().unwrap_or(0.0);
            if banned.contains(&(i as i64)) {
                adj = f32::NEG_INFINITY;
            }
            if adj > max_adj {
                max_adj = adj;
                max_idx = i;
                chosen_raw = raw;
            }
            if raw > raw_max {
                sum_exp *= (raw_max - raw).exp();
                raw_max = raw;
            }
            sum_exp += (raw - raw_max).exp();
            if raw > heap_min {
                heap[heap_min_pos] = (raw, i);
                heap_min = f32::INFINITY;
                for (j, &(val, _)) in heap.iter().enumerate() {
                    if val < heap_min {
                        heap_min = val;
                        heap_min_pos = j;
                    }
                }
            }
        }
        let top_k = heap.iter().map(|(v, _)| *v as f64).collect();
        if !sum_exp.is_normal() {
            return Ok((max_idx as i64, 0.0, top_k));
        }
        let lse = sum_exp.ln() + raw_max;
        Ok((max_idx as i64, (chosen_raw - lse).exp(), top_k))
    }
}

/// Order of the repeat guard: greedy decoding never emits a token that
/// would re-complete a length-`REPEAT_GUARD_ORDER` token block already
/// produced in this utterance. This kills the runaway repetition loops ASR
/// models fall into on noisy or silent tails, without the indiscriminate WER
/// cost a flat repetition penalty imposes on legitimate word reuse. Distinct
/// from `lune_dict::ngram` (a word-level correction LM): this is a decode-time mask
/// over raw decoder token ids, not a probabilistic model.
const REPEAT_GUARD_ORDER: usize = 3;

/// Prefix-constrained phrase biasing: fill `out` with the boost for the next
/// decode step. A phrase's entry token is always boosted at `score` (a match
/// can start anywhere); its k-th token is boosted at `2 × score` only when
/// the decode tail already equals the phrase's first k tokens. Unlike a flat
/// per-token boost, subwords like "ton" never get promoted in the middle of
/// unrelated words — the bias follows the phrase, not the alphabet.
fn phrase_boost_tokens(
    generated: &[i64],
    seqs: &[Vec<i64>],
    score: f32,
    out: &mut HashMap<i64, f32>,
) {
    out.clear();
    for seq in seqs {
        if let Some(&first) = seq.first() {
            let e = out.entry(first).or_insert(0.0);
            *e = e.max(score);
        }
        // Longest matching prefix wins; stop at the first (longest) hit.
        let max_k = seq.len().saturating_sub(1).min(generated.len());
        for k in (1..=max_k).rev() {
            if generated[generated.len() - k..] == seq[..k] {
                let e = out.entry(seq[k]).or_insert(0.0);
                *e = e.max(score * 2.0);
                break;
            }
        }
    }
}

/// Fill `out` with the token ids that would re-complete a length-`n` block
/// given the tail of `generated`. Empty in the common case, so the hot argmax
/// path stays on its single-pass fast branch.
fn banned_next_tokens(generated: &[i64], n: usize, out: &mut HashSet<i64>) {
    out.clear();
    if n < 2 || generated.len() < n {
        return;
    }
    let prefix = &generated[generated.len() - (n - 1)..];
    for w in generated.windows(n) {
        if w[..n - 1] == *prefix {
            out.insert(w[n - 1]);
        }
    }
}

/// Detect a degenerate tail loop: the last `4 * n` tokens are an `n`-token
/// block repeated four times, for some `n` in 1..=8. Returns the block
/// length so the caller can trim the redundant copies. Requires ≥16 tokens
/// before it bothers checking.
fn repeated_cycle_len(ids: &[i64]) -> Option<usize> {
    let len = ids.len();
    if len < 16 {
        return None;
    }
    (1..=8).find(|&n| {
        let window = 4 * n;
        len >= window && {
            let base = len - window;
            (1..4).all(|r| (0..n).all(|k| ids[base + k] == ids[base + r * n + k]))
        }
    })
}

/// Build mRoPE position_ids `[3, 1, seq_len]` — all 3 dims identical for audio-only.
fn mrope_position_ids(seq_len: usize) -> Array3<i64> {
    let row: Vec<i64> = (0..seq_len as i64).collect();
    let mut data = Vec::with_capacity(3 * seq_len);
    for _ in 0..3 {
        data.extend_from_slice(&row);
    }
    Array3::from_shape_vec((3, 1, seq_len), data).unwrap()
}

type OrtInput<'a> = (
    std::borrow::Cow<'a, str>,
    ort::session::SessionInputValue<'a>,
);

/// Append per-layer empty KV cache inputs for prefill (past_len=0).
fn append_empty_kv(inputs: &mut Vec<OrtInput<'_>>) -> Result<()> {
    for i in 0..NUM_LAYERS {
        let k = ndarray::ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), f16::ZERO);
        let v = ndarray::ArrayD::from_elem(IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]), f16::ZERO);
        inputs.push((
            std::borrow::Cow::Borrowed(kv_name(i, true)),
            Value::from_array(k).ctx("empty k{i}")?.into(),
        ));
        inputs.push((
            std::borrow::Cow::Borrowed(kv_name(i, false)),
            Value::from_array(v).ctx("empty v{i}")?.into(),
        ));
    }
    Ok(())
}

fn append_cached_kv(inputs: &mut Vec<OrtInput<'_>>, kv_cache: Vec<ArrayD<f16>>) -> Result<()> {
    for (i, arr) in kv_cache.into_iter().enumerate() {
        let layer = i / 2;
        let is_key = i % 2 == 0;
        inputs.push((
            std::borrow::Cow::Borrowed(kv_name(layer, is_key)),
            Value::from_array(arr)
                .ctx("cached kv layer {layer}")?
                .into(),
        ));
    }
    Ok(())
}

fn bind_empty_kv(binding: &mut ort::session::IoBinding) -> Result<()> {
    for i in 0..NUM_LAYERS {
        let k = Value::from_array(ndarray::ArrayD::from_elem(
            IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]),
            f16::ZERO,
        ))
        .ctx("empty k{i}")?;
        let v = Value::from_array(ndarray::ArrayD::from_elem(
            IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]),
            f16::ZERO,
        ))
        .ctx("empty v{i}")?;
        binding
            .bind_input(kv_name(i, true), &k)
            .ctx("bind empty key layer {i}")?;
        binding
            .bind_input(kv_name(i, false), &v)
            .ctx("bind empty value layer {i}")?;
    }
    Ok(())
}

/// Static KV cache tensor names: [past_k, past_v, present_k, present_v] × NUM_LAYERS.
/// Initialized once, safe `&'static str` references via OnceLock.
fn kv_cache_names() -> &'static [String] {
    use std::sync::OnceLock;
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut v = Vec::with_capacity(NUM_LAYERS * 4);
        for i in 0..NUM_LAYERS {
            v.push(format!("past_key_values.{i}.key"));
            v.push(format!("past_key_values.{i}.value"));
            v.push(format!("present_key_values.{i}.key"));
            v.push(format!("present_key_values.{i}.value"));
        }
        v
    })
}

fn kv_name(layer: usize, is_key: bool) -> &'static str {
    let idx = layer * 4 + if is_key { 0 } else { 1 };
    // Safety: kv_cache_names() returns &'static [String], element .as_str() is 'static.
    unsafe { &*(kv_cache_names()[idx].as_str() as *const str) }
}

fn present_kv_name(layer: usize, is_key: bool) -> &'static str {
    let idx = layer * 4 + if is_key { 2 } else { 3 };
    // SAFETY: kv_cache_names() returns &'static [String], element .as_str() is 'static.
    unsafe { &*(kv_cache_names()[idx].as_str() as *const str) }
}

fn memory_info(
    device: AllocationDevice,
    memory_type: MemoryType,
    context: &str,
) -> Result<MemoryInfo> {
    MemoryInfo::new(device, 0, AllocatorType::Device, memory_type)
        .ctx(&format!("{context} memory info"))
}

fn bind_decoder_outputs_to_device(
    binding: &mut ort::session::IoBinding,
    use_gpu: bool,
) -> Result<()> {
    let logits_mem = memory_info(AllocationDevice::CPU, MemoryType::Default, "logits")?;
    binding
        .bind_output_to_device("logits", &logits_mem)
        .ctx("bind logits output")?;

    #[cfg(feature = "cuda")]
    let kv_mem = if use_gpu {
        memory_info(AllocationDevice::CUDA, MemoryType::Default, "kv")?
    } else {
        memory_info(AllocationDevice::CPU, MemoryType::Default, "kv")?
    };
    #[cfg(not(feature = "cuda"))]
    let kv_mem = {
        let _ = use_gpu;
        memory_info(AllocationDevice::CPU, MemoryType::Default, "kv")?
    };

    for i in 0..NUM_LAYERS {
        binding
            .bind_output_to_device(present_kv_name(i, true), &kv_mem)
            .ctx("bind present key layer {i}")?;
        binding
            .bind_output_to_device(present_kv_name(i, false), &kv_mem)
            .ctx("bind present value layer {i}")?;
    }
    Ok(())
}

enum DecodeKvCache {
    Bound(Vec<DynValue>),
    Cpu(Vec<ArrayD<f16>>),
}

impl Qwen3AsrBackend {
    fn transcribe_pcm(
        &self,
        pcm: &[f32],
        timestamp_ms: u64,
        id: u64,
        language: Option<&str>,
    ) -> Result<Vec<AsrSegment>> {
        let start = std::time::Instant::now();
        let (boost_seqs, boost_score) = self
            .boost_seqs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        // Boost for the first decoded token: no history yet, entry tokens only.
        let mut boost: HashMap<i64, f32> = HashMap::new();
        phrase_boost_tokens(&[], &boost_seqs, boost_score, &mut boost);

        // 0. Loudness normalization — quiet dictation otherwise decodes
        //    measurably worse (absolute mel scale; see `normalize_loudness`).
        let pcm = normalize_loudness(pcm);

        // 1. Mel spectrogram [1, 128, T] — FP16 native.
        //    Pad mel output to 3000 frames (30s) matching the reference
        //    WhisperFeatureExtractor. The encoder's chunked attention windows
        //    were trained on this geometry. Pads at mel level to avoid
        //    computing FFT on silence.
        const MEL_MIN_FRAMES: usize = 3000;
        let mel_t0 = std::time::Instant::now();
        let spec = self.mel.extract_from_f32_padded(&pcm, MEL_MIN_FRAMES);
        let mel_3d = ndarray::Array3::from_shape_vec((1, spec.n_mels, spec.n_frames), spec.data)
            .ctx("mel reshape")?;
        let mel_val = Value::from_array(mel_3d).ctx("mel value")?;
        let mel_ms = mel_t0.elapsed().as_secs_f64() * 1000.0;

        let mut models = self.models.lock().map_err(|e| lock_err("models", e))?;

        // 2. Encoder → audio features [1, N, D] (FP16 native).
        // Shrink the CUDA arena at the start of each dictation so chunks left
        // over from the previous utterance's variable-shape decode loop are
        // released before this utterance grows the arena again. Without this,
        // each dictation's high-water mark accumulates across the session.
        let encoder_t0 = std::time::Instant::now();
        let audio_feat = {
            let run_opts = make_shrink_run_options(self.use_gpu)?;
            let out = models
                .encoder
                .run_with_options(ort::inputs!("mel" => mel_val), &run_opts)
                .ctx("encoder")?;
            extract_f16_array(&out[0])?
        };
        let encoder_ms = encoder_t0.elapsed().as_secs_f64() * 1000.0;
        let n_audio = audio_feat.shape()[1];
        let hidden_size = audio_feat.shape()[2];

        // 3. Build prompt token IDs (dynamic system prompt + context)
        let prompt_t0 = std::time::Instant::now();
        let (prefix_ids, suffix_ids) = self.build_prompt_ids(language);
        let mut input_ids: Vec<i64> =
            Vec::with_capacity(prefix_ids.len() + n_audio + suffix_ids.len());
        input_ids.extend_from_slice(&prefix_ids);
        let audio_offset = input_ids.len();
        input_ids.extend(std::iter::repeat_n(AUDIO_PAD_ID, n_audio));
        input_ids.extend_from_slice(&suffix_ids);
        let seq_len = input_ids.len();

        // 4. Embed tokens → [1, seq_len, D] (FP16), scatter encoder output
        let embeds = {
            let ids_val =
                Value::from_array(Array2::from_shape_vec((1, seq_len), input_ids).ctx("ids")?)
                    .ctx("ids val")?;
            let out = models
                .embed_tokens
                .run(ort::inputs!("input_ids" => ids_val))
                .ctx("embed_tokens")?;
            let emb = extract_f16_array(&out[0])?;
            let dims = emb.shape().to_vec();
            let mut arr = emb
                .into_shape_with_order(IxDyn(&[dims[0], dims[1], dims[2]]))
                .ctx("embeds reshape")?;
            // Scatter: replace audio_pad positions with encoder output (both f16).
            // Use contiguous slice copies instead of per-element indexed writes.
            let emb_slice = arr
                .as_slice_mut()
                .ok_or_else(|| Error::asr("embeds not contiguous"))?;
            let feat_slice = audio_feat
                .as_slice()
                .ok_or_else(|| Error::asr("audio_feat not contiguous"))?;
            for t in 0..n_audio {
                let dst = (audio_offset + t) * hidden_size;
                let src = t * hidden_size;
                emb_slice[dst..dst + hidden_size]
                    .copy_from_slice(&feat_slice[src..src + hidden_size]);
            }
            arr
        };
        let prompt_ms = prompt_t0.elapsed().as_secs_f64() * 1000.0;

        // 5. Prefill: run decoder with full embeddings + empty KV cache
        let empty_banned: HashSet<i64> = HashSet::new();
        let prefill_t0 = std::time::Instant::now();
        let (first_token, first_conf, first_logits, mut kv_cache) = {
            let embeds_val = Value::from_array(embeds).ctx("embeds val")?;
            let pos_val = Value::from_array(mrope_position_ids(seq_len)).ctx("pos val")?;
            let mask_val = Value::from_array(Array2::<i64>::ones((1, seq_len))).ctx("mask val")?;

            if self.decoder_io_binding {
                let mut binding = models
                    .decoder
                    .create_binding()
                    .ctx("decoder prefill binding")?;
                binding
                    .bind_input("inputs_embeds", &embeds_val)
                    .ctx("bind prefill embeds")?;
                binding
                    .bind_input("position_ids", &pos_val)
                    .ctx("bind prefill position_ids")?;
                binding
                    .bind_input("attention_mask", &mask_val)
                    .ctx("bind prefill attention_mask")?;
                bind_empty_kv(&mut binding)?;
                bind_decoder_outputs_to_device(&mut binding, self.use_gpu)?;

                let mut out = models
                    .decoder
                    .run_binding(&binding)
                    .ctx("decoder prefill")?;
                let (tok, conf, logits_k) =
                    argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost, &empty_banned)?;

                let mut cache: Vec<DynValue> = Vec::with_capacity(NUM_LAYERS * 2);
                for i in 0..NUM_LAYERS {
                    cache.push(
                        out.remove(present_kv_name(i, true))
                            .ok_or_else(|| Error::asr(format!("missing present key layer {i}")))?,
                    );
                    cache.push(
                        out.remove(present_kv_name(i, false)).ok_or_else(|| {
                            Error::asr(format!("missing present value layer {i}"))
                        })?,
                    );
                }
                (tok, conf, logits_k, DecodeKvCache::Bound(cache))
            } else {
                let mut dec_inputs = ort::inputs!(
                    "inputs_embeds" => embeds_val,
                    "position_ids" => pos_val,
                    "attention_mask" => mask_val
                );
                append_empty_kv(&mut dec_inputs)?;

                let mut out = models.decoder.run(dec_inputs).ctx("decoder prefill")?;
                let (tok, conf, logits_k) =
                    argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost, &empty_banned)?;

                let mut cache: Vec<ArrayD<f16>> = Vec::with_capacity(NUM_LAYERS * 2);
                for i in 0..NUM_LAYERS {
                    let key = out
                        .remove(present_kv_name(i, true))
                        .ok_or_else(|| Error::asr(format!("missing present key layer {i}")))?;
                    let value = out
                        .remove(present_kv_name(i, false))
                        .ok_or_else(|| Error::asr(format!("missing present value layer {i}")))?;
                    cache.push(extract_f16_array(&key)?);
                    cache.push(extract_f16_array(&value)?);
                }
                (tok, conf, logits_k, DecodeKvCache::Cpu(cache))
            }
        };
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;

        let mut tokens: Vec<(i64, f32, Vec<f64>)> = vec![(first_token, first_conf, first_logits)];
        let mut pos = seq_len;
        let mut decode_steps = 0usize;
        let mut gen_ids: Vec<i64> = tokens.iter().map(|(id, _, _)| *id).collect();
        let mut banned: HashSet<i64> = HashSet::new();

        // Pre-allocate reusable position array for the decode loop.
        let mut pos_arr = Array3::<i64>::zeros((3, 1, 1));
        // Preallocated all-ones attention mask. The decode mask is always ones,
        // so each step binds a length-`total_len` prefix *view* of this row —
        // zero per-step allocation. Width covers the worst case pos + 1.
        let mask_ones = Array2::<i64>::ones((1, seq_len + MAX_NEW_TOKENS));
        let mut decoder_binding = if self.decoder_io_binding {
            Some(models.decoder.create_binding().ctx("decoder binding")?)
        } else {
            None
        };

        // 6. Autoregressive greedy decode loop
        let decode_t0 = std::time::Instant::now();
        loop {
            let last_tok = tokens.last().unwrap().0;
            if EOS_IDS.contains(&last_tok) || tokens.len() >= MAX_NEW_TOKENS {
                break;
            }
            // Repetition guard: if the tail has collapsed into a repeated
            // block, drop the redundant copies and stop — emitting the loop
            // verbatim is worse than truncating it.
            if let Some(n) = repeated_cycle_len(&gen_ids) {
                let keep = gen_ids.len() - 3 * n;
                tokens.truncate(keep);
                tracing::warn!(
                    cycle = n,
                    kept = keep,
                    "repetition loop detected, trimming tail"
                );
                break;
            }
            // Repeat-guard mask for the upcoming token (empty in the common
            // case, so the argmax fast path is preserved).
            banned_next_tokens(&gen_ids, REPEAT_GUARD_ORDER, &mut banned);
            // Refresh the prefix-constrained phrase bias for this step.
            if !boost_seqs.is_empty() {
                phrase_boost_tokens(&gen_ids, &boost_seqs, boost_score, &mut boost);
            }

            let (tok, conf, logits_k) = {
                // Embed single token (FP16 native, scoped to drop before decoder)
                let emb = {
                    let id_val =
                        Value::from_array(Array2::from_elem((1, 1), last_tok)).ctx("id val")?;
                    let emb_out = models
                        .embed_tokens
                        .run(ort::inputs!("input_ids" => id_val))
                        .ctx("embed step")?;
                    extract_f16_array(&emb_out[0])?
                };

                // Reuse position array — just update the value.
                pos_arr.fill(pos as i64);

                let total_len = pos + 1;
                let emb_val = Value::from_array(emb).ctx("emb val")?;
                let pos_val = Value::from_array(pos_arr.clone()).ctx("pos val")?;
                let mask_ref =
                    TensorRef::from_array_view(mask_ones.slice(ndarray::s![.., ..total_len]))
                        .ctx("mask view")?;

                match &mut kv_cache {
                    DecodeKvCache::Bound(cache) => {
                        let binding = decoder_binding
                            .as_mut()
                            .ok_or_else(|| Error::asr("missing decoder binding"))?;
                        binding.clear();
                        binding
                            .bind_input("inputs_embeds", &emb_val)
                            .ctx("bind embeds")?;
                        binding
                            .bind_input("position_ids", &pos_val)
                            .ctx("bind position_ids")?;
                        binding
                            .bind_input("attention_mask", &mask_ref)
                            .ctx("bind attention_mask")?;
                        // Move ORT-owned KV values directly into the next decoder run.
                        // Present KV outputs are bound to device memory below.
                        for (i, v) in cache.drain(..).enumerate() {
                            let layer = i / 2;
                            let is_key = i % 2 == 0;
                            binding
                                .bind_input(kv_name(layer, is_key), &v)
                                .ctx("bind kv layer {layer}")?;
                        }
                        bind_decoder_outputs_to_device(binding, self.use_gpu)?;

                        let mut out = models.decoder.run_binding(binding).ctx("decoder step")?;

                        let result = argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost, &banned)?;

                        for i in 0..NUM_LAYERS {
                            cache.push(out.remove(present_kv_name(i, true)).ok_or_else(|| {
                                Error::asr(format!("missing present key layer {i}"))
                            })?);
                            cache.push(out.remove(present_kv_name(i, false)).ok_or_else(|| {
                                Error::asr(format!("missing present value layer {i}"))
                            })?);
                        }
                        result
                    }
                    DecodeKvCache::Cpu(cache) => {
                        let mut dec_inputs = ort::inputs!(
                            "inputs_embeds" => emb_val,
                            "position_ids" => pos_val,
                            "attention_mask" => mask_ref
                        );
                        append_cached_kv(&mut dec_inputs, std::mem::take(cache))?;

                        let mut out = models.decoder.run(dec_inputs).ctx("decoder step")?;

                        let result = argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost, &banned)?;

                        for i in 0..NUM_LAYERS {
                            let key = out.remove(present_kv_name(i, true)).ok_or_else(|| {
                                Error::asr(format!("missing present key layer {i}"))
                            })?;
                            let value = out.remove(present_kv_name(i, false)).ok_or_else(|| {
                                Error::asr(format!("missing present value layer {i}"))
                            })?;
                            cache.push(extract_f16_array(&key)?);
                            cache.push(extract_f16_array(&value)?);
                        }
                        result
                    }
                }
            };
            tokens.push((tok, conf, logits_k));
            gen_ids.push(tok);
            decode_steps += 1;
            pos += 1;
        }
        let decode_ms = decode_t0.elapsed().as_secs_f64() * 1000.0;

        // Arena janitor: ORT shrink run-options only act on the arenas of the
        // session that runs, and every session here owns a private CUDA arena
        // (no CUDA allocator is registered at env level). The decoder's arena
        // is where the decode ladder lives — KV growth plus the all-position
        // prefill-logits scratch, ~1 GB high-water. After dropping the KV and
        // binding, every chunk in it is fully free, so one trivial 1-token
        // decoder run carrying shrink options releases the lot back to the
        // driver. Costs one decode step (~10 ms) per utterance; idle VRAM
        // between dictations drops by the full ladder.
        drop(decoder_binding);
        drop(kv_cache);
        if self.use_gpu {
            let mut janitor = || -> Result<()> {
                let shrink = make_shrink_run_options(true)?;
                let id_val =
                    Value::from_array(Array2::from_elem((1, 1), EOS_IDS[0])).ctx("janitor id")?;
                let emb_out = models
                    .embed_tokens
                    .run(ort::inputs!("input_ids" => id_val))
                    .ctx("janitor embed")?;
                let emb = extract_f16_array(&emb_out[0])?;
                drop(emb_out);
                let emb_val = Value::from_array(emb).ctx("janitor emb val")?;
                let pos_val =
                    Value::from_array(Array3::<i64>::zeros((3, 1, 1))).ctx("janitor pos")?;
                let mask_val =
                    Value::from_array(Array2::<i64>::ones((1, 1))).ctx("janitor mask")?;
                let mut inputs = ort::inputs!(
                    "inputs_embeds" => emb_val,
                    "position_ids" => pos_val,
                    "attention_mask" => mask_val
                );
                append_empty_kv(&mut inputs)?;
                models
                    .decoder
                    .run_with_options(inputs, &shrink)
                    .ctx("janitor run")?;
                Ok(())
            };
            if let Err(err) = janitor() {
                tracing::debug!(error = %err, "post-utterance arena janitor skipped");
            }
        }

        drop(models);

        let elapsed = start.elapsed();
        let profile = Qwen3AsrProfile {
            audio_ms: pcm.len() * 1000 / whisper::SAMPLE_RATE,
            total_ms: elapsed.as_secs_f64() * 1000.0,
            mel_ms,
            encoder_ms,
            prompt_ms,
            prefill_ms,
            decode_ms,
            decode_steps,
            decoder_io_binding: self.decoder_io_binding,
        };
        if let Ok(mut last_profile) = self.last_profile.lock() {
            *last_profile = Some(profile);
        }
        tracing::debug!(
            elapsed_ms = profile.total_ms,
            audio_ms = profile.audio_ms,
            mel_ms = profile.mel_ms,
            encoder_ms = profile.encoder_ms,
            prompt_ms = profile.prompt_ms,
            prefill_ms = profile.prefill_ms,
            decode_ms = profile.decode_ms,
            decode_steps = profile.decode_steps,
            decoder_io_binding = profile.decoder_io_binding,
            tokens = tokens.len(),
            "Qwen3 inference"
        );
        // 7. Decode tokens → text, parse output, build segment
        let (text, detected_language, mean_conf, min_conf, word_confidences) =
            self.decode_output(&tokens);
        if text.is_empty() {
            return Ok(vec![]);
        }

        Ok(vec![AsrSegment {
            id,
            text,
            is_final: true,
            start_ms: None,
            end_ms: Some(timestamp_ms),
            confidence: mean_conf,
            min_token_confidence: min_conf,
            word_confidences,
            detected_language,
        }])
    }

    fn decode_output(
        &self,
        tokens: &[(i64, f32, Vec<f64>)],
    ) -> (
        String,
        Option<String>,
        Option<f32>,
        Option<f32>,
        Vec<lune_core::asr::WordConfidence>,
    ) {
        use lune_core::hwce::TokenSpan;

        let content_start = tokens
            .iter()
            .position(|(id, _, _)| *id == ASR_TEXT_ID)
            .map(|i| i + 1)
            .unwrap_or(0);

        let is_meta = |id: i64| -> bool {
            EOS_IDS.contains(&id)
                || id == IM_START
                || id == IM_END
                || id == NEWLINE
                || id == ASR_TEXT_ID
                || id == AUDIO_START
                || id == AUDIO_END
        };

        let raw_token_ids: Vec<u32> = tokens
            .iter()
            .map(|(id, _, _)| *id)
            .filter(|&id| {
                !EOS_IDS.contains(&id)
                    && id != IM_START
                    && id != IM_END
                    && id != NEWLINE
                    && id != AUDIO_START
                    && id != AUDIO_END
            })
            .map(|id| id as u32)
            .collect();
        let raw_text = self
            .tokenizer
            .decode(&raw_token_ids, true)
            .unwrap_or_default();
        let (detected_language, parsed_text) = parse_auto_asr_output(&raw_text);

        let content: Vec<&(i64, f32, Vec<f64>)> = tokens[content_start..]
            .iter()
            .skip_while(|(id, _, _)| is_meta(*id))
            .filter(|(id, _, _)| !is_meta(*id))
            .collect();

        if content.is_empty() {
            return (parsed_text, detected_language, None, None, Vec::new());
        }

        let token_ids: Vec<u32> = content.iter().map(|(id, _, _)| *id as u32).collect();
        let mut text = self
            .tokenizer
            .decode(&token_ids, true)
            .unwrap_or_default()
            .trim()
            .to_string();
        if !parsed_text.is_empty() {
            text = parsed_text;
        }
        for prefix in ["You are an automatic speech recognition system"] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest.trim_start_matches(['.', '\n', ' ']).to_string();
            }
        }

        let confs: Vec<f32> = content.iter().map(|(_, c, _)| *c).collect();
        let mean = Some(confs.iter().sum::<f32>() / confs.len() as f32);
        let min = confs.iter().copied().reduce(f32::min);

        // Build words + token spans for MABLE.
        let mut words: Vec<String> = Vec::new();
        let mut spans: Vec<TokenSpan> = Vec::new();
        let mut current_word = String::new();
        let mut span_start = 0usize;

        for (tok_idx, &(id, _, _)) in content.iter().enumerate() {
            let piece = self
                .tokenizer
                .decode(&[*id as u32], false)
                .unwrap_or_default();
            if piece.is_empty() {
                continue;
            }
            let starts_new_word = piece.starts_with(' ') || piece.starts_with('\u{2581}');
            if starts_new_word && !current_word.is_empty() {
                let w: String = current_word
                    .trim()
                    .chars()
                    .filter(|c| !c.is_ascii_punctuation())
                    .collect();
                if !w.is_empty() {
                    words.push(w);
                    spans.push(TokenSpan::new(span_start, tok_idx));
                }
                current_word.clear();
                span_start = tok_idx;
            }
            let trimmed = piece.trim_start();
            current_word.push_str(trimmed);
        }
        if !current_word.is_empty() {
            let w: String = current_word
                .trim()
                .chars()
                .filter(|c| !c.is_ascii_punctuation())
                .collect();
            if !w.is_empty() {
                words.push(w);
                spans.push(TokenSpan::new(span_start, content.len()));
            }
        }

        let hwce_tokens: Vec<lune_core::hwce::TokenInput> = content
            .iter()
            .map(|&&(id, conf, ref lk)| lune_core::hwce::TokenInput {
                token_id: id as u32,
                confidence: conf,
                top_k_logits: lk.clone(),
            })
            .collect();

        let hwce_results = lune_core::hwce::estimate(
            &words,
            &hwce_tokens,
            &spans,
            lune_core::hwce::Config::default(),
        );

        let word_confidences: Vec<lune_core::asr::WordConfidence> = hwce_results
            .into_iter()
            .map(|wr| lune_core::asr::WordConfidence {
                word: wr.word,
                confidence: 1.0 - wr.word_error_risk,
                word_error_risk: wr.word_error_risk,
            })
            .collect();

        (text, detected_language, mean, min, word_confidences)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn parse_auto_asr_output(raw: &str) -> (Option<String>, String) {
    let raw = raw.trim();

    // Preferred path: split on the `<asr_text>` marker. The prefix may
    // optionally start with `language ` ("language Chinese<asr_text>...")
    // or be a bare language name ("Chinese<asr_text>...") — both occur in
    // the wild depending on whether the model emitted the leading word.
    if let Some(asr_pos) = raw.find("<asr_text>") {
        let prefix = raw[..asr_pos].trim();
        let lang = prefix.strip_prefix("language ").unwrap_or(prefix).trim();
        let text = raw[asr_pos + "<asr_text>".len()..].trim();
        return (non_empty_string(lang), text.to_string());
    }

    // Legacy fallback: "language X some text" with no marker.
    let Some(rest) = raw.strip_prefix("language ") else {
        return (None, raw.to_string());
    };
    let mut lang_end = 0;
    for (i, c) in rest.char_indices() {
        if c.is_whitespace() || !c.is_alphabetic() {
            lang_end = i;
            break;
        }
        lang_end = i + c.len_utf8();
    }
    if lang_end == 0 {
        return (None, raw.to_string());
    }
    let lang = &rest[..lang_end];
    let text = rest[lang_end..].trim();
    (non_empty_string(lang), text.to_string())
}

/// Map a BCP-47 tag (or English name) to the language label Qwen3-ASR
/// expects in its `language X` prompt. Returns `None` for unrecognised
/// tags so the model falls back to auto-detection rather than receiving
/// a malformed hint.
fn bcp47_to_qwen3_language(tag: &str) -> Option<String> {
    let t = tag.trim().to_ascii_lowercase();
    if t.is_empty() || t == "auto" {
        return None;
    }
    let primary = t.split(['-', '_']).next().unwrap_or(&t);
    let name = match primary {
        "en" => "English",
        "zh" | "cmn" | "yue" => "Chinese",
        "ja" => "Japanese",
        "ko" => "Korean",
        "es" => "Spanish",
        "fr" => "French",
        "de" => "German",
        "it" => "Italian",
        "pt" => "Portuguese",
        "ru" => "Russian",
        "ar" => "Arabic",
        "hi" => "Hindi",
        "vi" => "Vietnamese",
        "th" => "Thai",
        "id" => "Indonesian",
        "nl" => "Dutch",
        "pl" => "Polish",
        "tr" => "Turkish",
        // Unknown tag. Only pass through values that look like a spelled-out
        // language name (a single all-alphabetic word) — anything carrying a
        // region/script subtag, digits, or separators is a malformed hint, so
        // fall back to auto-detection (`None`) per the documented contract.
        _ => {
            let trimmed = tag.trim();
            return if !trimmed.is_empty() && trimmed.chars().all(|c| c.is_alphabetic()) {
                Some(title_case(trimmed))
            } else {
                None
            };
        }
    };
    Some(name.to_string())
}

fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_word_start = true;
    for c in s.chars() {
        if c.is_whitespace() {
            at_word_start = true;
            out.push(c);
        } else if at_word_start {
            out.extend(c.to_uppercase());
            at_word_start = false;
        } else {
            out.extend(c.to_lowercase());
        }
    }
    out
}

fn non_empty_string(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn lock_err<T>(what: &str, e: std::sync::PoisonError<T>) -> Error {
    Error::asr(format!("{what} lock poisoned: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auto_output_with_asr_text_marker() {
        let (lang, text) = parse_auto_asr_output("language English<asr_text>Hello world.");
        assert_eq!(lang.as_deref(), Some("English"));
        assert_eq!(text, "Hello world.");
    }

    #[test]
    fn parse_auto_output_without_marker() {
        let (lang, text) = parse_auto_asr_output("language French Bonjour tout le monde.");
        assert_eq!(lang.as_deref(), Some("French"));
        assert_eq!(text, "Bonjour tout le monde.");
    }

    #[test]
    fn parse_auto_output_bare_language_prefix() {
        // Model sometimes omits the "language " literal and emits just
        // the bare language name before <asr_text>.
        let (lang, text) = parse_auto_asr_output("Chinese<asr_text>穿衣服。");
        assert_eq!(lang.as_deref(), Some("Chinese"));
        assert_eq!(text, "穿衣服。");
    }

    #[test]
    fn parse_auto_output_marker_only_no_lang() {
        let (lang, text) = parse_auto_asr_output("<asr_text>Hello.");
        assert_eq!(lang, None);
        assert_eq!(text, "Hello.");
    }

    #[test]
    fn bcp47_maps_common_tags() {
        use super::bcp47_to_qwen3_language as m;
        assert_eq!(m("en"), Some("English".into()));
        assert_eq!(m("en-US"), Some("English".into()));
        assert_eq!(m("zh-CN"), Some("Chinese".into()));
        assert_eq!(m("ja"), Some("Japanese".into()));
        assert_eq!(m("auto"), None);
        assert_eq!(m(""), None);
        // Unknown but name-like tags pass through title-cased; malformed tags
        // (region/script subtags, digits, separators) fall back to auto (None)
        // rather than emitting a bogus `language X` hint.
        assert_eq!(m("English"), Some("English".into()));
        assert_eq!(m("ENGLISH"), Some("English".into()));
        assert_eq!(m("Klingon"), Some("Klingon".into()));
        assert_eq!(m("zz-ZZ"), None);
        assert_eq!(m("xx9"), None);
    }

    #[test]
    fn loudness_norm_boosts_quiet_speech_to_target() {
        // Quiet sine at RMS ≈ 0.008 → boosted toward 0.08 (10× cap applies).
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| 0.0113 * (i as f32 * 0.2).sin())
            .collect();
        let out = normalize_loudness(&pcm);
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.05, "quiet speech should be boosted, rms={rms}");
        assert!(out.iter().all(|x| x.abs() <= 1.0));
    }

    #[test]
    fn loudness_norm_leaves_normal_level_untouched() {
        let pcm: Vec<f32> = (0..16_000).map(|i| 0.14 * (i as f32 * 0.2).sin()).collect();
        let out = normalize_loudness(&pcm);
        assert!(
            matches!(out, std::borrow::Cow::Borrowed(_)),
            "healthy-level audio must pass through without copy"
        );
    }

    #[test]
    fn loudness_norm_ignores_silence_and_respects_peak() {
        let silent = vec![0.0f32; 1600];
        assert!(matches!(
            normalize_loudness(&silent),
            std::borrow::Cow::Borrowed(_)
        ));
        // Quiet RMS but a single hot transient: gain limited by peak.
        let mut pcm = vec![0.005f32; 16_000];
        pcm[3] = 0.9;
        let out = normalize_loudness(&pcm);
        assert!(out.iter().all(|x| x.abs() <= 1.0));
        assert!(out[3].abs() <= 0.99, "peak limited, got {}", out[3]);
    }

    #[test]
    fn phrase_boost_entry_tokens_always_active() {
        let seqs = vec![vec![10, 11, 12], vec![20, 21]];
        let mut out = HashMap::new();
        phrase_boost_tokens(&[], &seqs, 1.5, &mut out);
        assert_eq!(out.get(&10), Some(&1.5));
        assert_eq!(out.get(&20), Some(&1.5));
        assert!(
            !out.contains_key(&11),
            "mid-phrase token must not be boosted without prefix"
        );
        assert!(!out.contains_key(&12));
    }

    #[test]
    fn phrase_boost_continuation_requires_matching_tail() {
        let seqs = vec![vec![10, 11, 12]];
        let mut out = HashMap::new();
        // Tail matches first two tokens → third gets the doubled boost.
        phrase_boost_tokens(&[5, 10, 11], &seqs, 1.0, &mut out);
        assert_eq!(out.get(&12), Some(&2.0));
        // Unrelated tail → only the entry token.
        phrase_boost_tokens(&[5, 6, 7], &seqs, 1.0, &mut out);
        assert_eq!(out.get(&10), Some(&1.0));
        assert!(!out.contains_key(&12));
    }

    #[test]
    fn phrase_boost_longest_prefix_wins() {
        // Tail [10, 11] matches prefix len 2 → boost seq[2], not seq[1].
        let seqs = vec![vec![10, 11, 12, 13]];
        let mut out = HashMap::new();
        phrase_boost_tokens(&[10, 11], &seqs, 1.0, &mut out);
        assert_eq!(out.get(&12), Some(&2.0));
        assert!(!out.contains_key(&13));
    }

    #[test]
    fn repeat_guard_bans_seen_completions() {
        // "a b c a b" → last bigram [a,b] previously continued with c → ban c.
        let mut out = HashSet::new();
        banned_next_tokens(&[1, 2, 3, 1, 2], 3, &mut out);
        assert_eq!(out.len(), 1);
        assert!(out.contains(&3));
    }

    #[test]
    fn repeat_guard_empty_without_repeat() {
        let mut out = HashSet::new();
        banned_next_tokens(&[1, 2, 3, 4], 3, &mut out);
        assert!(out.is_empty(), "no repeated bigram → nothing banned");
        banned_next_tokens(&[1, 2], 3, &mut out);
        assert!(out.is_empty(), "too short → nothing banned");
    }

    #[test]
    fn repeated_cycle_len_detects_and_sizes_loop() {
        // 12 unique tokens then "7 8" repeated 4× = 20 tokens; cycle len 2.
        let mut ids: Vec<i64> = (0..12).collect();
        for _ in 0..4 {
            ids.push(7);
            ids.push(8);
        }
        assert_eq!(repeated_cycle_len(&ids), Some(2));
        // Trimming keeps one clean cycle: len - 3*n.
        assert_eq!(ids.len() - 3 * 2, 14);
    }

    #[test]
    fn repeated_cycle_len_ignores_normal_text() {
        let ids: Vec<i64> = (0..40).map(|i| i % 9).collect();
        assert_eq!(repeated_cycle_len(&ids), None);
    }
}
