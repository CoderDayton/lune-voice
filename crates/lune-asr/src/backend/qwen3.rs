//! Qwen3-ASR 0.6B ONNX backend — encoder-decoder with autoregressive greedy decoding.
//!
//! Model: custom ONNX export via `scripts/export_onnx.py` (encoder + embed_tokens + decoder).
//! Mel: Whisper-compatible 128-bin log-mel at 16 kHz (see `whisper_mel.rs`).
//! Decoding: greedy argmax with per-token softmax confidence.

use std::path::PathBuf;
use std::sync::Mutex;

use half::f16;
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{DynValue, Value};
use tokenizers::Tokenizer;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::error::{Error, Result};

use crate::mel::whisper;
use crate::ort_utils::OrtModelLoader;

// ── Model constants ─────────────────────────────────────────────────────────

const VOCAB_SIZE: usize = 151_936;
const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const MAX_NEW_TOKENS: usize = 448;

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

const SYSTEM_PROMPT: &str = "";

/// Maximum audio buffer (5 min at 16 kHz).
const MAX_WINDOW_SAMPLES: usize = 300 * 16_000;

// ── Types ───────────────────────────────────────────────────────────────────

pub struct Qwen3AsrConfig {
    pub model_dir: PathBuf,
    pub use_gpu: bool,
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
    boost_map: Mutex<std::collections::HashMap<i64, f32>>,
    /// Per-session context phrases injected into the system prompt.
    /// e.g. ["Toledans", "Rodolfo"] → "Preferred terms: Toledans, Rodolfo"
    context_phrases: Mutex<Vec<String>>,
}

pub struct Qwen3Session {
    pending: Vec<f32>,
    total_samples: u64,
    sample_rate: u32,
    next_id: u64,
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
        let loader = OrtModelLoader::new(config.model_dir.clone(), config.use_gpu);

        let encoder = loader.load_first(&["qwen3-asr-encoder.onnx"], "qwen3-asr-encoder")?;
        let embed_tokens =
            loader.load_first(&["qwen3-asr-embed-tokens.onnx"], "qwen3-asr-embed-tokens")?;
        let decoder = loader.load_first(&["qwen3-asr-decoder.onnx"], "qwen3-asr-decoder")?;

        let tok_path = config.model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| Error::Asr(format!("load tokenizer: {e}")))?;

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
            boost_map: Mutex::new(std::collections::HashMap::new()),
            context_phrases: Mutex::new(Vec::new()),
        })
    }

    pub fn last_profile(&self) -> Option<Qwen3AsrProfile> {
        self.last_profile.lock().ok().and_then(|p| *p)
    }

    /// Set word boost phrases — dual-layer: system prompt context + logit bias.
    /// Phrases are injected into the system prompt as preferred terms AND
    /// tokenized for additive logit bias on decode.
    pub fn set_boost_phrases(&self, phrases: &[String], score: f32) {
        // Layer 1: context phrases for system prompt injection
        *self
            .context_phrases
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = phrases.to_vec();

        // Layer 2: token-level logit bias
        let mut map = self.boost_map.lock().unwrap_or_else(|e| e.into_inner());
        map.clear();
        for phrase in phrases {
            if let Ok(encoding) = self.tokenizer.encode(phrase.as_str(), false) {
                for &id in encoding.get_ids() {
                    map.entry(id as i64).or_insert(score);
                }
            }
        }
        if !map.is_empty() {
            tracing::debug!(
                phrases = phrases.len(),
                tokens = map.len(),
                "boost map updated"
            );
        }
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
        // Language is controlled via suffix, not the system prompt.
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
        let session = Qwen3Session {
            pending: Vec::with_capacity(MAX_WINDOW_SAMPLES),
            total_samples: 0,
            sample_rate: config.sample_rate,
            next_id: 1,
        };
        tracing::info!(
            sample_rate = config.sample_rate,
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

    async fn poll_results(&self, _session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        Ok(vec![]) // Batch backend — no incremental results.
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let (pcm, ts_ms, id) = {
            let mut sess = session.lock().map_err(|e| lock_err("session", e))?;
            if sess.pending.is_empty() {
                return Ok(vec![]);
            }
            let ts = sess.total_samples * 1000 / sess.sample_rate as u64;
            let id = sess.next_id;
            sess.next_id += 1;
            let pcm = std::mem::take(&mut sess.pending);
            (pcm, ts, id)
        };

        if pcm.is_empty() {
            return Ok(vec![]);
        }

        self.transcribe_pcm(&pcm, ts_ms, id, None)
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
    let (shape, data) = value
        .try_extract_tensor::<f16>()
        .map_err(|e| Error::Asr(format!("extract f16: {e}")))?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())
        .map_err(|e| Error::Asr(format!("array reshape: {e}")))
}

const MABLE_TOP_K: usize = 16;

/// Fused argmax + softmax confidence + top-k logit extraction on f16 logits tensor.
/// Returns (token_id, softmax_confidence, top_k_logits_as_f64).
fn argmax_on_logits(
    value: &ort::value::Value,
    vocab_size: usize,
    boost: &std::collections::HashMap<i64, f32>,
) -> Result<(i64, f32, Vec<f64>)> {
    let (shape, data) = value
        .try_extract_tensor::<f16>()
        .map_err(|e| Error::Asr(format!("extract logits: {e}")))?;
    // Last position's logits: last `vocab_size` elements.
    let total = shape.iter().map(|&d| d as usize).product::<usize>();
    let start = total.saturating_sub(vocab_size);
    let logits = &data[start..];

    if logits.is_empty() {
        return Err(Error::Asr("empty logits tensor".into()));
    }
    if logits.len() != vocab_size {
        return Err(Error::Asr(format!(
            "logits slice {} != vocab {vocab_size}",
            logits.len()
        )));
    }

    // Top-k min-heap: (value, index), sorted ascending so heap[0] is the min.
    let mut heap = [(f32::NEG_INFINITY, 0usize); MABLE_TOP_K];
    let mut heap_min = f32::NEG_INFINITY;
    let mut heap_min_pos = 0usize;

    if boost.is_empty() {
        // Fast path: single pass argmax + LSE + top-k on f16 → f32.
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            let f = v.to_f32();
            if f > max_val {
                max_val = f;
                max_idx = i;
            }
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
        let sum_exp: f32 = logits.iter().map(|&x| (x.to_f32() - max_val).exp()).sum();
        if !sum_exp.is_normal() {
            let top_k = heap.iter().map(|(v, _)| *v as f64).collect();
            return Ok((max_idx as i64, 0.0, top_k));
        }
        let lse = sum_exp.ln() + max_val;
        let top_k = heap.iter().map(|(v, _)| *v as f64).collect();
        Ok((max_idx as i64, (max_val - lse).exp(), top_k))
    } else {
        // Boosted path: track unboosted raw value inline for confidence.
        let mut max_idx = 0usize;
        let mut max_boosted = f32::NEG_INFINITY;
        let mut raw_max = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            let raw = v.to_f32();
            let boosted = raw + boost.get(&(i as i64)).copied().unwrap_or(0.0);
            if boosted > max_boosted {
                max_boosted = boosted;
                max_idx = i;
                raw_max = raw;
            }
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
        let sum_exp: f32 = logits.iter().map(|&x| (x.to_f32() - raw_max).exp()).sum();
        if !sum_exp.is_normal() {
            let top_k = heap.iter().map(|(v, _)| *v as f64).collect();
            return Ok((max_idx as i64, 0.0, top_k));
        }
        let lse = sum_exp.ln() + raw_max;
        let top_k = heap.iter().map(|(v, _)| *v as f64).collect();
        Ok((max_idx as i64, (raw_max - lse).exp(), top_k))
    }
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
            Value::from_array(k)
                .map_err(|e| Error::Asr(format!("empty k{i}: {e}")))?
                .into(),
        ));
        inputs.push((
            std::borrow::Cow::Borrowed(kv_name(i, false)),
            Value::from_array(v)
                .map_err(|e| Error::Asr(format!("empty v{i}: {e}")))?
                .into(),
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
                .map_err(|e| Error::Asr(format!("cached kv layer {layer}: {e}")))?
                .into(),
        ));
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
    unsafe { &*(kv_cache_names()[idx].as_str() as *const str) }
}

fn memory_info(
    device: AllocationDevice,
    memory_type: MemoryType,
    context: &str,
) -> Result<MemoryInfo> {
    MemoryInfo::new(device, 0, AllocatorType::Device, memory_type)
        .map_err(|e| Error::Asr(format!("{context} memory info: {e}")))
}

fn bind_decoder_outputs_to_device(
    binding: &mut ort::session::IoBinding,
    use_gpu: bool,
) -> Result<()> {
    let logits_mem = memory_info(AllocationDevice::CPU, MemoryType::Default, "logits")?;
    binding
        .bind_output_to_device("logits", &logits_mem)
        .map_err(|e| Error::Asr(format!("bind logits output: {e}")))?;

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
            .map_err(|e| Error::Asr(format!("bind present key layer {i}: {e}")))?;
        binding
            .bind_output_to_device(present_kv_name(i, false), &kv_mem)
            .map_err(|e| Error::Asr(format!("bind present value layer {i}: {e}")))?;
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
        let boost = self
            .boost_map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        // 1. Mel spectrogram [1, 128, T] — FP16 native.
        //    Pad mel output to 3000 frames (30s) matching the reference
        //    WhisperFeatureExtractor. The encoder's chunked attention windows
        //    were trained on this geometry. Pads at mel level to avoid
        //    computing FFT on silence.
        const MEL_MIN_FRAMES: usize = 3000;
        let mel_t0 = std::time::Instant::now();
        let spec = self.mel.extract_from_f32_padded(pcm, MEL_MIN_FRAMES);
        let mel_3d = ndarray::Array3::from_shape_vec((1, spec.n_mels, spec.n_frames), spec.data)
            .map_err(|e| Error::Asr(format!("mel reshape: {e}")))?;
        let mel_val =
            Value::from_array(mel_3d).map_err(|e| Error::Asr(format!("mel value: {e}")))?;
        let mel_ms = mel_t0.elapsed().as_secs_f64() * 1000.0;

        let mut models = self.models.lock().map_err(|e| lock_err("models", e))?;

        // 2. Encoder → audio features [1, N, D] (FP16 native)
        let encoder_t0 = std::time::Instant::now();
        let audio_feat = {
            let out = models
                .encoder
                .run(ort::inputs!("mel" => mel_val))
                .map_err(|e| Error::Asr(format!("encoder: {e}")))?;
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
            let ids_val = Value::from_array(
                Array2::from_shape_vec((1, seq_len), input_ids)
                    .map_err(|e| Error::Asr(format!("ids: {e}")))?,
            )
            .map_err(|e| Error::Asr(format!("ids val: {e}")))?;
            let out = models
                .embed_tokens
                .run(ort::inputs!("input_ids" => ids_val))
                .map_err(|e| Error::Asr(format!("embed_tokens: {e}")))?;
            let emb = extract_f16_array(&out[0])?;
            let dims = emb.shape().to_vec();
            let mut arr = emb
                .into_shape_with_order(IxDyn(&[dims[0], dims[1], dims[2]]))
                .map_err(|e| Error::Asr(format!("embeds reshape: {e}")))?;
            // Scatter: replace audio_pad positions with encoder output (both f16).
            // Use contiguous slice copies instead of per-element indexed writes.
            let emb_slice = arr
                .as_slice_mut()
                .ok_or_else(|| Error::Asr("embeds not contiguous".into()))?;
            let feat_slice = audio_feat
                .as_slice()
                .ok_or_else(|| Error::Asr("audio_feat not contiguous".into()))?;
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
        let prefill_t0 = std::time::Instant::now();
        let (first_token, first_conf, first_logits, mut kv_cache) =
            {
                let mut dec_inputs = ort::inputs!(
                    "inputs_embeds" => Value::from_array(embeds)
                        .map_err(|e| Error::Asr(format!("embeds val: {e}")))?,
                    "position_ids" => Value::from_array(mrope_position_ids(seq_len))
                        .map_err(|e| Error::Asr(format!("pos val: {e}")))?,
                    "attention_mask" => Value::from_array(Array2::<i64>::ones((1, seq_len)))
                        .map_err(|e| Error::Asr(format!("mask val: {e}")))?
                );
                append_empty_kv(&mut dec_inputs)?;

                let mut out = models
                    .decoder
                    .run(dec_inputs)
                    .map_err(|e| Error::Asr(format!("decoder prefill: {e}")))?;

                let (tok, conf, logits_k) = argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost)?;

                // Extract per-layer KV cache
                let cache =
                    if self.decoder_io_binding {
                        let mut cache: Vec<DynValue> = Vec::with_capacity(NUM_LAYERS * 2);
                        for i in 0..NUM_LAYERS {
                            cache.push(out.remove(present_kv_name(i, true)).ok_or_else(|| {
                                Error::Asr(format!("missing present key layer {i}"))
                            })?);
                            cache.push(out.remove(present_kv_name(i, false)).ok_or_else(|| {
                                Error::Asr(format!("missing present value layer {i}"))
                            })?);
                        }
                        DecodeKvCache::Bound(cache)
                    } else {
                        let mut cache: Vec<ArrayD<f16>> = Vec::with_capacity(NUM_LAYERS * 2);
                        for i in 0..NUM_LAYERS {
                            let key = out.remove(present_kv_name(i, true)).ok_or_else(|| {
                                Error::Asr(format!("missing present key layer {i}"))
                            })?;
                            let value = out.remove(present_kv_name(i, false)).ok_or_else(|| {
                                Error::Asr(format!("missing present value layer {i}"))
                            })?;
                            cache.push(extract_f16_array(&key)?);
                            cache.push(extract_f16_array(&value)?);
                        }
                        DecodeKvCache::Cpu(cache)
                    };
                (tok, conf, logits_k, cache)
            };
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1000.0;

        let mut tokens: Vec<(i64, f32, Vec<f64>)> = vec![(first_token, first_conf, first_logits)];
        let mut pos = seq_len;
        let mut decode_steps = 0usize;

        // Pre-allocate reusable position array for the decode loop.
        let mut pos_arr = Array3::<i64>::zeros((3, 1, 1));
        let mut decoder_binding = if self.decoder_io_binding {
            Some(
                models
                    .decoder
                    .create_binding()
                    .map_err(|e| Error::Asr(format!("decoder binding: {e}")))?,
            )
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
            // Repetition guard: check directly on tokens slice (no allocation).
            let len = tokens.len();
            if len >= 16 {
                let found_repeat = (1..=8).any(|n| {
                    let window = 4 * n;
                    if len < window {
                        return false;
                    }
                    let base = len - window;
                    (1..4).all(|r| (0..n).all(|k| tokens[base + k].0 == tokens[base + r * n + k].0))
                });
                if found_repeat {
                    tracing::warn!(tokens_so_far = len, "repetition loop detected, stopping");
                    break;
                }
            }

            let (tok, conf, logits_k) = {
                // Embed single token (FP16 native, scoped to drop before decoder)
                let emb = {
                    let id_val = Value::from_array(Array2::from_elem((1, 1), last_tok))
                        .map_err(|e| Error::Asr(format!("id val: {e}")))?;
                    let emb_out = models
                        .embed_tokens
                        .run(ort::inputs!("input_ids" => id_val))
                        .map_err(|e| Error::Asr(format!("embed step: {e}")))?;
                    extract_f16_array(&emb_out[0])?
                };

                // Reuse position array — just update the value.
                pos_arr.fill(pos as i64);

                let total_len = pos + 1;
                let emb_val =
                    Value::from_array(emb).map_err(|e| Error::Asr(format!("emb val: {e}")))?;
                let pos_val = Value::from_array(pos_arr.clone())
                    .map_err(|e| Error::Asr(format!("pos val: {e}")))?;
                let mask_val = Value::from_array(Array2::<i64>::ones((1, total_len)))
                    .map_err(|e| Error::Asr(format!("mask val: {e}")))?;

                match &mut kv_cache {
                    DecodeKvCache::Bound(cache) => {
                        let binding = decoder_binding
                            .as_mut()
                            .ok_or_else(|| Error::Asr("missing decoder binding".into()))?;
                        binding.clear();
                        binding
                            .bind_input("inputs_embeds", &emb_val)
                            .map_err(|e| Error::Asr(format!("bind embeds: {e}")))?;
                        binding
                            .bind_input("position_ids", &pos_val)
                            .map_err(|e| Error::Asr(format!("bind position_ids: {e}")))?;
                        binding
                            .bind_input("attention_mask", &mask_val)
                            .map_err(|e| Error::Asr(format!("bind attention_mask: {e}")))?;
                        // Move ORT-owned KV values directly into the next decoder run.
                        // Present KV outputs are bound to device memory below.
                        for (i, v) in cache.drain(..).enumerate() {
                            let layer = i / 2;
                            let is_key = i % 2 == 0;
                            binding
                                .bind_input(kv_name(layer, is_key), &v)
                                .map_err(|e| Error::Asr(format!("bind kv layer {layer}: {e}")))?;
                        }
                        bind_decoder_outputs_to_device(binding, self.use_gpu)?;

                        let mut out = models
                            .decoder
                            .run_binding(binding)
                            .map_err(|e| Error::Asr(format!("decoder step: {e}")))?;

                        let result = argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost)?;

                        for i in 0..NUM_LAYERS {
                            cache.push(out.remove(present_kv_name(i, true)).ok_or_else(|| {
                                Error::Asr(format!("missing present key layer {i}"))
                            })?);
                            cache.push(out.remove(present_kv_name(i, false)).ok_or_else(|| {
                                Error::Asr(format!("missing present value layer {i}"))
                            })?);
                        }
                        result
                    }
                    DecodeKvCache::Cpu(cache) => {
                        let mut dec_inputs = ort::inputs!(
                            "inputs_embeds" => emb_val,
                            "position_ids" => pos_val,
                            "attention_mask" => mask_val
                        );
                        append_cached_kv(&mut dec_inputs, std::mem::take(cache))?;

                        let mut out = models
                            .decoder
                            .run(dec_inputs)
                            .map_err(|e| Error::Asr(format!("decoder step: {e}")))?;

                        let result = argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost)?;

                        for i in 0..NUM_LAYERS {
                            let key = out.remove(present_kv_name(i, true)).ok_or_else(|| {
                                Error::Asr(format!("missing present key layer {i}"))
                            })?;
                            let value = out.remove(present_kv_name(i, false)).ok_or_else(|| {
                                Error::Asr(format!("missing present value layer {i}"))
                            })?;
                            cache.push(extract_f16_array(&key)?);
                            cache.push(extract_f16_array(&value)?);
                        }
                        result
                    }
                }
            };
            tokens.push((tok, conf, logits_k));
            decode_steps += 1;
            pos += 1;
        }
        let decode_ms = decode_t0.elapsed().as_secs_f64() * 1000.0;

        drop(models);

        let elapsed = start.elapsed();
        let profile = Qwen3AsrProfile {
            audio_ms: pcm.len() * 1000 / 16_000,
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
        let (text, mean_conf, min_conf, word_confidences) = self.decode_output(&tokens);
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
        }])
    }

    fn decode_output(
        &self,
        tokens: &[(i64, f32, Vec<f64>)],
    ) -> (
        String,
        Option<f32>,
        Option<f32>,
        Vec<lune_core::asr::WordConfidence>,
    ) {
        use lune_core::mable_wc::{self, TokenSpan};

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

        let content: Vec<&(i64, f32, Vec<f64>)> = tokens[content_start..]
            .iter()
            .skip_while(|(id, _, _)| is_meta(*id))
            .filter(|(id, _, _)| !is_meta(*id))
            .collect();

        if content.is_empty() {
            return (String::new(), None, None, Vec::new());
        }

        let token_ids: Vec<u32> = content.iter().map(|(id, _, _)| *id as u32).collect();
        let mut text = self
            .tokenizer
            .decode(&token_ids, true)
            .unwrap_or_default()
            .trim()
            .to_string();
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

        let herc_tokens: Vec<mable_wc::TokenInput> = content
            .iter()
            .map(|&&(_, conf, ref lk)| mable_wc::TokenInput {
                confidence: conf,
                top_k_logits: lk.clone(),
            })
            .collect();

        let herc_wcs = mable_wc::estimate(
            &words,
            &herc_tokens,
            &spans,
            mable_wc::HercConfig::default(),
        );

        use crate::confidence::common_words::COMMON_WORDS;
        use std::collections::HashSet;
        use std::sync::LazyLock;
        static CW: LazyLock<HashSet<&str>> =
            LazyLock::new(|| COMMON_WORDS.iter().copied().collect());
        const UNCOMMON_BOOST: f32 = 0.15;

        let word_confidences: Vec<lune_core::asr::WordConfidence> = herc_wcs
            .into_iter()
            .map(|wc| {
                let mut fs = wc.fix_score as f32;
                if fs > 0.10 && !CW.contains(wc.word.to_lowercase().as_str()) {
                    fs = (fs + UNCOMMON_BOOST).min(1.0);
                }
                lune_core::asr::WordConfidence {
                    word: wc.word,
                    confidence: wc.confidence as f32,
                    fix_score: fs,
                }
            })
            .collect();

        (text, mean, min, word_confidences)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn lock_err<T>(what: &str, e: std::sync::PoisonError<T>) -> Error {
    Error::Asr(format!("{what} lock poisoned: {e}"))
}

#[cfg(test)]
mod tests {}
