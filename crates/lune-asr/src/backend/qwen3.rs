//! Qwen3-ASR 0.6B ONNX backend — encoder-decoder with autoregressive greedy decoding.
//!
//! Model: custom ONNX export via `scripts/export_onnx.py` (encoder + embed_tokens + decoder).
//! Mel: Whisper-compatible 128-bin log-mel at 16 kHz (see `whisper_mel.rs`).
//! Decoding: greedy argmax with per-token softmax confidence.

use std::path::PathBuf;
use std::sync::Mutex;

use half::f16;
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig, WordConfidence};
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

/// Base system prompt — language-neutral, minimal. Language is controlled
/// via the `language X` suffix parameter, not the system prompt.
const SYSTEM_PROMPT: &str = "\
You are an ASR engine. \
Transcribe the spoken audio exactly in the spoken language. \
Do not translate. \
Output only the transcript text. \
Do not add words not clearly present. \
If no speech, return empty string.";

/// Maximum audio buffer (5 min at 16 kHz).
const MAX_WINDOW_SAMPLES: usize = 300 * 16_000;
/// Silence onset padding (~200 ms at 16 kHz).
const ONSET_PAD_SAMPLES: usize = 3_200;
const SILENCE_RMS_THRESHOLD: f32 = 0.008;

// ── Types ───────────────────────────────────────────────────────────────────

pub struct Qwen3AsrConfig {
    pub model_dir: PathBuf,
    pub use_gpu: bool,
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
    /// Language hint for this session. Some("English") forces English,
    /// None lets the model auto-detect.
    language: Option<String>,
}

// ── Constructor ─────────────────────────────────────────────────────────────

impl Qwen3AsrBackend {
    pub fn new(config: Qwen3AsrConfig) -> Result<Self> {
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
            boost_map: Mutex::new(std::collections::HashMap::new()),
            context_phrases: Mutex::new(Vec::new()),
        })
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
            // Tokenize "language English" (or whatever)
            let lang_text = format!("language {lang}");
            if let Ok(enc) = self.tokenizer.encode(lang_text.as_str(), false) {
                suffix.extend(enc.get_ids().iter().map(|&id| id as i64));
            }
        }
        suffix.push(ASR_TEXT_ID);

        (prefix, suffix)
    }
}

// ── AsrBackend impl ─────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl AsrBackend for Qwen3AsrBackend {
    type SessionHandle = Mutex<Qwen3Session>;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        // Map BCP-47 prefix → Qwen3 canonical language name. None → auto-detect.
        const LANG_MAP: &[(&str, &str)] = &[
            ("en", "English"),
            ("fr", "French"),
            ("de", "German"),
            ("es", "Spanish"),
            ("it", "Italian"),
            ("zh", "Chinese"),
            ("cn", "Chinese"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
            ("pt", "Portuguese"),
            ("ru", "Russian"),
        ];
        let language = config.language.as_deref().and_then(|l| {
            let lower = l.to_ascii_lowercase();
            LANG_MAP
                .iter()
                .find(|(prefix, _)| lower.starts_with(prefix))
                .map(|(_, name)| name.to_string())
        });
        let session = Qwen3Session {
            pending: Vec::with_capacity(MAX_WINDOW_SAMPLES),
            total_samples: 0,
            sample_rate: config.sample_rate,
            next_id: 1,
            language,
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
        let (pcm, ts_ms, id, language) = {
            let mut sess = session.lock().map_err(|e| lock_err("session", e))?;
            if sess.pending.is_empty() {
                return Ok(vec![]);
            }
            let ts = sess.total_samples * 1000 / sess.sample_rate as u64;
            let id = sess.next_id;
            sess.next_id += 1;
            let pcm = sess.pending.clone();
            let lang = sess.language.clone();
            sess.pending.clear();
            (pcm, ts, id, lang)
        };

        let trimmed = trim_leading_silence(&pcm);
        if trimmed.is_empty() {
            return Ok(vec![]);
        }

        self.transcribe_pcm(trimmed, ts_ms, id, language.as_deref())
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

/// Fused argmax + softmax confidence directly on f16 logits tensor.
/// Avoids allocating a full f32 copy (~600KB per step). Operates on the
/// last `vocab_size` elements of the last position in the logits tensor.
fn argmax_on_logits(
    value: &ort::value::Value,
    vocab_size: usize,
    boost: &std::collections::HashMap<i64, f32>,
) -> Result<(i64, f32)> {
    let (shape, data) = value
        .try_extract_tensor::<f16>()
        .map_err(|e| Error::Asr(format!("extract logits: {e}")))?;
    // Last position's logits: last `vocab_size` elements.
    let total = shape.iter().map(|&d| d as usize).product::<usize>();
    let start = total.saturating_sub(vocab_size);
    let logits = &data[start..];

    if boost.is_empty() {
        // Fast path: single pass argmax + LSE on f16 → f32.
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            let f = v.to_f32();
            if f > max_val {
                max_val = f;
                max_idx = i;
            }
        }
        let lse: f32 = logits
            .iter()
            .map(|&x| (x.to_f32() - max_val).exp())
            .sum::<f32>()
            .ln()
            + max_val;
        Ok((max_idx as i64, (max_val - lse).exp()))
    } else {
        // Boosted path: need unboosted max for confidence.
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            let boosted = v.to_f32() + boost.get(&(i as i64)).copied().unwrap_or(0.0);
            if boosted > max_val {
                max_val = boosted;
                max_idx = i;
            }
        }
        let raw_max = logits[max_idx].to_f32();
        let lse: f32 = logits
            .iter()
            .map(|&x| (x.to_f32() - raw_max).exp())
            .sum::<f32>()
            .ln()
            + raw_max;
        Ok((max_idx as i64, (raw_max - lse).exp()))
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

        // 1. Mel spectrogram [1, 128, T] — FP16 native, variable length.
        //    Pad PCM with a short tail of silence so the mel's dynamic-range
        //    normalization includes a proper floor region (matching Whisper's
        //    behaviour where silence has mel-floor values, not zeros).
        const TAIL_PAD: usize = 16_000; // 1s of silence context
        let padded_len = pcm.len() + TAIL_PAD;
        let mut pcm_padded = vec![0.0f32; padded_len];
        pcm_padded[..pcm.len()].copy_from_slice(pcm);
        let spec = self.mel.extract_from_f32(&pcm_padded);
        let mel_3d = ndarray::Array3::from_shape_vec((1, spec.n_mels, spec.n_frames), spec.data)
            .map_err(|e| Error::Asr(format!("mel reshape: {e}")))?;
        let mel_val =
            Value::from_array(mel_3d).map_err(|e| Error::Asr(format!("mel value: {e}")))?;

        let mut models = self.models.lock().map_err(|e| lock_err("models", e))?;

        // 2. Encoder → audio features [1, N, D] (FP16 native)
        let audio_feat = {
            let out = models
                .encoder
                .run(ort::inputs!("mel" => mel_val))
                .map_err(|e| Error::Asr(format!("encoder: {e}")))?;
            extract_f16_array(&out[0])?
        };
        let n_audio = audio_feat.shape()[1];
        let hidden_size = audio_feat.shape()[2];

        // 3. Build prompt token IDs (dynamic system prompt + context)
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

        // 5. Prefill: run decoder with full embeddings + empty KV cache
        let (first_token, first_conf, mut kv_cache) = {
            let mut dec_inputs = ort::inputs!(
                "inputs_embeds" => Value::from_array(embeds)
                    .map_err(|e| Error::Asr(format!("embeds val: {e}")))?,
                "position_ids" => Value::from_array(mrope_position_ids(seq_len))
                    .map_err(|e| Error::Asr(format!("pos val: {e}")))?,
                "attention_mask" => Value::from_array(Array2::<i64>::ones((1, seq_len)))
                    .map_err(|e| Error::Asr(format!("mask val: {e}")))?
            );
            append_empty_kv(&mut dec_inputs)?;

            let out = models
                .decoder
                .run(dec_inputs)
                .map_err(|e| Error::Asr(format!("decoder prefill: {e}")))?;

            let (tok, conf) = argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost)?;

            // Extract per-layer KV cache
            let mut cache: Vec<ArrayD<f16>> = Vec::with_capacity(NUM_LAYERS * 2);
            for i in 0..NUM_LAYERS {
                cache.push(extract_f16_array(&out[present_kv_name(i, true)])?);
                cache.push(extract_f16_array(&out[present_kv_name(i, false)])?);
            }
            (tok, conf, cache)
        };

        let mut tokens: Vec<(i64, f32)> = vec![(first_token, first_conf)];
        let mut pos = seq_len;

        // Pre-allocate reusable position array for the decode loop.
        let mut pos_arr = Array3::<i64>::zeros((3, 1, 1));

        // 6. Autoregressive greedy decode loop
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

            let (tok, conf) = {
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
                // emb is already [1, 1, D] from embed_tokens — no reshape needed.
                let mut dec_inputs = ort::inputs!(
                    "inputs_embeds" => Value::from_array(emb)
                        .map_err(|e| Error::Asr(format!("emb val: {e}")))?,
                    "position_ids" => Value::from_array(pos_arr.clone())
                        .map_err(|e| Error::Asr(format!("pos val: {e}")))?,
                    "attention_mask" => Value::from_array(Array2::<i64>::ones((1, total_len)))
                        .map_err(|e| Error::Asr(format!("mask val: {e}")))?
                );
                // Move KV cache into ORT (no clone — saves ~44 MB/step).
                // Arrays are replaced with new present values after the run.
                for i in 0..NUM_LAYERS {
                    let k = std::mem::take(&mut kv_cache[i * 2]);
                    let v = std::mem::take(&mut kv_cache[i * 2 + 1]);
                    dec_inputs.push((
                        std::borrow::Cow::Borrowed(kv_name(i, true)),
                        Value::from_array(k)
                            .map_err(|e| Error::Asr(format!("k{i} val: {e}")))?
                            .into(),
                    ));
                    dec_inputs.push((
                        std::borrow::Cow::Borrowed(kv_name(i, false)),
                        Value::from_array(v)
                            .map_err(|e| Error::Asr(format!("v{i} val: {e}")))?
                            .into(),
                    ));
                }

                let out = models
                    .decoder
                    .run(dec_inputs)
                    .map_err(|e| Error::Asr(format!("decoder step: {e}")))?;

                let result = argmax_on_logits(&out["logits"], VOCAB_SIZE, &boost)?;

                for i in 0..NUM_LAYERS {
                    kv_cache[i * 2] = extract_f16_array(&out[present_kv_name(i, true)])?;
                    kv_cache[i * 2 + 1] = extract_f16_array(&out[present_kv_name(i, false)])?;
                }
                result
            };
            tokens.push((tok, conf));
            pos += 1;
        }

        drop(models);

        let elapsed = start.elapsed();
        tracing::debug!(
            elapsed_ms = elapsed.as_millis(),
            audio_ms = pcm.len() * 1000 / 16_000,
            tokens = tokens.len(),
            "Qwen3 inference"
        );
        // 7. Decode tokens → text, parse output, build segment
        let (text, word_confs) = self.decode_output(&tokens);
        if text.is_empty() {
            return Ok(vec![]);
        }

        let mean_conf = if word_confs.is_empty() {
            None
        } else {
            Some(word_confs.iter().map(|w| w.confidence).sum::<f32>() / word_confs.len() as f32)
        };
        let min_conf = word_confs.iter().map(|w| w.confidence).reduce(f32::min);

        Ok(vec![AsrSegment {
            id,
            text,
            is_final: true,
            start_ms: None,
            end_ms: Some(timestamp_ms),
            confidence: mean_conf,
            min_token_confidence: min_conf,
            word_confidences: word_confs,
            hypotheses: vec![],
        }])
    }

    /// Decode generated tokens to text and per-word confidences.
    ///
    /// Skips metadata tokens (everything up to and including `<asr_text>`),
    /// then aggregates BPE subwords into words using space-boundary detection.
    fn decode_output(&self, tokens: &[(i64, f32)]) -> (String, Vec<WordConfidence>) {
        // Find content start: skip tokens up to and including <asr_text>.
        // Also skip any IM_START/IM_END/NEWLINE metadata tokens that the
        // model may echo before the actual transcript.
        let content_start = tokens
            .iter()
            .position(|(id, _)| *id == ASR_TEXT_ID)
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

        let content: Vec<(i64, f32)> = tokens[content_start..]
            .iter()
            .copied()
            .skip_while(|(id, _)| is_meta(*id))
            .filter(|(id, _)| !is_meta(*id))
            .collect();

        if content.is_empty() {
            return (String::new(), vec![]);
        }

        // Decode full text
        let token_ids: Vec<u32> = content.iter().map(|(id, _)| *id as u32).collect();
        let mut text = self
            .tokenizer
            .decode(&token_ids, true)
            .unwrap_or_default()
            .trim()
            .to_string();
        // Safety net: strip echoed system prompt fragments the model may
        // regurgitate before the actual transcript.
        for prefix in [
            SYSTEM_PROMPT,
            "You are an automatic speech recognition system",
        ] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest.trim_start_matches(['.', '\n', ' ']).to_string();
            }
        }

        // Build per-word confidences via O(1) token lookup (not decode).
        // BPE pieces starting with space (' ' or 'Ġ') mark word boundaries.
        let mut words: Vec<WordConfidence> = Vec::new();
        let mut current_word = String::new();
        let mut current_conf = f32::INFINITY;

        for &(token_id, conf) in &content {
            let piece = self
                .tokenizer
                .id_to_token(token_id as u32)
                .unwrap_or_default();
            // BPE uses 'Ġ' (U+0120) as space prefix; normalize to actual space.
            let is_word_start = piece.starts_with('Ġ') || piece.starts_with(' ');
            if is_word_start && !current_word.is_empty() {
                words.push(WordConfidence::new(
                    std::mem::take(&mut current_word),
                    current_conf,
                ));
                let clean = piece.trim_start_matches('Ġ').trim_start();
                current_word = clean.to_string();
                current_conf = conf;
            } else {
                let clean = piece.trim_start_matches('Ġ');
                current_word.push_str(clean);
                current_conf = current_conf.min(conf);
            }
        }
        if !current_word.is_empty() {
            words.push(WordConfidence::new(current_word, current_conf));
        }

        (text, words)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Trim leading silence, keeping a small onset pad before speech.
fn trim_leading_silence(pcm: &[f32]) -> &[f32] {
    const FRAME: usize = 160;
    let mut speech_start = 0;
    for (i, chunk) in pcm.chunks(FRAME).enumerate() {
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        if rms > SILENCE_RMS_THRESHOLD {
            speech_start = i * FRAME;
            break;
        }
    }
    let offset = speech_start.saturating_sub(ONSET_PAD_SAMPLES);
    &pcm[offset..]
}

fn lock_err<T>(what: &str, e: std::sync::PoisonError<T>) -> Error {
    Error::Asr(format!("{what} lock poisoned: {e}"))
}
