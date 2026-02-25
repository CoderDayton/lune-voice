use std::path::PathBuf;
use std::sync::Mutex;

use qwen_asr::config;
use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe::{self, StreamState};

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::error::{Error, Result};

/// Runtime configuration for the local Qwen ASR backend.
#[derive(Debug, Clone)]
pub struct QwenAsrConfig {
    /// Directory containing Qwen3-ASR safetensors and vocab.
    pub model_dir: PathBuf,
    /// Streaming chunk size in seconds.
    pub stream_chunk_sec: f32,
    /// Number of unstable tail tokens kept for rollback.
    pub stream_rollback: i32,
    /// Number of chunks kept "unfixed" before committing text.
    pub stream_unfixed_chunks: i32,
    /// Decoder cap per chunk.
    pub stream_max_new_tokens: i32,
    /// Whether to condition each chunk on previously committed text.
    pub past_text_conditioning: bool,
}

impl Default for QwenAsrConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            stream_chunk_sec: 0.5,
            stream_rollback: 5,
            stream_unfixed_chunks: 2,
            stream_max_new_tokens: 32,
            past_text_conditioning: true,
        }
    }
}

/// Local ASR backend using `qwen-asr` incremental streaming decode.
pub struct QwenAsrBackend {
    ctx: Mutex<QwenCtx>,
}

struct SessionState {
    config: AsrSessionConfig,
    /// Full accumulated PCM buffer (16 kHz mono f32).
    audio_buf: Vec<f32>,
    stream: StreamState,
    next_id: u64,
}

pub struct QwenAsrSession {
    state: Mutex<SessionState>,
}

impl QwenAsrBackend {
    pub fn new(config: QwenAsrConfig) -> Result<Self> {
        if !config.model_dir.exists() {
            return Err(Error::ModelNotFound(config.model_dir));
        }

        let model_dir = config.model_dir.to_string_lossy().to_string();
        let mut ctx = QwenCtx::load(&model_dir)
            .ok_or_else(|| Error::Asr(format!("failed to load qwen-asr model from {model_dir}")))?;

        ctx.stream_chunk_sec = config.stream_chunk_sec;
        ctx.stream_rollback = config.stream_rollback;
        ctx.stream_unfixed_chunks = config.stream_unfixed_chunks;
        ctx.stream_max_new_tokens = config.stream_max_new_tokens;
        ctx.past_text_conditioning = config.past_text_conditioning;
        // Keep transcription deterministic and avoid model-internal silence trimming.
        ctx.skip_silence = false;

        tracing::info!(
            model_dir = %model_dir,
            chunk_sec = ctx.stream_chunk_sec,
            rollback = ctx.stream_rollback,
            unfixed_chunks = ctx.stream_unfixed_chunks,
            max_new_tokens = ctx.stream_max_new_tokens,
            past_text_conditioning = ctx.past_text_conditioning,
            "initialized Qwen ASR backend"
        );

        Ok(Self {
            ctx: Mutex::new(ctx),
        })
    }
}

#[cold]
#[inline(never)]
fn lock_err(context: &'static str, e: impl std::fmt::Display) -> Error {
    Error::Asr(format!("{context}: {e}"))
}

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

#[inline]
fn resolve_qwen_language(language: &str) -> Option<String> {
    config::normalize_language(language).or_else(|| map_language_tag(language).map(str::to_string))
}

#[async_trait::async_trait]
impl AsrBackend for QwenAsrBackend {
    type SessionHandle = QwenAsrSession;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        if config.sample_rate != 16_000 {
            return Err(Error::Asr(format!(
                "qwen-asr expects 16000 Hz mono PCM, got {} Hz",
                config.sample_rate
            )));
        }

        {
            let mut ctx = self.ctx.lock().map_err(|e| lock_err("ctx lock", e))?;
            // Clear any previous session prompt/language state.
            let _ = ctx.set_prompt("");

            if let Some(lang) = config.language.as_deref().and_then(resolve_qwen_language) {
                ctx.set_force_language(&lang).map_err(|_| {
                    Error::Asr(format!(
                        "failed to apply forced language '{lang}' for qwen-asr"
                    ))
                })?;
            } else {
                let _ = ctx.set_force_language("");
            }
        }

        Ok(QwenAsrSession {
            state: Mutex::new(SessionState {
                config,
                audio_buf: Vec::with_capacity(16_000 * 8),
                stream: StreamState::new(),
                next_id: 1,
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
        Ok(())
    }

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;
        if state.audio_buf.is_empty() {
            return Ok(vec![]);
        }

        let mut ctx = self.ctx.lock().map_err(|e| lock_err("ctx lock", e))?;

        let sample_rate = state.config.sample_rate;
        let SessionState {
            ref audio_buf,
            ref mut stream,
            ..
        } = *state;

        let start_samples = stream.audio_cursor();

        let text = transcribe::stream_push_audio(&mut ctx, audio_buf, stream, false)
            .ok_or_else(|| Error::Asr("qwen-asr stream decode failed".into()))?;

        let end_samples = stream.audio_cursor();
        let id = state.next_id;
        state.next_id = id.saturating_add(1);

        let text = text.trim();
        if text.is_empty() {
            return Ok(vec![]);
        }

        let start_ms = (start_samples as u64).saturating_mul(1000) / sample_rate as u64;
        let end_ms = (end_samples as u64).saturating_mul(1000) / sample_rate as u64;

        Ok(vec![AsrSegment {
            id,
            confidence: None,
            // `stream_push_audio` only emits committed/stable deltas.
            is_final: true,
            start_ms: Some(start_ms),
            end_ms: Some(end_ms),
            text: text.to_owned(),
        }])
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        let _ = session;
        Ok(())
    }

    async fn reset_context(&self, session: &Self::SessionHandle) -> Result<()> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| lock_err("session lock", e))?;
        state.audio_buf.clear();
        state.stream.reset();
        Ok(())
    }

    fn segment_capacity_hint(&self) -> usize {
        1
    }
}
