use std::sync::Mutex;
use std::time::Duration;

use serde::Deserialize;
use url::Url;

use lune_core::asr::{AsrBackend, AsrSegment, AsrSessionConfig};
use lune_core::config::RedactedString;
use lune_core::error::{Error, Result};

/// Configuration for the OpenAI-compatible ASR backend.
#[derive(Debug, Clone)]
pub struct OpenAiAsrConfig {
    /// Base URL, e.g. `https://api.openai.com/v1`.
    pub base_url: Url,
    /// API key (stored redacted to prevent accidental logging).
    pub api_key: RedactedString,
    /// Model name, e.g. `"whisper-1"`.
    pub model: String,
    /// Per-request timeout (default 30 s).
    pub timeout: Duration,
    /// Minimum PCM samples buffered before triggering a request.
    ///
    /// Defaults to 0.5 s at 16 kHz = 8000 samples.
    pub min_samples_before_request: usize,
}

impl Default for OpenAiAsrConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".parse().expect("valid URL"),
            api_key: RedactedString::new(""),
            model: "whisper-1".into(),
            timeout: Duration::from_secs(30),
            min_samples_before_request: 8000, // 0.5 s @ 16 kHz
        }
    }
}

/// ASR session state — held behind a mutex inside the session handle.
struct SessionState {
    config: AsrSessionConfig,
    /// Raw PCM f32 samples accumulated via `push_audio`.
    audio_buf: Vec<f32>,
    /// Finalized segments ready to be drained by `poll_results`.
    pending: Vec<AsrSegment>,
    next_id: u64,
}

/// The session handle returned by `start_session`.
pub struct OpenAiAsrSession {
    state: Mutex<SessionState>,
}

/// OpenAI-compatible ASR backend.
///
/// Buffers PCM audio in each session and sends it to
/// `POST /v1/audio/transcriptions` when `poll_results` is called (if
/// enough audio has accumulated). Encodes audio as 16-bit PCM WAV.
///
/// On transient HTTP errors, retries up to 3 times with 2 s initial
/// exponential backoff. On persistent failure, returns raw empty segments
/// with a WARN log rather than propagating the error.
#[derive(Debug)]
pub struct OpenAiCompatibleAsrBackend {
    client: reqwest::Client,
    config: OpenAiAsrConfig,
}

impl OpenAiCompatibleAsrBackend {
    pub fn new(config: OpenAiAsrConfig) -> Result<Self> {
        // Reject plaintext (non-TLS) unless targeting loopback.
        let scheme = config.base_url.scheme();
        let host = config.base_url.host_str().unwrap_or("");
        let is_loopback = host == "127.0.0.1" || host == "localhost" || host == "::1";
        if scheme != "https" && !is_loopback {
            return Err(Error::Asr(format!(
                "OpenAI ASR base_url must use HTTPS (got {scheme}://); \
                 plaintext only allowed for loopback addresses"
            )));
        }

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::Asr(format!("build http client: {e}")))?;

        tracing::info!(
            base_url = %config.base_url,
            model = %config.model,
            "initialized OpenAI-compatible ASR backend"
        );

        Ok(Self { client, config })
    }
}

#[async_trait::async_trait]
impl AsrBackend for OpenAiCompatibleAsrBackend {
    type SessionHandle = OpenAiAsrSession;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        tracing::debug!(language = ?config.language, sample_rate = config.sample_rate, "ASR session started");
        Ok(OpenAiAsrSession {
            state: Mutex::new(SessionState {
                config,
                audio_buf: Vec::new(),
                pending: Vec::new(),
                next_id: 0,
            }),
        })
    }

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut state = session
            .state
            .lock()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
        state.audio_buf.extend_from_slice(pcm);
        Ok(())
    }

    async fn poll_results(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        let samples_len = {
            let state = session
                .state
                .lock()
                .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
            state.audio_buf.len()
        };

        // Only send a request if we have enough audio buffered.
        if samples_len >= self.config.min_samples_before_request {
            let (pcm, lang, sample_rate, base_id) = {
                let mut state = session
                    .state
                    .lock()
                    .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
                let pcm = std::mem::take(&mut state.audio_buf);
                let lang = state.config.language.clone();
                let sr = state.config.sample_rate;
                let id = state.next_id;
                (pcm, lang, sr, id)
            };

            match self
                .transcribe(&pcm, sample_rate, lang.as_deref(), base_id)
                .await
            {
                Ok(mut new_segs) => {
                    let count = new_segs.len() as u64;
                    let mut state = session
                        .state
                        .lock()
                        .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
                    state.next_id += count;
                    state.pending.append(&mut new_segs);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "ASR API call failed; dropping audio chunk");
                    // Put the audio back so it can be retried.
                    let mut state = session
                        .state
                        .lock()
                        .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
                    state.audio_buf.extend_from_slice(&pcm);
                }
            }
        }

        let mut state = session
            .state
            .lock()
            .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
        Ok(std::mem::take(&mut state.pending))
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        // Flush any remaining audio.
        let (pcm, lang, sample_rate, base_id) = {
            let mut state = session
                .state
                .lock()
                .map_err(|e| Error::Asr(format!("session lock: {e}")))?;
            let pcm = std::mem::take(&mut state.audio_buf);
            let lang = state.config.language.clone();
            let sr = state.config.sample_rate;
            let id = state.next_id;
            (pcm, lang, sr, id)
        };

        if !pcm.is_empty()
            && let Err(e) = self
                .transcribe(&pcm, sample_rate, lang.as_deref(), base_id)
                .await
        {
            tracing::warn!(error = %e, "ASR flush on session end failed");
        }

        tracing::debug!("ASR session ended");
        Ok(())
    }
}

impl OpenAiCompatibleAsrBackend {
    /// Send PCM audio to `/v1/audio/transcriptions`, retry on transient errors.
    async fn transcribe(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        base_id: u64,
    ) -> Result<Vec<AsrSegment>> {
        let wav = encode_wav_16bit(pcm, sample_rate);
        let url = self
            .config
            .base_url
            .join("/v1/audio/transcriptions")
            .map_err(|e| Error::Asr(format!("url join: {e}")))?;

        let mut last_err = String::new();
        let mut delay = Duration::from_secs(2);

        for attempt in 0..3u8 {
            if attempt > 0 {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }

            let part = reqwest::multipart::Part::bytes(wav.clone())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| Error::Asr(format!("mime: {e}")))?;

            let mut form = reqwest::multipart::Form::new()
                .part("file", part)
                .text("model", self.config.model.clone())
                .text("response_format", "verbose_json");

            if let Some(lang) = language {
                form = form.text("language", lang.to_owned());
            }

            let resp = self
                .client
                .post(url.clone())
                .header(
                    "Authorization",
                    format!("Bearer {}", self.config.api_key.expose()),
                )
                .multipart(form)
                .send()
                .await;

            match resp {
                Err(e) => {
                    last_err = e.to_string();
                    tracing::warn!(attempt, error = %e, "ASR request failed");
                }
                Ok(r) if r.status().is_server_error() || r.status() == 429 => {
                    last_err = format!("HTTP {}", r.status());
                    tracing::warn!(attempt, status = %r.status(), "ASR transient error");
                }
                Ok(r) if !r.status().is_success() => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    return Err(Error::Asr(format!("ASR API error {status}: {body}")));
                }
                Ok(r) => {
                    let body: VerboseJsonResponse = r
                        .json()
                        .await
                        .map_err(|e| Error::Asr(format!("parse response: {e}")))?;
                    return Ok(body.into_segments(base_id));
                }
            }
        }

        Err(Error::Asr(format!(
            "ASR failed after 3 attempts: {last_err}"
        )))
    }
}

// ── WAV encoding ──────────────────────────────────────────────────────────────

/// Encode f32 PCM samples as a 16-bit mono WAV file in memory.
pub fn encode_wav_16bit(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = pcm.len();
    let data_bytes = num_samples * 2; // 2 bytes per 16-bit sample
    let file_size = 36 + data_bytes;

    let mut buf = Vec::with_capacity(44 + data_bytes);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(file_size as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for &s in pcm {
        let sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&sample.to_le_bytes());
    }

    buf
}

// ── Response parsing ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct VerboseJsonResponse {
    #[serde(default)]
    segments: Vec<VerboseSegment>,
    // Fallback: if segments is empty, the whole text is a single segment.
    text: String,
}

#[derive(Debug, Deserialize)]
struct VerboseSegment {
    id: u64,
    text: String,
    start: f64,
    end: f64,
}

impl VerboseJsonResponse {
    fn into_segments(self, base_id: u64) -> Vec<AsrSegment> {
        if self.segments.is_empty() {
            // Plain json fallback: single final segment.
            if self.text.trim().is_empty() {
                return vec![];
            }
            return vec![AsrSegment {
                id: base_id,
                text: self.text.trim().to_owned(),
                is_final: true,
                start_ms: None,
                end_ms: None,
            }];
        }

        self.segments
            .into_iter()
            .map(|s| AsrSegment {
                id: base_id + s.id,
                text: s.text.trim().to_owned(),
                is_final: true,
                start_ms: Some((s.start * 1000.0) as u64),
                end_ms: Some((s.end * 1000.0) as u64),
            })
            .filter(|s| !s.text.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_wav_header() {
        let pcm = vec![0.0f32; 160]; // 10 ms at 16 kHz
        let wav = encode_wav_16bit(&pcm, 16000);

        // RIFF magic
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");

        // Data length = 160 samples * 2 bytes = 320
        let data_len = u32::from_le_bytes(wav[40..44].try_into().unwrap());
        assert_eq!(data_len, 320);

        // Total file = 44 header + 320 data
        assert_eq!(wav.len(), 364);
    }

    #[test]
    fn encode_wav_clamps_samples() {
        let pcm = vec![2.0f32, -2.0f32]; // out-of-range
        let wav = encode_wav_16bit(&pcm, 16000);

        let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
        let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
        assert_eq!(s0, 32767);
        assert_eq!(s1, -32767);
    }

    #[test]
    fn verbose_response_into_segments() {
        let resp = VerboseJsonResponse {
            text: "Hello world.".into(),
            segments: vec![
                VerboseSegment {
                    id: 0,
                    text: "Hello".into(),
                    start: 0.0,
                    end: 0.5,
                },
                VerboseSegment {
                    id: 1,
                    text: "world.".into(),
                    start: 0.5,
                    end: 1.0,
                },
            ],
        };
        let segs = resp.into_segments(10);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].id, 10);
        assert_eq!(segs[0].text, "Hello");
        assert_eq!(segs[0].start_ms, Some(0));
        assert_eq!(segs[1].id, 11);
        assert_eq!(segs[1].end_ms, Some(1000));
        assert!(segs[0].is_final);
    }

    #[test]
    fn verbose_response_fallback_to_text() {
        let resp = VerboseJsonResponse {
            text: "  Fallback text.  ".into(),
            segments: vec![],
        };
        let segs = resp.into_segments(0);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "Fallback text.");
    }

    #[test]
    fn rejects_plaintext_non_loopback() {
        let config = OpenAiAsrConfig {
            base_url: "http://example.com/v1".parse().unwrap(),
            ..Default::default()
        };
        let result = OpenAiCompatibleAsrBackend::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("HTTPS"));
    }

    #[test]
    fn allows_plaintext_loopback() {
        let config = OpenAiAsrConfig {
            base_url: "http://127.0.0.1:8080/v1".parse().unwrap(),
            ..Default::default()
        };
        assert!(OpenAiCompatibleAsrBackend::new(config).is_ok());
    }

    #[tokio::test]
    async fn graceful_degradation_on_unreachable() {
        let config = OpenAiAsrConfig {
            base_url: "http://127.0.0.1:1/v1".parse().unwrap(),
            timeout: Duration::from_millis(100),
            min_samples_before_request: 0, // trigger immediately
            ..Default::default()
        };
        let backend = OpenAiCompatibleAsrBackend::new(config).unwrap();
        let session = backend
            .start_session(AsrSessionConfig::default())
            .await
            .unwrap();

        backend.push_audio(&session, &[0.0f32; 100]).await.unwrap();
        // poll_results should not panic or propagate the error — it logs WARN.
        let segs = backend.poll_results(&session).await.unwrap();
        assert!(segs.is_empty());
    }
}
