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
    /// Finalized segments ready to be drained by `poll_final`.
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
/// `POST /v1/audio/transcriptions` when `poll_final` is called. Encodes
/// audio as 16-bit PCM WAV.
///
/// On transient HTTP errors, retries up to 3 times with 2 s initial
/// exponential backoff. An opportunistic flush that still fails re-buffers
/// the audio with a WARN log; a forced flush (`poll_final`, the batch
/// pipeline's one transcription shot) propagates the error so the session
/// reports a failure instead of a silent "no speech detected".
#[derive(Debug)]
pub struct OpenAiCompatibleAsrBackend {
    client: reqwest::Client,
    config: OpenAiAsrConfig,
    /// Pre-joined transcription URL, built once at construction time.
    transcription_url: Url,
    /// Pre-formatted Bearer token, built once to avoid per-request alloc.
    /// Held as a `RedactedString` so a `Debug` of the backend — or a stray log
    /// — cannot leak the key, and the buffer is zeroized on drop.
    auth_header: RedactedString,
}

impl OpenAiCompatibleAsrBackend {
    pub fn new(config: OpenAiAsrConfig) -> Result<Self> {
        // Reject plaintext (non-TLS) unless targeting a local endpoint
        // (loopback, private LAN, `.local`/`.localhost`). The ASR key
        // requirement and the LLM backend use the same `is_local_endpoint`
        // test, so a LAN Ollama/llama-server is plaintext-safe and a public
        // host must use HTTPS.
        let scheme = config.base_url.scheme();
        if scheme != "https" && !lune_core::config::is_local_endpoint(&config.base_url) {
            return Err(Error::asr(format!(
                "OpenAI ASR base_url must use HTTPS (got {scheme}://); \
                 plaintext only allowed for loopback and private local addresses"
            )));
        }

        // Pre-build the transcription URL once; avoids per-request parse+alloc.
        let transcription_url = build_transcription_url(&config.base_url)?;

        // Pre-format the auth header value once.
        let auth_header = RedactedString::new(format!("Bearer {}", config.api_key.expose()));

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| Error::asr(format!("build http client: {e}")))?;

        tracing::info!(
            base_url = %config.base_url,
            model = %config.model,
            "initialized OpenAI-compatible ASR backend"
        );

        Ok(Self {
            client,
            config,
            transcription_url,
            auth_header,
        })
    }
}

/// Lock helper: maps poison error to `Error::Asr` with a fixed message.
#[inline(always)]
fn lock_err<T>(r: std::sync::LockResult<T>) -> Result<T> {
    r.map_err(|_| Error::asr("session lock poisoned"))
}

#[async_trait::async_trait]
impl AsrBackend for OpenAiCompatibleAsrBackend {
    type SessionHandle = OpenAiAsrSession;

    async fn start_session(&self, config: AsrSessionConfig) -> Result<Self::SessionHandle> {
        tracing::debug!(language = ?config.language, sample_rate = config.sample_rate, "ASR session started");
        // Pre-allocate audio buffer: 5 s at typical 16 kHz to avoid reallocs.
        let capacity = (config.sample_rate as usize).saturating_mul(5);
        Ok(OpenAiAsrSession {
            state: Mutex::new(SessionState {
                config,
                audio_buf: Vec::with_capacity(capacity),
                pending: Vec::with_capacity(8),
                next_id: 0,
            }),
        })
    }

    async fn push_audio(&self, session: &Self::SessionHandle, pcm: &[f32]) -> Result<()> {
        let mut state = lock_err(session.state.lock())?;
        state.audio_buf.extend_from_slice(pcm);
        Ok(())
    }

    async fn poll_final(&self, session: &Self::SessionHandle) -> Result<Vec<AsrSegment>> {
        self.flush_audio(session, true).await?;
        let mut state = lock_err(session.state.lock())?;
        Ok(std::mem::take(&mut state.pending))
    }

    async fn end_session(&self, session: Self::SessionHandle) -> Result<()> {
        let _ = session;
        tracing::debug!("ASR session ended");
        Ok(())
    }
}

impl OpenAiCompatibleAsrBackend {
    async fn flush_audio(&self, session: &OpenAiAsrSession, force: bool) -> Result<()> {
        let samples_len = lock_err(session.state.lock())?.audio_buf.len();
        if samples_len == 0 || (!force && samples_len < self.config.min_samples_before_request) {
            return Ok(());
        }

        let (mut pcm, lang, sample_rate, base_id) = {
            let mut state = lock_err(session.state.lock())?;
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
            Ok(new_segs) => {
                let count = new_segs.len() as u64;
                let mut state = lock_err(session.state.lock())?;
                state.next_id += count;
                state.pending.extend(new_segs);
            }
            Err(e) => {
                tracing::warn!(error = %e, "ASR API call failed; restoring audio for retry");
                let mut state = lock_err(session.state.lock())?;
                if state.audio_buf.is_empty() {
                    state.audio_buf = pcm;
                } else {
                    pcm.append(&mut state.audio_buf);
                    state.audio_buf = pcm;
                }
                // A forced flush is the utterance's only transcription shot:
                // swallowing the failure surfaces upstream as "no speech
                // detected". Propagate it so the session reports the error.
                if force {
                    drop(state);
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Send PCM audio to `/v1/audio/transcriptions`, retry on transient errors.
    async fn transcribe(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        base_id: u64,
    ) -> Result<Vec<AsrSegment>> {
        // Encode WAV once; cloned cheaply via Arc-backed Bytes on each attempt.
        let wav = encode_wav_16bit(pcm, sample_rate);

        let mut last_err = String::new();
        let mut delay = INITIAL_BACKOFF;
        // Set from `Retry-After` when the server tells us how long to wait;
        // overrides our own schedule for that one attempt.
        let mut server_delay: Option<Duration> = None;

        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                let wait = server_delay.take().unwrap_or_else(|| jittered(delay));
                tokio::time::sleep(wait).await;
                delay *= 2;
            }

            let part = reqwest::multipart::Part::bytes(wav.clone())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| Error::asr(format!("mime: {e}")))?;

            let mut form = reqwest::multipart::Form::new()
                .part("file", part)
                .text("model", self.config.model.clone())
                .text("response_format", "verbose_json");

            if let Some(lang) = language {
                form = form.text("language", lang.to_owned());
            }

            let resp = self
                .client
                .post(self.transcription_url.clone())
                .header("Authorization", self.auth_header.expose())
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
                    server_delay = retry_after(r.headers());
                    tracing::warn!(
                        attempt,
                        status = %r.status(),
                        retry_after_secs = server_delay.map(|d| d.as_secs()),
                        "ASR transient error"
                    );
                }
                Ok(r) if !r.status().is_success() => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    return Err(Error::asr(format!("ASR API error {status}: {body}")));
                }
                Ok(r) => {
                    let body: VerboseJsonResponse = r
                        .json()
                        .await
                        .map_err(|e| Error::asr(format!("parse response: {e}")))?;
                    return Ok(body.into_segments(base_id));
                }
            }
        }

        Err(Error::asr(format!(
            "ASR failed after {MAX_ATTEMPTS} attempts: {last_err}"
        )))
    }
}

/// Attempts made before a transient failure becomes a hard error.
const MAX_ATTEMPTS: u8 = 3;

/// First backoff window; doubled on each subsequent attempt.
const INITIAL_BACKOFF: Duration = Duration::from_secs(2);

/// Ceiling applied to a server-supplied `Retry-After`. The header is remote
/// input: an endpoint that asks us to sleep for an hour would hang the
/// utterance far past any point the user still wants it.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

/// Equal-jitter backoff: keep half the window fixed so the delay still grows,
/// and randomise the other half.
///
/// A desktop client has no herd to stampede, so this is not about protecting the
/// server. It matters when the endpoint is a *local* llama-server or Ollama that
/// just restarted: without jitter every retry in the process lands on the same
/// tick, so the requests keep colliding on the same cold-start window instead of
/// spreading across it.
fn jittered(base: Duration) -> Duration {
    const NANOS_PER_SEC: u128 = 1_000_000_000;

    let half = base / 2;
    // Nanosecond phase as the entropy source. Jitter needs spread, not
    // unpredictability, so a CSPRNG (and the dependency it costs) would be
    // wasted here.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    // Scale the sub-second phase across the whole half-window rather than using
    // it as raw nanoseconds: the phase only spans 1 s, so adding it directly
    // would confine a 2 s half-window's jitter to its first second and leave the
    // rest of the window unused — the spread shrinking exactly as backoff grows.
    let spread = u128::from(nanos) * half.as_nanos() / NANOS_PER_SEC;
    half + Duration::from_nanos(u64::try_from(spread).unwrap_or(u64::MAX))
}

/// Parse a `Retry-After` delay, clamped to [`MAX_RETRY_AFTER`].
///
/// Only the delta-seconds form is honoured; the HTTP-date form is rare on the
/// APIs this talks to and our own backoff is a fine fallback for it.
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let secs: u64 = raw.trim().parse().ok()?;
    Some(Duration::from_secs(secs).min(MAX_RETRY_AFTER))
}

fn build_transcription_url(base_url: &Url) -> Result<Url> {
    let mut url = base_url.clone();
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| Error::asr("base_url cannot be a cannot-be-a-base URL"))?;
    segments.push("audio");
    segments.push("transcriptions");
    drop(segments);
    Ok(url)
}

// ── WAV encoding ──────────────────────────────────────────────────────────────

/// Encode f32 PCM samples as a 16-bit mono WAV file in memory.
///
/// The output buffer is sized exactly once; each sample is converted via
/// a branchless clamp+scale and written as little-endian i16.
pub fn encode_wav_16bit(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = pcm.len();
    let data_bytes = num_samples * 2; // 2 bytes per 16-bit sample
    let file_size = 36 + data_bytes;

    // Allocate exact capacity upfront — no reallocs.
    let mut buf = Vec::with_capacity(44 + data_bytes);

    // RIFF header (12 bytes)
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(file_size as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk (24 bytes)
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk header (8 bytes)
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_bytes as u32).to_le_bytes());

    // Sample data: write all as raw bytes in one pass.
    // Reserve space and use unsafe ptr write to avoid bounds checks in hot loop.
    let header_len = buf.len(); // should be 44
    buf.resize(header_len + data_bytes, 0u8);
    let dst = &mut buf[header_len..];
    for (i, &s) in pcm.iter().enumerate() {
        // Branchless clamp: avoids branch misprediction on out-of-range values.
        let clamped = s.clamp(-1.0, 1.0);
        let sample = (clamped * 32767.0) as i16;
        let bytes = sample.to_le_bytes();
        dst[i * 2] = bytes[0];
        dst[i * 2 + 1] = bytes[1];
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
            let trimmed = self.text.trim();
            if trimmed.is_empty() {
                return vec![];
            }
            return vec![AsrSegment {
                id: base_id,
                text: trimmed.to_owned(),
                is_final: true,
                start_ms: None,
                end_ms: None,
                confidence: None,
                min_token_confidence: None,
                word_confidences: vec![],
                detected_language: None,
            }];
        }

        // Pre-allocate exact capacity; filter step may reduce but avoids realloc growth.
        let mut out = Vec::with_capacity(self.segments.len());
        for s in self.segments {
            let text = s.text.trim().to_owned();
            if !text.is_empty() {
                out.push(AsrSegment {
                    id: base_id + s.id,
                    text,
                    is_final: true,
                    start_ms: Some((s.start * 1000.0) as u64),
                    end_ms: Some((s.end * 1000.0) as u64),
                    confidence: None,
                    min_token_confidence: None,
                    word_confidences: vec![],
                    detected_language: None,
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_stays_within_the_backoff_window() {
        let base = Duration::from_secs(4);
        // Never below half the window (the delay must still grow across
        // attempts) and never above it (backoff must stay bounded).
        for _ in 0..256 {
            let d = jittered(base);
            assert!(d >= base / 2, "{d:?} collapsed the backoff");
            assert!(d <= base, "{d:?} overshot the window");
        }
    }

    #[test]
    fn jitter_actually_varies() {
        let base = Duration::from_secs(4);
        // A constant "jitter" would defeat the point; sample until we see two
        // different values rather than asserting on one draw.
        let first = jittered(base);
        let varied = (0..1000).any(|_| jittered(base) != first);
        assert!(varied, "jitter produced a constant delay");
    }

    #[test]
    fn jitter_handles_a_degenerate_window() {
        // `half` is zero here; the modulus must not divide by zero.
        assert_eq!(jittered(Duration::ZERO), Duration::ZERO);
    }

    #[test]
    fn retry_after_is_honoured_and_clamped() {
        use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("5"));
        assert_eq!(retry_after(&headers), Some(Duration::from_secs(5)));

        // A hostile or misconfigured endpoint must not park the utterance.
        headers.insert(RETRY_AFTER, HeaderValue::from_static("86400"));
        assert_eq!(retry_after(&headers), Some(MAX_RETRY_AFTER));
    }

    #[test]
    fn unparseable_retry_after_falls_back_to_our_own_backoff() {
        use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

        let mut headers = HeaderMap::new();
        assert_eq!(retry_after(&headers), None);

        // The HTTP-date form is deliberately not parsed.
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(retry_after(&headers), None);
    }

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
        // IPv6 loopback must also be recognized (host_str returns `[::1]`).
        let config = OpenAiAsrConfig {
            base_url: "http://[::1]:8080/v1".parse().unwrap(),
            ..Default::default()
        };
        assert!(OpenAiCompatibleAsrBackend::new(config).is_ok());
    }

    /// A private-LAN server is "local" by `is_local_endpoint` (the same test
    /// the LLM backend uses), so plaintext there must be accepted — matching
    /// the LLM guard and the keyless-readiness logic.
    #[test]
    fn allows_plaintext_private_lan() {
        let config = OpenAiAsrConfig {
            base_url: "http://192.168.1.50:8000/v1".parse().unwrap(),
            ..Default::default()
        };
        assert!(OpenAiCompatibleAsrBackend::new(config).is_ok());
    }

    #[test]
    fn preserves_custom_base_path_prefix() {
        let url = build_transcription_url(&"https://example.com/proxy/openai/v1".parse().unwrap())
            .unwrap();
        assert_eq!(
            url.as_str(),
            "https://example.com/proxy/openai/v1/audio/transcriptions"
        );
    }

    #[tokio::test]
    async fn unreachable_api_surfaces_error_from_poll_final() {
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
        // poll_final must propagate the failure — Ok(empty) would read as
        // "no speech detected" upstream and hide the API outage.
        let err = backend.poll_final(&session).await.unwrap_err();
        assert!(err.to_string().contains("ASR failed"), "got: {err}");
    }

    #[tokio::test]
    async fn poll_final_restores_audio_after_failed_flush() {
        let config = OpenAiAsrConfig {
            base_url: "http://127.0.0.1:1/v1".parse().unwrap(),
            timeout: Duration::from_millis(100),
            min_samples_before_request: usize::MAX,
            ..Default::default()
        };
        let backend = OpenAiCompatibleAsrBackend::new(config).unwrap();
        let session = backend
            .start_session(AsrSessionConfig::default())
            .await
            .unwrap();

        backend.push_audio(&session, &[0.0f32; 100]).await.unwrap();
        // The failure propagates, but the audio must be re-buffered so a
        // caller-driven retry still has the full utterance.
        assert!(backend.poll_final(&session).await.is_err());
        assert_eq!(session.state.lock().unwrap().audio_buf.len(), 100);
    }
}
