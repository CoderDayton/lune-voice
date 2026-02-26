use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::{AudioConfig, AudioStream, Vad, VadConfig};
use lune_commands::VoiceCommandRouter;
use lune_core::Result;
use tokio::sync::RwLock;

use lune_core::asr::{AsrBackend, AsrBackendKind, AsrSessionConfig};
use lune_core::config::{StylePreset, UserSettings, builtin_registry};
use lune_core::error::Error;
use lune_core::llm::TextPostProcessor;
use lune_core::session::{DictationSession, SessionEvent, SessionStatus};
use lune_dict::DictionaryManager;
use lune_output::TextOutputController;
use lune_persist::Database;

use crate::llm_worker::{self, LlmRequest};

/// Timeout for waiting on final LLM response during teardown.
const TEARDOWN_LLM_TIMEOUT: Duration = Duration::from_secs(3);
/// Hard cap for a dictation session.
const MAX_SESSION_LENGTH: Duration = Duration::from_secs(60);
/// Minimum new samples before calling poll_results (~200 ms at 16 kHz).
const MIN_SAMPLES_BEFORE_POLL: usize = 3_200;

/// Silence duration (in samples at 16 kHz) after speech that triggers a flush.
/// 200 ms = 3_200 samples — matches natural conversation turn-taking gap
/// (Stivers et al. 2009: 208 ms mean across 10 languages). Short enough
/// for responsive end-of-utterance detection while clearing inter-word pauses.
const SILENCE_FLUSH_SAMPLES: usize = 3_200;
/// Keep forwarding a short silence tail after speech so brief words can still
/// reach backend chunking thresholds and get committed.
const ASR_PUSH_HANGOVER_SAMPLES: usize = 8_000; // 500ms @ 16kHz
/// Suppress immediate duplicate flushes from repeated backend emits.
const DUPLICATE_FLUSH_WINDOW: Duration = Duration::from_millis(1200);

/// Configuration for the session pipeline.
#[derive(Clone)]
pub struct SessionManagerConfig {
    pub asr_config: AsrSessionConfig,
    pub asr_backend_kind: AsrBackendKind,
    /// Confidence threshold for LLM bypass. `None` = always send to LLM.
    pub confidence_threshold: Option<f32>,
    /// Min single-token confidence threshold. `None` = no word-level gate.
    pub min_token_threshold: Option<f32>,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            asr_config: AsrSessionConfig::default(),
            asr_backend_kind: AsrBackendKind::ParakeetTdt,
            confidence_threshold: Some(0.95),
            min_token_threshold: Some(0.7),
        }
    }
}

struct ActiveHandle {
    stop_tx: oneshot::Sender<()>,
    session_id: Uuid,
}

#[derive(Default)]
struct SessionCounters {
    teardown_forced_flushes: u32,
}

/// Orchestrates the full dictation pipeline.
///
/// Only one session may be active at a time. Call `start()` to begin and
/// `stop()` to finalize gracefully. Events are delivered over the channel
/// passed to `start()`.
pub struct SessionManager<A: AsrBackend> {
    asr: Arc<A>,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    config: SessionManagerConfig,
    settings: Arc<RwLock<UserSettings>>,
    active: Arc<Mutex<Option<ActiveHandle>>>,
}

impl<A: AsrBackend + 'static> SessionManager<A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        asr: Arc<A>,
        router: Arc<VoiceCommandRouter>,
        post_processor: Arc<dyn TextPostProcessor>,
        dict: Arc<Mutex<DictionaryManager>>,
        persist: Arc<Mutex<Database>>,
        config: SessionManagerConfig,
        settings: Arc<RwLock<UserSettings>>,
    ) -> Self {
        Self {
            asr,
            router,
            post_processor,
            dict,
            persist,
            config,
            settings,
            active: Arc::new(Mutex::new(None)),
        }
    }

    /// Start a new session. Returns `Err` if one is already active.
    pub async fn start(&self, event_tx: mpsc::Sender<SessionEvent>) -> Result<Uuid> {
        let mut lock = self.active.lock().await;
        if lock.is_some() {
            return Err(Error::Session("session already active".into()));
        }

        let session_id = Uuid::new_v4();
        let (stop_tx, stop_rx) = oneshot::channel();

        let current_style_id = {
            let settings = self.settings.read().await;
            settings.default_style.clone()
        };
        let style_preset = builtin_registry()
            .get(&current_style_id)
            .unwrap_or_else(|| {
                builtin_registry()
                    .get("general")
                    .expect("builtin 'general' preset must exist")
            });

        {
            let session = DictationSession {
                id: session_id,
                started_at: Utc::now(),
                ended_at: None,
                style: style_preset.id.clone(),
                backend: self.config.asr_backend_kind,
                status: SessionStatus::Active,
                transcript: None,
            };
            if let Err(e) = self.persist.lock().await.insert_session(&session) {
                warn!(error = %e, "failed to persist session start");
            }
        }

        let asr = Arc::clone(&self.asr);
        let router = Arc::clone(&self.router);
        let post_processor = Arc::clone(&self.post_processor);
        let dict = Arc::clone(&self.dict);
        let persist = Arc::clone(&self.persist);
        let config = self.config.clone();
        let active_arc = Arc::clone(&self.active);

        tokio::task::spawn_local(run_session(
            session_id,
            asr,
            router,
            post_processor,
            dict,
            persist,
            config,
            style_preset,
            event_tx,
            stop_rx,
            active_arc,
        ));

        *lock = Some(ActiveHandle {
            stop_tx,
            session_id,
        });
        info!(%session_id, "session started");
        Ok(session_id)
    }

    pub async fn stop(&self) {
        let mut lock = self.active.lock().await;
        if let Some(handle) = lock.take() {
            debug!(session_id = %handle.session_id, "sending stop signal");
            let _ = handle.stop_tx.send(());
        }
    }

    #[inline]
    pub async fn cancel(&self) {
        self.stop().await;
    }

    #[inline]
    pub async fn active_session_id(&self) -> Option<Uuid> {
        self.active.lock().await.as_ref().map(|h| h.session_id)
    }
}

// ── Pipeline task ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn run_session<A: AsrBackend + 'static>(
    session_id: Uuid,
    asr: Arc<A>,
    router: Arc<VoiceCommandRouter>,
    post_processor: Arc<dyn TextPostProcessor>,
    dict: Arc<Mutex<DictionaryManager>>,
    persist: Arc<Mutex<Database>>,
    config: SessionManagerConfig,
    style_preset: Arc<StylePreset>,
    event_tx: mpsc::Sender<SessionEvent>,
    mut stop_rx: oneshot::Receiver<()>,
    active_arc: Arc<Mutex<Option<ActiveHandle>>>,
) {
    let mut status = SessionStatus::Completed;
    let mut style_preset = style_preset;

    let (llm_tx, mut llm_rx) = llm_worker::spawn_llm_worker(Arc::clone(&post_processor));

    // ── Start audio capture ───────────────────────────────────────────────
    let audio_stream = match AudioStream::new(AudioConfig::default()) {
        Ok(s) => s,
        Err(e) => {
            let _ = event_tx.send(SessionEvent::Error(e.to_string())).await;
            finalize_db(
                session_id,
                &persist,
                SessionStatus::Failed(e.to_string()),
                None,
                None,
            )
            .await;
            active_arc.lock().await.take();
            return;
        }
    };

    // ── Start ASR session ─────────────────────────────────────────────────
    let asr_session = match asr.start_session(config.asr_config).await {
        Ok(s) => s,
        Err(e) => {
            let _ = event_tx.send(SessionEvent::Error(e.to_string())).await;
            finalize_db(
                session_id,
                &persist,
                SessionStatus::Failed(e.to_string()),
                None,
                None,
            )
            .await;
            active_arc.lock().await.take();
            return;
        }
    };

    let mut vad = Vad::new(VadConfig::default(), 16_000);

    let _ = event_tx.send(SessionEvent::Started).await;

    let mut output_ctrl = TextOutputController::new(event_tx.clone());
    let mut session_text = String::with_capacity(512);
    let mut pending_nonfinal = String::with_capacity(128);
    let mut counters = SessionCounters::default();
    let mut sent_to_llm = false;
    let mut last_flushed_text = String::new();
    let mut last_flush_at = tokio::time::Instant::now() - DUPLICATE_FLUSH_WINDOW;
    // Streaming LLM state: first chunk of a new response needs a separator
    // from previous session text; subsequent chunks are typed as-is.
    let mut llm_first_chunk = true;

    // Reused each audio frame to avoid allocation on the ~50Hz partial path.
    let mut partial_buf = String::with_capacity(512);
    let mut audio_batch: Vec<f32> = Vec::with_capacity(4096);
    let mut audio_poll_tick = tokio::time::interval(Duration::from_millis(10));
    audio_poll_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Shared pipeline gate: accumulate samples + VAD speech flag before calling poll_results.
    let mut samples_since_last_poll: usize = 0;
    let mut had_speech_since_last_poll: bool = false;

    // VAD-driven speech→silence tracking (no wall-clock timers).
    let mut was_speaking = false;
    let mut silence_samples: usize = 0;
    let mut asr_push_hangover_samples: usize = 0;
    // Absolute session cap to avoid indefinitely active toggle sessions.
    let session_max_dur = tokio::time::sleep(MAX_SESSION_LENGTH);
    tokio::pin!(session_max_dur);
    // Resetting Qwen context on every utterance causes repeated streaming
    // cold-start behavior; keep it warm within a session.
    let reset_asr_context_on_flush = !matches!(config.asr_backend_kind, AsrBackendKind::QwenAsr);

    // ── Main loop ─────────────────────────────────────────────────────────
    loop {
        tokio::select! {
            biased;

            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }

            // ── Hard session cap reached → stop session ────────────────
            () = &mut session_max_dur => {
                info!(
                    %session_id,
                    max_session_secs = MAX_SESSION_LENGTH.as_secs(),
                    "max session length reached; stopping session"
                );
                break;
            }

            // ── LLM streaming chunk or completion ────────────────────
            Some(resp) = llm_rx.recv() => {
                if !resp.corrected_text.is_empty() {
                    if llm_first_chunk {
                        // First chunk: insert separator from previous session text.
                        commit_final_text(&mut output_ctrl, &mut session_text, &resp.corrected_text).await;
                        llm_first_chunk = false;
                    } else {
                        // Subsequent streaming chunks: already word-spaced.
                        output_ctrl.emit_final(resp.corrected_text.clone()).await;
                        session_text.push_str(&resp.corrected_text);
                    }
                }
                if resp.is_final {
                    llm_first_chunk = true;
                }
            }

            // ── Poll captured audio → feed ASR, process speech ───────
            _ = audio_poll_tick.tick() => {
                let Some(first_chunk) = audio_stream.try_next_chunk() else {
                    continue;
                };

                audio_batch.clear();
                audio_batch.extend_from_slice(&first_chunk.samples);
                while let Some(extra) = audio_stream.try_next_chunk() {
                    audio_batch.extend_from_slice(&extra.samples);
                    if audio_batch.len() >= 16_000 * 2 {
                        break;
                    }
                }

                // TEN VAD is the sole speech authority — no energy pre-gate.
                let (batch_has_speech, _energy) = vad.process_chunk(&audio_batch);

                if batch_has_speech {
                    asr_push_hangover_samples = ASR_PUSH_HANGOVER_SAMPLES;
                } else {
                    asr_push_hangover_samples =
                        asr_push_hangover_samples.saturating_sub(audio_batch.len());
                }
                let should_push_audio = batch_has_speech || asr_push_hangover_samples > 0;

                if should_push_audio
                    && let Err(e) = asr.push_audio(&asr_session, &audio_batch).await
                {
                    warn!(error = %e, "push_audio error");
                    continue;
                }

                // Shared pipeline gate: only poll for audio actually sent to ASR.
                if should_push_audio {
                    samples_since_last_poll += audio_batch.len();
                }
                had_speech_since_last_poll |= batch_has_speech;

                let segments = if samples_since_last_poll >= MIN_SAMPLES_BEFORE_POLL {
                    let should_poll = had_speech_since_last_poll || should_push_audio;
                    samples_since_last_poll = 0;
                    had_speech_since_last_poll = false;

                    if should_poll {
                        let poll_start = std::time::Instant::now();
                        match asr.poll_results(&asr_session).await {
                            Ok(s) => {
                                debug!(
                                    elapsed_ms = poll_start.elapsed().as_millis(),
                                    segment_count = s.len(),
                                    "poll_results"
                                );
                                s
                            }
                            Err(e) => { warn!(error = %e, "poll_results error"); continue; }
                        }
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };

                if !segments.is_empty() {
                    for seg in &segments {
                        let route = router.route_full(&seg.text);
                        let asr_text = if route.commands.is_empty() {
                            seg.text.as_str()
                        } else {
                            route.remaining.as_str()
                        };
                        let remaining = asr_text.trim();
                        for cmd in &route.commands {
                            if let lune_core::command::VoiceCommand::SwitchStyle(style_id) = cmd {
                                if let Some(next_preset) = builtin_registry().get(style_id) {
                                    style_preset = next_preset;
                                    info!(
                                        %session_id,
                                        style = %style_preset.id,
                                        "switched style preset via voice command"
                                    );
                                } else {
                                    warn!(
                                        %session_id,
                                        style = %style_id,
                                        "unknown style preset in voice command"
                                    );
                                }
                            }
                            let payload = serde_json::to_string(cmd)
                                .unwrap_or_else(|_| format!("{cmd:?}"));
                            let _ = event_tx
                                .try_send(SessionEvent::CommandDetected(payload));
                        }

                        if seg.is_final {
                            if !remaining.is_empty() {
                                // Two-level confidence gating: sentence mean AND
                                // min-token check (arXiv:2407.21414). High-confidence
                                // segments bypass LLM to avoid overcorrection.
                                let gate_pass = config
                                    .confidence_threshold
                                    .is_some_and(|t| {
                                        let min_t = config.min_token_threshold.unwrap_or(0.0);
                                        seg.is_confident_gated(t, min_t)
                                    });

                                if gate_pass {
                                    let applied = {
                                        let guard = dict.lock().await;
                                        guard.apply(remaining)
                                    };
                                    debug!(
                                        confidence = ?seg.confidence,
                                        text = %applied,
                                        "confidence gate: bypassing LLM"
                                    );
                                    eprintln!(
                                        "\x1b[36m[GATE]\x1b[0m conf={:.3} bypass LLM: {applied}",
                                        seg.confidence.unwrap_or(1.0)
                                    );
                                    commit_final_text(
                                        &mut output_ctrl,
                                        &mut session_text,
                                        &applied,
                                    )
                                    .await;
                                } else {
                                    let flushed = flush_to_llm_if_new(
                                        remaining,
                                        &dict,
                                        &style_preset,
                                        &llm_tx,
                                        &mut last_flushed_text,
                                        &mut last_flush_at,
                                    )
                                    .await;
                                    sent_to_llm |= flushed;
                                }
                            }
                            pending_nonfinal.clear();
                        } else {
                            partial_buf.clear();
                            partial_buf.push_str(&session_text);
                            if !remaining.is_empty() {
                                if !partial_buf.is_empty() { partial_buf.push(' '); }
                                partial_buf.push_str(remaining);
                                pending_nonfinal.clear();
                                pending_nonfinal.push_str(remaining);
                            } else {
                                pending_nonfinal.clear();
                            }
                            let _ = event_tx.try_send(SessionEvent::PartialText(partial_buf.clone()));
                        }
                    }
                }

                // ── VAD end-of-speech flush ──────────────────────
                // Use VAD (not energy) for speech state tracking so
                // background noise can't keep was_speaking stuck on.
                if batch_has_speech {
                    silence_samples = 0;
                    was_speaking = true;
                } else if was_speaking {
                    silence_samples += audio_batch.len();
                    if silence_samples >= SILENCE_FLUSH_SAMPLES {
                        if !pending_nonfinal.is_empty() {
                            debug!(%session_id, "VAD end-of-speech flush");
                            let flushed = flush_to_llm_if_new(
                                &pending_nonfinal,
                                &dict,
                                &style_preset,
                                &llm_tx,
                                &mut last_flushed_text,
                                &mut last_flush_at,
                            )
                            .await;
                            sent_to_llm |= flushed;
                            pending_nonfinal.clear();
                            if reset_asr_context_on_flush
                                && let Err(e) = asr.reset_context(&asr_session).await
                            {
                                warn!(error = %e, "reset_context error");
                            }
                        }
                        // Always reset — even if nothing to flush — so VAD
                        // inference stops running on every tick after speech.
                        was_speaking = false;
                        silence_samples = 0;
                    }
                }
            }
        }
    }

    // ── Teardown ──────────────────────────────────────────────────────────

    // Drain any remaining captured audio through ASR to capture the last word(s).
    while let Some(chunk) = audio_stream.try_next_chunk() {
        let _ = asr.push_audio(&asr_session, &chunk.samples).await;
    }
    if let Ok(segments) = asr.poll_results(&asr_session).await {
        for seg in &segments {
            let route = router.route_full(&seg.text);
            let asr_text = if route.commands.is_empty() {
                seg.text.as_str()
            } else {
                route.remaining.as_str()
            };
            let text = asr_text.trim();

            for cmd in &route.commands {
                let payload = serde_json::to_string(cmd).unwrap_or_else(|_| format!("{cmd:?}"));
                let _ = event_tx.try_send(SessionEvent::CommandDetected(payload));
            }

            if !text.is_empty() {
                if seg.is_final {
                    let gate_pass = config
                        .confidence_threshold
                        .is_some_and(|t| seg.is_confident(t));

                    if gate_pass {
                        let applied = {
                            let guard = dict.lock().await;
                            guard.apply(text)
                        };
                        commit_final_text(&mut output_ctrl, &mut session_text, &applied).await;
                    } else {
                        let flushed = flush_to_llm_if_new(
                            text,
                            &dict,
                            &style_preset,
                            &llm_tx,
                            &mut last_flushed_text,
                            &mut last_flush_at,
                        )
                        .await;
                        sent_to_llm |= flushed;
                    }
                } else {
                    pending_nonfinal.clear();
                    pending_nonfinal.push_str(text);
                }
            }
        }
    }

    if !pending_nonfinal.is_empty() {
        let flushed = flush_to_llm_if_new(
            &pending_nonfinal,
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
        )
        .await;
        pending_nonfinal.clear();
        if flushed {
            sent_to_llm = true;
            counters.teardown_forced_flushes = counters.teardown_forced_flushes.saturating_add(1);
        }
    }

    // Drop sender so the LLM worker knows no more requests are coming.
    drop(llm_tx);

    if sent_to_llm {
        let _ = tokio::time::timeout(TEARDOWN_LLM_TIMEOUT, async {
            while let Some(resp) = llm_rx.recv().await {
                if !resp.corrected_text.is_empty() {
                    if llm_first_chunk {
                        commit_final_text(
                            &mut output_ctrl,
                            &mut session_text,
                            &resp.corrected_text,
                        )
                        .await;
                        llm_first_chunk = false;
                    } else {
                        output_ctrl.emit_final(resp.corrected_text.clone()).await;
                        session_text.push_str(&resp.corrected_text);
                    }
                }
                if resp.is_final {
                    llm_first_chunk = true;
                }
            }
        })
        .await;
    }

    if let Err(e) = asr.end_session(asr_session).await {
        warn!(error = %e, "end_session error");
        status = SessionStatus::Failed(e.to_string());
    }

    let transcript = if session_text.is_empty() {
        None
    } else {
        Some(session_text)
    };
    finalize_db(session_id, &persist, status, Some(Utc::now()), transcript).await;

    info!(
        %session_id,
        teardown_forced_flushes = counters.teardown_forced_flushes,
        "session processing counters"
    );

    let _ = event_tx.send(SessionEvent::Stopped).await;
    active_arc.lock().await.take();
    info!(%session_id, "session stopped");
}

/// Forward ASR text to the LLM worker without ASR-side post-processing.
async fn flush_to_llm(
    utterance: &str,
    dict: &Mutex<DictionaryManager>,
    style_preset: &Arc<StylePreset>,
    llm_tx: &mpsc::Sender<LlmRequest>,
) {
    let (text, dictionary) = {
        let guard = dict.lock().await;
        let applied = guard.apply(utterance);
        let relevant = guard
            .top_relevant(&applied, 20)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        (applied, relevant)
    };
    let req = LlmRequest {
        text,
        style_preset: Arc::clone(style_preset),
        dictionary,
    };
    if let Err(e) = llm_tx.send(req).await {
        warn!(error = %e, "LLM worker unavailable");
    }
}

async fn flush_to_llm_if_new(
    utterance: &str,
    dict: &Mutex<DictionaryManager>,
    style_preset: &Arc<StylePreset>,
    llm_tx: &mpsc::Sender<LlmRequest>,
    last_flushed_text: &mut String,
    last_flush_at: &mut tokio::time::Instant,
) -> bool {
    let normalized = normalize_whitespace(utterance);
    if normalized.is_empty() {
        return false;
    }
    let now = tokio::time::Instant::now();
    if normalized == *last_flushed_text
        && now.duration_since(*last_flush_at) <= DUPLICATE_FLUSH_WINDOW
    {
        debug!(text = %normalized, "suppressing duplicate ASR flush");
        return false;
    }

    flush_to_llm(&normalized, dict, style_preset, llm_tx).await;
    *last_flushed_text = normalized;
    *last_flush_at = now;
    true
}

#[inline]
fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Commit corrected text without ever emitting a standalone leading space token.
async fn commit_final_text(
    output_ctrl: &mut TextOutputController,
    session_text: &mut String,
    corrected_text: &str,
) {
    let trimmed = corrected_text.trim_start();
    if trimmed.is_empty() {
        return;
    }

    let needs_space = !session_text.is_empty();
    let mut payload = String::with_capacity(trimmed.len() + usize::from(needs_space));
    if needs_space {
        payload.push(' ');
    }
    payload.push_str(trimmed);

    output_ctrl.emit_final(payload.clone()).await;
    session_text.push_str(&payload);
}

#[cold]
async fn finalize_db(
    session_id: Uuid,
    persist: &Mutex<Database>,
    status: SessionStatus,
    ended_at: Option<chrono::DateTime<Utc>>,
    transcript: Option<String>,
) {
    let mut db = persist.lock().await;
    match db.get_session(session_id) {
        Ok(Some(mut s)) => {
            s.status = status;
            s.ended_at = ended_at;
            s.transcript = transcript;
            if let Err(e) = db.update_session(&s) {
                warn!(error = %e, %session_id, "failed to update session in DB");
            }
        }
        Ok(None) => warn!(%session_id, "session not found in DB during finalize"),
        Err(e) => warn!(error = %e, %session_id, "DB error during finalize"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::asr::{AsrSegment, AsrSessionConfig};
    use lune_core::llm::{PostProcessInput, PostProcessOutput};
    use tokio::sync::mpsc;

    struct NoopAsr;

    #[async_trait::async_trait]
    impl AsrBackend for NoopAsr {
        type SessionHandle = ();

        async fn start_session(&self, _: AsrSessionConfig) -> Result<()> {
            Ok(())
        }
        async fn push_audio(&self, _: &(), _: &[f32]) -> Result<()> {
            Ok(())
        }
        async fn poll_results(&self, _: &()) -> Result<Vec<AsrSegment>> {
            Ok(vec![])
        }
        async fn end_session(&self, _: ()) -> Result<()> {
            Ok(())
        }
    }

    struct EchoPostProcessor;

    #[async_trait::async_trait]
    impl TextPostProcessor for EchoPostProcessor {
        async fn transform(&self, input: &PostProcessInput) -> Result<PostProcessOutput> {
            Ok(PostProcessOutput {
                cleaned_text: input.raw_text.clone(),
                metadata: None,
            })
        }
    }

    fn make_manager() -> SessionManager<NoopAsr> {
        let db = lune_persist::Database::open_in_memory().unwrap();
        let dict =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        let settings = Arc::new(RwLock::new(lune_core::config::UserSettings::default()));

        SessionManager::new(
            Arc::new(NoopAsr),
            Arc::new(VoiceCommandRouter::new()),
            Arc::new(EchoPostProcessor),
            Arc::new(Mutex::new(dict)),
            Arc::new(Mutex::new(db)),
            SessionManagerConfig::default(),
            settings,
        )
    }

    fn local_set() -> tokio::task::LocalSet {
        tokio::task::LocalSet::new()
    }

    #[tokio::test]
    async fn start_returns_session_id() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx, _rx) = mpsc::channel(16);
                let id = mgr.start(tx).await.unwrap();
                assert!(!id.is_nil());
                mgr.stop().await;
            })
            .await;
    }

    #[tokio::test]
    async fn double_start_errors() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx1, _rx1) = mpsc::channel(16);
                let (tx2, _rx2) = mpsc::channel(16);
                mgr.start(tx1).await.unwrap();
                let err = mgr.start(tx2).await;
                assert!(err.is_err());
                mgr.stop().await;
            })
            .await;
    }

    #[tokio::test]
    async fn stop_clears_active() {
        local_set()
            .run_until(async {
                let mgr = make_manager();
                let (tx, _rx) = mpsc::channel(16);
                mgr.start(tx).await.unwrap();
                mgr.stop().await;
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                assert!(mgr.active_session_id().await.is_none());
            })
            .await;
    }

    #[tokio::test]
    async fn flush_to_llm_applies_dictionary_and_includes_relevant_entries() {
        let mut dict_mgr =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        dict_mgr.insert("lune voice", "Lune Voice", 10).unwrap();
        let dict = Mutex::new(dict_mgr);

        let (llm_tx, mut llm_rx) = mpsc::channel(1);
        let style_preset = Arc::new(StylePreset::general());

        flush_to_llm("i use lune voice daily", &dict, &style_preset, &llm_tx).await;

        let req = llm_rx.try_recv().expect("request should be enqueued");
        assert_eq!(req.text, "i use Lune Voice daily");
        assert_eq!(req.dictionary.len(), 1);
        assert_eq!(req.dictionary[0].phrase, "lune voice");
        assert_eq!(req.dictionary[0].replacement, "Lune Voice");
    }

    #[tokio::test]
    async fn commit_final_text_uses_single_separator_between_commits() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut output_ctrl = TextOutputController::new(tx);
        let mut session_text = String::new();

        commit_final_text(&mut output_ctrl, &mut session_text, "Hello").await;
        commit_final_text(&mut output_ctrl, &mut session_text, "world").await;

        let first = rx.try_recv().expect("first final event");
        let second = rx.try_recv().expect("second final event");
        assert!(matches!(first, SessionEvent::FinalizedText(t) if t == "Hello"));
        assert!(matches!(second, SessionEvent::FinalizedText(t) if t == " world"));
        assert_eq!(session_text, "Hello world");
    }

    #[tokio::test]
    async fn flush_to_llm_if_new_suppresses_duplicate_within_window() {
        let dict =
            lune_dict::DictionaryManager::new(lune_persist::Database::open_in_memory().unwrap())
                .unwrap();
        let dict = Mutex::new(dict);
        let style_preset = Arc::new(StylePreset::general());
        let (llm_tx, mut llm_rx) = mpsc::channel(4);
        let mut last_flushed_text = String::new();
        let mut last_flush_at = tokio::time::Instant::now() - DUPLICATE_FLUSH_WINDOW;

        let first = flush_to_llm_if_new(
            "hello there",
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
        )
        .await;
        let second = flush_to_llm_if_new(
            "hello there",
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
        )
        .await;

        assert!(first);
        assert!(!second);
        let req = llm_rx.try_recv().expect("first request should be present");
        assert_eq!(req.text, "hello there");
        assert!(llm_rx.try_recv().is_err());

        last_flush_at =
            tokio::time::Instant::now() - DUPLICATE_FLUSH_WINDOW - Duration::from_millis(5);
        let third = flush_to_llm_if_new(
            "hello there",
            &dict,
            &style_preset,
            &llm_tx,
            &mut last_flushed_text,
            &mut last_flush_at,
        )
        .await;
        assert!(third);
    }

    #[test]
    fn normalize_whitespace_collapses_runs() {
        assert_eq!(
            normalize_whitespace("  hello   world \n test\t"),
            "hello world test"
        );
    }
}
