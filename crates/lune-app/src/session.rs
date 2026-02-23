use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use lune_audio::AudioCaptureService;
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

/// Number of words in pending non-final text before flushing to LLM.
const WORDS_PER_FLUSH: usize = 8;

/// After a flush, re-arm the silence timer with this shorter window so any
/// last-word audio that lands in the fresh ASR window is captured quickly.
const TAIL_SILENCE_TIMEOUT: Duration = Duration::from_millis(250);

/// Maximum dictionary entries to include in LLM context.
const TOP_RELEVANT_DICT_ENTRIES: usize = 20;

/// Timeout for waiting on final LLM response during teardown.
const TEARDOWN_LLM_TIMEOUT: Duration = Duration::from_secs(3);

/// Words that are pure filler/noise and should not be sent to the LLM alone.
const NOISE_FILLERS: &[&str] = &[
    "um", "uh", "mm", "m", "hmm", "hm", "ah", "er", "yeah", "ok", "okay",
];

/// Check if a single word (after stripping trailing punctuation) is a noise filler.
///
/// Length gate: all fillers are ≤4 bytes ("okay" is the longest). Words longer
/// than that are rejected without scanning the table — this is the common case
/// for real speech words.
#[inline]
fn is_noise_filler(word: &str) -> bool {
    let stripped = word.trim_end_matches(|c: char| c.is_ascii_punctuation());
    let len = stripped.len();
    // All fillers are 1–4 ASCII bytes; reject anything outside that range.
    len >= 1
        && len <= 4
        && NOISE_FILLERS
            .iter()
            .any(|&f| stripped.eq_ignore_ascii_case(f))
}

/// Returns true iff every whitespace-separated token in `text` is a noise filler.
///
/// Fast path: all fillers are ≤4 chars. With spaces between them, the longest
/// all-noise string for N words is 4*N + (N-1) bytes. If `text.len()` exceeds
/// the theoretical max for a reasonable filler count, we skip iteration entirely.
fn is_noise_only(text: &str) -> bool {
    // "okay okay okay okay" = 4 words, 19 bytes. Beyond ~25 bytes, it's
    // extremely unlikely to be all fillers — but we use a generous 30-byte
    // cutoff (covers up to 6 fillers with punctuation/spaces).
    if text.len() > 30 || text.is_empty() {
        return false;
    }
    text.split_whitespace().all(is_noise_filler)
}

/// Count whitespace-separated words without iterator allocation.
/// Scans bytes directly — valid because ASCII whitespace is single-byte
/// and we only care about word boundaries, not Unicode word segmentation.
#[inline]
fn word_count(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut count = 0u32;
    let mut in_word = false;
    for &b in bytes {
        let is_ws = b == b' ' || b == b'\t' || b == b'\n' || b == b'\r';
        if is_ws {
            in_word = false;
        } else if !in_word {
            in_word = true;
            count += 1;
        }
    }
    count as usize
}

/// Configuration for the session pipeline.
#[derive(Clone)]
pub struct SessionManagerConfig {
    /// Silence window before finalizing an utterance (default 800 ms).
    pub silence_timeout: Duration,
    pub asr_config: AsrSessionConfig,
    pub asr_backend_kind: AsrBackendKind,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            silence_timeout: Duration::from_millis(1200),
            asr_config: AsrSessionConfig::default(),
            asr_backend_kind: AsrBackendKind::ParakeetTdt,
        }
    }
}

struct ActiveHandle {
    stop_tx: oneshot::Sender<()>,
    session_id: Uuid,
}

/// Orchestrates the full dictation pipeline.
///
/// Only one session may be active at a time. Call `start()` to begin and
/// `stop()` to finalize gracefully. Events are delivered over the channel
/// passed to `start()`.
pub struct SessionManager<A: AsrBackend> {
    audio: Arc<AudioCaptureService>,
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
        audio: Arc<AudioCaptureService>,
        asr: Arc<A>,
        router: Arc<VoiceCommandRouter>,
        post_processor: Arc<dyn TextPostProcessor>,
        dict: Arc<Mutex<DictionaryManager>>,
        persist: Arc<Mutex<Database>>,
        config: SessionManagerConfig,
        settings: Arc<RwLock<UserSettings>>,
    ) -> Self {
        Self {
            audio,
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

        let current_style_id = self.settings.read().await.default_style.clone();
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

        let audio = Arc::clone(&self.audio);
        let asr = Arc::clone(&self.asr);
        let router = Arc::clone(&self.router);
        let post_processor = Arc::clone(&self.post_processor);
        let dict = Arc::clone(&self.dict);
        let persist = Arc::clone(&self.persist);
        let config = self.config.clone();
        let active_arc = Arc::clone(&self.active);

        tokio::task::spawn_local(run_session(
            session_id,
            audio,
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
    audio: Arc<AudioCaptureService>,
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
    let silence_timeout = config.silence_timeout;
    let mut style_preset = style_preset;

    let (llm_tx, mut llm_rx) = llm_worker::spawn_llm_worker(Arc::clone(&post_processor));

    // ── Start audio capture ───────────────────────────────────────────────
    let (audio_tx, mut audio_rx) = mpsc::channel::<lune_core::audio::AudioFrame>(1024);

    let _audio_handle = match audio.start_session(audio_tx) {
        Ok(h) => h,
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

    let _ = event_tx.send(SessionEvent::Started).await;

    let mut output_ctrl = TextOutputController::new(event_tx.clone());
    let mut session_text = String::with_capacity(512);
    let mut utterance = String::with_capacity(128);
    let mut pending_nonfinal = String::with_capacity(128);

    // Reused each audio frame to avoid allocation on the ~50Hz partial path.
    let mut partial_buf = String::with_capacity(512);
    // Cached prefix (session_text + " " + utterance) to avoid rebuilding it
    // every frame when only pending_nonfinal changes. Invalidated on flush/commit.
    let mut partial_prefix = String::with_capacity(512);
    let mut partial_prefix_valid = false;
    let mut audio_batch: Vec<f32> = Vec::with_capacity(4096);

    // Word-count accumulator state.
    let mut utterance_wc: usize = 0;
    let mut pending_wc: usize = 0;

    // Resettable silence timer: arms on every speech event, fires when
    // no speech is detected for `silence_timeout` ms.
    // Starts dormant (Duration::MAX) — armed only after first speech.
    let silence_dur = tokio::time::sleep(Duration::MAX);
    tokio::pin!(silence_dur);
    let mut silence_armed = false;
    // After a flush, the next silence window uses TAIL_SILENCE_TIMEOUT (short)
    // so any last-word audio lands quickly.  Reset to false when speech arrives.
    let mut tail_mode = false;

    // ── Main loop ─────────────────────────────────────────────────────────
    loop {
        tokio::select! {
            biased;

            _ = &mut stop_rx => {
                debug!(%session_id, "stop signal received");
                break;
            }

            // ── LLM complete → emit and commit to transcript ─────────
            Some(resp) = llm_rx.recv() => {
                if !resp.corrected_text.is_empty() {
                    if !session_text.is_empty() {
                        output_ctrl.emit_final(String::from(" ")).await;
                        session_text.push(' ');
                    }
                    output_ctrl.emit_final(resp.corrected_text.clone()).await;
                    session_text.push_str(&resp.corrected_text);
                    partial_prefix_valid = false;
                }
            }

            // ── Silence timer fired → flush utterance ─────────────────
            () = &mut silence_dur, if silence_armed => {
                silence_armed = false;
                let was_tail = std::mem::replace(&mut tail_mode, false);
                if !pending_nonfinal.is_empty() {
                    if !utterance.is_empty() { utterance.push(' '); }
                    utterance.push_str(&pending_nonfinal);
                    utterance_wc += pending_wc;
                    pending_nonfinal.clear();
                    pending_wc = 0;
                }
                if !utterance.is_empty() {
                    let noise = is_noise_only(&utterance);
                    if noise {
                        debug!(%session_id, text = %utterance, "silence: noise-only, discarding");
                    } else {
                        debug!(%session_id, tail = was_tail, "silence flush");
                        flush_to_llm(&utterance, &dict, &style_preset, &llm_tx).await;
                    }
                    utterance.clear();
                    utterance_wc = 0;
                    partial_prefix_valid = false;
                    if let Err(e) = asr.reset_context(&asr_session).await {
                        warn!(error = %e, "reset_context error");
                    }
                    // Real flush (not tail, not noise): arm a short window so any
                    // last-word audio that lands in the fresh window flushes fast.
                    if !was_tail && !noise {
                        tail_mode = true;
                        silence_dur.as_mut().reset(
                            tokio::time::Instant::now() + TAIL_SILENCE_TIMEOUT,
                        );
                        silence_armed = true;
                    }
                }
            }

            // ── Audio frame → feed ASR, process speech ────────────────
            frame = audio_rx.recv() => {
                let Some(frame) = frame else {
                    debug!(%session_id, "audio channel closed");
                    break;
                };

                audio_batch.clear();
                audio_batch.extend_from_slice(&frame.samples);
                while let Ok(extra) = audio_rx.try_recv() {
                    audio_batch.extend_from_slice(&extra.samples);
                }

                if let Err(e) = asr.push_audio(&asr_session, &audio_batch).await {
                    warn!(error = %e, "push_audio error");
                    continue;
                }

                let segments = match asr.poll_results(&asr_session).await {
                    Ok(s) => s,
                    Err(e) => { warn!(error = %e, "poll_results error"); continue; }
                };

                let mut had_speech = false;
                if !segments.is_empty() {
                    for seg in &segments {
                        let route = router.route_full(&seg.text);
                        let remaining = route.remaining.trim();
                        if !remaining.is_empty() || !route.commands.is_empty() {
                            had_speech = true;
                        }

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
                                if !utterance.is_empty() { utterance.push(' '); }
                                utterance.push_str(remaining);
                                utterance_wc += word_count(remaining);
                                partial_prefix_valid = false;
                            }
                            pending_nonfinal.clear();
                            pending_wc = 0;
                        } else {
                            // Rebuild the cached prefix only when session_text
                            // or utterance has changed (rare vs every frame).
                            if !partial_prefix_valid {
                                partial_prefix.clear();
                                partial_prefix.push_str(&session_text);
                                if !utterance.is_empty() {
                                    if !partial_prefix.is_empty() { partial_prefix.push(' '); }
                                    partial_prefix.push_str(&utterance);
                                }
                                partial_prefix_valid = true;
                            }
                            partial_buf.clear();
                            partial_buf.push_str(&partial_prefix);
                            if !remaining.is_empty() {
                                if !partial_buf.is_empty() { partial_buf.push(' '); }
                                partial_buf.push_str(remaining);
                                pending_nonfinal.clear();
                                pending_nonfinal.push_str(remaining);
                                pending_wc = word_count(remaining);
                            }
                            let _ = event_tx.try_send(SessionEvent::PartialText(partial_buf.clone()));
                        }
                    }
                }

                // ── Silence timer: reset on speech, arm if not already ─
                if had_speech {
                    // Real speech: exit tail mode — the user is still talking,
                    // normal silence_timeout applies from here.
                    tail_mode = false;
                    silence_dur.as_mut().reset(
                        tokio::time::Instant::now() + silence_timeout,
                    );
                    silence_armed = true;
                }

                // ── Word-count flush ───────────────────────────────────
                if utterance_wc + pending_wc >= WORDS_PER_FLUSH {
                    if !pending_nonfinal.is_empty() {
                        if !utterance.is_empty() { utterance.push(' '); }
                        utterance.push_str(&pending_nonfinal);
                        pending_nonfinal.clear();
                        pending_wc = 0;
                    }
                    let noise = is_noise_only(&utterance);
                    if noise {
                        debug!(%session_id, text = %utterance, "word-count flush: noise-only, discarding");
                    } else {
                        debug!(%session_id, word_count = utterance_wc, "word-count flush");
                        flush_to_llm(&utterance, &dict, &style_preset, &llm_tx).await;
                    }
                    utterance.clear();
                    utterance_wc = 0;
                    partial_prefix_valid = false;
                    if let Err(e) = asr.reset_context(&asr_session).await {
                        warn!(error = %e, "reset_context error");
                    }
                    // Arm short tail window: user may have said one more word
                    // right as the flush fired.
                    tail_mode = true;
                    silence_dur.as_mut().reset(
                        tokio::time::Instant::now() + TAIL_SILENCE_TIMEOUT,
                    );
                    silence_armed = true;
                }
            }
        }
    }

    // ── Teardown ──────────────────────────────────────────────────────────

    // Drain any remaining audio frames through ASR to capture the last word(s).
    // The user may have stopped dictation right after speaking — audio frames
    // could be buffered in the channel but not yet fed to ASR.
    while let Ok(frame) = audio_rx.try_recv() {
        let _ = asr.push_audio(&asr_session, &frame.samples).await;
    }
    if let Ok(segments) = asr.poll_results(&asr_session).await {
        for seg in &segments {
            let route = router.route_full(&seg.text);
            let text = route.remaining.trim();

            for cmd in &route.commands {
                let payload = serde_json::to_string(cmd).unwrap_or_else(|_| format!("{cmd:?}"));
                let _ = event_tx.try_send(SessionEvent::CommandDetected(payload));
            }

            if !text.is_empty() {
                if seg.is_final {
                    if !utterance.is_empty() {
                        utterance.push(' ');
                    }
                    utterance.push_str(text);
                } else {
                    pending_nonfinal.clear();
                    pending_nonfinal.push_str(text);
                }
            }
        }
    }

    if !pending_nonfinal.is_empty() {
        if !utterance.is_empty() {
            utterance.push(' ');
        }
        utterance.push_str(&pending_nonfinal);
    }

    // Send any remaining utterance to LLM (skip if noise-only).
    let has_final_flush = !utterance.is_empty() && !is_noise_only(&utterance);
    if has_final_flush {
        flush_to_llm(&utterance, &dict, &style_preset, &llm_tx).await;
    }

    // Drop sender so the LLM worker knows no more requests are coming.
    drop(llm_tx);

    if has_final_flush {
        let _ = tokio::time::timeout(TEARDOWN_LLM_TIMEOUT, async {
            while let Some(resp) = llm_rx.recv().await {
                if !resp.corrected_text.is_empty() {
                    if !session_text.is_empty() {
                        output_ctrl.emit_final(String::from(" ")).await;
                    }
                    output_ctrl.emit_final(resp.corrected_text.clone()).await;
                    if !session_text.is_empty() {
                        session_text.push(' ');
                    }
                    session_text.push_str(&resp.corrected_text);
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

    let _ = event_tx.send(SessionEvent::Stopped).await;
    active_arc.lock().await.take();
    info!(%session_id, "session stopped");
}

/// Build context string from the sliding window, apply dictionary, and send to LLM.
async fn flush_to_llm(
    utterance: &str,
    dict: &Mutex<DictionaryManager>,
    style_preset: &Arc<StylePreset>,
    llm_tx: &mpsc::Sender<LlmRequest>,
) {
    let (after_dict, relevant) = {
        let guard = dict.lock().await;
        let applied = guard.apply(utterance);
        let rel = guard
            .top_relevant(&applied, TOP_RELEVANT_DICT_ENTRIES)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        (applied, rel)
    };

    let req = LlmRequest {
        text: after_dict,
        style_preset: Arc::clone(style_preset),
        dictionary: relevant,
    };
    if let Err(e) = llm_tx.send(req).await {
        warn!(error = %e, "LLM worker unavailable");
    }
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
        let audio_config = lune_core::audio::AudioCaptureConfig::default();
        let settings = Arc::new(RwLock::new(lune_core::config::UserSettings::default()));

        SessionManager::new(
            Arc::new(AudioCaptureService::new(audio_config)),
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

    #[test]
    fn noise_detection_basics() {
        assert!(is_noise_filler("um"));
        assert!(is_noise_filler("Um"));
        assert!(is_noise_filler("UM"));
        assert!(is_noise_filler("uh"));
        assert!(is_noise_filler("er"));
        assert!(is_noise_filler("Er"));
        assert!(is_noise_filler("ER"));
        assert!(is_noise_filler("hmm"));
        assert!(is_noise_filler("ok"));
        assert!(is_noise_filler("okay"));
        assert!(is_noise_filler("yeah"));
        assert!(!is_noise_filler("hello"));
        assert!(!is_noise_filler("the"));
        assert!(!is_noise_filler(""));
        assert!(!is_noise_filler("umbrella"));
        assert!(!is_noise_filler("error"));
    }

    #[test]
    fn noise_filler_with_trailing_punct() {
        assert!(is_noise_filler("um,"));
        assert!(is_noise_filler("uh."));
        assert!(is_noise_filler("okay!"));
        assert!(is_noise_filler("hmm..."));
        assert!(is_noise_filler("er,"));
        assert!(!is_noise_filler("hello."));
    }

    #[test]
    fn noise_only_detection() {
        assert!(is_noise_only("um uh"));
        assert!(is_noise_only("hmm"));
        assert!(is_noise_only("ok yeah um"));
        assert!(is_noise_only("er um uh"));
        assert!(!is_noise_only("um hello"));
        assert!(!is_noise_only("the quick brown fox"));
        assert!(!is_noise_only(""));
    }

    #[test]
    fn word_count_examples() {
        assert_eq!(word_count(""), 0);
        assert_eq!(word_count("hello"), 1);
        assert_eq!(word_count("hello world"), 2);
        assert_eq!(word_count("  hello   world  "), 2);
        assert_eq!(word_count("a b c d e f g h"), 8);
    }

    #[test]
    fn word_count_whitespace_variants() {
        assert_eq!(word_count("a\tb\nc"), 3);
        assert_eq!(word_count("\t\n "), 0);
        assert_eq!(word_count(" a "), 1);
    }
}
