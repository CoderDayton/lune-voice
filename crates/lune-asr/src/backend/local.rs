// SPDX-License-Identifier: GPL-3.0-or-later

//! What the two on-device ggml backends — whisper.cpp ([`super::whisper_cpp`])
//! and Parakeet ([`super::parakeet_cpp`]) — have in common.
//!
//! Both decode a whole utterance at once from 16 kHz mono `f32`, both trim it
//! to its speech with the same standalone silero VAD, both report per-token
//! probabilities that fold into the same per-word confidences, and both load a
//! ggml file through a C++ loader that dies rather than reports on a malformed
//! one. Keeping those pieces here means a tuning change (a VAD threshold, the
//! word-confidence rule) reaches both engines at once instead of drifting.
//!
//! Without the `whisper-cpp` feature both engines are stubs and nothing here
//! is called outside its own tests.

#![cfg_attr(not(feature = "whisper-cpp"), allow(dead_code))]

use lune_core::asr::WordConfidence;
use lune_core::error::Result;

/// Sample rate both engines decode at (16 kHz mono `f32`).
pub(crate) const SAMPLE_RATE: usize = 16_000;

// ── Model file sanity check ─────────────────────────────────────────────────

/// On-disk magic of a ggml model file: `0x67676d6c` ("ggml") stored
/// little-endian, so the first four bytes read `l m g g`. Whisper and Parakeet
/// models share it.
pub(crate) const GGML_MAGIC: [u8; 4] = *b"lmgg";

/// Smallest plausible ggml model. Well under any real speech model, and only
/// there to reject an empty or barely-started file with a clear message.
pub(crate) const MIN_GGML_BYTES: u64 = 4096;

/// Reject a model file whisper.cpp would die on rather than report.
///
/// Its loaders raise a C++ exception or trip `GGML_ASSERT` on malformed input.
/// Neither can be caught from Rust — an exception crossing the `extern "C"`
/// boundary is undefined behaviour and the assert calls `abort()` — so the
/// process disappears with no dialog and no error. Validating first is the only
/// lever available.
///
/// This is a cheap sanity check, not an integrity check: the download path
/// already verifies a SHA-256 (`lune-persist`'s hub). What it catches is a file
/// truncated after that — a full disk, a crash mid-write, a hand-placed file.
pub fn check_ggml_file(path: &std::path::Path) -> Result<()> {
    use lune_core::error::Error;
    use std::io::Read;

    let meta = std::fs::metadata(path).map_err(|e| {
        Error::asr(format!(
            "cannot read the model file {}: {e}",
            path.display()
        ))
    })?;
    if meta.len() < MIN_GGML_BYTES {
        return Err(Error::asr(format!(
            "the model file {} is only {} bytes and cannot be a speech model — \
             it is most likely a partial download; delete it and download the model again",
            path.display(),
            meta.len()
        )));
    }
    let mut head = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut head))
        .map_err(|e| {
            Error::asr(format!(
                "cannot read the model file {}: {e}",
                path.display()
            ))
        })?;
    if head != GGML_MAGIC {
        return Err(Error::asr(format!(
            "the model file {} is not in ggml format — delete it and download the model again",
            path.display()
        )));
    }
    Ok(())
}

// ── Per-word confidence ─────────────────────────────────────────────────────

/// Folds a decode's tokens into per-word confidences and the segment-level
/// mean and minimum.
///
/// A word's `confidence` is its weakest token, so one garbled token flags the
/// whole word; `mean_confidence` is the mean over the same tokens, so a
/// downstream consumer can tell a uniformly risky word from one outlier token
/// (a glued punctuation token, typically) dragging a fine word down — see
/// [`WordConfidence::mean_confidence`]. Whitespace-only words are dropped.
///
/// Feeds the corrector's risk gating, which protects uncertain words and drives
/// `unresolved_words`; leaving the per-word list empty makes that gating inert.
#[derive(Debug)]
pub(crate) struct WordConfidences {
    words: Vec<WordConfidence>,
    word: String,
    word_min: f32,
    word_sum: f32,
    word_count: u32,
    total_sum: f32,
    total_min: f32,
    total_count: u32,
}

impl WordConfidences {
    pub(crate) fn new() -> Self {
        Self {
            words: Vec::new(),
            word: String::new(),
            word_min: 1.0,
            word_sum: 0.0,
            word_count: 0,
            total_sum: 0.0,
            total_min: 1.0,
            total_count: 0,
        }
    }

    /// Add one decoded token: its text and its probability. `starts_word`
    /// closes the word in progress first — whisper marks that with a leading
    /// space, Parakeet with SentencePiece's word-start flag. A probability
    /// outside `0..=1` (NaN included) is clamped rather than trusted.
    pub(crate) fn push(&mut self, piece: &str, probability: f32, starts_word: bool) {
        let p = if probability.is_nan() {
            0.0
        } else {
            probability.clamp(0.0, 1.0)
        };
        self.total_sum += p;
        self.total_min = self.total_min.min(p);
        self.total_count += 1;
        if starts_word {
            self.flush();
        }
        self.word.push_str(piece);
        self.word_min = self.word_min.min(p);
        self.word_sum += p;
        self.word_count += 1;
    }

    fn flush(&mut self) {
        let trimmed = self.word.trim();
        if !trimmed.is_empty() {
            let confidence = self.word_min;
            let mean_confidence = if self.word_count > 0 {
                self.word_sum / self.word_count as f32
            } else {
                confidence
            };
            self.words.push(WordConfidence {
                word: trimmed.to_string(),
                confidence,
                word_error_risk: 1.0 - confidence,
                mean_confidence,
            });
        }
        self.word.clear();
        self.word_min = 1.0;
        self.word_sum = 0.0;
        self.word_count = 0;
    }

    /// The finished words, plus the segment's mean and minimum token
    /// probability — both `None` when no token was pushed.
    pub(crate) fn finish(mut self) -> (Vec<WordConfidence>, Option<f32>, Option<f32>) {
        self.flush();
        if self.total_count == 0 {
            return (self.words, None, None);
        }
        let mean = self.total_sum / self.total_count as f32;
        (self.words, Some(mean), Some(self.total_min))
    }
}

/// Refuse to load a model when ggml has no CPU device to run it on.
///
/// Only a broken plugin build gets here — its bundled `backends/` directory
/// missing or unreadable — but there both native loaders throw a C++
/// exception from their state setup, which aborts the process with no
/// message. Checked first, it becomes an error the app can show.
#[cfg(feature = "whisper-cpp")]
pub(crate) fn ensure_cpu_backend() -> Result<()> {
    if crate::gpu_backend::cpu_device_registered() {
        return Ok(());
    }
    Err(lune_core::error::Error::asr(
        "no CPU compute backend is loaded, so no speech model can run — the \
         app's backends folder is missing or damaged; reinstall Lune Voice",
    ))
}

// ── Chunking at pauses ──────────────────────────────────────────────────────

/// Ranges of a `total`-sample clip to decode one by one, in order and with no
/// gap between them.
///
/// `speech` holds the VAD speech spans in clip samples. A clip longer than
/// `max` is cut in the middle of a pause between two spans, so each piece
/// keeps its own pauses whole. Two spans that touch count: the VAD pads each
/// span, which closes a short pause. Among the pauses that leave both sides at
/// least `min`, one in the back half of the window wins, then the longest,
/// then the latest. A stretch with no usable pause stays in one piece.
///
/// `lead` samples are decoded in front of the first piece and count against
/// its window.
pub(crate) fn plan_chunks_within(
    speech: &[(usize, usize)],
    total: usize,
    lead: usize,
    max: usize,
    min: usize,
) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut start = 0;
    loop {
        let room = max.saturating_sub(if start == 0 { lead } else { 0 });
        if total - start <= room {
            break;
        }
        let cut = speech
            .windows(2)
            .filter(|pair| pair[1].0 >= pair[0].1)
            .map(|pair| {
                let width = pair[1].0 - pair[0].1;
                (pair[0].1 + width / 2, width)
            })
            .filter(|&(mid, _)| mid >= start + min && mid <= start + room && mid + min <= total)
            .max_by_key(|&(mid, width)| (mid - start >= room / 2, width, mid));
        let Some((mid, _)) = cut else {
            break;
        };
        chunks.push((start, mid));
        start = mid;
    }
    chunks.push((start, total));
    chunks
}

// ── Endpoint trimming ───────────────────────────────────────────────────────

/// Padding kept around the VAD speech span when endpoint-trimming, so soft
/// onsets/offsets aren't clipped. Asymmetric on purpose: the onset gets more
/// lead-in than the tail — a bit of silence before the first word lets the
/// decoder anchor the opening phrase, which measurably recovers softly-spoken
/// opening words (A/B on whisper: recovered a dropped opening word across
/// quiet/loud and with pre-roll lead-in). The tail keeps the original margin;
/// the producer-side flush in `capture.rs` already guards the very last
/// phonemes.
const VAD_PAD_LEAD_MS: usize = 400;
const VAD_PAD_TAIL_MS: usize = 240;

/// silero VAD probability threshold. Below whisper.cpp's 0.5 default so a soft
/// first-word onset (fricatives/plosives, or audio still inside the AGC
/// warm-up) is detected earlier and not clipped mid-word.
const VAD_THRESHOLD: f32 = 0.35;

/// Minimum voiced duration for a VAD segment (ms). Below whisper.cpp's 250
/// default so short commands ("yes", "no", "stop", "undo") aren't discarded
/// wholesale — that path returns zero segments and drops the whole utterance.
const VAD_MIN_SPEECH_MS: i32 = 100;

/// Padding the VAD itself keeps around each segment (ms). Raised from the 30
/// default for extra onset/offset lead-in; stacks with the pads above.
const VAD_SPEECH_PAD_MS: i32 = 120;

/// The stretch of a buffer that holds its speech.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpeechSpan {
    /// First sample to decode: the first onset less the lead pad.
    pub start: usize,
    /// One past the last sample to decode: the last offset plus the tail pad.
    pub end: usize,
    /// The first onset itself, before padding.
    pub onset: usize,
    /// Every VAD speech span, in buffer samples.
    pub spans: Vec<(usize, usize)>,
}

/// What endpoint trimming found in a buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VadTrim {
    /// Decode `start..end` of the buffer.
    Speech(SpeechSpan),
    /// The VAD heard nothing: there is nothing to decode.
    Silence,
    /// No VAD, or it failed: decode the whole buffer.
    Full,
}

/// The part of a buffer a decode runs on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodeSlice {
    pub start: usize,
    pub end: usize,
    /// Where speech begins, in buffer samples. `None` when no VAD ran, so the
    /// onset is unknown — not the same as speech at sample zero.
    pub onset: Option<usize>,
    /// Every VAD speech span, in buffer samples; empty without a VAD.
    pub spans: Vec<(usize, usize)>,
}

impl DecodeSlice {
    /// The speech spans re-based onto the slice: relative to `start` and
    /// clamped to its length, ready to hand to a chunk planner.
    pub(crate) fn relative_spans(&self) -> Vec<(usize, usize)> {
        let len = self.end - self.start;
        self.spans
            .iter()
            .map(|&(from, to)| {
                (
                    from.saturating_sub(self.start).min(len),
                    to.saturating_sub(self.start).min(len),
                )
            })
            .collect()
    }

    /// Where the slice starts in the buffer, in milliseconds, so timestamps
    /// relative to the slice can be put back onto the buffer the caller pushed.
    pub(crate) fn offset_ms(&self) -> u64 {
        (self.start as u64 * 1000) / SAMPLE_RATE as u64
    }
}

impl VadTrim {
    /// The slice to decode out of a `len`-sample buffer, or `None` for silence.
    /// A degenerate span (start at or past end) falls back to the whole buffer
    /// rather than dropping the utterance.
    pub(crate) fn decode_slice(self, len: usize) -> Option<DecodeSlice> {
        match self {
            Self::Silence => None,
            Self::Speech(span) if span.start < span.end => Some(DecodeSlice {
                start: span.start,
                end: span.end,
                onset: Some(span.onset),
                spans: span.spans,
            }),
            Self::Speech(_) | Self::Full => Some(DecodeSlice {
                start: 0,
                end: len,
                onset: None,
                spans: Vec::new(),
            }),
        }
    }
}

/// Turn VAD segment times (centiseconds, as whisper.cpp's VAD reports them)
/// into the padded span of a `len`-sample buffer to decode. `None` when there
/// are no segments.
pub(crate) fn speech_span_from_segments(segments: &[(f32, f32)], len: usize) -> Option<SpeechSpan> {
    let (first, last) = (segments.first()?, segments.last()?);
    let to_sample = |cs: f32| (cs * SAMPLE_RATE as f32 / 100.0) as isize;
    let pad_lead = (VAD_PAD_LEAD_MS * SAMPLE_RATE / 1000) as isize;
    let pad_tail = (VAD_PAD_TAIL_MS * SAMPLE_RATE / 1000) as isize;
    let onset = to_sample(first.0).max(0) as usize;
    let start = (to_sample(first.0) - pad_lead).max(0) as usize;
    let end = ((to_sample(last.1) + pad_tail).max(0) as usize).min(len);
    let spans = segments
        .iter()
        .map(|&(from, to)| {
            (
                (to_sample(from).max(0) as usize).min(len),
                (to_sample(to).max(0) as usize).min(len),
            )
        })
        .collect();
    Some(SpeechSpan {
        start: start.min(len),
        end,
        onset: onset.min(len),
        spans,
    })
}

#[cfg(feature = "whisper-cpp")]
pub(crate) use vad::EndpointVad;

#[cfg(feature = "whisper-cpp")]
mod vad {
    use super::{
        VAD_MIN_SPEECH_MS, VAD_SPEECH_PAD_MS, VAD_THRESHOLD, VadTrim, check_ggml_file,
        speech_span_from_segments,
    };
    use std::path::Path;
    use std::sync::Mutex;
    use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

    /// whisper.cpp's standalone silero VAD, used to cut a buffer down to
    /// [first speech onset .. last offset] plus padding as ONE contiguous
    /// slice. That drops leading/trailing silence (less hallucination; anchors
    /// the decode at the first word) while preserving every INTERNAL pause —
    /// the cue both decoders use for punctuation. whisper's in-decode
    /// `enable_vad` is deliberately not used: it concatenates speech spans and
    /// strips those pauses, which reads back as run-on sentences.
    pub(crate) struct EndpointVad(Mutex<WhisperVadContext>);

    impl EndpointVad {
        /// Load the VAD at `path`. A missing or unusable model degrades to
        /// decoding the whole buffer, with a warning, rather than failing the
        /// backend: losing the trim is survivable.
        pub(crate) fn load(path: &Path, n_threads: i32) -> Option<Self> {
            // Same loader, same failure mode as the main model: a truncated
            // silero file aborts the process rather than returning an error,
            // so validate before handing over the path.
            if let Err(e) = check_ggml_file(path) {
                tracing::warn!(error = %e, "VAD model unusable; decoding without endpoint trim");
                return None;
            }
            let mut params = WhisperVadContextParams::default();
            // Always on the CPU: silero is tiny (no GPU benefit) and standing
            // up a second GPU context at init is a crash source.
            params.set_use_gpu(false);
            params.set_n_threads(n_threads);
            match WhisperVadContext::new(&path.to_string_lossy(), params) {
                Ok(vad) => Some(Self(Mutex::new(vad))),
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load the VAD; decoding without endpoint trim");
                    None
                }
            }
        }

        /// Find the speech in `audio`.
        pub(crate) fn trim(&self, audio: &[f32]) -> VadTrim {
            // Recover from a poisoned lock rather than silently decoding
            // without endpoint trimming for the rest of the process. The VAD
            // context is reset on every use, so a panic in a previous decode
            // leaves nothing behind worth protecting — and losing the trim
            // quietly shows up as hallucinated leading silence, which is very
            // hard to trace back to here.
            let mut vad = self.0.lock().unwrap_or_else(|e| e.into_inner());
            let mut params = WhisperVadParams::default();
            params.set_threshold(VAD_THRESHOLD);
            params.set_min_speech_duration(VAD_MIN_SPEECH_MS);
            params.set_speech_pad(VAD_SPEECH_PAD_MS);
            match vad.segments_from_samples(params, audio) {
                Ok(segs) => {
                    let times: Vec<(f32, f32)> = (0..segs.num_segments())
                        .map(|i| {
                            (
                                segs.get_segment_start_timestamp(i).unwrap_or(0.0),
                                segs.get_segment_end_timestamp(i).unwrap_or(0.0),
                            )
                        })
                        .collect();
                    match speech_span_from_segments(&times, audio.len()) {
                        Some(span) => VadTrim::Speech(span),
                        None => VadTrim::Silence,
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "VAD failed; decoding the full buffer");
                    VadTrim::Full
                }
            }
        }
    }
}

// ── Native logging ──────────────────────────────────────────────────────────

/// Route whisper.cpp, Parakeet and ggml native logs into `tracing` instead of
/// raw stderr, so they obey the global log filter.
///
/// Idempotent, and must run before anything that can make ggml talk —
/// `gpu_backend::load_plugins` above all, whose `dlopen` failures are the only
/// account of why a GPU plugin did not load. On a GUI launch raw stderr goes
/// nowhere, so a diagnostic emitted before this call is lost outright.
///
/// Parakeet needs its own hook, and not only for its own messages: loading a
/// Parakeet model re-points ggml's process-global log callback at Parakeet's
/// (`parakeet_backend_init_gpu` calls `ggml_log_set` with it). Left at its
/// stderr default, that would silently undo the ggml hook for whisper too.
#[cfg(feature = "whisper-cpp")]
pub fn install_logging_hooks() {
    static LOG_HOOK: std::sync::Once = std::sync::Once::new();
    LOG_HOOK.call_once(|| {
        whisper_rs::install_logging_hooks();
        // SAFETY: `parakeet_log_set` stores the callback and user-data pointer
        // in a process-global and forwards them to `ggml_log_set`. The callback
        // is a `'static` fn and the user data is null, so neither can dangle.
        unsafe {
            whisper_rs::whisper_rs_sys::parakeet_log_set(
                Some(native_log_trampoline),
                std::ptr::null_mut(),
            );
        }
    });
}

/// No-op: the stub build links no native decoder, so there is nothing to route.
#[cfg(not(feature = "whisper-cpp"))]
pub fn install_logging_hooks() {}

/// Forwards one native log line to `tracing`, at the level it was logged at.
///
/// Logged under the `ggml` target rather than `parakeet`: `parakeet_log_set`
/// installs this as ggml's process-global callback, so it also carries every
/// ggml line whisper and the GPU plugin loader emit.
#[cfg(feature = "whisper-cpp")]
unsafe extern "C" fn native_log_trampoline(
    level: whisper_rs::whisper_rs_sys::ggml_log_level,
    text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    use whisper_rs::whisper_rs_sys as sys;
    if text.is_null() {
        return;
    }
    // SAFETY: ggml and Parakeet pass a NUL-terminated string that outlives
    // this call; it is copied out before returning.
    let text = unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy();
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    #[allow(non_upper_case_globals)]
    match level {
        sys::ggml_log_level_GGML_LOG_LEVEL_ERROR => tracing::error!(target: "ggml", "{text}"),
        sys::ggml_log_level_GGML_LOG_LEVEL_WARN => tracing::warn!(target: "ggml", "{text}"),
        sys::ggml_log_level_GGML_LOG_LEVEL_INFO => tracing::info!(target: "ggml", "{text}"),
        sys::ggml_log_level_GGML_LOG_LEVEL_DEBUG => tracing::debug!(target: "ggml", "{text}"),
        _ => tracing::trace!(target: "ggml", "{text}"),
    }
}

// ── Threading ───────────────────────────────────────────────────────────────

/// Decode-thread cap. Both decoders scale sub-linearly past a handful of
/// threads, and the audio and UI threads must not be starved.
const MAX_THREADS: usize = 8;

/// How many threads a decode uses on this machine.
pub(crate) fn decode_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get().min(MAX_THREADS) as i32)
        .unwrap_or(4)
}

/// Run CPU/GPU-heavy decode work without stalling the async runtime.
///
/// Lune Voice runs on Tauri's multi-thread tokio runtime, so `block_in_place`
/// hands the worker off to the blocking pool and other tasks keep running
/// during the multi-second decode. There is no `spawn_blocking` here because a
/// decoder session holds raw C pointers and is pinned to the thread that holds
/// its lock. On a current-thread runtime (unit tests) or outside a runtime the
/// closure runs inline — a current-thread *production* runtime would stall its
/// single thread for the whole decode, so we warn loudly rather than freezing
/// silently.
pub(crate) fn run_blocking<T>(f: impl FnOnce() -> T) -> T {
    use tokio::runtime::{Handle, RuntimeFlavor};
    match Handle::try_current() {
        Ok(h) if h.runtime_flavor() == RuntimeFlavor::MultiThread => tokio::task::block_in_place(f),
        Ok(h) if h.runtime_flavor() == RuntimeFlavor::CurrentThread => {
            // Reachable only in tests. A current-thread production runtime must
            // not be used here — it would block its single thread for the
            // whole decode. Warn once so this is visible, then run inline
            // (block_in_place would panic on the current_thread flavor).
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "speech decode running on a current-thread runtime blocks the async \
                     runtime; Lune Voice must run on a multi-thread runtime"
                );
            });
            f()
        }
        _ => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const MS: usize = SAMPLE_RATE / 1000;

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("lune-ggml-check-{name}"));
        let mut f = std::fs::File::create(&path).expect("create temp model");
        f.write_all(bytes).expect("write temp model");
        path
    }

    #[test]
    fn a_well_formed_model_passes() {
        let mut bytes = GGML_MAGIC.to_vec();
        bytes.resize(MIN_GGML_BYTES as usize + 1, 0);
        let path = write_temp("ok.bin", &bytes);
        assert!(check_ggml_file(&path).is_ok());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_truncated_download_is_rejected_by_size() {
        // What a full disk or a crash mid-write leaves behind. whisper.cpp
        // would abort the process on this, so it must never reach the loader.
        let path = write_temp("short.bin", &GGML_MAGIC);
        let err = check_ggml_file(&path).expect_err("a 4-byte model must be refused");
        assert!(
            err.to_string().contains("partial download"),
            "the message must tell the user what to do, got: {err}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_file_that_is_not_ggml_is_rejected_by_magic() {
        // Right size, wrong contents — an HTML error page saved under the
        // model's name is the realistic version of this.
        let mut bytes = b"<!DOCTYPE html>".to_vec();
        bytes.resize(MIN_GGML_BYTES as usize + 1, b' ');
        let path = write_temp("html.bin", &bytes);
        let err = check_ggml_file(&path).expect_err("a non-ggml file must be refused");
        assert!(err.to_string().contains("not in ggml format"), "got: {err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_missing_model_is_reported_not_panicked() {
        let path = std::env::temp_dir().join("lune-ggml-check-absent.bin");
        let _ = std::fs::remove_file(&path);
        assert!(check_ggml_file(&path).is_err());
    }

    fn words(acc: WordConfidences) -> Vec<(String, f32, f32)> {
        acc.finish()
            .0
            .into_iter()
            .map(|w| (w.word, w.confidence, w.mean_confidence))
            .collect()
    }

    #[test]
    fn tokens_group_into_words_at_each_word_start() {
        let mut acc = WordConfidences::new();
        acc.push(" Hel", 0.9, true);
        acc.push("lo", 0.5, false);
        acc.push(",", 0.7, false);
        acc.push(" world", 1.0, true);
        let got = words(acc);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, "Hello,");
        assert!((got[0].1 - 0.5).abs() < 1e-6, "min token wins");
        assert!((got[0].2 - 0.7).abs() < 1e-6, "mean of 0.9, 0.5, 0.7");
        assert_eq!(got[1].0, "world");
    }

    #[test]
    fn segment_stats_cover_every_token() {
        let mut acc = WordConfidences::new();
        acc.push(" a", 0.2, true);
        acc.push(" b", 0.8, true);
        let (w, mean, min) = acc.finish();
        assert_eq!(w.len(), 2);
        assert!((mean.unwrap() - 0.5).abs() < 1e-6);
        assert!((min.unwrap() - 0.2).abs() < 1e-6);
        assert!((w[0].word_error_risk - 0.8).abs() < 1e-6);
    }

    #[test]
    fn no_tokens_means_no_confidence_rather_than_full_confidence() {
        let (w, mean, min) = WordConfidences::new().finish();
        assert!(w.is_empty());
        assert_eq!((mean, min), (None, None));
    }

    #[test]
    fn whitespace_only_words_are_dropped() {
        let mut acc = WordConfidences::new();
        acc.push(" ", 0.4, true);
        acc.push(" ok", 0.9, true);
        assert_eq!(words(acc).len(), 1);
    }

    #[test]
    fn out_of_range_probabilities_are_clamped() {
        let mut acc = WordConfidences::new();
        acc.push(" a", f32::NAN, true);
        acc.push(" b", 1.7, true);
        acc.push(" c", -0.3, true);
        let (w, _, min) = acc.finish();
        assert_eq!(w[0].confidence, 0.0);
        assert_eq!(w[1].confidence, 1.0);
        assert_eq!(w[2].confidence, 0.0);
        assert_eq!(min, Some(0.0));
    }

    /// Parakeet's 60 s window: a 5-minute dictation with a pause every ~10 s
    /// comes back as pieces that all fit, cut only inside pauses.
    #[test]
    fn a_long_dictation_is_cut_at_pauses_into_windows_that_fit() {
        let second = SAMPLE_RATE;
        let (max, min) = (60 * second, 3 * second);
        let mut speech = Vec::new();
        let mut at = 0;
        while at + 10 * second <= 300 * second {
            speech.push((at, at + 9 * second));
            at += 10 * second;
        }
        let total = at;
        let chunks = plan_chunks_within(&speech, total, 0, max, min);
        assert!(chunks.len() >= 5, "{chunks:?}");
        assert_eq!(chunks.first().map(|c| c.0), Some(0));
        assert_eq!(chunks.last().map(|c| c.1), Some(total));
        for pair in chunks.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "no gap, no overlap");
            // Every cut lands inside a pause, never inside a speech span.
            let cut = pair[0].1;
            assert!(
                !speech.iter().any(|&(a, b)| a < cut && cut < b),
                "cut at {cut} splits speech"
            );
        }
        for (start, end) in &chunks {
            assert!(end - start <= max);
        }
        // A clip that fits is left whole.
        assert_eq!(
            plan_chunks_within(&speech[..3], 30 * second, 0, max, min),
            vec![(0, 30 * second)]
        );
    }

    #[test]
    fn a_vad_span_is_padded_and_clamped_to_the_buffer() {
        // Speech from 1.00 s to 2.00 s and 2.50 s to 3.00 s of a 3.1 s buffer.
        let len = 3_100 * MS;
        let span = speech_span_from_segments(&[(100.0, 200.0), (250.0, 300.0)], len)
            .expect("segments make a span");
        assert_eq!(span.onset, 1_000 * MS);
        assert_eq!(span.start, (1_000 - VAD_PAD_LEAD_MS) * MS);
        // 3.00 s + 240 ms tail would run past the buffer; it stops at the end.
        assert_eq!(span.end, len);
        assert_eq!(
            span.spans,
            vec![(1_000 * MS, 2_000 * MS), (2_500 * MS, 3_000 * MS)]
        );
    }

    #[test]
    fn speech_at_the_very_start_keeps_sample_zero() {
        let span = speech_span_from_segments(&[(0.0, 50.0)], 16_000).expect("span");
        assert_eq!(span.start, 0);
        assert_eq!(span.onset, 0);
    }

    #[test]
    fn no_segments_is_no_span() {
        assert_eq!(speech_span_from_segments(&[], 16_000), None);
    }

    #[test]
    fn silence_decodes_nothing_and_no_vad_decodes_everything() {
        assert_eq!(VadTrim::Silence.decode_slice(100), None);
        let full = VadTrim::Full.decode_slice(100).expect("full buffer");
        assert_eq!((full.start, full.end, full.onset), (0, 100, None));
        let degenerate = VadTrim::Speech(SpeechSpan {
            start: 50,
            end: 50,
            onset: 50,
            spans: vec![],
        });
        let fallback = degenerate.decode_slice(100).expect("whole buffer");
        assert_eq!((fallback.start, fallback.end), (0, 100));
    }

    #[test]
    fn a_trimmed_slice_rebases_its_spans_and_reports_its_offset() {
        let slice = VadTrim::Speech(SpeechSpan {
            start: 8_000,
            end: 40_000,
            onset: 14_400,
            spans: vec![(14_400, 30_000), (32_000, 48_000)],
        })
        .decode_slice(48_000)
        .expect("speech");
        assert_eq!(slice.onset, Some(14_400));
        assert_eq!(slice.offset_ms(), 500);
        // The second span runs past the slice's end and is clamped to it.
        assert_eq!(
            slice.relative_spans(),
            vec![(6_400, 22_000), (24_000, 32_000)]
        );
    }
}
