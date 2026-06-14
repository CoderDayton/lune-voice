//! Hybrid VAD: Earshot (fast acoustic gate) → Silero (neural verify) → chunker.
//!
//! Lives in `lune-app` because it needs both `ort` (Silero) and `lune-audio`
//! (Earshot/PCEN), and is consumed by the capture loop next door
//! (`crate::capture`). `lune-audio` stays ONNX-free.
//!
//! [`HybridVad`] is the orchestrator. Per audio chunk it reframes to Silero's
//! 512-sample frames, runs the two-stage gate, and emits [`GateAction`]s the
//! capture loop applies to the (batch) ASR backend:
//!   1. **Earshot** (stage 1) — cheap energy/GMM activity pre-gate (high
//!      recall) deciding which frames are worth the neural pass; also feeds the
//!      UI level meter.
//!   2. **Silero** (stage 2) — neural speech probability; rejects fans,
//!      keyboard, music, static that Earshot lets through.
//!   3. **Chunker** (stage 3) — onset/offset/hangover state machine that turns
//!      the probability stream into stable speech segments.
//!
//! Segments are **buffered**, not streamed: the orchestrator accumulates a
//! segment and only emits its audio once the chunker finalizes it (and only if
//! it clears `min_chunk_ms`), which is what lets sub-`min_chunk` noise blips be
//! dropped. Because the ASR decode is batch (`poll_final` at session end),
//! delaying the per-segment push costs no latency.

mod chunker;
mod config;
mod silero;

use std::collections::VecDeque;

use tracing::warn;

use lune_audio::{AudioChunk, Vad, VadConfig};
use lune_core::audio::SAMPLE_RATE;
use lune_core::config::PipelineConfig;

use self::chunker::{Chunker, ChunkerEvent};
use self::silero::{SILERO_FRAME, Silero};

pub use self::config::{DEFAULT_VAD_CONFIG, VadPipelineConfig};

/// Minimum total gated gap that counts as an intentional pause rather than a
/// breath. Below a full second every compositional hesitation became a comma
/// in real dictation, so only second-plus gaps inject pause prosody.
const PAUSE_TOTAL_MIN_SAMPLES: usize = SAMPLE_RATE; // 1000 ms
/// Ceiling on the reconstructed pause the model hears. Longer pauses carry no
/// extra punctuation information and only cost encoder frames.
const PAUSE_TOTAL_CAP_SAMPLES: usize = (SAMPLE_RATE * 3) / 2; // 1500 ms

/// What the gate decided the capture loop should do with the ASR stream.
#[derive(Debug, Clone, PartialEq)]
pub enum GateAction {
    /// Verified speech audio (preroll + segment) to push to the ASR backend.
    PushSamples(Vec<f32>),
    /// Inject this many silence samples to reconstruct an inter-segment pause
    /// (only emitted when `pause_injection` is on).
    InjectPause(usize),
    /// A segment just closed — informational, for metrics/logging.
    SegmentFinalized,
}

/// The two-stage hybrid speech gate. One per session; [`reset`](Self::reset)
/// to reuse across runs (e.g. the tuning benchmark).
pub struct HybridVad {
    /// Stage 1 — Earshot energy/GMM gate (+ PCEN). Reused from `lune-audio`.
    earshot: Vad,
    /// Kept so [`reset`](Self::reset) can rebuild Earshot in place (it has no
    /// reset of its own).
    earshot_config: VadConfig,
    sample_rate: u32,
    /// Stage 2 — neural verifier. `None` when `silero_enabled` is off
    /// (Earshot-only fallback through the same chunker).
    silero: Option<Silero>,
    /// Stage 3 — onset/offset/hangover segment state machine.
    chunker: Chunker,
    config: VadPipelineConfig,

    /// Leftover samples (< one Silero frame) awaiting the next chunk.
    sample_buf: Vec<f32>,
    /// Rolling pre-onset audio prepended to a segment so hard word onsets
    /// ("hello") aren't clipped to "ello". Bounded to `preroll_cap` samples.
    preroll: VecDeque<f32>,
    preroll_cap: usize,
    /// The segment currently being accumulated.
    segment: Vec<f32>,
    /// Gated silence seen since the last kept segment (for pause injection).
    idle_samples: usize,
    /// Snapshot of `idle_samples` at the current segment's onset.
    segment_lead_silence: usize,
    /// Preroll actually prepended to the current segment (already "heard").
    segment_preroll_samples: usize,
    /// Whether any segment has been kept this session (gates pause injection).
    had_kept_segment: bool,
    /// Peak Earshot energy since the last [`take_peak_energy`](Self::take_peak_energy).
    peak_energy: f32,
}

impl HybridVad {
    /// Build the gate from the resolved pipeline config. Mirrors the old
    /// `init_vad`: Earshot reuses `speech_threshold`/`speech_threshold_end`
    /// and the PCEN toggle; the Silero/chunker knobs come from
    /// [`VadPipelineConfig::from_pipeline`].
    pub fn new(
        pipeline: &PipelineConfig,
        vad_pcen_enabled: bool,
    ) -> std::result::Result<Self, String> {
        Self::with_config(VadPipelineConfig::from_pipeline(pipeline), vad_pcen_enabled)
    }

    /// Build directly from a resolved [`VadPipelineConfig`] — the entry point
    /// for the tuning benchmark, which sweeps the config fields. Earshot's own
    /// start/end thresholds are irrelevant here (the gate reads the raw
    /// `vad_score`, not Earshot's hysteresis), so only the PCEN toggle is
    /// threaded through.
    pub fn with_config(
        config: VadPipelineConfig,
        vad_pcen_enabled: bool,
    ) -> std::result::Result<Self, String> {
        let mut earshot_config = VadConfig::default();
        if !vad_pcen_enabled {
            earshot_config.pcen = None;
        }
        let sample_rate = config.sample_rate as u32;
        let earshot = Vad::new(earshot_config.clone(), sample_rate)
            .map_err(|e| format!("earshot init: {e}"))?;

        let silero = if config.silero_enabled {
            Some(Silero::new()?)
        } else {
            None
        };
        let chunker = Chunker::new(&config);
        let preroll_cap = config.preroll_samples();

        Ok(Self {
            earshot,
            earshot_config,
            sample_rate,
            silero,
            chunker,
            config,
            sample_buf: Vec::with_capacity(2 * SILERO_FRAME),
            preroll: VecDeque::with_capacity(preroll_cap + SILERO_FRAME),
            preroll_cap,
            segment: Vec::new(),
            idle_samples: 0,
            segment_lead_silence: 0,
            segment_preroll_samples: 0,
            had_kept_segment: false,
            peak_energy: 0.0,
        })
    }

    /// Feed one captured chunk through the gate. Reframes to Silero frames and
    /// returns the actions the caller applies to the ASR stream (often empty —
    /// segments emit only at finalize).
    pub fn process_chunk(&mut self, chunk: &AudioChunk) -> Vec<GateAction> {
        let mut actions = Vec::new();
        self.sample_buf.extend_from_slice(&chunk.samples);
        while self.sample_buf.len() >= SILERO_FRAME {
            let frame: Vec<f32> = self.sample_buf.drain(..SILERO_FRAME).collect();
            self.process_frame(&frame, &mut actions);
        }
        actions
    }

    fn process_frame(&mut self, frame: &[f32], actions: &mut Vec<GateAction>) {
        // Stage 1: Earshot activity + energy (drives the level meter too).
        let er = self.earshot.process_chunk(frame);
        self.peak_energy = self.peak_energy.max(er.energy);
        let earshot_active = er.vad_score >= self.config.earshot_activity_threshold;

        // Stage 2: speech probability. Skip the neural pass when Earshot
        // pre-filters an idle frame out — but always run it inside a segment so
        // the offset/hangover decision sees true silence. A failed Silero call
        // is fail-safe: treat the frame as silence rather than force speech.
        let prob = if let Some(silero) = self.silero.as_mut() {
            let run = self.chunker.in_speech() || !self.config.earshot_prefilter || earshot_active;
            if run {
                match silero.predict(frame) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!(error = %e, "silero predict failed; treating frame as silence");
                        0.0
                    }
                }
            } else {
                // Pre-filtered out: keep Silero's rolling context continuous so
                // the first verified frame after this gap isn't misaligned.
                silero.observe(frame);
                0.0
            }
        } else {
            // Earshot-only fallback: collapse the activity gate to 0/1 prob so
            // the same chunker drives segmentation.
            if earshot_active { 1.0 } else { 0.0 }
        };

        // Stage 3: segment state machine.
        match self.chunker.step(prob) {
            ChunkerEvent::Idle => {
                self.push_preroll(frame);
                if self.had_kept_segment {
                    self.idle_samples += frame.len();
                }
            }
            ChunkerEvent::Begin => {
                self.segment.clear();
                self.segment_lead_silence = self.idle_samples;
                self.idle_samples = 0;
                self.segment_preroll_samples = self.preroll.len();
                self.segment.extend(self.preroll.drain(..));
                self.segment.extend_from_slice(frame);
            }
            ChunkerEvent::Continue => {
                self.segment.extend_from_slice(frame);
            }
            ChunkerEvent::Finalize { keep } => {
                self.segment.extend_from_slice(frame);
                self.finalize_segment(keep, actions);
            }
        }
    }

    fn push_preroll(&mut self, frame: &[f32]) {
        if self.preroll_cap == 0 {
            return;
        }
        self.preroll.extend(frame.iter().copied());
        while self.preroll.len() > self.preroll_cap {
            self.preroll.pop_front();
        }
    }

    fn finalize_segment(&mut self, keep: bool, actions: &mut Vec<GateAction>) {
        if keep {
            if self.config.pause_injection && self.had_kept_segment {
                let inject = pause_injection_samples(
                    self.segment_lead_silence,
                    self.segment_preroll_samples,
                    0,
                );
                if inject > 0 {
                    actions.push(GateAction::InjectPause(inject));
                }
            }
            let seg = std::mem::take(&mut self.segment);
            actions.push(GateAction::PushSamples(seg));
            actions.push(GateAction::SegmentFinalized);
            self.had_kept_segment = true;
        } else {
            // Dropped as a noise blip: fold its lead silence back into the
            // running gap so pause timing across the drop stays continuous.
            if self.had_kept_segment {
                self.idle_samples = self.idle_samples.saturating_add(self.segment_lead_silence);
            }
            self.segment.clear();
        }
        self.segment_lead_silence = 0;
        self.segment_preroll_samples = 0;
    }

    /// Flush any open segment at end of stream. Appends the sub-frame tail to
    /// the open segment first so the final word's phonemes aren't clipped
    /// (mirrors the old producer tail-flush).
    pub fn finish(&mut self) -> Vec<GateAction> {
        let mut actions = Vec::new();
        if self.chunker.in_speech() && !self.sample_buf.is_empty() {
            let tail = std::mem::take(&mut self.sample_buf);
            self.segment.extend_from_slice(&tail);
        }
        if let ChunkerEvent::Finalize { keep } = self.chunker.finish() {
            self.finalize_segment(keep, &mut actions);
        }
        self.sample_buf.clear();
        actions
    }

    /// True while a segment is open — lets the caller gate its tail-flush.
    pub fn in_segment(&self) -> bool {
        self.chunker.in_speech()
    }

    /// Peak Earshot energy since the last call, then resets it (level meter).
    pub fn take_peak_energy(&mut self) -> f32 {
        let p = self.peak_energy;
        self.peak_energy = 0.0;
        p
    }

    /// Reset all per-session state, reusing the loaded Silero model. Earshot
    /// has no in-place reset, so it is rebuilt from the stored config.
    pub fn reset(&mut self) -> std::result::Result<(), String> {
        self.earshot = Vad::new(self.earshot_config.clone(), self.sample_rate)
            .map_err(|e| format!("earshot reset: {e}"))?;
        if let Some(s) = self.silero.as_mut() {
            s.reset();
        }
        self.chunker.reset();
        self.sample_buf.clear();
        self.preroll.clear();
        self.segment.clear();
        self.idle_samples = 0;
        self.segment_lead_silence = 0;
        self.segment_preroll_samples = 0;
        self.had_kept_segment = false;
        self.peak_energy = 0.0;
        Ok(())
    }
}

/// How much silence to inject at a segment boundary so the (batch) model hears
/// the pause the gate stripped — exactly.
///
/// `gap_fed_samples` is silence already fed live as part of the segment (zero
/// in the buffered Silero model, where the hangover tail is inside the
/// segment). `preroll_samples` is the pre-onset audio prepended to the new
/// segment — also already heard. The heard pause becomes
/// `min(true gap, PAUSE_TOTAL_CAP_SAMPLES)` sample-for-sample.
fn pause_injection_samples(
    silence_samples: usize,
    preroll_samples: usize,
    gap_fed_samples: usize,
) -> usize {
    if silence_samples == 0 {
        return 0;
    }
    let total_gap = gap_fed_samples + silence_samples;
    if total_gap < PAUSE_TOTAL_MIN_SAMPLES {
        return 0;
    }
    let already_heard = gap_fed_samples + preroll_samples;
    silence_samples
        .saturating_sub(preroll_samples)
        .min(PAUSE_TOTAL_CAP_SAMPLES.saturating_sub(already_heard))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Chunk-quantized fed-gap values (80 ms chunks); preroll = 4 × 80 ms.
    const CHUNK: usize = SAMPLE_RATE * 80 / 1000;
    const PREROLL: usize = 4 * CHUNK;

    #[test]
    fn pause_injection_breath_gaps_skipped() {
        assert_eq!(pause_injection_samples(0, 0, 3 * CHUNK), 0);
        // Total gap 560 ms < 1000 ms floor: a breath.
        assert_eq!(pause_injection_samples(PREROLL, PREROLL, 3 * CHUNK), 0);
    }

    #[test]
    fn pause_injection_heard_equals_true_gap() {
        for gap_fed in [CHUNK, 3 * CHUNK, 6 * CHUNK] {
            let silence = SAMPLE_RATE - gap_fed; // 1 s total gap
            let inject = pause_injection_samples(silence, PREROLL, gap_fed);
            assert_eq!(gap_fed + inject + PREROLL, SAMPLE_RATE, "gap_fed={gap_fed}");
        }
    }

    #[test]
    fn pause_injection_caps_total_heard() {
        for gap_fed in [CHUNK, 3 * CHUNK] {
            let silence = SAMPLE_RATE * 5 - gap_fed;
            let inject = pause_injection_samples(silence, PREROLL, gap_fed);
            assert_eq!(
                gap_fed + inject + PREROLL,
                PAUSE_TOTAL_CAP_SAMPLES,
                "gap_fed={gap_fed}"
            );
        }
        assert_eq!(pause_injection_samples(SAMPLE_RATE, 0, SAMPLE_RATE * 2), 0);
    }

    fn load_fixture(name: &str) -> Vec<f32> {
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("fixtures/audio")
            .join(name);
        let reader = hound::WavReader::open(&wav).expect("open fixture");
        let spec = reader.spec();
        match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap() as f32 / max)
                    .collect()
            }
        }
    }

    fn run_fixture(hybrid: &mut HybridVad, samples: &[f32]) -> usize {
        let mut pushed = 0usize;
        for (i, chunk) in samples.chunks(1280).enumerate() {
            let c = AudioChunk {
                samples: chunk.to_vec(),
                timestamp_ms: (i as u64) * 80,
            };
            for a in hybrid.process_chunk(&c) {
                if let GateAction::PushSamples(s) = a {
                    pushed += s.len();
                }
            }
        }
        for a in hybrid.finish() {
            if let GateAction::PushSamples(s) = a {
                pushed += s.len();
            }
        }
        pushed
    }

    /// Regression for the Silero context ABI: real speech must pass the gate
    /// (most of a ~23 s clean clip), proving the model actually discriminates.
    #[test]
    fn real_speech_produces_segments() {
        let samples = load_fixture("librispeech_clean.wav");
        let mut hybrid = HybridVad::new(&PipelineConfig::default(), true)
            .expect("hybrid vad builds with bundled model");
        let pushed = run_fixture(&mut hybrid, &samples);
        // Clean ~23 s speech is mostly voiced; insist on at least half passing.
        assert!(
            pushed > samples.len() / 2,
            "real speech must pass the gate: pushed {pushed} of {}",
            samples.len()
        );
    }

    /// End-to-end smoke test through the real bundled Silero model: pure
    /// silence must never produce a kept speech segment.
    #[test]
    fn silence_yields_no_segments() {
        let mut hybrid = HybridVad::new(&PipelineConfig::default(), true)
            .expect("hybrid vad builds with bundled model");
        let chunk = AudioChunk {
            samples: vec![0.0f32; 1280],
            timestamp_ms: 0,
        };
        let mut pushed = 0usize;
        // ~2 s of silence.
        for i in 0..25 {
            let mut c = chunk.clone();
            c.timestamp_ms = i * 80;
            for action in hybrid.process_chunk(&c) {
                if let GateAction::PushSamples(s) = action {
                    pushed += s.len();
                }
            }
        }
        for action in hybrid.finish() {
            if let GateAction::PushSamples(s) = action {
                pushed += s.len();
            }
        }
        assert_eq!(pushed, 0, "silence must not be pushed to ASR as speech");
        assert!(!hybrid.in_segment());
    }
}
