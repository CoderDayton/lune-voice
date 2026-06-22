//! Hybrid VAD: Earshot (fast acoustic gate) → Silero (neural verify) → span gate.
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
//!   3. **Span gate** (stage 3) — opens on onset and stays open through short
//!      pauses, closing only on a `long_gap` run of trailing silence.
//!
//! The gate is **contiguous**: while a span is open the orchestrator appends
//! *every* frame — speech and sub-`long_gap` silence alike — so the natural
//! rhythm of an utterance survives to the recognizer. This is what stops
//! adjacent words fusing (`MakeSure`): stripping the inter-word pauses is what
//! destroyed the word boundaries Qwen3-ASR reads. A span is emitted only once
//! it closes and only if it clears `min_speech_ms`, so noise blips drop. The
//! long dead-air between spans is discarded; a splice there is harmless because
//! the words are seconds apart. Because the ASR decode is batch (`poll_final`
//! at session end), delaying the per-span push costs no latency.

mod chunker;
mod config;
mod silero;

use std::collections::VecDeque;

use tracing::warn;

use lune_audio::{AudioChunk, Vad, VadConfig};

use self::chunker::{SpanEvent, SpanGate};
use self::silero::{SILERO_FRAME, Silero};

pub use self::config::{DEFAULT_VAD_CONFIG, VadPipelineConfig};

/// What the gate decided the capture loop should do with the ASR stream.
#[derive(Debug, Clone, PartialEq)]
pub enum GateAction {
    /// Verified speech audio (preroll + span, intra-utterance pauses intact) to
    /// push to the ASR backend.
    PushSamples(Vec<f32>),
    /// A span just closed — informational, for metrics/logging.
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
    /// (Earshot-only fallback through the same span gate).
    silero: Option<Silero>,
    /// Stage 3 — contiguous-span state machine.
    span_gate: SpanGate,
    config: VadPipelineConfig,
    /// Earshot gate hangover: frames the gate stays open after the last active
    /// frame, so Silero receives a continuous, unbroken run of audio across
    /// brief intra-utterance dips instead of being re-gated mid-word.
    gate_hold_frames: usize,
    /// Frames the Earshot gate is still open (counts down once Earshot quiets).
    earshot_hold: usize,

    /// Leftover samples (< one Silero frame) awaiting the next chunk.
    sample_buf: Vec<f32>,
    /// Rolling pre-onset audio prepended to a span so hard word onsets
    /// ("hello") aren't clipped to "ello". Bounded to `preroll_cap` samples.
    preroll: VecDeque<f32>,
    preroll_cap: usize,
    /// The span currently being accumulated (speech + intra-utterance silence).
    span: Vec<f32>,
    /// Trailing silence (samples) accrued at the tail of the open span, trimmed
    /// to the hangover window when the span closes.
    trailing_silence_samples: usize,
    /// Peak Earshot energy since the last [`take_peak_energy`](Self::take_peak_energy).
    peak_energy: f32,
}

impl HybridVad {
    /// Build the gate for a session. The gate is self-tuned via
    /// [`DEFAULT_VAD_CONFIG`] (returned by [`VadPipelineConfig::from_pipeline`]);
    /// only `vad_pcen_enabled` is taken from the caller. `pipeline` is accepted
    /// for a stable call site but no longer drives the gate's thresholds — the
    /// legacy `speech_threshold` fed the old Earshot-only `is_speech` gate the
    /// raw-`vad_score` gate replaced.
    pub fn new(
        pipeline: &lune_core::config::PipelineConfig,
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
        let span_gate = SpanGate::new(&config);
        let preroll_cap = config.preroll_samples();
        // Hold the Earshot gate open at least as long as the chunker's hangover
        // so Silero keeps verifying right up to the moment a span finalizes.
        let gate_hold_frames = config.hangover_frames().max(1);

        Ok(Self {
            earshot,
            earshot_config,
            sample_rate,
            silero,
            span_gate,
            config,
            gate_hold_frames,
            earshot_hold: 0,
            sample_buf: Vec::with_capacity(2 * SILERO_FRAME),
            preroll: VecDeque::with_capacity(preroll_cap + SILERO_FRAME),
            preroll_cap,
            span: Vec::new(),
            trailing_silence_samples: 0,
            peak_energy: 0.0,
        })
    }

    /// Feed one captured chunk through the gate. Reframes to Silero frames and
    /// returns the actions the caller applies to the ASR stream (often empty —
    /// spans emit only at close).
    pub fn process_chunk(&mut self, chunk: &AudioChunk) -> Vec<GateAction> {
        let mut actions = Vec::new();
        self.sample_buf.extend_from_slice(&chunk.samples);
        while self.sample_buf.len() >= SILERO_FRAME {
            let frame: Vec<f32> = self.sample_buf.drain(..SILERO_FRAME).collect();
            self.process_frame(&frame, &mut actions);
        }
        actions
    }

    /// Diagnostic: per-frame raw stage scores `(earshot_vad_score, silero_prob)`
    /// for every 512-sample frame in `chunk` (plus buffered leftovers). Unlike
    /// the live path, Silero runs UNCONDITIONALLY (no Earshot pre-filter) so both
    /// stages' independent opinions are visible — for measuring agreement and
    /// false-positive rates, not for gating. Mutates Earshot/Silero rolling
    /// context like the real path, so call [`reset`](Self::reset) between clips.
    pub fn debug_stage_scores(&mut self, chunk: &AudioChunk) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        self.sample_buf.extend_from_slice(&chunk.samples);
        while self.sample_buf.len() >= SILERO_FRAME {
            let frame: Vec<f32> = self.sample_buf.drain(..SILERO_FRAME).collect();
            let er = self.earshot.process_chunk(&frame);
            let prob = match self.silero.as_mut() {
                Some(s) => s.predict(&frame).unwrap_or(0.0),
                None => 0.0,
            };
            out.push((er.vad_score, prob));
        }
        out
    }

    fn process_frame(&mut self, frame: &[f32], actions: &mut Vec<GateAction>) {
        // Stage 1 — Earshot acoustic gate (also feeds the level meter). It opens
        // on raw activity and holds open for `gate_hold_frames`, handing Silero a
        // continuous, *unchunked* run of audio across brief intra-utterance dips.
        let er = self.earshot.process_chunk(frame);
        self.peak_energy = self.peak_energy.max(er.energy);
        let earshot_active = er.vad_score >= self.config.earshot_activity_threshold;
        if earshot_active {
            self.earshot_hold = self.gate_hold_frames;
        } else if self.earshot_hold > 0 {
            self.earshot_hold -= 1;
        }
        let gate_open = earshot_active || self.earshot_hold > 0;

        // Stage 2 — Silero verifies the audio the gate passes. Its invocation
        // depends only on the Earshot gate (forward flow). A failed call is
        // fail-safe (treated as silence). Gate-skipped frames still advance
        // Silero's rolling context so the next verified frame isn't misaligned.
        let prob = if let Some(silero) = self.silero.as_mut() {
            if !self.config.earshot_prefilter || gate_open {
                match silero.predict(frame) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!(error = %e, "silero predict failed; treating frame as silence");
                        0.0
                    }
                }
            } else {
                silero.observe(frame);
                0.0
            }
        } else {
            // Earshot-only fallback: collapse the raw gate to a 0/1 probability
            // so the same span gate still owns segmentation.
            if earshot_active { 1.0 } else { 0.0 }
        };

        // Stage 3 — the span gate decides boundaries; the orchestrator owns the
        // audio. While in-span, *every* frame is appended (speech and silence)
        // so intra-utterance pauses survive. The closing tail is trimmed to the
        // hangover window.
        match self.span_gate.step(prob) {
            SpanEvent::Idle => self.push_preroll(frame),
            SpanEvent::Open => {
                self.span.clear();
                self.span.extend(self.preroll.drain(..));
                self.span.extend_from_slice(frame);
                self.trailing_silence_samples = 0;
            }
            SpanEvent::Append => self.append_frame(frame, prob),
            SpanEvent::Flush => {
                // Length cap: emit the audio so far to bound memory, but the span
                // CONTINUES. No hangover trim, no pre-roll re-prepend, no
                // `SegmentFinalized` — and `trailing_silence_samples` is preserved
                // so the intra-silence cap stays continuous across the boundary.
                // This keeps a length split transparent to the recognizer instead
                // of injecting a ~700 ms tail+pre-roll silence that reads as a
                // sentence break.
                self.append_frame(frame, prob);
                let span = std::mem::take(&mut self.span);
                if !span.is_empty() {
                    actions.push(GateAction::PushSamples(span));
                }
            }
            SpanEvent::Close { keep } => {
                // The closing frame is a silence frame (only a long-gap run or EOS
                // reaches `Close` now): `append_frame` drops it when the
                // intra-silence cap is already saturated, otherwise `close_span`'s
                // hangover trim removes it.
                self.append_frame(frame, prob);
                self.close_span(keep, &[], actions);
            }
        }
    }

    /// Append an in-span frame, tracking the trailing-silence run that
    /// `close_span` trims to the hangover window: a frame at or below
    /// `silero_offset` extends the run, a louder one resets it.
    ///
    /// Intra-utterance silence cap: once the current silence run reaches
    /// `max_intra_silence_ms`, further silence frames in that run are dropped
    /// rather than appended, so the recognizer hears a word-boundary pause but
    /// not a sentence-length one (which Qwen3 would punctuate). The cap shapes
    /// only the *audio*; the span gate still sees every frame's probability and
    /// owns the open/close (`long_gap`) decision. `0` disables the cap.
    fn append_frame(&mut self, frame: &[f32], prob: f32) {
        if prob <= self.config.silero_offset {
            let cap = self.config.max_intra_silence_samples();
            if cap > 0 && self.trailing_silence_samples >= cap {
                // This pause already contributed `cap` of silence — drop the
                // excess instead of feeding the model sentence-end prosody.
                return;
            }
            self.span.extend_from_slice(frame);
            self.trailing_silence_samples += frame.len();
        } else {
            self.span.extend_from_slice(frame);
            self.trailing_silence_samples = 0;
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

    /// Close the open span: trim its trailing silence to the hangover window,
    /// append any end-of-stream `tail`, and emit the audio if it cleared
    /// `min_speech` (the [`SpanGate`] already folded that into `keep`).
    ///
    /// When `max_intra_silence_ms < hangover_ms`, `append_frame` has already
    /// bounded the trailing run below the hangover window, so the trim below is
    /// inert and the effective tail is the cap — see `max_intra_silence_ms`.
    fn close_span(&mut self, keep: bool, tail: &[f32], actions: &mut Vec<GateAction>) {
        let hangover = self.config.hangover_samples();
        if self.trailing_silence_samples > hangover {
            let drop = self.trailing_silence_samples - hangover;
            let new_len = self.span.len().saturating_sub(drop);
            self.span.truncate(new_len);
        }
        self.span.extend_from_slice(tail);
        self.trailing_silence_samples = 0;
        if keep && !self.span.is_empty() {
            let span = std::mem::take(&mut self.span);
            actions.push(GateAction::PushSamples(span));
            actions.push(GateAction::SegmentFinalized);
        } else {
            self.span.clear();
        }
    }

    /// Flush any open span at end of stream. Appends the sub-frame tail to the
    /// open span (after the trailing-silence trim) so the final word's phonemes
    /// aren't clipped.
    pub fn finish(&mut self) -> Vec<GateAction> {
        let mut actions = Vec::new();
        let tail = std::mem::take(&mut self.sample_buf);
        if let SpanEvent::Close { keep } = self.span_gate.finish() {
            self.close_span(keep, &tail, &mut actions);
        }
        actions
    }

    /// True while a span is open — lets the caller gate its tail-flush.
    pub fn in_segment(&self) -> bool {
        self.span_gate.in_span()
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
        self.span_gate.reset();
        self.earshot_hold = 0;
        self.sample_buf.clear();
        self.preroll.clear();
        self.span.clear();
        self.trailing_silence_samples = 0;
        self.peak_energy = 0.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lune_core::config::PipelineConfig;

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
    /// silence must never produce a kept speech span.
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

    /// Drive a long silence run through the private `append_frame` seam with
    /// explicit probabilities (no model) and close the span: the cap must bound
    /// the silence kept in the emitted audio so the recognizer hears a
    /// word-boundary pause, not a sentence-length one.
    #[test]
    fn intra_silence_cap_bounds_emitted_span() {
        let cfg = VadPipelineConfig {
            silero_enabled: false,
            max_intra_silence_ms: 240,
            // Larger than the cap, so the hangover trim can't mask the cap effect.
            hangover_ms: 2_000,
            min_speech_ms: 0,
            ..DEFAULT_VAD_CONFIG
        };
        let mut vad = HybridVad::with_config(cfg, false).expect("build hybrid vad");
        let voiced = vec![0.2f32; SILERO_FRAME];
        let silent = vec![0.0f32; SILERO_FRAME];

        // Open a span with one voiced frame, then feed a 40-frame silence run.
        vad.span.extend_from_slice(&voiced);
        vad.trailing_silence_samples = 0;
        for _ in 0..40 {
            vad.append_frame(&silent, 0.0); // 0.0 <= silero_offset -> silence
        }

        let mut actions = Vec::new();
        vad.close_span(true, &[], &mut actions);
        let pushed: usize = actions
            .iter()
            .filter_map(|a| match a {
                GateAction::PushSamples(s) => Some(s.len()),
                GateAction::SegmentFinalized => None,
            })
            .sum();

        // Silence is appended only until the run reaches the cap; with 512-sample
        // frames that is the first frame boundary >= cap (240 ms -> 256 ms).
        let cap_samples = (240usize * 16_000) / 1_000;
        let max_silence = cap_samples.div_ceil(SILERO_FRAME) * SILERO_FRAME;
        assert_eq!(
            pushed,
            SILERO_FRAME + max_silence,
            "cap must bound the silence kept in the emitted span"
        );
        assert!(
            pushed < SILERO_FRAME + 40 * SILERO_FRAME,
            "excess silence beyond the cap must be dropped, not appended"
        );
    }

    /// `max_intra_silence_ms = 0` disables the cap: every silence frame in the
    /// span is retained (the pre-cap behaviour).
    #[test]
    fn intra_silence_cap_zero_retains_all_silence() {
        let cfg = VadPipelineConfig {
            silero_enabled: false,
            max_intra_silence_ms: 0,
            hangover_ms: 2_000,
            min_speech_ms: 0,
            ..DEFAULT_VAD_CONFIG
        };
        let mut vad = HybridVad::with_config(cfg, false).expect("build hybrid vad");
        let voiced = vec![0.2f32; SILERO_FRAME];
        let silent = vec![0.0f32; SILERO_FRAME];
        vad.span.extend_from_slice(&voiced);
        vad.trailing_silence_samples = 0;
        for _ in 0..40 {
            vad.append_frame(&silent, 0.0);
        }
        let mut actions = Vec::new();
        vad.close_span(true, &[], &mut actions);
        let pushed: usize = actions
            .iter()
            .filter_map(|a| match a {
                GateAction::PushSamples(s) => Some(s.len()),
                GateAction::SegmentFinalized => None,
            })
            .sum();
        assert_eq!(
            pushed,
            SILERO_FRAME + 40 * SILERO_FRAME,
            "cap=0 must retain every silence frame"
        );
    }
}
