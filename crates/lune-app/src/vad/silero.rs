//! Silero VAD v6 — the neural speech verifier (stage 2 of the hybrid gate).
//!
//! The model is baked into the binary via `include_bytes!` and run on CPU with
//! `ort`. v6 ONNX I/O (verified against the committed model):
//!   inputs:  `input` f32 `[1, CONTEXT + SILERO_FRAME]` — each 512-sample
//!            window must be prepended with the trailing [`CONTEXT`] (64)
//!            samples of the previous window (zeros for the first), 576 total
//!            @ 16 kHz. Feeding a bare 512 makes the model see misaligned
//!            input and return ~0 for everything, including clear speech.
//!            `state` f32 `[2, 1, 128]` (LSTM state, carried across frames),
//!            `sr`    i64 scalar (16000)
//!   outputs: `output` f32 `[1, 1]` speech probability, `stateN` the new state.

use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::sync::Mutex;

/// Bundled Silero VAD v6 model (~2.3 MB, MIT) — see `crates/lune-app/data/`.
const SILERO_ONNX: &[u8] = include_bytes!("../../data/silero_vad.onnx");

/// Samples per Silero frame at 16 kHz (32 ms). The model expects exactly this.
pub(crate) const SILERO_FRAME: usize = 512;

const SAMPLE_RATE: i64 = 16_000;
const STATE_DIM: usize = 128;
/// Samples of the previous window prepended to each frame (Silero v5+
/// convention). Without it the model is effectively deaf to the input.
const CONTEXT: usize = 64;

/// Stateful Silero VAD runner. One per session; reset between sessions.
pub(crate) struct Silero {
    /// `None` only transiently, after the session is parked back into the pool
    /// on drop.
    session: Option<Session>,
    /// LSTM state `[2, 1, 128]`, carried across frames, zeroed on `reset`.
    state: Array3<f32>,
    /// Trailing [`CONTEXT`] samples of the previous window, prepended to the
    /// next one. Zeroed on `reset`.
    context: [f32; CONTEXT],
}

/// A single warm Silero ONNX session, parked here between dictations. Building
/// it (`commit_from_memory` + Level3 graph optimization) is the dominant
/// per-dictation startup cost; the session is a pure function — all VAD state is
/// threaded in/out as tensors (`state`/`stateN`), nothing lives inside it — so
/// it is safe to hand from one dictation to the next. Only one dictation runs at
/// a time, so a pool of one suffices; a concurrent extra `Silero` just builds
/// its own and frees it.
static SILERO_SESSION_POOL: Mutex<Option<Session>> = Mutex::new(None);

/// Take the parked warm session if present, else `None` (the caller builds).
fn pooled_session() -> Option<Session> {
    SILERO_SESSION_POOL.lock().ok().and_then(|mut p| p.take())
}

/// Build a fresh Silero ONNX session — the expensive path run once per process.
fn build_session() -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("silero builder: {e}"))?
        .with_log_level(ort::logging::LogLevel::Error)
        .map_err(|e| format!("silero log level: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("silero opt level: {e}"))?
        // Silero is a tiny LSTM run per-frame; one thread is lowest-latency.
        .with_intra_threads(1)
        .map_err(|e| format!("silero intra threads: {e}"))?
        .commit_from_memory(SILERO_ONNX)
        .map_err(|e| format!("silero load: {e}"))
}

/// Build and park a session ahead of the first dictation so `Silero::new` on
/// the hot path reuses it. No-op if one is already parked. Safe to call from a
/// startup background thread.
pub(crate) fn prewarm() -> Result<(), String> {
    if SILERO_SESSION_POOL.lock().is_ok_and(|p| p.is_some()) {
        return Ok(());
    }
    let session = build_session()?;
    if let Ok(mut pool) = SILERO_SESSION_POOL.lock()
        && pool.is_none()
    {
        *pool = Some(session);
    }
    Ok(())
}

impl Silero {
    pub(crate) fn new() -> Result<Self, String> {
        // Reuse the warm session if one is parked; only the first dictation
        // (or one after a leaked session) pays the graph-build cost.
        let session = match pooled_session() {
            Some(s) => s,
            None => build_session()?,
        };
        Ok(Self {
            session: Some(session),
            state: Array3::zeros((2, 1, STATE_DIM)),
            context: [0.0; CONTEXT],
        })
    }

    /// Zero the LSTM state and rolling context — call at the start of each
    /// session/stream.
    pub(crate) fn reset(&mut self) {
        self.state.fill(0.0);
        self.context = [0.0; CONTEXT];
    }

    /// Advance the rolling context without running inference. Used for frames
    /// the Earshot pre-filter skips, so the context stays continuous and the
    /// first verified frame after a gap sees the true preceding samples.
    pub(crate) fn observe(&mut self, frame: &[f32]) {
        self.update_context(frame);
    }

    fn update_context(&mut self, frame: &[f32]) {
        let n = frame.len();
        if n >= CONTEXT {
            self.context.copy_from_slice(&frame[n - CONTEXT..]);
        } else {
            self.context.copy_within(n.., 0);
            self.context[CONTEXT - n..].copy_from_slice(frame);
        }
    }

    /// Speech probability `[0,1]` for one [`SILERO_FRAME`]-sample frame.
    /// Threads the LSTM state across calls.
    pub(crate) fn predict(&mut self, frame: &[f32]) -> Result<f32, String> {
        // Prepend the previous window's trailing context (Silero v5+ ABI).
        let mut buf = Vec::with_capacity(CONTEXT + frame.len());
        buf.extend_from_slice(&self.context);
        buf.extend_from_slice(frame);
        let len = buf.len();
        let input = Array2::from_shape_vec((1, len), buf)
            .map_err(|e| format!("silero input shape: {e}"))?;
        let sr = ndarray::arr0(SAMPLE_RATE);

        let session = self.session.as_mut().ok_or("silero session unavailable")?;
        let outputs = session
            .run(ort::inputs![
                "input" => Value::from_array(input).map_err(|e| format!("silero input val: {e}"))?,
                "state" => Value::from_array(self.state.clone()).map_err(|e| format!("silero state val: {e}"))?,
                "sr" => Value::from_array(sr).map_err(|e| format!("silero sr val: {e}"))?,
            ])
            .map_err(|e| format!("silero run: {e}"))?;

        let (_, prob) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("silero output: {e}"))?;
        let speech = prob.first().copied().unwrap_or(0.0);

        let (_, new_state) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("silero stateN: {e}"))?;
        let new_state = new_state.to_vec();
        drop(outputs); // release the &mut self.session borrow before mutating self

        self.state = Array3::from_shape_vec((2, 1, STATE_DIM), new_state)
            .map_err(|e| format!("silero state reshape: {e}"))?;
        self.update_context(frame);
        Ok(speech)
    }
}

impl Drop for Silero {
    fn drop(&mut self) {
        // Park the built session for the next dictation instead of freeing it,
        // so the next `Silero::new` skips the graph build. Keep at most one.
        let Some(session) = self.session.take() else {
            return;
        };
        if let Ok(mut pool) = SILERO_SESSION_POOL.lock()
            && pool.is_none()
        {
            *pool = Some(session);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_runs_and_carries_state() {
        let mut s = Silero::new().expect("bundled v6 model should load on CPU");
        let silence = vec![0.0f32; SILERO_FRAME];
        let p = s.predict(&silence).expect("predict runs");
        assert!((0.0..=1.0).contains(&p), "prob must be in [0,1], got {p}");
        // Reset must not panic and state must keep producing valid probs.
        s.reset();
        let p2 = s.predict(&silence).expect("predict after reset");
        assert!((0.0..=1.0).contains(&p2));
    }

    #[test]
    fn second_instance_reuses_pooled_session() {
        // First instance runs, then parks its session on drop.
        {
            let mut a = Silero::new().expect("first build");
            let _ = a.predict(&vec![0.0f32; SILERO_FRAME]).expect("predict a");
        }
        // A fresh instance must work whether it reuses the parked session or
        // builds its own — the session is stateless, so reuse is transparent.
        let mut b = Silero::new().expect("reuse or rebuild");
        let p = b.predict(&vec![0.0f32; SILERO_FRAME]).expect("predict b");
        assert!((0.0..=1.0).contains(&p));
    }
}
