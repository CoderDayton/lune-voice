// SPDX-License-Identifier: GPL-3.0-or-later

//! Best-effort UI sound cues, played from the always-running backend.
//!
//! The desktop windows are not a reliable place to play audio: the main
//! window is destroyed on close (and never built when the app starts
//! minimized), and the pill overlay may not exist. The Rust process is the
//! only surface kept alive across the whole session (`prevent_exit` keeps it
//! running for the tray, hotkeys, and dictation), so cue playback lives here.
//!
//! Cues are small 16-bit mono PCM WAVs embedded in the binary, decoded with
//! `hound` and played through a short-lived `cpal` output stream. Playback is
//! fire-and-forget on a detached thread and every failure is swallowed at
//! debug level — a missing output device must never disrupt dictation.

use std::io::Cursor;
use std::sync::OnceLock;
use std::sync::mpsc;
use std::time::Duration;

use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tracing::debug;

/// A dictation lifecycle sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cue {
    /// Recording started (entered the listening state).
    Start,
    /// Recording finished and text was delivered.
    Stop,
    /// The session errored.
    Error,
}

const START_WAV: &[u8] = include_bytes!("../assets/sounds/start_recording.wav");
const STOP_WAV: &[u8] = include_bytes!("../assets/sounds/stop_recording.wav");
const ERROR_WAV: &[u8] = include_bytes!("../assets/sounds/error.wav");

impl Cue {
    fn bytes(self) -> &'static [u8] {
        match self {
            Cue::Start => START_WAV,
            Cue::Stop => STOP_WAV,
            Cue::Error => ERROR_WAV,
        }
    }
}

/// Serializes cue playback onto one long-lived worker thread. Spawning a
/// thread per cue let Start/Stop/Error opens of the same default output device
/// overlap (rapid hotkey taps, a session that starts then immediately errors),
/// which on ALSA without dmix fails "device busy" or garbles the cues.
///
/// The queue is bounded: a cue is feedback about something that just happened,
/// so if the worker is still playing when several more arrive, dropping them is
/// right and queueing them is wrong — an unbounded channel would play a Stop
/// chime seconds later, during the *next* utterance.
static CUE_TX: OnceLock<mpsc::SyncSender<(Cue, f32)>> = OnceLock::new();

/// Cues that may wait for the worker before new ones are dropped. One in
/// flight plus a couple queued covers a normal start→stop→error burst.
const CUE_QUEUE_DEPTH: usize = 2;

fn cue_sender() -> &'static mpsc::SyncSender<(Cue, f32)> {
    CUE_TX.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel::<(Cue, f32)>(CUE_QUEUE_DEPTH);
        std::thread::spawn(move || {
            while let Ok((cue, volume)) = rx.recv() {
                if let Err(e) = play_blocking(cue, volume) {
                    debug!(?cue, error = %e, "sound cue playback skipped");
                }
            }
        });
        tx
    })
}

/// Play a cue at `volume` (0.0–1.0) without blocking the caller. A volume at
/// or below zero is silent and skips playback entirely. Failures are logged at
/// debug and otherwise ignored.
pub fn play(cue: Cue, volume: f32) {
    if volume <= 0.0 {
        return;
    }
    // `try_send`, never `send`: this runs on the dictation path and must not
    // block on the audio device. A full queue means the cue is already stale.
    if cue_sender().try_send((cue, volume.min(1.0))).is_err() {
        debug!(?cue, "sound cue dropped — playback queue full");
    }
}

fn play_blocking(cue: Cue, volume: f32) -> Result<(), String> {
    let mono = decode_mono(cue.bytes())?;
    if mono.samples.is_empty() {
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    let sample_format = supported.sample_format();
    let dst_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let buffer_range = *supported.buffer_size();
    let default_config: cpal::StreamConfig = supported.into();

    let mut samples = resample_linear(&mono.samples, mono.sample_rate, dst_rate);
    let gain = perceptual_gain(volume);
    if gain < 1.0 {
        for s in &mut samples {
            *s *= gain;
        }
    }
    let duration = Duration::from_secs_f64(samples.len() as f64 / dst_rate.max(1) as f64);

    // BufferSize::Default resolves to a tiny ALSA period on some backends,
    // underrunning every period and crackling. Request a comfortable explicit
    // buffer, clamped to the device's supported range.
    let target = std::env::var("LUNE_SND_BUF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let fixed = match buffer_range {
        cpal::SupportedBufferSize::Range { min, max } => target.clamp(min, max),
        cpal::SupportedBufferSize::Unknown => target,
    };
    let fixed_config = cpal::StreamConfig {
        buffer_size: cpal::BufferSize::Fixed(fixed),
        ..default_config
    };
    debug!(
        ?sample_format,
        src_rate = mono.sample_rate,
        dst_rate,
        channels,
        fixed_buffer = fixed,
        "sound cue: negotiated output config"
    );

    // Fall back to the default buffer if the backend rejects an explicit size.
    let stream =
        build_stream(sample_format, &device, &fixed_config, &samples, channels).or_else(|e| {
            debug!(error = %e, "fixed-buffer output failed; retrying with default buffer");
            build_stream(sample_format, &device, &default_config, &samples, channels)
        })?;
    stream.play().map_err(|e| e.to_string())?;

    // Playback is driven by the cpal callback; hold the stream open for the
    // known clip length plus a short tail so the device drains before drop.
    std::thread::sleep(duration + Duration::from_millis(150));
    Ok(())
}

/// Map the volume slider position (0.0–1.0) to an amplitude gain.
///
/// The slider is perceptual but the DAC is linear: scaling samples by the raw
/// slider value leaves 50% at only −6 dB — barely quieter — with the whole
/// useful range crammed into the bottom fifth of the travel. Squaring the
/// position puts half-slider at −12 dB, close to the ≈−10 dB listeners judge
/// as "half as loud", while keeping the endpoints exact (0 → silence,
/// 1 → unity, so the default volume is bit-identical to before).
fn perceptual_gain(volume: f32) -> f32 {
    let v = volume.clamp(0.0, 1.0);
    v * v
}

/// Decoded mono PCM, normalized to f32 in [-1.0, 1.0].
struct MonoPcm {
    samples: Vec<f32>,
    sample_rate: u32,
}

fn decode_mono(bytes: &[u8]) -> Result<MonoPcm, String> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            // Saturate/clamp: a malformed header with 0 bits/sample would
            // otherwise overflow the subtract and shift out of range.
            let shift = spec.bits_per_sample.saturating_sub(1).min(62);
            let scale = (1i64 << shift) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale).unwrap_or(0.0))
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
    };

    let samples = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok(MonoPcm {
        samples,
        sample_rate: spec.sample_rate,
    })
}

/// Linearly resample a mono signal. Adequate for short notification cues —
/// the rate ratios are small and pitch accuracy is not critical.
fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let last = input.len() - 1;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = input[idx.min(last)];
        let b = input[(idx + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Dispatch stream construction on the device's output sample format.
fn build_stream(
    sample_format: SampleFormat,
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: &[f32],
    channels: usize,
) -> Result<cpal::Stream, String> {
    match sample_format {
        SampleFormat::F32 => build_output::<f32>(device, config, samples, channels),
        SampleFormat::I16 => build_output::<i16>(device, config, samples, channels),
        SampleFormat::U16 => build_output::<u16>(device, config, samples, channels),
        other => Err(format!("unsupported output sample format: {other:?}")),
    }
}

/// Build an output stream that plays `samples` (mono) across all output
/// channels, then falls silent. The stream stops when dropped by the caller.
fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: &[f32],
    channels: usize,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let samples = samples.to_vec();
    let len = samples.len();
    let mut pos = 0usize;
    device
        .build_output_stream(
            *config,
            move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
                for frame in out.chunks_mut(channels.max(1)) {
                    let s = if pos < len {
                        let v = samples[pos];
                        pos += 1;
                        v
                    } else {
                        0.0
                    };
                    let value = T::from_sample(s);
                    for ch in frame.iter_mut() {
                        *ch = value;
                    }
                }
            },
            |err| debug!(error = %err, "sound output stream error (underrun/xrun?)"),
            None,
        )
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_cues_decode_to_nonempty_finite_mono() {
        for cue in [Cue::Start, Cue::Stop, Cue::Error] {
            let pcm = decode_mono(cue.bytes()).expect("decode");
            assert!(!pcm.samples.is_empty(), "{cue:?} decoded empty");
            assert_eq!(pcm.sample_rate, 44_100, "{cue:?} unexpected rate");
            assert!(
                pcm.samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0),
                "{cue:?} samples out of range"
            );
        }
    }

    #[test]
    fn perceptual_gain_keeps_endpoints_exact() {
        assert_eq!(perceptual_gain(0.0), 0.0);
        assert_eq!(perceptual_gain(1.0), 1.0);
    }

    #[test]
    fn perceptual_gain_puts_half_slider_near_half_loudness() {
        // −12 dB at 50% travel (vs −6 dB linear), near the −10 dB "half as
        // loud" psychoacoustic mark.
        let db = 20.0 * perceptual_gain(0.5).log10();
        assert!((db + 12.0).abs() < 0.1, "got {db} dB");
    }

    #[test]
    fn perceptual_gain_is_monotonic_and_clamps_out_of_range() {
        let mut prev = -1.0;
        for i in 0..=20 {
            let g = perceptual_gain(i as f32 / 20.0);
            assert!(g >= prev, "gain not monotonic at step {i}");
            prev = g;
        }
        assert_eq!(perceptual_gain(-0.5), 0.0);
        assert_eq!(perceptual_gain(1.5), 1.0);
    }

    #[test]
    fn resample_is_identity_when_rates_match() {
        let input = vec![0.0, 0.5, -0.5, 1.0];
        assert_eq!(resample_linear(&input, 44_100, 44_100), input);
    }

    #[test]
    fn resample_empty_stays_empty() {
        assert!(resample_linear(&[], 44_100, 48_000).is_empty());
    }

    #[test]
    fn resample_scales_length_by_rate_ratio() {
        let input = vec![0.0; 100];
        let out = resample_linear(&input, 44_100, 48_000);
        let expected = (100.0_f64 * 48_000.0 / 44_100.0).round() as usize;
        assert_eq!(out.len(), expected);
    }

    #[test]
    fn resample_interpolates_linearly() {
        // Upsampling [0, 1] 2x should place a ~0.5 midpoint.
        let out = resample_linear(&[0.0, 1.0], 1, 2);
        assert!(out.len() >= 3);
        assert!(
            (out[1] - 0.5).abs() < 0.2,
            "midpoint not interpolated: {out:?}"
        );
    }
}
