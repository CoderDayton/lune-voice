use crate::{AudioChunk, AudioError, ring::RingBuffer};
use audioadapter_buffers::direct::SequentialSlice;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};
use rubato::{Async, FixedAsync, PolynomialDegree, Resampler};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tracing::{debug, warn};

/// Configuration for audio capture
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Target sample rate (Hz). Default: 16000 (optimal for ASR)
    pub target_sample_rate: u32,

    /// Chunk duration in milliseconds. Default: 80ms
    ///
    /// Smaller chunks reduce onset-detection latency at the cost of more
    /// frequent ring-buffer pushes.  VAD and ASR run on the session thread,
    /// so the audio callback stays cheap regardless of chunk size.
    pub chunk_duration_ms: u32,

    /// Ring buffer capacity (number of chunks). Default: 128 (~10 seconds at 80ms)
    pub buffer_capacity: usize,

    /// Preferred input device by name. `None` (default) uses the host default
    /// input. A name that is no longer present falls back to the default.
    pub input_device: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            chunk_duration_ms: 80,
            buffer_capacity: 128,
            input_device: None,
        }
    }
}

/// Outcome of matching a saved input-device preference against the devices
/// actually present on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSelection {
    /// The device name to open, or `None` to fall back to the host default.
    pub device: Option<String>,
    /// `true` when a specific device was requested but is no longer present —
    /// the caller should surface a non-fatal warning and use the default.
    pub fell_back: bool,
}

/// Names of the input devices cpal currently reports on the default host, in
/// enumeration order. Empty if enumeration fails (the default device is still
/// usable). Used to populate the device picker and to validate a saved choice.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices
            .filter_map(|d| d.description().ok().map(|desc| desc.name().to_owned()))
            .collect(),
        Err(e) => {
            warn!("enumerating input devices failed: {e}");
            Vec::new()
        }
    }
}

/// Resolve a persisted input-device preference against the devices currently
/// available. `requested` is the saved device name (`None` = host default);
/// `available` is the list of present input-device names.
///
/// Device identity on Linux is by name (cpal exposes no portable persistent
/// id), so a renamed or unplugged device won't match and we fall back.
pub fn resolve_input_device(requested: Option<&str>, available: &[String]) -> DeviceSelection {
    match requested {
        None => DeviceSelection {
            device: None,
            fell_back: false,
        },
        Some(name) if available.iter().any(|d| d == name) => DeviceSelection {
            device: Some(name.to_string()),
            fell_back: false,
        },
        Some(_) => DeviceSelection {
            device: None,
            fell_back: true,
        },
    }
}

/// Audio capture stream with automatic resampling
pub struct AudioStream {
    config: AudioConfig,
    ring: Arc<RingBuffer<AudioChunk>>,
    /// When set, the audio callback emits its sub-chunk tail as a final short
    /// chunk on its next invocation. See [`AudioStream::request_flush`].
    flush: Arc<AtomicBool>,
    /// Kept alive to keep the cpal callback running. `None` for file-backed streams.
    _stream: Option<cpal::Stream>,
}

fn validate_config(config: &AudioConfig) -> Result<(), AudioError> {
    if config.target_sample_rate == 0 {
        return Err(AudioError::StreamBuild(
            "target_sample_rate must be greater than 0".into(),
        ));
    }
    if config.chunk_duration_ms == 0 {
        return Err(AudioError::StreamBuild(
            "chunk_duration_ms must be greater than 0".into(),
        ));
    }
    if config.buffer_capacity == 0 {
        return Err(AudioError::StreamBuild(
            "buffer_capacity must be greater than 0".into(),
        ));
    }
    // Catch overflow up front so the audio thread never panics on a
    // misconfigured rate × duration combination. computed_chunk_samples
    // also doubles as the validity check for the result being > 0.
    let _ = computed_chunk_samples(config)?;
    Ok(())
}

/// Resolve `target_sample_rate × chunk_duration_ms / 1000` as a usize,
/// surfacing overflow as a `StreamBuild` error rather than panicking on
/// the audio thread. Casting to usize before multiplying widens to 64 bits
/// on common targets and elides u32 overflow; `checked_mul` covers 32-bit.
fn computed_chunk_samples(config: &AudioConfig) -> Result<usize, AudioError> {
    let rate = config.target_sample_rate as usize;
    let dur_ms = config.chunk_duration_ms as usize;
    let samples = rate
        .checked_mul(dur_ms)
        .and_then(|n| n.checked_div(1000))
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            AudioError::StreamBuild(format!(
                "chunk_samples is invalid for rate={} Hz × duration={} ms",
                config.target_sample_rate, config.chunk_duration_ms
            ))
        })?;
    Ok(samples)
}

fn convert_i16_samples_into(data: &[i16], out: &mut Vec<f32>) {
    out.clear();
    out.extend(data.iter().map(|&s| s as f32 / 32768.0));
}

fn convert_u16_samples_into(data: &[u16], out: &mut Vec<f32>) {
    out.clear();
    out.extend(data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
}

// Channel-imbalance thresholds applied per audio callback:
//
// * `POWER_RATIO_LIMIT` (16.0) — if one channel carries >16× the energy of
//   the weakest, channels are imbalanced beyond ~12 dB and averaging would
//   attenuate the dominant content. Fall back to the strongest channel.
// * `STEREO_CORR_FLOOR` (0.1) — required correlation between L and R before
//   we trust their sum. Below this the channels are largely uncorrelated or
//   antiphase, and summing risks partial-to-total cancellation. Stereo-only
//   gate (multi-channel layouts don't have a single correlation coefficient).
const POWER_RATIO_LIMIT: f32 = 16.0;
const STEREO_CORR_FLOOR: f32 = 0.1;
const POWER_FLOOR: f32 = 1.0e-12;

fn clean_sample(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn downmix_mono_into(data: &[f32], channels: usize, out: &mut Vec<f32>) {
    out.clear();
    if channels <= 1 {
        out.extend(data.iter().map(|&s| clean_sample(s)));
        return;
    }

    if channels == 2 {
        let frames = data.len() / 2;
        if frames == 0 {
            return;
        }

        // Single pass: emit the optimistic average while accumulating channel
        // power. If the imbalance/correlation checks reject averaging we
        // overwrite the buffer below. The expected case (coherent stereo) is
        // therefore one pass over `data`.
        out.reserve(frames);
        let mut left_power = 0.0f32;
        let mut right_power = 0.0f32;
        let mut cross = 0.0f32;
        for frame in data.chunks_exact(2) {
            let left = clean_sample(frame[0]);
            let right = clean_sample(frame[1]);
            left_power += left * left;
            right_power += right * right;
            cross += left * right;
            out.push(0.5 * (left + right));
        }

        let strongest_power = left_power.max(right_power);
        let weakest_power = left_power.min(right_power);
        let power_ratio = strongest_power / weakest_power.max(POWER_FLOOR);
        let stereo_corr = cross / (left_power * right_power).sqrt().max(POWER_FLOOR);
        let average_channels = power_ratio <= POWER_RATIO_LIMIT && stereo_corr >= STEREO_CORR_FLOOR;

        if !average_channels {
            let strongest_channel = if left_power >= right_power { 0 } else { 1 };
            for (slot, frame) in out.iter_mut().zip(data.chunks_exact(2)) {
                *slot = clean_sample(frame[strongest_channel]);
            }
        }
        return;
    }

    let frames = data.len() / channels;
    if frames == 0 {
        return;
    }
    if channels > 32 {
        // Beyond 32 channels (uncommon — ambisonic / mixdown busses) skip
        // imbalance detection and average; the fixed-size power array below
        // would not fit, and these layouts are not realistic capture devices.
        out.extend(
            data.chunks_exact(channels)
                .map(|frame| frame.iter().map(|&s| clean_sample(s)).sum::<f32>() / channels as f32),
        );
        return;
    }

    // Same single-pass strategy as the stereo branch: emit the optimistic
    // average while accumulating per-channel power, then overwrite if the
    // imbalance check rejects averaging.
    let mut channel_power = [0.0f32; 32];
    let inv_channels = 1.0 / channels as f32;
    out.reserve(frames);
    for frame in data.chunks_exact(channels) {
        let mut sum = 0.0f32;
        for (ch, &sample) in frame.iter().enumerate() {
            let s = clean_sample(sample);
            channel_power[ch] += s * s;
            sum += s;
        }
        out.push(sum * inv_channels);
    }

    let powers = &channel_power[..channels];
    let (strongest_channel, strongest_power) = powers
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .unwrap_or((0, 0.0));
    let weakest_power = powers.iter().copied().fold(f32::INFINITY, f32::min);
    let power_ratio = strongest_power / weakest_power.max(POWER_FLOOR);
    if power_ratio > POWER_RATIO_LIMIT {
        for (slot, frame) in out.iter_mut().zip(data.chunks_exact(channels)) {
            *slot = clean_sample(frame[strongest_channel]);
        }
    }
}

impl AudioStream {
    /// Create a new audio stream with the given configuration
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        validate_config(&config)?;
        let host = cpal::default_host();

        // Enumerate inputs once and keep the device handles, so the saved-device
        // resolution and the actual selection are derived from the same list —
        // they can't disagree, and we avoid enumerating twice.
        let devices = host
            .input_devices()
            .map(|it| it.collect::<Vec<_>>())
            .unwrap_or_default();
        let available: Vec<String> = devices
            .iter()
            .filter_map(|d| d.description().ok().map(|desc| desc.name().to_owned()))
            .collect();

        // Honor the saved device preference, falling back to the host default
        // when it's absent (unplugged/renamed) — never fatal.
        let selection = resolve_input_device(config.input_device.as_deref(), &available);
        if selection.fell_back {
            warn!(
                "saved input device {:?} not found; using system default",
                config.input_device
            );
        }
        let device = match &selection.device {
            Some(name) => devices
                .into_iter()
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name() == name)
                        .unwrap_or(false)
                })
                .or_else(|| host.default_input_device())
                .ok_or(AudioError::NoDevice)?,
            None => host.default_input_device().ok_or(AudioError::NoDevice)?,
        };

        let device_name = device
            .description()
            .map(|d| d.name().to_owned())
            .unwrap_or_else(|_| "Unknown".into());
        debug!("Using audio device: {device_name}");

        // Query supported configs and pick the best one
        let supported_config = device
            .default_input_config()
            .map_err(|e| AudioError::StreamBuild(e.to_string()))?;

        let native_sample_rate = supported_config.sample_rate();
        let channels = supported_config.channels();

        debug!(
            "Native format: {native_sample_rate}Hz, {channels} channels, {:?}",
            supported_config.sample_format()
        );

        // Build stream config
        let chunk_samples = computed_chunk_samples(&config)?;
        let stream_config = StreamConfig {
            channels,
            sample_rate: native_sample_rate,
            buffer_size: BufferSize::Default,
        };

        let ring = Arc::new(
            RingBuffer::new(config.buffer_capacity)
                .map_err(|e| AudioError::StreamBuild(e.into()))?,
        );
        let ring_clone = Arc::clone(&ring);

        let target_rate = config.target_sample_rate;
        let start_time = Instant::now();
        let flush = Arc::new(AtomicBool::new(false));
        let flush_cb = Arc::clone(&flush);

        // Shared state for the audio callback
        let mut resampler = if native_sample_rate != target_rate {
            Some(
                Async::<f32>::new_poly(
                    target_rate as f64 / native_sample_rate as f64,
                    1.2, // Max relative ratio deviation
                    PolynomialDegree::Septic,
                    chunk_samples,
                    1, // Mono after downmix
                    FixedAsync::Input,
                )
                .map_err(|e: rubato::ResamplerConstructionError| {
                    AudioError::Resampler(e.to_string())
                })?,
            )
        } else {
            None
        };

        // Scratch buffers pre-allocated once and reused every callback —
        // avoids per-frame heap allocation on the real-time audio thread.
        let mut input_resample_buf = Vec::with_capacity(chunk_samples * 2);
        let mut output_buf = Vec::with_capacity(chunk_samples * 2);
        let mut mono_buf = Vec::<f32>::with_capacity(chunk_samples * 4);
        let mut resample_input_slice = Vec::<f32>::with_capacity(chunk_samples * 2);

        // Build the stream based on sample format
        // Macro to reduce boilerplate for stream building across sample formats.
        // Each format (F32, I16, U16) differs only in the callback signature and whether
        // sample conversion is needed. This macro captures the common pattern of building
        // a stream, error handling, and calling process_audio_callback.
        macro_rules! build_stream {
            // F32 variant: no conversion needed, pass data directly
            (f32) => {
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[f32], _: &_| {
                            process_audio_callback(
                                data,
                                channels as usize,
                                &mut resampler,
                                &mut input_resample_buf,
                                &mut resample_input_slice,
                                &mut output_buf,
                                &mut mono_buf,
                                chunk_samples,
                                &ring_clone,
                                start_time,
                                &flush_cb,
                            );
                        },
                        |err| warn!("Audio stream error: {err}"),
                        None,
                    )
                    .map_err(|e| AudioError::StreamBuild(e.to_string()))?
            };
            // I16/U16 variants: allocate conversion buffer and convert before processing
            ($sample_type:ty, $convert_fn:expr) => {{
                let mut sample_buf = Vec::<f32>::with_capacity(chunk_samples * 4);
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[$sample_type], _: &_| {
                            $convert_fn(data, &mut sample_buf);
                            process_audio_callback(
                                &sample_buf,
                                channels as usize,
                                &mut resampler,
                                &mut input_resample_buf,
                                &mut resample_input_slice,
                                &mut output_buf,
                                &mut mono_buf,
                                chunk_samples,
                                &ring_clone,
                                start_time,
                                &flush_cb,
                            );
                        },
                        |err| warn!("Audio stream error: {err}"),
                        None,
                    )
                    .map_err(|e| AudioError::StreamBuild(e.to_string()))?
            }};
        }

        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => build_stream!(f32),
            SampleFormat::I16 => build_stream!(i16, convert_i16_samples_into),
            SampleFormat::U16 => build_stream!(u16, convert_u16_samples_into),
            format => {
                return Err(AudioError::StreamBuild(format!(
                    "Unsupported sample format: {format:?}"
                )));
            }
        };

        stream
            .play()
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        debug!("Audio stream started successfully");

        Ok(Self {
            config,
            ring,
            flush,
            _stream: Some(stream),
        })
    }

    /// Create an `AudioStream` pre-loaded with PCM samples (16 kHz mono f32).
    ///
    /// Chunks are pushed into the ring buffer immediately so `try_next_chunk`
    /// drains them at the consumer's pace — no real-time audio thread involved.
    /// Useful for integration tests and offline file processing.
    pub fn from_samples(samples: &[f32], config: AudioConfig) -> Result<Self, AudioError> {
        validate_config(&config)?;
        let chunk_samples = computed_chunk_samples(&config)?;
        // Auto-size buffer to fit all chunks so nothing is dropped.
        let num_chunks = samples.len().div_ceil(chunk_samples);
        let capacity = num_chunks.max(config.buffer_capacity);
        let ring =
            Arc::new(RingBuffer::new(capacity).map_err(|e| AudioError::StreamBuild(e.into()))?);
        let mut offset_ms = 0u64;
        for chunk in samples.chunks(chunk_samples) {
            ring.push(AudioChunk {
                samples: chunk.to_vec(),
                timestamp_ms: offset_ms,
            });
            offset_ms += u64::from(config.chunk_duration_ms);
        }
        Ok(Self {
            config,
            ring,
            flush: Arc::new(AtomicBool::new(false)),
            _stream: None,
        })
    }

    /// Get the next audio chunk (async, blocks until data available)
    pub async fn next_chunk(&self) -> Option<AudioChunk> {
        self.ring.pop().await
    }

    /// Try to get the next chunk without blocking
    pub fn try_next_chunk(&self) -> Option<AudioChunk> {
        self.ring.try_pop()
    }

    /// Returns `true` if this stream was created from pre-loaded samples
    /// (no live audio thread). When the ring buffer is empty, no more audio
    /// will ever arrive.
    pub fn is_offline(&self) -> bool {
        self._stream.is_none()
    }

    /// Get current buffer fill percentage (0.0 to 1.0)
    /// Values near 1.0 indicate backpressure (consumer is slow)
    pub fn buffer_fill(&self) -> f32 {
        self.ring.len() as f32 / self.config.buffer_capacity as f32
    }

    /// Discard every chunk currently buffered in the ring. Call before a
    /// new consumer starts on a long-lived stream so stale audio captured
    /// during the idle period isn't transcribed.
    pub fn drain(&self) {
        while self.ring.try_pop().is_some() {}
    }

    /// Drop chunks older than the most recent `keep` chunks. Use on
    /// session start to discard idle-period audio while preserving the
    /// short window captured between hotkey press and consumer startup.
    pub fn drain_keep_last(&self, keep: usize) {
        let to_drop = self.ring.len().saturating_sub(keep);
        for _ in 0..to_drop {
            if self.ring.try_pop().is_none() {
                break;
            }
        }
    }

    /// Ask the audio callback to emit its buffered sub-chunk tail (up to one
    /// chunk duration of audio) as a final short chunk on its next invocation.
    /// Without this, audio captured after the last full chunk boundary —
    /// often the final phonemes of the last word — is silently dropped when
    /// the session stops. No-op for file-backed streams (`from_samples`
    /// already pushes its partial tail chunk).
    pub fn request_flush(&self) {
        self.flush.store(true, Ordering::Release);
    }

    /// Lightweight handle that shares this stream's ring buffer.
    ///
    /// Multiple handles can coexist; whichever consumer pops first wins.
    /// In practice the runtime holds the underlying `AudioStream` long
    /// enough to outlive any handles, and only one session consumes at a
    /// time.
    pub fn handle(&self) -> AudioStreamHandle {
        AudioStreamHandle {
            config: self.config.clone(),
            ring: Arc::clone(&self.ring),
            flush: Arc::clone(&self.flush),
        }
    }
}

/// Cheap-to-clone reader that shares an `AudioStream`'s ring buffer.
///
/// Lifetime of the underlying live capture is tied to the source
/// `AudioStream` — drop the source and no new chunks arrive (already
/// queued chunks remain consumable until drained).
#[derive(Clone)]
pub struct AudioStreamHandle {
    config: AudioConfig,
    ring: Arc<RingBuffer<AudioChunk>>,
    flush: Arc<AtomicBool>,
}

impl AudioStreamHandle {
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// See `AudioStream::request_flush`.
    pub fn request_flush(&self) {
        self.flush.store(true, Ordering::Release);
    }

    pub async fn next_chunk(&self) -> Option<AudioChunk> {
        self.ring.pop().await
    }

    pub fn try_next_chunk(&self) -> Option<AudioChunk> {
        self.ring.try_pop()
    }

    pub fn drain(&self) {
        while self.ring.try_pop().is_some() {}
    }

    /// See `AudioStream::drain_keep_last`.
    pub fn drain_keep_last(&self, keep: usize) {
        let to_drop = self.ring.len().saturating_sub(keep);
        for _ in 0..to_drop {
            if self.ring.try_pop().is_none() {
                break;
            }
        }
    }

    pub fn buffer_fill(&self) -> f32 {
        self.ring.len() as f32 / self.config.buffer_capacity as f32
    }
}

/// Core audio processing callback (runs in real-time audio thread)
#[allow(clippy::too_many_arguments)]
fn process_audio_callback(
    data: &[f32],
    channels: usize,
    resampler: &mut Option<Async<f32>>,
    input_resample_buf: &mut Vec<f32>,
    resample_input_slice: &mut Vec<f32>,
    output_buf: &mut Vec<f32>,
    mono_buf: &mut Vec<f32>,
    chunk_samples: usize,
    ring: &RingBuffer<AudioChunk>,
    start_time: Instant,
    flush: &AtomicBool,
) {
    // 1. Downmix to mono. Averaging coherent channels is fine, but some
    // devices expose antiphase or highly imbalanced channels; in those cases
    // keep the strongest channel so speech is not attenuated before ASR.
    downmix_mono_into(data, channels, mono_buf);

    // 2. Resample if needed
    if let Some(r) = resampler {
        input_resample_buf.extend_from_slice(mono_buf);
        // Cap resample buffer to prevent unbounded growth if the resampler
        // repeatedly errors. ~1s of 48kHz audio is the upper sane bound for a
        // backlog before we're better off dropping than queueing forever.
        const MAX_RESAMPLE_BACKLOG: usize = 48_000;
        if input_resample_buf.len() > MAX_RESAMPLE_BACKLOG {
            warn!(
                "Resample backlog {} exceeds cap; dropping oldest",
                input_resample_buf.len()
            );
            let drop_n = input_resample_buf.len() - MAX_RESAMPLE_BACKLOG;
            input_resample_buf.drain(..drop_n);
        }
        let required_input = r.input_frames_next();
        while input_resample_buf.len() >= required_input {
            // Reuse resample_input_slice instead of allocating per iteration.
            resample_input_slice.clear();
            resample_input_slice.extend_from_slice(&input_resample_buf[..required_input]);
            input_resample_buf.drain(..required_input);
            let frames = resample_input_slice.len();
            let Ok(adapter) = SequentialSlice::new(resample_input_slice.as_slice(), 1, frames)
            else {
                // Adapter construction shouldn't fail with valid contiguous input,
                // but never panic in the audio thread — drop this iteration's frames.
                warn!("SequentialSlice construction failed; skipping {frames} frames");
                continue;
            };
            match r.process(&adapter, 0, None) {
                Ok(output) => output_buf.extend_from_slice(&output.take_data()),
                Err(e) => {
                    warn!("Resampling error: {e}");
                    // Reset internal pending data so a transient resampler error
                    // doesn't keep us in a degraded state forever.
                    input_resample_buf.clear();
                    break;
                }
            }
        }
    } else {
        output_buf.extend_from_slice(mono_buf);
    }

    while output_buf.len() >= chunk_samples {
        let mut chunk_data = Vec::with_capacity(chunk_samples);
        chunk_data.extend(output_buf.drain(..chunk_samples));

        let timestamp_ms = start_time.elapsed().as_millis() as u64;
        let chunk = AudioChunk {
            samples: chunk_data,
            timestamp_ms,
        };

        // RingBuffer.push handles eviction atomically — no TOCTOU race.
        ring.push(chunk);
    }

    // Flush request: emit the sub-chunk tail so the session's final word
    // isn't clipped at the last full-chunk boundary. Consumers accept
    // arbitrary chunk lengths (VAD carries partial frames; ASR appends raw
    // samples), so a short final chunk is safe.
    if flush.swap(false, Ordering::AcqRel) {
        // Drain the resampler backlog first: on a non-16 kHz device the
        // partial input frame held here (up to ~27 ms on a 48 kHz mic —
        // often the last phonemes) would otherwise never reach the ring.
        // Zero-padding to a full frame costs a brief silent tail, which the
        // downstream VAD endpoint-trim removes.
        if let Some(r) = resampler
            && !input_resample_buf.is_empty()
        {
            let required_input = r.input_frames_next();
            input_resample_buf.resize(required_input, 0.0);
            let frames = input_resample_buf.len();
            match SequentialSlice::new(input_resample_buf.as_slice(), 1, frames) {
                Ok(adapter) => match r.process(&adapter, 0, None) {
                    Ok(output) => output_buf.extend_from_slice(&output.take_data()),
                    Err(e) => warn!("Flush resampling error: {e}"),
                },
                Err(_) => warn!("SequentialSlice construction failed during flush"),
            }
            input_resample_buf.clear();
        }
        if !output_buf.is_empty() {
            let samples = std::mem::take(output_buf);
            let timestamp_ms = start_time.elapsed().as_millis() as u64;
            ring.push(AudioChunk {
                samples,
                timestamp_ms,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_samples_rejects_zero_chunk_duration() {
        let err = AudioStream::from_samples(
            &[0.0f32; 16],
            AudioConfig {
                chunk_duration_ms: 0,
                ..AudioConfig::default()
            },
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("chunk_duration_ms"));
    }

    #[test]
    fn from_samples_rejects_zero_sample_rate() {
        let err = AudioStream::from_samples(
            &[0.0f32; 16],
            AudioConfig {
                target_sample_rate: 0,
                ..AudioConfig::default()
            },
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("target_sample_rate"));
    }

    /// Regression: a zero ring buffer would silently drop every chunk the
    /// audio callback pushed. `validate_config` must reject it up front
    /// — both audio-device construction and `from_samples` rely on this
    /// single source of truth.
    #[test]
    fn validate_config_rejects_zero_buffer_capacity() {
        let err = validate_config(&AudioConfig {
            buffer_capacity: 0,
            ..AudioConfig::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("buffer_capacity"), "got: {err}");
    }

    /// Regression: `target_sample_rate × chunk_duration_ms / 1000` can
    /// truncate to zero for low rates × short durations (e.g. 500 Hz × 1 ms).
    /// `validate_config` must catch this — otherwise the audio thread
    /// later panics on `chunks_exact(0)` or pushes empty buffers into
    /// the ring forever.
    #[test]
    fn validate_config_rejects_chunk_samples_truncating_to_zero() {
        let err = validate_config(&AudioConfig {
            target_sample_rate: 500,
            chunk_duration_ms: 1,
            ..AudioConfig::default()
        })
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("chunk_samples"),
            "expected message to mention chunk_samples; got: {msg}"
        );
    }

    /// The prompt's named edge case: a saved device that has since been
    /// unplugged or renamed must NOT be opened — resolution falls back to the
    /// host default and flags it so the caller can warn (never panic/error).
    #[test]
    fn resolve_input_device_falls_back_when_requested_missing() {
        let available = vec!["Built-in Microphone".to_string(), "HDMI".to_string()];
        let sel = resolve_input_device(Some("USB Headset"), &available);
        assert_eq!(
            sel,
            DeviceSelection {
                device: None,
                fell_back: true,
            }
        );
    }

    #[test]
    fn resolve_input_device_selects_requested_when_present() {
        let available = vec!["Built-in Microphone".to_string(), "USB Headset".to_string()];
        let sel = resolve_input_device(Some("USB Headset"), &available);
        assert_eq!(
            sel,
            DeviceSelection {
                device: Some("USB Headset".to_string()),
                fell_back: false,
            }
        );
    }

    #[test]
    fn resolve_input_device_uses_default_when_none_requested() {
        let available = vec!["Built-in Microphone".to_string()];
        let sel = resolve_input_device(None, &available);
        assert_eq!(
            sel,
            DeviceSelection {
                device: None,
                fell_back: false,
            }
        );
    }

    #[test]
    fn convert_u16_samples_centers_midpoint() {
        let mut converted = Vec::new();
        convert_u16_samples_into(&[0, 32768, u16::MAX], &mut converted);
        assert!(converted[0] <= -1.0);
        assert_eq!(converted[1], 0.0);
        assert!(converted[2] > 0.99);
    }

    #[test]
    fn convert_i16_samples_scales_signed_range() {
        let mut converted = Vec::new();
        convert_i16_samples_into(&[i16::MIN, 0, i16::MAX], &mut converted);
        assert_eq!(converted[0], -1.0);
        assert_eq!(converted[1], 0.0);
        assert!(converted[2] > 0.99);
    }

    #[test]
    fn downmix_averages_coherent_stereo() {
        let data = [0.2, 0.4, -0.4, -0.2];
        let mut out = Vec::new();
        downmix_mono_into(&data, 2, &mut out);
        assert_eq!(out, vec![0.3, -0.3]);
    }

    #[test]
    fn downmix_preserves_antiphase_stereo() {
        let data = [0.5, -0.5, -0.25, 0.25];
        let mut out = Vec::new();
        downmix_mono_into(&data, 2, &mut out);
        assert_eq!(out, vec![0.5, -0.25]);
    }

    #[test]
    fn downmix_uses_strongest_channel_when_imbalanced() {
        let data = [0.02, 0.6, -0.02, -0.6];
        let mut out = Vec::new();
        downmix_mono_into(&data, 2, &mut out);
        assert_eq!(out, vec![0.6, -0.6]);
    }
}
