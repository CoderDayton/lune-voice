// SPDX-License-Identifier: GPL-3.0-or-later

use crate::devices::{
    capture_buffer_size, capture_host, enumerate_input_devices, resolve_input_device,
};
use crate::{AudioChunk, AudioError, ring::RingBuffer};
use audioadapter_buffers::direct::SequentialSlice;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tracing::{debug, info, warn};

/// Configuration for audio capture
#[derive(Debug, Clone, Default)]
pub struct AudioConfig {
    /// Preferred input device by name. `None` (default) uses the host default
    /// input. A name that is no longer present falls back to the default.
    pub input_device: Option<String>,
}

const TARGET_SAMPLE_RATE: u32 = 16_000;
const CHUNK_DURATION_MS: usize = 80;
const CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * CHUNK_DURATION_MS / 1000;
const BUFFER_CAPACITY: usize = 128;

/// Audio capture stream with automatic resampling
pub struct AudioStream {
    handle: AudioStreamHandle,
    /// Set by cpal's error callback. A device that disappears (unplugged USB
    /// headset, a suspended dock) stops delivering audio but leaves the stream
    /// object alive and apparently fine, so without this flag a prewarmed
    /// stream is kept forever and every later dictation dies on the starvation
    /// watchdog until an unrelated setting is touched.
    failed: Arc<AtomicBool>,
    /// The configured input device was not present, so this stream is on the
    /// system default instead. Transcripts are degraded rather than absent,
    /// which is exactly why it has to be reported rather than inferred.
    fell_back: bool,
    /// Kept alive to keep the cpal callback running. `None` for file-backed streams.
    _stream: Option<cpal::Stream>,
}

fn convert_i16_samples_into(data: &[i16], out: &mut Vec<f32>) {
    out.clear();
    out.extend(data.iter().map(|&s| s as f32 / 32768.0));
}

fn convert_i32_samples_into(data: &[i32], out: &mut Vec<f32>) {
    out.clear();
    out.extend(data.iter().map(|&s| s as f32 / 2_147_483_648.0));
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
        let host = capture_host();
        info!(host = ?host.id(), "opening capture stream");

        // Enumerate inputs once and keep the device handles, so the saved-device
        // resolution and the actual selection are derived from the same list —
        // they can't disagree, and we avoid enumerating twice.
        let mut devices = enumerate_input_devices(&host);
        let available: Vec<String> = devices.iter().map(|d| d.name.clone()).collect();

        // Honor the saved device preference, falling back to the host default
        // when it's absent (unplugged/renamed) — never fatal.
        let selection = resolve_input_device(config.input_device.as_deref(), &available);
        if selection.fell_back {
            warn!(
                "saved input device {:?} not found; using system default",
                config.input_device
            );
        }
        let device = match selection.device {
            Some(idx) => devices.swap_remove(idx).device,
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
        let chunk_samples = CHUNK_SAMPLES;
        let stream_config = StreamConfig {
            channels,
            sample_rate: native_sample_rate,
            buffer_size: capture_buffer_size(
                &host,
                native_sample_rate,
                supported_config.buffer_size(),
            ),
        };

        let ring = Arc::new(
            RingBuffer::new(BUFFER_CAPACITY).map_err(|e| AudioError::StreamBuild(e.into()))?,
        );
        let failed = Arc::new(AtomicBool::new(false));
        let failed_cb = Arc::clone(&failed);
        let ring_cb = Arc::clone(&ring);
        let flush = Arc::new(FlushState::new(true));
        let flush_cb = Arc::clone(&flush);

        let mut processor = CaptureProcessor::new(
            channels as usize,
            native_sample_rate,
            TARGET_SAMPLE_RATE,
            chunk_samples,
        )?;

        macro_rules! build_stream {
            (f32) => {
                device
                    .build_input_stream(
                        stream_config,
                        move |data: &[f32], _: &_| processor.process(data, &ring_cb, &flush_cb),
                        {
                            let failed = Arc::clone(&failed_cb);
                            move |err| {
                                warn!("Audio stream error: {err}");
                                failed.store(true, Ordering::Release);
                            }
                        },
                        None,
                    )
                    .map_err(|e| AudioError::StreamBuild(e.to_string()))?
            };
            ($sample_type:ty, $convert_fn:expr) => {{
                let mut sample_buf = Vec::<f32>::with_capacity(chunk_samples * 4);
                device
                    .build_input_stream(
                        stream_config,
                        move |data: &[$sample_type], _: &_| {
                            $convert_fn(data, &mut sample_buf);
                            processor.process(&sample_buf, &ring_cb, &flush_cb);
                        },
                        {
                            let failed = Arc::clone(&failed_cb);
                            move |err| {
                                warn!("Audio stream error: {err}");
                                failed.store(true, Ordering::Release);
                            }
                        },
                        None,
                    )
                    .map_err(|e| AudioError::StreamBuild(e.to_string()))?
            }};
        }

        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => build_stream!(f32),
            SampleFormat::I16 => build_stream!(i16, convert_i16_samples_into),
            SampleFormat::I32 => build_stream!(i32, convert_i32_samples_into),
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
            handle: AudioStreamHandle { ring, flush },
            failed,
            fell_back: selection.fell_back,
            _stream: Some(stream),
        })
    }

    /// Create an `AudioStream` pre-loaded with PCM samples (16 kHz mono f32).
    ///
    /// Chunks are pushed into the ring buffer immediately so `try_next_chunk`
    /// drains them at the consumer's pace — no real-time audio thread involved.
    /// Useful for integration tests and offline file processing.
    pub fn from_samples(samples: &[f32], _config: AudioConfig) -> Result<Self, AudioError> {
        let chunk_samples = CHUNK_SAMPLES;
        // Auto-size buffer to fit all chunks so nothing is dropped.
        let num_chunks = samples.len().div_ceil(chunk_samples);
        let capacity = num_chunks.max(BUFFER_CAPACITY);
        let ring =
            Arc::new(RingBuffer::new(capacity).map_err(|e| AudioError::StreamBuild(e.into()))?);
        for chunk in samples.chunks(chunk_samples) {
            ring.push(AudioChunk {
                samples: chunk.to_vec(),
            });
        }
        Ok(Self {
            handle: AudioStreamHandle {
                ring,
                flush: Arc::new(FlushState::new(false)),
            },
            failed: Arc::new(AtomicBool::new(false)),
            fell_back: false,
            _stream: None,
        })
    }

    /// Try to get the next chunk without blocking
    pub fn try_next_chunk(&self) -> Option<AudioChunk> {
        self.handle.try_next_chunk()
    }

    /// Returns `true` if this stream was created from pre-loaded samples
    /// (no live audio thread). When the ring buffer is empty, no more audio
    /// will ever arrive.
    pub fn is_offline(&self) -> bool {
        self._stream.is_none()
    }

    /// Chunks discarded because the ring buffer filled: audio a reader lost
    /// because it fell behind.
    pub fn dropped_chunks(&self) -> u64 {
        self.handle.dropped_chunks()
    }

    /// Mark this stream as actively read for as long as the guard lives.
    ///
    /// A pre-warmed stream outlives every session, so between dictations its
    /// ring is a rolling pre-roll window and evicting the oldest chunk is
    /// expected. Only inside a guard does an eviction mean a reader lost audio.
    pub fn consume(&self) -> ConsumerGuard {
        self.handle.consume()
    }

    /// Whether the configured input device was missing and the system default
    /// was used in its place.
    pub fn fell_back(&self) -> bool {
        self.fell_back
    }

    /// Whether cpal has reported an error on this stream.
    ///
    /// Sticky: a device does not come back on its own, so once this is true the
    /// stream has to be rebuilt rather than waited on.
    pub fn has_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Current buffer fill fraction (0.0 to 1.0). Values near 1.0 indicate
    /// backpressure — the consumer is slow.
    pub fn buffer_fill(&self) -> f32 {
        self.handle.buffer_fill()
    }

    /// Drop chunks older than the most recent `keep` chunks. Use on
    /// session start to discard idle-period audio while preserving the
    /// short window captured between hotkey press and consumer startup.
    pub fn drain_keep_last(&self, keep: usize) {
        self.handle.drain_keep_last(keep);
    }

    /// Ask the audio callback to emit its buffered sub-chunk tail (up to one
    /// chunk duration of audio) as a final short chunk on its next invocation.
    /// Without this, audio captured after the last full chunk boundary —
    /// often the final phonemes of the last word — is silently dropped when
    /// the session stops. No-op for file-backed streams (`from_samples`
    /// already pushes its partial tail chunk).
    pub fn request_flush(&self) {
        self.handle.request_flush();
    }

    pub fn flush_pending(&self) -> bool {
        self.handle.flush_pending()
    }

    /// Lightweight handle that shares this stream's ring buffer.
    ///
    /// Multiple handles can coexist; whichever consumer pops first wins.
    /// In practice the runtime holds the underlying `AudioStream` long
    /// enough to outlive any handles, and only one session consumes at a
    /// time.
    pub fn handle(&self) -> AudioStreamHandle {
        self.handle.clone()
    }
}

/// Cheap-to-clone reader that shares an `AudioStream`'s ring buffer.
///
/// Lifetime of the underlying live capture is tied to the source
/// `AudioStream` — drop the source and no new chunks arrive (already
/// queued chunks remain consumable until drained).
#[derive(Clone)]
pub struct AudioStreamHandle {
    ring: Arc<RingBuffer<AudioChunk>>,
    flush: Arc<FlushState>,
}

impl AudioStreamHandle {
    /// See [`AudioStream::dropped_chunks`].
    pub fn dropped_chunks(&self) -> u64 {
        self.ring.evicted()
    }

    /// See [`AudioStream::consume`].
    pub fn consume(&self) -> ConsumerGuard {
        ConsumerGuard::new(Arc::clone(&self.ring))
    }

    /// See `AudioStream::request_flush`.
    pub fn request_flush(&self) {
        self.flush.request();
    }

    pub fn flush_pending(&self) -> bool {
        self.flush.pending()
    }

    pub fn try_next_chunk(&self) -> Option<AudioChunk> {
        self.ring.try_pop()
    }

    /// See `AudioStream::drain_keep_last`.
    pub fn drain_keep_last(&self, keep: usize) {
        while self.ring.len() > keep {
            if self.ring.try_pop().is_none() {
                break;
            }
        }
    }

    /// See [`AudioStream::buffer_fill`].
    pub fn buffer_fill(&self) -> f32 {
        self.ring.len() as f32 / self.ring.capacity() as f32
    }
}

/// Registers an active reader on a ring buffer, deregistering on drop.
///
/// Holding one switches the ring from "idle pre-roll window" (evictions are
/// the design) to "someone is reading" (evictions are lost audio, counted and
/// warned about). Tie its lifetime to the capture loop, not to the handle:
/// the runtime keeps a pre-warmed handle alive the whole time the app runs.
#[must_use = "evictions stop counting as data loss as soon as this is dropped"]
pub struct ConsumerGuard {
    ring: Arc<RingBuffer<AudioChunk>>,
}

impl ConsumerGuard {
    fn new(ring: Arc<RingBuffer<AudioChunk>>) -> Self {
        ring.attach_consumer();
        Self { ring }
    }
}

impl Drop for ConsumerGuard {
    fn drop(&mut self) {
        self.ring.detach_consumer();
    }
}

struct FlushState {
    requested: AtomicU64,
    done: AtomicU64,
    live: bool,
}

impl FlushState {
    fn new(live: bool) -> Self {
        Self {
            requested: AtomicU64::new(0),
            done: AtomicU64::new(0),
            live,
        }
    }

    fn request(&self) {
        if self.live {
            self.requested.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn pending(&self) -> bool {
        self.done.load(Ordering::Acquire) < self.requested.load(Ordering::Acquire)
    }

    fn outstanding(&self) -> Option<u64> {
        let req = self.requested.load(Ordering::Acquire);
        (req != self.done.load(Ordering::Relaxed)).then_some(req)
    }

    fn ack(&self, req: u64) {
        self.done.store(req, Ordering::Release);
    }
}

fn build_resampler(native: u32, target: u32, chunk: usize) -> Result<Async<f32>, AudioError> {
    let params = SincInterpolationParameters::new(128, WindowFunction::BlackmanHarris2)
        .oversampling_factor(256)
        .interpolation(SincInterpolationType::Linear);
    Async::<f32>::new_sinc(
        f64::from(target) / f64::from(native),
        1.0,
        &params,
        chunk,
        1,
        FixedAsync::Input,
    )
    .map_err(|e| AudioError::Resampler(e.to_string()))
}

fn resample_into(
    r: &mut Async<f32>,
    input: &[f32],
    partial: Option<usize>,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let start = out.len();
    let max = r.output_frames_next();
    out.resize(start + max, 0.0);
    let adapter_in = SequentialSlice::new(input, 1, input.len()).map_err(|e| e.to_string())?;
    let mut adapter_out =
        SequentialSlice::new_mut(&mut out[start..], 1, max).map_err(|e| e.to_string())?;
    let indexing = partial.map(|n| Indexing::new().partial_len(n));
    let result = r.process_into_buffer(&adapter_in, &mut adapter_out, indexing.as_ref());
    let written = result.as_ref().map_or(0, |&(_, n)| n);
    out.truncate(start + written);
    result.map(|_| ()).map_err(|e| e.to_string())
}

struct CaptureProcessor {
    channels: usize,
    chunk_samples: usize,
    resampler: Option<Async<f32>>,
    mono: Vec<f32>,
    pending_in: Vec<f32>,
    out: Vec<f32>,
}

impl CaptureProcessor {
    fn new(
        channels: usize,
        native_rate: u32,
        target_rate: u32,
        chunk_samples: usize,
    ) -> Result<Self, AudioError> {
        let resampler = (native_rate != target_rate)
            .then(|| build_resampler(native_rate, target_rate, chunk_samples))
            .transpose()?;
        let block_out = resampler.as_ref().map_or(0, |r| r.output_frames_max());
        Ok(Self {
            channels,
            chunk_samples,
            resampler,
            mono: Vec::with_capacity(chunk_samples * 4),
            pending_in: Vec::with_capacity(chunk_samples * 4),
            out: Vec::with_capacity(chunk_samples * 4 + block_out),
        })
    }

    fn process(&mut self, data: &[f32], ring: &RingBuffer<AudioChunk>, flush: &FlushState) {
        downmix_mono_into(data, self.channels, &mut self.mono);

        if let Some(r) = self.resampler.as_mut() {
            self.pending_in.extend_from_slice(&self.mono);
            const MAX_RESAMPLE_BACKLOG: usize = 48_000;
            if self.pending_in.len() > MAX_RESAMPLE_BACKLOG {
                warn!(
                    "Resample backlog {} exceeds cap; dropping oldest",
                    self.pending_in.len()
                );
                let drop_n = self.pending_in.len() - MAX_RESAMPLE_BACKLOG;
                self.pending_in.drain(..drop_n);
            }
            let need = r.input_frames_next();
            let mut pos = 0;
            while self.pending_in.len() - pos >= need {
                if let Err(e) =
                    resample_into(r, &self.pending_in[pos..pos + need], None, &mut self.out)
                {
                    warn!("Resampling error: {e}");
                    pos = self.pending_in.len();
                    break;
                }
                pos += need;
            }
            self.pending_in.drain(..pos);
        } else {
            self.out.extend_from_slice(&self.mono);
        }

        let mut pos = 0;
        while self.out.len() - pos >= self.chunk_samples {
            ring.push(AudioChunk {
                samples: self.out[pos..pos + self.chunk_samples].to_vec(),
            });
            pos += self.chunk_samples;
        }
        self.out.drain(..pos);

        if let Some(req) = flush.outstanding() {
            self.flush_tail(ring);
            flush.ack(req);
        }
    }

    fn flush_tail(&mut self, ring: &RingBuffer<AudioChunk>) {
        if let Some(r) = self.resampler.as_mut() {
            let owed = (self.pending_in.len() as f64 * r.resample_ratio()).ceil() as usize
                + r.output_delay();
            let target_len = self.out.len() + owed;
            for _ in 0..3 {
                if self.out.len() >= target_len {
                    break;
                }
                let n = self.pending_in.len();
                if let Err(e) = resample_into(r, &self.pending_in, Some(n), &mut self.out) {
                    warn!("Flush resampling error: {e}");
                    break;
                }
                self.pending_in.clear();
            }
            self.out.truncate(target_len);
            self.pending_in.clear();
            r.reset();
        }
        if !self.out.is_empty() {
            ring.push(AudioChunk {
                samples: self.out.to_vec(),
            });
            self.out.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_fill_stays_within_range_for_an_oversized_clip() {
        let clip = vec![0.0f32; CHUNK_SAMPLES * BUFFER_CAPACITY * 2];
        let stream = AudioStream::from_samples(&clip, AudioConfig::default()).unwrap();
        let fill = stream.buffer_fill();
        assert!((0.0..=1.0).contains(&fill), "fill out of range: {fill}");
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
    fn convert_i32_samples_scales_signed_range() {
        let mut converted = Vec::new();
        convert_i32_samples_into(&[i32::MIN, 0, i32::MAX, 1 << 30], &mut converted);
        assert_eq!(converted[0], -1.0);
        assert_eq!(converted[1], 0.0);
        assert!(converted[2] > 0.99 && converted[2] <= 1.0);
        assert_eq!(converted[3], 0.5);
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

    const TARGET: u32 = 16_000;
    const CHUNK: usize = 1_280;

    fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
        let n = (f64::from(rate) * secs) as usize;
        (0..n)
            .map(|i| {
                (0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate)).sin())
                    as f32
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn drain(ring: &RingBuffer<AudioChunk>) -> Vec<f32> {
        let mut out = Vec::new();
        while let Some(chunk) = ring.try_pop() {
            out.extend_from_slice(&chunk.samples);
        }
        out
    }

    fn capture(proc: &mut CaptureProcessor, input: &[f32]) -> Vec<f32> {
        let ring = RingBuffer::new(4_096).unwrap();
        let flush = FlushState::new(true);
        for block in input.chunks(441) {
            proc.process(block, &ring, &flush);
        }
        flush.request();
        proc.process(&[], &ring, &flush);
        assert!(!flush.pending());
        drain(&ring)
    }

    #[test]
    fn resampler_rejects_content_above_the_output_nyquist() {
        for native in [48_000, 44_100] {
            let input = tone(12_000.0, native, 1.0);
            let mut proc = CaptureProcessor::new(1, native, TARGET, CHUNK).unwrap();
            let out = capture(&mut proc, &input);
            let settled = &out[400..out.len() - 400];
            let atten_db = 20.0 * (rms(settled) / rms(&input)).log10();
            assert!(
                atten_db < -60.0,
                "{native} Hz: 12 kHz tone only {atten_db:.1} dB down"
            );
        }
    }

    #[test]
    fn resampler_passes_an_in_band_tone_whole() {
        for native in [48_000, 44_100] {
            let input = tone(1_000.0, native, 1.0);
            let mut proc = CaptureProcessor::new(1, native, TARGET, CHUNK).unwrap();
            let delay = proc.resampler.as_ref().unwrap().output_delay();
            let out = capture(&mut proc, &input);

            let expected = (input.len() as f64 * f64::from(TARGET) / f64::from(native)).ceil();
            let len = out.len() as f64;
            assert!(
                (expected..=expected + delay as f64 + 2.0).contains(&len),
                "{native} Hz: {len} samples out, expected ~{expected} (+{delay} delay)"
            );

            let settled = &out[400..out.len() - 400];
            let gain_db = 20.0 * (rms(settled) / rms(&input)).log10();
            assert!(
                gain_db.abs() < 0.5,
                "{native} Hz: passband gain {gain_db:.2} dB"
            );

            let crossings = settled
                .windows(2)
                .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
                .count();
            let hz = crossings as f64 / (settled.len() as f64 / f64::from(TARGET));
            assert!((hz - 1_000.0).abs() < 5.0, "{native} Hz: measured {hz} Hz");
        }
    }

    #[test]
    fn a_flush_resets_the_resampler_for_the_next_session() {
        let first = tone(3_000.0, 48_000, 0.3);
        let second = tone(700.0, 48_000, 0.3);
        let mut reused = CaptureProcessor::new(1, 48_000, TARGET, CHUNK).unwrap();
        capture(&mut reused, &first);
        let mut fresh = CaptureProcessor::new(1, 48_000, TARGET, CHUNK).unwrap();
        assert_eq!(capture(&mut reused, &second), capture(&mut fresh, &second));
    }

    #[test]
    fn a_flush_is_acknowledged_once_the_tail_is_in_the_ring() {
        let ring = RingBuffer::new(64).unwrap();
        let flush = FlushState::new(true);
        let mut proc = CaptureProcessor::new(1, TARGET, TARGET, CHUNK).unwrap();
        proc.process(&vec![0.1; CHUNK + 100], &ring, &flush);
        assert_eq!(drain(&ring).len(), CHUNK);

        flush.request();
        assert!(flush.pending());
        proc.process(&[], &ring, &flush);
        assert!(!flush.pending());
        assert_eq!(drain(&ring).len(), 100, "the sub-chunk tail");
    }

    #[test]
    fn a_file_backed_stream_has_no_flush_to_wait_for() {
        let stream = AudioStream::from_samples(&[0.0; 4_000], AudioConfig::default()).unwrap();
        stream.request_flush();
        assert!(!stream.flush_pending());
        let handle = stream.handle();
        handle.request_flush();
        assert!(!handle.flush_pending());
    }

    #[test]
    fn drain_keep_last_leaves_only_the_newest() {
        let stream = AudioStream::from_samples(&[0.0; CHUNK * 10], AudioConfig::default()).unwrap();
        stream.drain_keep_last(3);
        assert_eq!(stream.handle.ring.len(), 3);
    }
}
