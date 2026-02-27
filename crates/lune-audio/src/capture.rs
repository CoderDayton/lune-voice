use crate::{AudioChunk, AudioError, ring::RingBuffer};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, SampleRate, StreamConfig};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use std::sync::Arc;
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
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 16_000,
            chunk_duration_ms: 80,
            buffer_capacity: 128,
        }
    }
}

/// Audio capture stream with automatic resampling
pub struct AudioStream {
    config: AudioConfig,
    ring: Arc<RingBuffer<AudioChunk>>,
    /// Kept alive to keep the cpal callback running. `None` for file-backed streams.
    _stream: Option<cpal::Stream>,
}

impl AudioStream {
    /// Create a new audio stream with the given configuration
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevice)?;

        let device_name = device.name().unwrap_or_else(|_| "Unknown".into());
        debug!("Using audio device: {}", device_name);

        // Query supported configs and pick the best one
        let supported_config = device
            .default_input_config()
            .map_err(|e| AudioError::StreamBuild(e.to_string()))?;

        let native_sample_rate = supported_config.sample_rate().0;
        let channels = supported_config.channels();

        debug!(
            "Native format: {}Hz, {} channels, {:?}",
            native_sample_rate,
            channels,
            supported_config.sample_format()
        );

        // Build stream config
        let chunk_samples = (config.target_sample_rate * config.chunk_duration_ms / 1000) as usize;
        let stream_config = StreamConfig {
            channels,
            sample_rate: SampleRate(native_sample_rate),
            buffer_size: BufferSize::Default,
        };

        let ring = Arc::new(RingBuffer::new(config.buffer_capacity));
        let ring_clone = Arc::clone(&ring);

        let target_rate = config.target_sample_rate;
        let start_time = Instant::now();

        // Shared state for the audio callback
        let mut resampler = if native_sample_rate != target_rate {
            Some(
                FastFixedIn::<f32>::new(
                    target_rate as f64 / native_sample_rate as f64,
                    1.2, // Max relative ratio deviation
                    PolynomialDegree::Septic,
                    chunk_samples,
                    1, // Mono after downmix
                )
                .map_err(|e| AudioError::Resampler(e.to_string()))?,
            )
        } else {
            None
        };

        let mut input_resample_buf = Vec::with_capacity(chunk_samples * 2);
        let mut output_buf = Vec::with_capacity(chunk_samples * 2);

        // Build the stream based on sample format
        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => device
                .build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &_| {
                        process_audio_callback(
                            data,
                            channels as usize,
                            &mut resampler,
                            &mut input_resample_buf,
                            &mut output_buf,
                            chunk_samples,
                            &ring_clone,
                            start_time,
                        );
                    },
                    |err| warn!("Audio stream error: {}", err),
                    None,
                )
                .map_err(|e| AudioError::StreamBuild(e.to_string()))?,

            SampleFormat::I16 => device
                .build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &_| {
                        let float_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / 32768.0).collect();
                        process_audio_callback(
                            &float_data,
                            channels as usize,
                            &mut resampler,
                            &mut input_resample_buf,
                            &mut output_buf,
                            chunk_samples,
                            &ring_clone,
                            start_time,
                        );
                    },
                    |err| warn!("Audio stream error: {}", err),
                    None,
                )
                .map_err(|e| AudioError::StreamBuild(e.to_string()))?,

            format => {
                return Err(AudioError::StreamBuild(format!(
                    "Unsupported sample format: {:?}",
                    format
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
            _stream: Some(stream),
        })
    }

    /// Create an `AudioStream` pre-loaded with PCM samples (16 kHz mono f32).
    ///
    /// Chunks are pushed into the ring buffer immediately so `try_next_chunk`
    /// drains them at the consumer's pace — no real-time audio thread involved.
    /// Useful for integration tests and offline file processing.
    pub fn from_samples(samples: &[f32], config: AudioConfig) -> Self {
        let chunk_samples = (config.target_sample_rate * config.chunk_duration_ms / 1000) as usize;
        // Auto-size buffer to fit all chunks so nothing is dropped.
        let num_chunks = samples.len().div_ceil(chunk_samples);
        let capacity = num_chunks.max(config.buffer_capacity);
        let ring = Arc::new(RingBuffer::new(capacity));
        let mut offset_ms = 0u64;
        for chunk in samples.chunks(chunk_samples) {
            ring.push(AudioChunk {
                samples: chunk.to_vec(),
                timestamp_ms: offset_ms,
            });
            offset_ms += u64::from(config.chunk_duration_ms);
        }
        Self {
            config,
            ring,
            _stream: None,
        }
    }

    /// Get the next audio chunk (async, blocks until data available)
    pub async fn next_chunk(&self) -> Option<AudioChunk> {
        self.ring.pop().await
    }

    /// Try to get the next chunk without blocking
    pub fn try_next_chunk(&self) -> Option<AudioChunk> {
        self.ring.try_pop()
    }

    /// Get current buffer fill percentage (0.0 to 1.0)
    /// Values near 1.0 indicate backpressure (consumer is slow)
    pub fn buffer_fill(&self) -> f32 {
        self.ring.len() as f32 / self.config.buffer_capacity as f32
    }
}

/// Core audio processing callback (runs in real-time audio thread)
#[allow(clippy::too_many_arguments)]
fn process_audio_callback(
    data: &[f32],
    channels: usize,
    resampler: &mut Option<FastFixedIn<f32>>,
    input_resample_buf: &mut Vec<f32>,
    output_buf: &mut Vec<f32>,
    chunk_samples: usize,
    ring: &RingBuffer<AudioChunk>,
    start_time: Instant,
) {
    // 1. Downmix to mono (simple average)
    let mono: Vec<f32> = if channels == 1 {
        data.to_vec()
    } else {
        data.chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // 2. Resample if needed
    if let Some(r) = resampler {
        input_resample_buf.extend_from_slice(&mono);
        let required_input = r.input_frames_next();
        while input_resample_buf.len() >= required_input {
            let input_chunk: Vec<f32> = input_resample_buf.drain(..required_input).collect();
            match r.process(&[input_chunk], None) {
                Ok(output) => output_buf.extend_from_slice(&output[0]),
                Err(e) => {
                    warn!("Resampling error: {}", e);
                    break;
                }
            }
        }
    } else {
        output_buf.extend_from_slice(&mono);
    }

    while output_buf.len() >= chunk_samples {
        let chunk_data: Vec<f32> = output_buf.drain(..chunk_samples).collect();

        let timestamp_ms = start_time.elapsed().as_millis() as u64;
        let chunk = AudioChunk {
            samples: chunk_data,
            timestamp_ms,
        };

        // If buffer is full, drop oldest chunk (backpressure handling).
        // No logging here — this is a real-time audio thread.
        if ring.is_full() {
            ring.try_pop();
        }

        ring.push(chunk);
    }
}
