use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use tokio::sync::mpsc;

use lune_core::audio::{AudioCaptureConfig, AudioFrame};
use lune_core::error::{Error, Result};

use crate::vad;

/// Handle to a running capture session. Drop to stop.
pub struct SessionHandle {
    running: Arc<AtomicBool>,
    _stream: cpal::Stream,
}

impl SessionHandle {
    /// Stop the capture session.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
        if let Err(e) = self._stream.pause() {
            tracing::warn!("failed to pause stream: {e}");
        }
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Shared state for the audio processing callback.
struct CallbackState {
    tx: mpsc::Sender<AudioFrame>,
    running: Arc<AtomicBool>,
    accum: Vec<f32>,
    start_time: Instant,
    target_rate: u32,
    device_rate: u32,
    frame_size: usize,
    channels: usize,
    vad_enabled: bool,
    vad_threshold: f32,
}

impl CallbackState {
    /// Process incoming mono f32 PCM samples.
    fn process_samples(&mut self, mono_f32: &[f32]) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        // Resample if device rate != target rate.
        let resampled;
        let samples = if self.device_rate != self.target_rate {
            resampled = resample_nearest(mono_f32, self.device_rate, self.target_rate);
            &resampled
        } else {
            mono_f32
        };

        self.accum.extend_from_slice(samples);

        // Emit complete frames.
        while self.accum.len() >= self.frame_size {
            let frame_samples: Vec<f32> = self.accum.drain(..self.frame_size).collect();

            if self.vad_enabled && !vad::is_speech(&frame_samples, self.vad_threshold) {
                continue;
            }

            let frame = AudioFrame {
                samples: frame_samples,
                timestamp_ms: self.start_time.elapsed().as_millis() as u64,
                sample_rate: self.target_rate,
            };

            if self.tx.try_send(frame).is_err() {
                tracing::debug!("audio frame dropped (channel full)");
            }
        }
    }

    /// Convert interleaved multi-channel data to mono f32.
    fn to_mono_f32(&self, data: &[f32]) -> Vec<f32> {
        if self.channels == 1 {
            data.to_vec()
        } else {
            data.chunks_exact(self.channels)
                .map(|ch| ch.iter().sum::<f32>() / self.channels as f32)
                .collect()
        }
    }

    /// Convert interleaved i16 multi-channel data to mono f32.
    fn i16_to_mono_f32(&self, data: &[i16]) -> Vec<f32> {
        let float_data: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
        self.to_mono_f32(&float_data)
    }
}

/// Audio capture service backed by cpal.
pub struct AudioCaptureService {
    config: AudioCaptureConfig,
}

impl AudioCaptureService {
    pub fn new(config: AudioCaptureConfig) -> Self {
        Self { config }
    }

    /// Start a capture session, sending `AudioFrame`s into `tx`.
    ///
    /// Returns a `SessionHandle` — drop it (or call `.stop()`) to end capture.
    pub fn start_session(&self, tx: mpsc::Sender<AudioFrame>) -> Result<SessionHandle> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| Error::Audio("no default input device".into()))?;

        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        tracing::info!(device = %device_name, "opening audio input device");

        let supported = device
            .default_input_config()
            .map_err(|e| Error::Audio(format!("no supported input config: {e}")))?;

        let device_channels = supported.channels();
        let device_sample_rate = supported.sample_rate().0;
        let device_format = supported.sample_format();

        tracing::info!(
            channels = device_channels,
            sample_rate = device_sample_rate,
            format = ?device_format,
            "device default config"
        );

        let stream_config = StreamConfig {
            channels: device_channels,
            sample_rate: cpal::SampleRate(device_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let running = Arc::new(AtomicBool::new(true));

        let frame_size = self.config.frame_size;
        let channels = device_channels as usize;

        let stream = match device_format {
            SampleFormat::F32 => {
                let mut state = CallbackState {
                    tx,
                    running: running.clone(),
                    accum: Vec::with_capacity(frame_size * 2),
                    start_time: Instant::now(),
                    target_rate: self.config.sample_rate,
                    device_rate: device_sample_rate,
                    frame_size,
                    channels,
                    vad_enabled: self.config.vad_enabled,
                    vad_threshold: self.config.vad_threshold,
                };
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            let mono = state.to_mono_f32(data);
                            state.process_samples(&mono);
                        },
                        err_callback,
                        None,
                    )
                    .map_err(|e| Error::Audio(format!("build_input_stream f32: {e}")))?
            }
            SampleFormat::I16 => {
                let mut state = CallbackState {
                    tx,
                    running: running.clone(),
                    accum: Vec::with_capacity(frame_size * 2),
                    start_time: Instant::now(),
                    target_rate: self.config.sample_rate,
                    device_rate: device_sample_rate,
                    frame_size,
                    channels,
                    vad_enabled: self.config.vad_enabled,
                    vad_threshold: self.config.vad_threshold,
                };
                device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[i16], _: &cpal::InputCallbackInfo| {
                            let mono = state.i16_to_mono_f32(data);
                            state.process_samples(&mono);
                        },
                        err_callback,
                        None,
                    )
                    .map_err(|e| Error::Audio(format!("build_input_stream i16: {e}")))?
            }
            fmt => {
                return Err(Error::Audio(format!("unsupported sample format: {fmt:?}")));
            }
        };

        stream
            .play()
            .map_err(|e| Error::Audio(format!("stream play: {e}")))?;

        tracing::info!("audio capture started");

        Ok(SessionHandle {
            running,
            _stream: stream,
        })
    }
}

fn err_callback(err: cpal::StreamError) {
    tracing::error!(%err, "audio stream error");
}

/// Nearest-neighbor resample from `src_rate` to `dst_rate`.
fn resample_nearest(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = (samples.len() as f64 / ratio).ceil() as usize;
    (0..out_len)
        .map(|i| {
            let src_idx = ((i as f64) * ratio).min((samples.len() - 1) as f64) as usize;
            samples[src_idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_same_rate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = resample_nearest(&input, 16000, 16000);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn resample_downsample() {
        let input: Vec<f32> = (0..48000).map(|i| i as f32).collect();
        let output = resample_nearest(&input, 48000, 16000);
        assert_eq!(output.len(), 16000);
    }

    #[test]
    fn resample_upsample() {
        let input: Vec<f32> = (0..8000).map(|i| i as f32).collect();
        let output = resample_nearest(&input, 8000, 16000);
        assert_eq!(output.len(), 16000);
    }

    #[test]
    fn service_construction() {
        let config = AudioCaptureConfig::default();
        let _service = AudioCaptureService::new(config);
    }

    #[test]
    fn callback_state_mono_passthrough() {
        let (tx, _rx) = mpsc::channel(16);
        let state = CallbackState {
            tx,
            running: Arc::new(AtomicBool::new(true)),
            accum: Vec::new(),
            start_time: Instant::now(),
            target_rate: 16000,
            device_rate: 16000,
            frame_size: 480,
            channels: 1,
            vad_enabled: false,
            vad_threshold: 0.01,
        };
        let data = vec![0.5f32; 480];
        let mono = state.to_mono_f32(&data);
        assert_eq!(mono.len(), 480);
    }

    #[test]
    fn callback_state_stereo_downmix() {
        let (tx, _rx) = mpsc::channel(16);
        let state = CallbackState {
            tx,
            running: Arc::new(AtomicBool::new(true)),
            accum: Vec::new(),
            start_time: Instant::now(),
            target_rate: 16000,
            device_rate: 16000,
            frame_size: 480,
            channels: 2,
            vad_enabled: false,
            vad_threshold: 0.01,
        };
        // Stereo: L=1.0, R=0.0 → mono = 0.5
        let data: Vec<f32> = (0..960)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mono = state.to_mono_f32(&data);
        assert_eq!(mono.len(), 480);
        assert!((mono[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn callback_state_i16_conversion() {
        let (tx, _rx) = mpsc::channel(16);
        let state = CallbackState {
            tx,
            running: Arc::new(AtomicBool::new(true)),
            accum: Vec::new(),
            start_time: Instant::now(),
            target_rate: 16000,
            device_rate: 16000,
            frame_size: 480,
            channels: 1,
            vad_enabled: false,
            vad_threshold: 0.01,
        };
        let data = vec![i16::MAX; 480];
        let mono = state.i16_to_mono_f32(&data);
        assert_eq!(mono.len(), 480);
        assert!((mono[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn callback_state_frame_emission() {
        let (tx, mut rx) = mpsc::channel(16);
        let mut state = CallbackState {
            tx,
            running: Arc::new(AtomicBool::new(true)),
            accum: Vec::new(),
            start_time: Instant::now(),
            target_rate: 16000,
            device_rate: 16000,
            frame_size: 480,
            channels: 1,
            vad_enabled: false,
            vad_threshold: 0.01,
        };

        // Push exactly one frame worth of loud samples.
        let samples = vec![0.5f32; 480];
        state.process_samples(&samples);

        let frame = rx.try_recv().unwrap();
        assert_eq!(frame.samples.len(), 480);
        assert_eq!(frame.sample_rate, 16000);
    }

    #[test]
    fn callback_state_vad_filters_silence() {
        let (tx, mut rx) = mpsc::channel(16);
        let mut state = CallbackState {
            tx,
            running: Arc::new(AtomicBool::new(true)),
            accum: Vec::new(),
            start_time: Instant::now(),
            target_rate: 16000,
            device_rate: 16000,
            frame_size: 480,
            channels: 1,
            vad_enabled: true,
            vad_threshold: 0.01,
        };

        // Push silence — should be filtered.
        let silence = vec![0.0f32; 480];
        state.process_samples(&silence);

        assert!(rx.try_recv().is_err());
    }
}
