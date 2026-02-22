use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use audioadapter_buffers::direct::SequentialSlice;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
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
    pre_resample: Vec<f32>,
    resampler: Option<Async<f32>>,
    resample_out_buf: Vec<f32>,
    resample_chunk: usize,
    start_time: Instant,
    target_rate: u32,
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

        match &mut self.resampler {
            Some(resampler) => {
                self.pre_resample.extend_from_slice(mono_f32);
                while self.pre_resample.len() >= self.resample_chunk {
                    let out_frames = resampler.output_frames_next();
                    if let (Ok(input), Ok(mut output)) = (
                        SequentialSlice::new(
                            &self.pre_resample[..self.resample_chunk],
                            1,
                            self.resample_chunk,
                        ),
                        SequentialSlice::new_mut(
                            &mut self.resample_out_buf[..out_frames],
                            1,
                            out_frames,
                        ),
                    ) && let Ok((_in_len, out_len)) =
                        resampler.process_into_buffer(&input, &mut output, None)
                    {
                        self.accum
                            .extend_from_slice(&self.resample_out_buf[..out_len]);
                    }
                    self.pre_resample.drain(..self.resample_chunk);
                }
            }
            None => {
                self.accum.extend_from_slice(mono_f32);
            }
        }

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
    #[inline]
    fn to_mono_f32(&self, data: &[f32]) -> Vec<f32> {
        if self.channels == 1 {
            data.to_vec()
        } else {
            // Multiply by inverse to avoid per-chunk division.
            let inv = 1.0f32 / self.channels as f32;
            data.chunks_exact(self.channels)
                .map(|ch| ch.iter().sum::<f32>() * inv)
                .collect()
        }
    }

    /// Convert interleaved i16 multi-channel data to mono f32.
    ///
    /// Avoids an intermediate Vec by combining i16→f32 conversion and downmix in one pass.
    #[inline]
    fn i16_to_mono_f32(&self, data: &[i16]) -> Vec<f32> {
        const INV_I16_MAX: f32 = 1.0 / i16::MAX as f32;
        if self.channels == 1 {
            data.iter().map(|&s| s as f32 * INV_I16_MAX).collect()
        } else {
            let inv_channels = 1.0f32 / self.channels as f32;
            data.chunks_exact(self.channels)
                .map(|ch| ch.iter().map(|&s| s as f32 * INV_I16_MAX).sum::<f32>() * inv_channels)
                .collect()
        }
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
        let target_rate = self.config.sample_rate;

        let make_state = |tx, running: Arc<AtomicBool>| -> Result<CallbackState> {
            let (resampler, resample_out_buf, resample_chunk) = if device_sample_rate != target_rate
            {
                let chunk_size = 1024usize;
                let params = SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 256,
                    window: WindowFunction::BlackmanHarris2,
                };
                let r = Async::<f32>::new_sinc(
                    target_rate as f64 / device_sample_rate as f64,
                    2.0,
                    &params,
                    chunk_size,
                    1, // mono
                    FixedAsync::Input,
                )
                .map_err(|e| Error::Audio(format!("create resampler: {e}")))?;
                let max_out = r.output_frames_max();
                let out_buf = vec![0.0f32; max_out];
                (Some(r), out_buf, chunk_size)
            } else {
                (None, Vec::new(), 0)
            };

            Ok(CallbackState {
                tx,
                running,
                accum: Vec::with_capacity(frame_size * 2),
                pre_resample: Vec::with_capacity(2048),
                resampler,
                resample_out_buf,
                resample_chunk,
                start_time: Instant::now(),
                target_rate,
                frame_size,
                channels,
                vad_enabled: self.config.vad_enabled,
                vad_threshold: self.config.vad_threshold,
            })
        };

        let stream = match device_format {
            SampleFormat::F32 => {
                let mut state = make_state(tx, running.clone())?;
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
                let mut state = make_state(tx, running.clone())?;
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

#[cold]
fn err_callback(err: cpal::StreamError) {
    tracing::error!(%err, "audio stream error");
}

/// Create a `CallbackState` with no resampler (same rate) for tests.
#[cfg(test)]
fn test_callback_state(tx: mpsc::Sender<AudioFrame>, channels: usize) -> CallbackState {
    CallbackState {
        tx,
        running: Arc::new(AtomicBool::new(true)),
        accum: Vec::new(),
        pre_resample: Vec::new(),
        resampler: None,
        resample_out_buf: Vec::new(),
        resample_chunk: 0,
        start_time: Instant::now(),
        target_rate: 16000,
        frame_size: 480,
        channels,
        vad_enabled: false,
        vad_threshold: 0.01,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_resampler_downsample() {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            Async::<f32>::new_sinc(16000.0 / 44100.0, 2.0, &params, 1024, 1, FixedAsync::Input)
                .unwrap();

        // 1024 samples at 44100 → ~372 samples at 16000.
        let input_data = vec![0.0f32; 1024];
        let input = SequentialSlice::new(&input_data, 1, 1024).unwrap();
        let output = resampler.process(&input, 0, None).unwrap();
        let out_data = output.take_data();
        let expected = (1024.0_f64 * 16000.0 / 44100.0).round() as usize;
        assert!((out_data.len() as i32 - expected as i32).unsigned_abs() <= 2);
    }

    #[test]
    fn service_construction() {
        let config = AudioCaptureConfig::default();
        let _service = AudioCaptureService::new(config);
    }

    #[test]
    fn callback_state_mono_passthrough() {
        let (tx, _rx) = mpsc::channel(16);
        let state = test_callback_state(tx, 1);
        let data = vec![0.5f32; 480];
        let mono = state.to_mono_f32(&data);
        assert_eq!(mono.len(), 480);
    }

    #[test]
    fn callback_state_stereo_downmix() {
        let (tx, _rx) = mpsc::channel(16);
        let state = test_callback_state(tx, 2);
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
        let state = test_callback_state(tx, 1);
        let data = vec![i16::MAX; 480];
        let mono = state.i16_to_mono_f32(&data);
        assert_eq!(mono.len(), 480);
        assert!((mono[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn callback_state_frame_emission() {
        let (tx, mut rx) = mpsc::channel(16);
        let mut state = test_callback_state(tx, 1);

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
        let mut state = test_callback_state(tx, 1);
        state.vad_enabled = true;

        // Push silence — should be filtered.
        let silence = vec![0.0f32; 480];
        state.process_samples(&silence);

        assert!(rx.try_recv().is_err());
    }
}
