//! VAD timeline inspection for a WAV file.
//!
//! Usage: cargo run --example vad_inspect -p lune-audio -- <path/to/file.wav>

use std::path::PathBuf;
use ten_vad_rs::TenVad;

const TEN_VAD_MODEL: &[u8] = include_bytes!("../models/ten-vad.onnx");
const HOP_SIZE: usize = 256;
const SAMPLE_RATE: u32 = 16_000;
const SPEECH_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone)]
struct VadFrame {
    start_ms: u32,
    prob: f32,
    is_speech: bool,
}

fn vad_timeline(samples: &[i16]) -> Vec<VadFrame> {
    let mut vad =
        TenVad::new_from_bytes(TEN_VAD_MODEL, SAMPLE_RATE).expect("failed to init TEN VAD");

    let ms_per_frame = (HOP_SIZE as f32 / SAMPLE_RATE as f32 * 1000.0) as u32;

    samples
        .chunks(HOP_SIZE)
        .enumerate()
        .filter_map(|(i, chunk)| {
            let mut frame = vec![0i16; HOP_SIZE];
            frame[..chunk.len()].copy_from_slice(chunk);

            match vad.process_frame(&frame) {
                Ok(prob) => Some(VadFrame {
                    start_ms: i as u32 * ms_per_frame,
                    prob,
                    is_speech: prob >= SPEECH_THRESHOLD,
                }),
                Err(_) => None,
            }
        })
        .collect()
}

fn speech_segments(
    timeline: &[VadFrame],
    min_speech_ms: u32,
    min_silence_ms: u32,
) -> Vec<(usize, usize)> {
    let ms_per_frame = (HOP_SIZE as f32 / SAMPLE_RATE as f32 * 1000.0) as u32;
    let min_speech_frames = (min_speech_ms / ms_per_frame).max(1) as usize;
    let min_silence_frames = (min_silence_ms / ms_per_frame).max(1) as usize;

    let mut segments: Vec<(usize, usize)> = vec![];
    let mut speech_start: Option<usize> = None;
    let mut silence_count = 0usize;

    for (i, frame) in timeline.iter().enumerate() {
        if frame.is_speech {
            silence_count = 0;
            if speech_start.is_none() {
                speech_start = Some(i);
            }
        } else {
            silence_count += 1;
            if let Some(start) = speech_start
                && silence_count >= min_silence_frames
            {
                let len = i - silence_count - start;
                if len >= min_speech_frames {
                    segments.push((start * HOP_SIZE, (i - silence_count) * HOP_SIZE));
                }
                speech_start = None;
                silence_count = 0;
            }
        }
    }

    if let Some(start) = speech_start {
        let end = timeline.len() * HOP_SIZE;
        if (timeline.len() - start) >= min_speech_frames {
            segments.push((start * HOP_SIZE, end));
        }
    }

    segments
}

fn print_vad_summary(timeline: &[VadFrame]) {
    println!("{:<10} {:<8} Speech", "Time(ms)", "Prob");
    println!("{}", "-".repeat(50));

    for frame in timeline.iter().step_by(10) {
        let bar_len = (frame.prob * 20.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
        println!(
            "{:<10} {:<8.3} {} {}",
            frame.start_ms,
            frame.prob,
            bar,
            if frame.is_speech { "SPEECH" } else { "silence" }
        );
    }
}

fn load_wav_i16(path: &PathBuf) -> Vec<i16> {
    let mut reader = hound::WavReader::open(path).expect("failed to open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "WAV must be 16kHz");
    assert_eq!(spec.channels, 1, "WAV must be mono");

    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.expect("read sample"))
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| (s.expect("read sample") * 32767.0) as i16)
            .collect(),
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            // default to workspace test_fixtures
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();
            root.join("test_fixtures/custom_recording.wav")
        });

    println!("VAD inspection: {}", path.display());
    let samples = load_wav_i16(&path);
    let duration_s = samples.len() as f32 / SAMPLE_RATE as f32;
    println!("  {:.1}s  ({} samples)\n", duration_s, samples.len());

    let timeline = vad_timeline(&samples);

    print_vad_summary(&timeline);

    println!();
    let segs = speech_segments(&timeline, 200, 300);
    println!("── Speech segments (min_speech=200ms, min_silence=300ms) ──");
    let mut total_speech_s = 0.0f32;
    for (i, (start, end)) in segs.iter().enumerate() {
        let start_s = *start as f32 / SAMPLE_RATE as f32;
        let end_s = *end as f32 / SAMPLE_RATE as f32;
        let dur_s = end_s - start_s;
        total_speech_s += dur_s;
        println!("  [{i}] {:.2}s – {:.2}s  ({:.2}s)", start_s, end_s, dur_s);
    }
    println!();
    println!(
        "  Total speech: {:.2}s / {:.1}s  ({:.0}%)",
        total_speech_s,
        duration_s,
        total_speech_s / duration_s * 100.0
    );
}
