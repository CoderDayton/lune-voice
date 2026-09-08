//! VAD timeline inspection for WAV files, with PCEN-AGC telemetry.
//!
//! Usage:
//!   cargo run --release --example vad_inspect -p lune-audio -- <path.wav>
//!   PCEN=0 cargo run ...                 # disable AGC frontend
//!   QUIET=1 cargo run ...                # skip the per-frame table

use earshot::Detector;
use lune_audio::{PcenAgc, PcenAgcConfig};
use std::path::PathBuf;

const HOP_SIZE: usize = lune_audio::PCEN_HOP_SIZE;
const SAMPLE_RATE: u32 = 16_000;
const SPEECH_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
struct Frame {
    start_ms: u32,
    prob: f32,
    is_speech: bool,
    raw_rms: f32,
    agc_rms: f32,
    gain: f32,
    sig_rms: f32,
    noise_rms: f32,
    snr_db: f32,
}

fn rms(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let m = buf.iter().map(|&x| x * x).sum::<f32>() / buf.len() as f32;
    m.sqrt()
}

fn run_timeline(samples: &[f32], pcen_on: bool) -> Vec<Frame> {
    let mut detector = Detector::default();
    let mut agc = if pcen_on {
        Some(PcenAgc::new(PcenAgcConfig::default(), SAMPLE_RATE, HOP_SIZE).expect("build agc"))
    } else {
        None
    };
    let ms_per_frame = (HOP_SIZE as f32 / SAMPLE_RATE as f32 * 1000.0) as u32;
    let mut work = vec![0.0f32; HOP_SIZE];
    let mut out = Vec::with_capacity(samples.len() / HOP_SIZE + 1);

    for (i, chunk) in samples.chunks(HOP_SIZE).enumerate() {
        if chunk.len() < HOP_SIZE {
            break;
        }
        let raw_rms = rms(chunk);
        work.clear();
        work.extend_from_slice(chunk);
        let (gain, sig_rms, noise_rms, snr_db) = match agc.as_mut() {
            Some(a) => {
                let st = a.process_frame(&mut work);
                (st.gain_applied, st.signal_rms, st.noise_rms, st.snr_db)
            }
            None => (1.0, 0.0, 0.0, 0.0),
        };
        let agc_rms = rms(&work);
        let prob = detector.predict_f32(&work);
        out.push(Frame {
            start_ms: i as u32 * ms_per_frame,
            prob,
            is_speech: prob >= SPEECH_THRESHOLD,
            raw_rms,
            agc_rms,
            gain,
            sig_rms,
            noise_rms,
            snr_db,
        });
    }
    out
}

fn speech_segments(
    timeline: &[Frame],
    min_speech_ms: u32,
    min_silence_ms: u32,
) -> Vec<(usize, usize)> {
    let ms_per_frame = (HOP_SIZE as f32 / SAMPLE_RATE as f32 * 1000.0) as u32;
    let min_speech_frames = (min_speech_ms / ms_per_frame).max(1) as usize;
    let min_silence_frames = (min_silence_ms / ms_per_frame).max(1) as usize;

    let mut segs: Vec<(usize, usize)> = vec![];
    let mut speech_start: Option<usize> = None;
    let mut silence = 0usize;

    for (i, f) in timeline.iter().enumerate() {
        if f.is_speech {
            silence = 0;
            if speech_start.is_none() {
                speech_start = Some(i);
            }
        } else {
            silence += 1;
            if let Some(s) = speech_start
                && silence >= min_silence_frames
            {
                let len = i - silence - s;
                if len >= min_speech_frames {
                    segs.push((s * HOP_SIZE, (i - silence) * HOP_SIZE));
                }
                speech_start = None;
                silence = 0;
            }
        }
    }
    if let Some(s) = speech_start {
        let end = timeline.len() * HOP_SIZE;
        if (timeline.len() - s) >= min_speech_frames {
            segs.push((s * HOP_SIZE, end));
        }
    }
    segs
}

fn print_per_frame_table(timeline: &[Frame], pcen_on: bool) {
    if pcen_on {
        println!(
            "{:>7}  {:>6}  {:>3}  {:>7}  {:>7}  {:>5}  {:>7}  {:>7}  {:>6}",
            "ms", "prob", "spk", "raw_rms", "agc_rms", "gain", "sig_rms", "nse_rms", "snr_dB",
        );
        println!("{}", "─".repeat(74));
        for f in timeline {
            println!(
                "{:>7}  {:>6.3}  {:>3}  {:>7.4}  {:>7.4}  {:>5.2}  {:>7.4}  {:>7.4}  {:>+6.1}",
                f.start_ms,
                f.prob,
                if f.is_speech { "yes" } else { "no" },
                f.raw_rms,
                f.agc_rms,
                f.gain,
                f.sig_rms,
                f.noise_rms,
                f.snr_db,
            );
        }
    } else {
        println!("{:>7}  {:>6}  {:>3}  {:>7}", "ms", "prob", "spk", "raw_rms");
        println!("{}", "─".repeat(30));
        for f in timeline {
            println!(
                "{:>7}  {:>6.3}  {:>3}  {:>7.4}",
                f.start_ms,
                f.prob,
                if f.is_speech { "yes" } else { "no" },
                f.raw_rms,
            );
        }
    }
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f32 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_summary(timeline: &[Frame], pcen_on: bool) {
    let n = timeline.len();
    if n == 0 {
        println!("(empty timeline)");
        return;
    }
    let speech = timeline.iter().filter(|f| f.is_speech).count();
    let mut probs: Vec<f32> = timeline.iter().map(|f| f.prob).collect();
    let mut raws: Vec<f32> = timeline.iter().map(|f| f.raw_rms).collect();
    probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    raws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    println!("── Summary ──");
    println!(
        "  frames:  {n}    speech: {speech} ({:.1}%)",
        100.0 * speech as f32 / n as f32
    );
    println!(
        "  prob   P50={:.3}  P95={:.3}",
        percentile(&probs, 0.50),
        percentile(&probs, 0.95)
    );
    println!(
        "  raw_rms  P50={:.4}  P95={:.4}",
        percentile(&raws, 0.50),
        percentile(&raws, 0.95)
    );

    if pcen_on {
        let mut agcs: Vec<f32> = timeline.iter().map(|f| f.agc_rms).collect();
        let mut gains: Vec<f32> = timeline.iter().map(|f| f.gain).collect();
        let mut snrs: Vec<f32> = timeline.iter().map(|f| f.snr_db).collect();
        agcs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        gains.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        snrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_gain: f32 = timeline.iter().map(|f| f.gain).sum::<f32>() / n as f32;
        let final_sig = timeline.last().map(|f| f.sig_rms).unwrap_or(0.0);
        let final_noise = timeline.last().map(|f| f.noise_rms).unwrap_or(0.0);
        println!(
            "  agc_rms  P50={:.4}  P95={:.4}",
            percentile(&agcs, 0.50),
            percentile(&agcs, 0.95)
        );
        println!(
            "  gain     P50={:.2}  P95={:.2}  mean={:.2}",
            percentile(&gains, 0.50),
            percentile(&gains, 0.95),
            mean_gain
        );
        println!(
            "  snr_dB   P50={:+.1}  P95={:+.1}",
            percentile(&snrs, 0.50),
            percentile(&snrs, 0.95)
        );
        println!(
            "  envelopes (final)  signal_rms={:.4}  noise_rms={:.4}",
            final_sig, final_noise
        );
    }
}

fn load_wav_f32(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("failed to open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "WAV must be 16kHz");
    assert_eq!(spec.channels, 1, "WAV must be mono");
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.expect("read sample") as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("read sample"))
            .collect(),
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();
            root.join("fixtures/audio/librispeech_clean.wav")
        });
    let pcen_on = std::env::var("PCEN").map(|v| v != "0").unwrap_or(true);
    let quiet = std::env::var("QUIET").map(|v| v != "0").unwrap_or(false);

    println!("VAD inspection: {}", path.display());
    println!("  PCEN frontend: {}", if pcen_on { "ON" } else { "off" });
    let samples = load_wav_f32(&path);
    let duration_s = samples.len() as f32 / SAMPLE_RATE as f32;
    println!("  {:.1}s ({} samples)\n", duration_s, samples.len());

    let timeline = run_timeline(&samples, pcen_on);

    if !quiet {
        print_per_frame_table(&timeline, pcen_on);
        println!();
    }
    print_summary(&timeline, pcen_on);

    println!();
    let segs = speech_segments(&timeline, 200, 300);
    println!("── Speech segments (min_speech=200ms, min_silence=300ms) ──");
    let mut total_speech_s = 0.0f32;
    for (i, (start, end)) in segs.iter().enumerate() {
        let start_s = *start as f32 / SAMPLE_RATE as f32;
        let end_s = *end as f32 / SAMPLE_RATE as f32;
        let dur_s = end_s - start_s;
        total_speech_s += dur_s;
        println!("[{i}] {:.2}s – {:.2}s ({:.2}s)", start_s, end_s, dur_s);
    }
    println!();
    println!(
        "  Total speech: {:.2}s / {:.1}s ({:.0}%)",
        total_speech_s,
        duration_s,
        total_speech_s / duration_s * 100.0
    );
}
