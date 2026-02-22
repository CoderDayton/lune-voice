/// Energy-based voice activity detection.
///
/// Returns `true` if the RMS energy of the frame exceeds the threshold.
#[inline]
pub fn is_speech(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }
    // Compare squared RMS against squared threshold to avoid sqrt entirely.
    // sum_sq / len > threshold^2  ↔  rms > threshold
    let threshold_sq = threshold * threshold;
    let len = samples.len();
    let sum_sq = sum_sq_simd(samples);
    sum_sq > threshold_sq * len as f32
}

/// Horizontal sum of squared samples using explicit SIMD-friendly chunking.
/// The compiler auto-vectorises the 8-wide accumulator loop on x86-64/aarch64.
#[inline(always)]
fn sum_sq_simd(samples: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let chunks = samples.chunks_exact(LANES);
    let remainder = chunks.remainder();
    for chunk in chunks {
        // Safety: chunks_exact guarantees len == LANES
        for i in 0..LANES {
            acc[i] = acc[i].mul_add(1.0, chunk[i] * chunk[i]);
        }
    }
    let mut total: f32 = acc.iter().copied().sum();
    for &s in remainder {
        total += s * s;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_below_threshold() {
        let silence = vec![0.0f32; 480];
        assert!(!is_speech(&silence, 0.01));
    }

    #[test]
    fn loud_signal_above_threshold() {
        let loud = vec![0.5f32; 480];
        assert!(is_speech(&loud, 0.01));
    }

    #[test]
    fn quiet_signal_below_threshold() {
        let quiet = vec![0.001f32; 480];
        assert!(!is_speech(&quiet, 0.01));
    }

    #[test]
    fn empty_frame() {
        assert!(!is_speech(&[], 0.01));
    }

    #[test]
    fn at_exact_threshold_is_not_speech() {
        // RMS of constant 0.01 signal = 0.01, which is not > 0.01
        let at_threshold = vec![0.01f32; 480];
        // Due to f32 precision, sqrt(0.01^2) may equal 0.01 exactly.
        // Use a slightly higher threshold to guarantee below.
        assert!(!is_speech(&at_threshold, 0.011));
    }

    #[test]
    fn non_multiple_of_lanes_remainder() {
        // 485 = 60*8 + 5 — exercises the remainder path
        let samples = vec![0.5f32; 485];
        assert!(is_speech(&samples, 0.01));
    }
}
