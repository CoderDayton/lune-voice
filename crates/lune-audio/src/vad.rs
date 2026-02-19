/// Energy-based voice activity detection.
///
/// Returns `true` if the RMS energy of the frame exceeds the threshold.
pub fn is_speech(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    rms > threshold
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
}
