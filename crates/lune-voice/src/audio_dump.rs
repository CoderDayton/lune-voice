// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tracing::warn;
use uuid::Uuid;

pub const SAVE_AUDIO_ENV: &str = "LUNE_SAVE_AUDIO";

const SAMPLE_RATE: u32 = 16_000;

pub(crate) fn dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::var_os(SAVE_AUDIO_ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)?;
        match std::fs::create_dir_all(&dir) {
            Ok(()) => {
                warn!(
                    dir = %dir.display(),
                    "{SAVE_AUDIO_ENV} is set: every dictation's audio and text is saved unencrypted"
                );
                Some(dir)
            }
            Err(e) => {
                warn!(dir = %dir.display(), error = %e, "{SAVE_AUDIO_ENV}: cannot create the directory, nothing will be saved");
                None
            }
        }
    })
    .as_deref()
}

pub(crate) fn save_wav(session_id: Uuid, samples: Vec<f32>) {
    let Some(dir) = dir() else { return };
    if samples.is_empty() {
        return;
    }
    tokio::task::spawn_blocking(move || {
        if let Err(e) = write_wav(dir, session_id, &samples) {
            warn!(%session_id, error = %e, "could not save the dictation's audio");
        }
    });
}

pub(crate) fn save_text(session_id: Uuid, text: &str) {
    let Some(dir) = dir() else { return };
    if let Err(e) = write_text(dir, session_id, text) {
        warn!(%session_id, error = %e, "could not save the dictation's text");
    }
}

fn write_wav(dir: &Path, session_id: Uuid, samples: &[f32]) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(dir.join(format!("{session_id}.wav")), spec)?;
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16)?;
    }
    writer.finalize()
}

fn write_text(dir: &Path, session_id: Uuid, text: &str) -> std::io::Result<()> {
    std::fs::write(dir.join(format!("{session_id}.hyp.txt")), text)
}

#[cfg(test)]
mod tests {
    use super::{write_text, write_wav};
    use uuid::Uuid;

    #[test]
    fn writes_a_16_bit_mono_wav_and_an_unchecked_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let id = Uuid::new_v4();
        write_wav(dir.path(), id, &[0.0, 0.5, -0.5, 2.0]).expect("wav");
        write_text(dir.path(), id, "hello there").expect("text");

        let mut reader =
            hound::WavReader::open(dir.path().join(format!("{id}.wav"))).expect("open");
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (1, 16_000, 16)
        );
        let samples: Vec<i16> = reader.samples().map(|s| s.expect("sample")).collect();
        assert_eq!(samples, [0, 16_384, -16_384, i16::MAX]);

        assert!(!dir.path().join(format!("{id}.txt")).exists());
        let text = std::fs::read_to_string(dir.path().join(format!("{id}.hyp.txt"))).expect("read");
        assert_eq!(text, "hello there");
    }
}
