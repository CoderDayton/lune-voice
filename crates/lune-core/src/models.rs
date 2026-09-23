// SPDX-License-Identifier: GPL-3.0-or-later

// whisper.cpp GGML model directories. Each holds a single `ggml-*.bin` file
// downloaded from the `ggerganov/whisper.cpp` HuggingFace repo. Small and Medium
// ship as q5 (q5_1 / q5_0) for a smaller footprint; Base and Large v3 Turbo are
// q8_0.
pub const DIR_WHISPER_BASE: &str = "whisper-base-q8_0";
pub const DIR_WHISPER_SMALL: &str = "whisper-small-q5_1";
pub const DIR_WHISPER_MEDIUM: &str = "whisper-medium-q5_0";
pub const DIR_WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo-q8_0";

// whisper.cpp's VAD model (silero v5, ggml). Downloaded from the
// `ggml-org/whisper-vad` HuggingFace repo and loaded into a standalone
// `WhisperVadContext` that trims leading/trailing silence off an utterance
// before decoding. It is deliberately not wired to the in-decode
// `FullParams::enable_vad`, which concatenates speech and strips the internal
// pauses whisper uses as its punctuation cue.
pub const DIR_WHISPER_VAD: &str = "whisper-vad-silero";

/// Directory under the data dir holding dlopen-able ggml compute backends: the
/// CPU variants shipped with the app, plus the CUDA plugin the user can add
/// after install. See `lune_asr::gpu_backend`.
pub const DIR_GPU_BACKEND: &str = "backends";
pub const WHISPER_VAD_GGML: &str = "ggml-silero-v5.1.2.bin";
