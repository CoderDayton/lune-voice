pub const DIR_STYLE_LLM: &str = "qwen3-styler-0.8b-gguf";

// whisper.cpp GGML model directories. Each holds a single `ggml-*.bin` file
// downloaded from the `ggerganov/whisper.cpp` HuggingFace repo. We ship the
// q8_0 quantizations — near-fp16 accuracy at ~half the footprint.
pub const DIR_WHISPER_SMALL: &str = "whisper-small-q8_0";
pub const DIR_WHISPER_MEDIUM: &str = "whisper-medium-q8_0";
pub const DIR_WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo-q8_0";

// whisper.cpp's built-in VAD model (silero v5, ggml). Downloaded from the
// `ggml-org/whisper-vad` HuggingFace repo and passed to `FullParams::enable_vad`
// so the whisper backend gates each utterance to speech before decoding.
pub const DIR_WHISPER_VAD: &str = "whisper-vad-silero";
pub const WHISPER_VAD_GGML: &str = "ggml-silero-v5.1.2.bin";
