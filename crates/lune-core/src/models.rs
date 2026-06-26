pub const DIR_QWEN3_ASR: &str = "qwen3-asr-0.6b";
pub const DIR_STYLE_LLM: &str = "qwen3-0.6b";

// whisper.cpp GGML model directories. Each holds a single `ggml-*.bin` file
// downloaded from the `ggerganov/whisper.cpp` HuggingFace repo. We ship the
// q8_0 quantizations — near-fp16 accuracy at ~half the footprint.
pub const DIR_WHISPER_SMALL: &str = "whisper-small-q8_0";
pub const DIR_WHISPER_MEDIUM: &str = "whisper-medium-q8_0";
pub const DIR_WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo-q8_0";
