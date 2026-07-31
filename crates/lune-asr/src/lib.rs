pub mod backend;
pub mod gpu_backend;

pub use backend::{
    AsrRegistry, AsrSessionHandle, OpenAiAsrConfig, OpenAiCompatibleAsrBackend, WhisperCppBackend,
    WhisperCppConfig, encode_wav_16bit,
};
