"""Compare Qwen3-ASR ONNX inference vs reference transformers pipeline.

Records a short WAV or uses an existing one, runs both pipelines, compares output.
"""
import sys
import os
import numpy as np
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
ONNX_DIR = Path.home() / ".local/share/lune-voice/models/qwen3-asr-0.6b"

# Find a test WAV — use the first arg, or look for fixture WAVs
wav_path = None
if len(sys.argv) > 1:
    wav_path = sys.argv[1]
else:
    fixture_dir = Path(__file__).parent.parent / "fixtures"
    if fixture_dir.exists():
        wavs = list(fixture_dir.glob("*.wav"))
        if wavs:
            wav_path = str(wavs[0])

if not wav_path:
    print("Usage: python test_asr_reference.py <path_to_wav>")
    print("  or place a .wav in fixtures/")
    sys.exit(1)

print(f"WAV: {wav_path}")


def load_wav_f32(path: str) -> tuple[np.ndarray, int]:
    """Load WAV as float32 mono 16kHz."""
    import soundfile as sf
    pcm, sr = sf.read(path, dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    return pcm, sr


def run_reference(pcm: np.ndarray, sr: int) -> str:
    """Run reference transformers pipeline."""
    import torch
    from transformers import AutoProcessor

    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
    except ImportError:
        print("qwen_asr package not available, skipping reference")
        return ""

    print("\n=== Reference (transformers) ===")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    # Build conversation in the expected format
    conversation = [
        {"role": "system", "content": "You are an ASR engine. Transcribe spoken audio exactly in the spoken language. Do not translate. Do not add words not clearly present."},
        {"role": "user", "content": [{"type": "audio", "audio": wav_path}]},
    ]

    # Use processor to prepare inputs
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    # Add language suffix
    text_ids = inputs["input_ids"]
    print(f"  Input IDs shape: {text_ids.shape}")
    print(f"  Input features shape: {inputs.get('input_features', inputs.get('audio_features', 'N/A'))}")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=448,
            do_sample=False,
            temperature=1e-6,
        )

    # Decode only new tokens
    new_tokens = generated[0, text_ids.shape[1]:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"  Output: {text}")
    return text


def run_onnx(pcm: np.ndarray, sr: int) -> str:
    """Run ONNX pipeline mimicking lune-voice's Rust code."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    print("\n=== ONNX (lune-voice style) ===")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    encoder = ort.InferenceSession(str(ONNX_DIR / "qwen3-asr-encoder.onnx"), providers=providers)
    embed_tokens = ort.InferenceSession(str(ONNX_DIR / "qwen3-asr-embed-tokens.onnx"), providers=providers)
    decoder = ort.InferenceSession(str(ONNX_DIR / "qwen3-asr-decoder.onnx"), providers=providers)
    tokenizer = Tokenizer.from_file(str(ONNX_DIR / "tokenizer.json"))

    print(f"  Provider: {encoder.get_providers()[0]}")

    # Constants matching qwen3.rs
    NUM_LAYERS = 28
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    MAX_NEW_TOKENS = 448
    EOS_IDS = {151643, 151645}
    AUDIO_PAD_ID = 151676
    ASR_TEXT_ID = 151704
    IM_START = 151644
    IM_END = 151645
    NEWLINE = 198
    AUDIO_START = 151669
    AUDIO_END = 151670

    # 1. Mel spectrogram — use whisper-compatible extraction
    from whisper_mel import WhisperMelExtractor
    mel_extractor = WhisperMelExtractor()

    # Pad with 1s tail silence (matching Rust code)
    tail_pad = np.zeros(16000, dtype=np.float32)
    pcm_padded = np.concatenate([pcm, tail_pad])
    mel = mel_extractor.extract(pcm_padded)
    print(f"  Mel shape: {mel.shape}")

    # 2. Encoder
    mel_input = mel.astype(np.float16)
    if mel_input.ndim == 2:
        mel_input = mel_input[np.newaxis, :, :]  # [1, 128, T]
    encoder_out = encoder.run(None, {"mel": mel_input})
    audio_feat = encoder_out[0]  # [1, N, D]
    n_audio = audio_feat.shape[1]
    hidden_size = audio_feat.shape[2]
    print(f"  Encoder output: {audio_feat.shape} (n_audio={n_audio}, D={hidden_size})")

    # 3. Build prompt IDs (matching qwen3.rs build_prompt_ids)
    sys_text = "You are an ASR engine. Transcribe spoken audio exactly in the spoken language. Do not translate. Do not add words not clearly present."
    sys_ids = tokenizer.encode(sys_text, add_special_tokens=False).ids

    prefix = [IM_START, 8948, NEWLINE] + list(sys_ids) + [IM_END, NEWLINE, IM_START, 872, NEWLINE, AUDIO_START]
    suffix = [AUDIO_END, IM_END, NEWLINE, IM_START, 77091, NEWLINE]
    lang_text = "language English"
    lang_ids = tokenizer.encode(lang_text, add_special_tokens=False).ids
    suffix += list(lang_ids)
    suffix.append(ASR_TEXT_ID)

    input_ids = prefix + [AUDIO_PAD_ID] * n_audio + suffix
    seq_len = len(input_ids)
    audio_offset = len(prefix)
    print(f"  Prompt: {len(prefix)} prefix + {n_audio} audio + {len(suffix)} suffix = {seq_len} tokens")

    # 4. Embed tokens + scatter encoder output
    ids_array = np.array([input_ids], dtype=np.int64)
    embed_out = embed_tokens.run(None, {"input_ids": ids_array})
    embeds = embed_out[0]  # [1, seq_len, D]
    # Scatter audio features into embed positions
    embeds[0, audio_offset:audio_offset + n_audio, :] = audio_feat[0]

    # 5. Prefill
    position_ids = np.tile(np.arange(seq_len, dtype=np.int64).reshape(1, -1), (3, 1, 1))
    attn_mask = np.ones((1, seq_len), dtype=np.int64)

    feeds = {
        "inputs_embeds": embeds.astype(np.float16),
        "position_ids": position_ids,
        "attention_mask": attn_mask,
    }
    for i in range(NUM_LAYERS):
        feeds[f"past_key_values.{i}.key"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
        feeds[f"past_key_values.{i}.value"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)

    outputs = decoder.run(None, feeds)
    logits = outputs[0]
    kv = outputs[1:]

    first_token = int(np.argmax(logits[0, -1, :]))
    tokens = [first_token]
    pos = seq_len

    # 6. Decode loop
    for step in range(MAX_NEW_TOKENS):
        if tokens[-1] in EOS_IDS:
            break

        tok_embed = embed_tokens.run(None, {"input_ids": np.array([[tokens[-1]]], dtype=np.int64)})[0]

        pos_ids = np.full((3, 1, 1), pos, dtype=np.int64)
        total_len = pos + 1
        mask = np.ones((1, total_len), dtype=np.int64)

        feeds = {
            "inputs_embeds": tok_embed.astype(np.float16),
            "position_ids": pos_ids,
            "attention_mask": mask,
        }
        for i in range(NUM_LAYERS):
            feeds[f"past_key_values.{i}.key"] = kv[i * 2]
            feeds[f"past_key_values.{i}.value"] = kv[i * 2 + 1]

        outputs = decoder.run(None, feeds)
        logits = outputs[0]
        kv = outputs[1:]

        next_token = int(np.argmax(logits[0, -1, :]))
        tokens.append(next_token)
        pos += 1

    # Debug: show all generated tokens
    print(f"  Raw tokens ({len(tokens)}):")
    for i, t in enumerate(tokens):
        piece = tokenizer.decode([t], skip_special_tokens=False)
        print(f"    [{i}] id={t} piece={repr(piece)}")
        if i > 30:
            print(f"    ... ({len(tokens) - i - 1} more)")
            break

    # Filter EOS/meta tokens (matching Rust decode_output: unwrap_or(0) fallback)
    META_IDS = EOS_IDS | {IM_START, IM_END, NEWLINE, ASR_TEXT_ID, AUDIO_START, AUDIO_END}
    content_tokens = [t for t in tokens if t not in META_IDS]

    text = tokenizer.decode(content_tokens, skip_special_tokens=True).strip()
    print(f"  Output: {text}")
    print(f"  Tokens: {len(tokens)} total, {len(content_tokens)} content")
    return text


# Need a whisper mel extractor matching the Rust code
# Write a minimal one
class WhisperMelExtractor:
    """Minimal whisper-compatible mel extractor matching lune-voice whisper.rs."""

    def __init__(self, sr=16000, n_fft=400, hop=160, win=400, n_mels=128):
        import scipy.signal
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.n_mels = n_mels
        self.window = scipy.signal.windows.hann(win, sym=False)
        self.mel_filters = self._slaney_filterbank(sr, n_fft, n_mels)

    def _slaney_filterbank(self, sr, n_fft, n_mels):
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0

        def hz_to_mel(hz):
            return np.where(hz < min_log_hz, hz / f_sp, min_log_mel + np.log(hz / min_log_hz) / logstep)

        def mel_to_hz(mel):
            return np.where(mel < min_log_mel, mel * f_sp, min_log_hz * np.exp(logstep * (mel - min_log_mel)))

        n_bins = n_fft // 2 + 1
        mel_min = hz_to_mel(np.array(0.0))
        mel_max = hz_to_mel(np.array(sr / 2.0))
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        fft_freqs = np.arange(n_bins) * sr / n_fft

        fb = np.zeros((n_mels, n_bins), dtype=np.float32)
        for m in range(n_mels):
            left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
            enorm = 2.0 / max(right - left, 1e-10)
            for k in range(n_bins):
                up = (fft_freqs[k] - left) / max(center - left, 1e-10)
                down = (right - fft_freqs[k]) / max(right - center, 1e-10)
                fb[m, k] = max(0, min(up, down)) * enorm
        return fb

    def extract(self, pcm):
        """Returns mel spectrogram [128, T] in float32."""
        # Reflect pad
        pad = self.n_fft // 2
        padded = np.pad(pcm, (pad, pad), mode="reflect")

        # STFT
        n_frames = (len(padded) - self.n_fft) // self.hop + 1
        # Drop last frame (Whisper convention)
        n_frames -= 1
        n_bins = self.n_fft // 2 + 1

        power = np.zeros((n_frames, n_bins), dtype=np.float32)
        win_offset = (self.n_fft - self.win) // 2
        for i in range(n_frames):
            start = i * self.hop
            frame = np.zeros(self.n_fft, dtype=np.float32)
            frame[win_offset:win_offset + self.win] = padded[start:start + self.win] * self.window
            spectrum = np.fft.rfft(frame)
            power[i] = np.abs(spectrum) ** 2

        # Mel filterbank
        mel = self.mel_filters @ power.T  # [n_mels, n_frames]

        # Whisper log normalize
        mel = np.maximum(mel, 1e-10)
        mel = np.log10(mel)
        peak = mel.max()
        mel = np.maximum(mel, peak - 8.0)
        mel = (mel / 4.0) + 1.0

        return mel


# Monkey-patch into the module namespace so run_onnx can use it
sys.modules["whisper_mel"] = type(sys)("whisper_mel")
sys.modules["whisper_mel"].WhisperMelExtractor = WhisperMelExtractor

pcm, sr = load_wav_f32(wav_path)
print(f"Audio: {len(pcm)} samples, {sr} Hz, {len(pcm)/sr:.2f}s")

if sr != 16000:
    print(f"WARNING: Sample rate is {sr}, expected 16000. Results may differ.")

onnx_text = run_onnx(pcm, sr)

print("\n" + "=" * 60)
print("Attempting reference model comparison...")
try:
    ref_text = run_reference(pcm, sr)
    if ref_text and onnx_text:
        print(f"\n{'='*60}")
        print(f"Reference: {ref_text}")
        print(f"ONNX:      {onnx_text}")
        print(f"Match: {ref_text.strip() == onnx_text.strip()}")
except Exception as e:
    print(f"Reference failed: {e}")
    print("(This is OK — the ONNX result above is the important one)")
