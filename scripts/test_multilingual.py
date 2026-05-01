"""Quick test: Qwen3-ASR ONNX on non-English audio (matching current Rust code)."""
import sys
import numpy as np
from pathlib import Path

ONNX_DIR = Path.home() / ".local/share/lune-voice/models/qwen3-asr-0.6b"

fixtures = {
    "french": Path("test_fixtures/voxpopuli_french.wav"),
    "german": Path("test_fixtures/voxpopuli_german.wav"),
    "spanish": Path("test_fixtures/voxpopuli_spanish.wav"),
    "italian": Path("test_fixtures/voxpopuli_italian.wav"),
}

if len(sys.argv) > 1:
    lang = sys.argv[1]
    fixtures = {lang: fixtures[lang]}

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


def load_wav(path):
    import soundfile as sf
    pcm, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    return pcm, sr


def load_reference(path):
    txt = path.with_suffix(".txt")
    if txt.exists():
        return txt.read_text().strip()
    return ""


def whisper_mel(pcm):
    """Minimal whisper-compatible 128-bin log-mel at 16kHz."""
    import scipy.signal

    sr, n_fft, hop, win, n_mels = 16000, 400, 160, 400, 128
    window = scipy.signal.windows.hann(win, sym=False)

    # Slaney mel filterbank
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz2mel = lambda hz: np.where(hz < min_log_hz, hz / f_sp, min_log_mel + np.log(hz / min_log_hz) / logstep)
    mel2hz = lambda m: np.where(m < min_log_mel, m * f_sp, min_log_hz * np.exp(logstep * (m - min_log_mel)))
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(hz2mel(0.0), hz2mel(sr / 2.0), n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    fft_freqs = np.arange(n_bins) * sr / n_fft
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = hz_pts[m], hz_pts[m+1], hz_pts[m+2]
        enorm = 2.0 / max(r - l, 1e-10)
        for k in range(n_bins):
            fb[m, k] = max(0, min((fft_freqs[k]-l)/max(c-l,1e-10), (r-fft_freqs[k])/max(r-c,1e-10))) * enorm

    pad = n_fft // 2
    padded = np.pad(pcm, (pad, pad), mode="reflect")
    n_frames = (len(padded) - n_fft) // hop + 1 - 1
    power = np.zeros((n_frames, n_bins), dtype=np.float32)
    for i in range(n_frames):
        frame = padded[i*hop:i*hop+n_fft] * window
        power[i] = np.abs(np.fft.rfft(frame)) ** 2
    mel = fb @ power.T
    mel = np.maximum(mel, 1e-10)
    mel = np.log10(mel)
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = mel / 4.0 + 1.0

    # Pad to 3000 frames (30s) like Rust code
    if mel.shape[1] < 3000:
        mel = np.pad(mel, ((0,0),(0, 3000 - mel.shape[1])), constant_values=mel.min())
    return mel


def run_onnx(wav_path):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    encoder = ort.InferenceSession(str(ONNX_DIR / "qwen3-asr-encoder.onnx"), providers=providers)
    embed_tokens = ort.InferenceSession(str(ONNX_DIR / "qwen3-asr-embed-tokens.onnx"), providers=providers)
    decoder = ort.InferenceSession(str(ONNX_DIR / "qwen3-asr-decoder.onnx"), providers=providers)
    tokenizer = Tokenizer.from_file(str(ONNX_DIR / "tokenizer.json"))

    pcm, sr = load_wav(wav_path)
    tail_pad = np.zeros(16000, dtype=np.float32)
    pcm = np.concatenate([pcm, tail_pad])

    # 1. Mel
    mel = whisper_mel(pcm)
    mel_input = mel.astype(np.float16)[np.newaxis, :, :]
    print(f"  Mel: {mel_input.shape}")

    # 2. Encoder
    audio_feat = encoder.run(None, {"mel": mel_input})[0]
    n_audio = audio_feat.shape[1]
    print(f"  Encoder: {audio_feat.shape}")

    # 3. Prompt — empty system, no forced language (matching current Rust)
    prefix = [IM_START, 8948, NEWLINE, IM_END, NEWLINE, IM_START, 872, NEWLINE, AUDIO_START]
    suffix = [AUDIO_END, IM_END, NEWLINE, IM_START, 77091, NEWLINE]

    input_ids = prefix + [AUDIO_PAD_ID] * n_audio + suffix
    seq_len = len(input_ids)
    audio_offset = len(prefix)
    print(f"  Prompt: {len(prefix)}+{n_audio}+{len(suffix)} = {seq_len}")

    # 4. Embed + scatter
    embeds = embed_tokens.run(None, {"input_ids": np.array([input_ids], dtype=np.int64)})[0]
    embeds[0, audio_offset:audio_offset+n_audio, :] = audio_feat[0]

    # 5. Prefill
    pos_ids = np.tile(np.arange(seq_len, dtype=np.int64).reshape(1,-1), (3,1,1))
    feeds = {
        "inputs_embeds": embeds.astype(np.float16),
        "position_ids": pos_ids,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
    }
    for i in range(NUM_LAYERS):
        feeds[f"past_key_values.{i}.key"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)
        feeds[f"past_key_values.{i}.value"] = np.zeros((1, NUM_KV_HEADS, 0, HEAD_DIM), dtype=np.float16)

    outputs = decoder.run(None, feeds)
    logits = outputs[0]
    kv = outputs[1:]
    tokens = [int(np.argmax(logits[0, -1, :]))]
    pos = seq_len

    # 6. Decode
    for _ in range(MAX_NEW_TOKENS):
        if tokens[-1] in EOS_IDS:
            break
        tok_embed = embed_tokens.run(None, {"input_ids": np.array([[tokens[-1]]], dtype=np.int64)})[0]
        feeds = {
            "inputs_embeds": tok_embed.astype(np.float16),
            "position_ids": np.full((3,1,1), pos, dtype=np.int64),
            "attention_mask": np.ones((1, pos+1), dtype=np.int64),
        }
        for i in range(NUM_LAYERS):
            feeds[f"past_key_values.{i}.key"] = kv[i*2]
            feeds[f"past_key_values.{i}.value"] = kv[i*2+1]
        outputs = decoder.run(None, feeds)
        logits = outputs[0]
        kv = outputs[1:]
        tokens.append(int(np.argmax(logits[0, -1, :])))
        pos += 1

    # 7. Parse — find <asr_text> and decode after it
    raw_ids = [t for t in tokens if t not in EOS_IDS]
    raw_text = tokenizer.decode(raw_ids, skip_special_tokens=False)

    asr_pos = next((i for i, t in enumerate(tokens) if t == ASR_TEXT_ID), None)
    if asr_pos is not None:
        content = [t for t in tokens[asr_pos+1:] if t not in EOS_IDS and t != IM_END]
    else:
        content = [t for t in tokens if t not in EOS_IDS and t != IM_END and t != ASR_TEXT_ID]
    clean = tokenizer.decode(content, skip_special_tokens=True).strip()

    return raw_text, clean


for lang, wav in fixtures.items():
    if not wav.exists():
        print(f"[skip] {lang}: {wav} not found")
        continue

    ref = load_reference(wav)
    print(f"\n{'='*60}")
    print(f"Language: {lang}")
    print(f"Reference: {ref}")
    print(f"{'='*60}")

    raw, clean = run_onnx(wav)
    print(f"  Raw:   {raw}")
    print(f"  Clean: {clean}")
    print()
