"""Inline probe: is the correction LLM bad, or just over-quantized?

Runs the EXACT production correction prompt (BASE_SYSTEM_PROMPT + user
format + /no_think) with greedy decoding against every quantization variant
of Qwen3-0.6B on disk, on real failure cases from the benches.

Run: uv run scripts/probe_llm_quant.py
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = Path.home() / ".local/share/lune-voice/models/qwen3-0.6b"
VARIANTS = {
    "q4f16 (current)": MODEL_DIR / "model_q4f16.onnx",
    "int8": MODEL_DIR / "onnx" / "model_int8.onnx",
}
MAX_NEW_TOKENS = 256

# Verbatim from crates/lune-core/src/config/style_llm.rs BASE_SYSTEM_PROMPT.
SYSTEM = (
    "You are an ASR transcript corrector.\n"
    "Output only the corrected transcript, in the same language as the input. "
    "Make the smallest edits that yield natural written text.\n\n"
    "Apply when clearly warranted:\n"
    "- Fix misspellings, fused or split words, and obvious ASR substitutions.\n"
    "- Fix homophones only when the surrounding context makes the right word unambiguous.\n"
    "- Insert short missing function words and fix simple agreement, plurals, and contractions.\n"
    "- If a dictionary list is provided below, use the exact target spelling for any matched phrase.\n\n"
    "Punctuation rules:\n"
    "- Preserve any punctuation already present in the input.\n"
    "- Inside one sentence, use commas for clause boundaries, list separators, and natural pauses — not periods.\n"
    "- Split into a new sentence only when the content clearly shifts to a separate complete thought.\n"
    "- End each sentence with exactly one terminal mark (. ! ?). Use ? only for actual questions; reserve ! for clear emphasis.\n"
    "- Capitalize the first word of each sentence and proper nouns. Do not invent capitalization mid-sentence.\n\n"
    "Do not:\n"
    "- Translate, paraphrase, or reorder content.\n"
    "- Replace a word with an unrelated word, or invent facts, names, or details.\n"
    "- Censor, soften, or remove profanity or sensitive terms.\n"
    "- Answer questions or reply conversationally — the input is a transcript, not a request.\n"
    "- Insert interior periods, semicolons, em-dashes, ellipses, or quotation marks unless the input clearly demands them.\n\n"
    "If a word looks wrong but the correct alternative is unclear, keep the original.\n\n"
    "Examples (English; the same minimal-edit policy applies in any language):\n"
    "Input: thanks\n"
    "Output: Thanks.\n"
    "Input: we should meat at noon\n"
    "Output: We should meet at noon.\n"
    "Input: pick up bread eggs and milk on the way home\n"
    "Output: Pick up bread, eggs, and milk on the way home.\n"
    "Input: did you see john where is he\n"
    "Output: Did you see John? Where is he?\n"
    "Input: i finished the report send it over when you get a sec ill be at lunch till one\n"
    "Output: I finished the report. Send it over when you get a sec. I'll be at lunch till one.\n"
    "Input: hey just checking in on the deploy did the migration finish if not we should roll back before the demo at three im on slack let me know\n"
    "Output: Hey, just checking in on the deploy. Did the migration finish? If not, we should roll back before the demo at three. I'm on Slack, let me know."
)

# Real failure cases from the benches (post-heuristic text, as the LLM sees it).
CASES = [
    ("run-on (punct gate)", "Turn on the light then close the door it is getting dark in here now"),
    ("interrogative (punct gate)", "What time is it"),
    ("profanity short (167)", "Fuck you, gonna do about your little fucking bitch. Fuck you."),
    ("noisy long-form snippet", "I don't wanna to hear anymore from you, Mister. If you stay here, you take orders from me, and that includes leaving the girl alone."),
    ("already correct (must copy)", "The morning light filtered through the kitchen window as she poured her second cup of coffee."),
    ("letter dictation (punc_001)", "Dear Mr. Thompson, I am writing to confirm our appointment on Friday, June 14th at 3:30 p.m."),
]


def user_prompt(transcript: str) -> str:
    return f"Correct the following English transcript:\n{transcript}\n\n/no_think"


def chatml(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load(path: Path):
    sess = ort.InferenceSession(
        str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]
    num_layers = sum(1 for n in input_names if n.endswith(".key"))
    has_pos = any("position" in n for n in input_names)
    kv_meta = next(i for i in sess.get_inputs() if "past_key_values.0.key" in i.name)
    num_heads = kv_meta.shape[1] if isinstance(kv_meta.shape[1], int) else 8
    head_dim = kv_meta.shape[3] if isinstance(kv_meta.shape[3], int) else 128
    kv_dtype = np.float16 if "float16" in kv_meta.type.lower() else np.float32
    return sess, num_layers, has_pos, num_heads, head_dim, kv_dtype


def generate(sess, tok, prompt, num_layers, has_pos, num_heads, head_dim, kv_dtype, eos_ids):
    ids = tok.encode(prompt).ids
    feeds = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
    }
    if has_pos:
        feeds["position_ids"] = np.arange(len(ids), dtype=np.int64).reshape(1, -1)
    for i in range(num_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros((1, num_heads, 0, head_dim), dtype=kv_dtype)
        feeds[f"past_key_values.{i}.value"] = np.zeros((1, num_heads, 0, head_dim), dtype=kv_dtype)

    outputs = sess.run(None, feeds)
    logits, kv = outputs[0], outputs[1:]
    generated = []
    total = len(ids)
    for _ in range(MAX_NEW_TOKENS):
        nxt = int(np.argmax(logits[0, -1, :]))
        if nxt in eos_ids:
            break
        generated.append(nxt)
        total += 1
        feeds = {
            "input_ids": np.array([[nxt]], dtype=np.int64),
            "attention_mask": np.ones((1, total), dtype=np.int64),
        }
        if has_pos:
            feeds["position_ids"] = np.array([[total - 1]], dtype=np.int64)
        for i in range(num_layers):
            feeds[f"past_key_values.{i}.key"] = kv[i * 2]
            feeds[f"past_key_values.{i}.value"] = kv[i * 2 + 1]
        outputs = sess.run(None, feeds)
        logits, kv = outputs[0], outputs[1:]
    text = tok.decode(generated, skip_special_tokens=True)
    # Qwen3 /no_think still emits an empty think block — strip it.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    eos_ids = {tid for t in ["<|im_end|>", "<|endoftext|>"] if (tid := tok.token_to_id(t)) is not None}

    for label, path in VARIANTS.items():
        if not path.exists():
            print(f"\n### {label}: MISSING ({path})")
            continue
        print(f"\n{'#' * 70}\n### {label}  ({path.stat().st_size // (1024 * 1024)} MiB)\n{'#' * 70}")
        sess, num_layers, has_pos, num_heads, head_dim, kv_dtype = load(path)
        print(f"provider={sess.get_providers()[0]} layers={num_layers} kv={kv_dtype.__name__}")
        for name, transcript in CASES:
            prompt = chatml(SYSTEM, user_prompt(transcript))
            out = generate(sess, tok, prompt, num_layers, has_pos, num_heads, head_dim, kv_dtype, eos_ids)
            print(f"\n--- {name}")
            print(f"  in : {transcript}")
            print(f"  out: {out}")


if __name__ == "__main__":
    main()
