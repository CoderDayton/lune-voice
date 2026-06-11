"""Quick standalone test: can the ONNX Qwen2.5-1.5B-Instruct correct a transcript?"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path

MODEL_DIR = Path.home() / ".local/share/lune-voice/models/qwen3-0.6b"
MODEL_PATH = MODEL_DIR / "model_q4f16.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

MAX_NEW_TOKENS = 256

TRANSCRIPT = (
    "i was gona go to the store but i didnt no weather it was open or not "
    "so i desided to stay home and wach a moovie insted. my frend called me "
    "and sed he wood come over latr so we cud hang out togther."
)

PROMPTS = [
    ("minimal", "Correct this transcript:\n\n" + TRANSCRIPT),
    (
        "structured",
        "You are an ASR error correction model. Fix transcription errors "
        "in the following text. Preserve meaning and phrasing, only fix "
        "clear spelling/word errors. Output only the corrected text.\n\n"
        "Transcript:\n" + TRANSCRIPT,
    ),
]


def build_chatml_prompt(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model():
    print(f"Loading model from {MODEL_PATH}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"Active provider: {sess.get_providers()[0]}")

    input_names = [i.name for i in sess.get_inputs()]
    num_layers = sum(1 for n in input_names if n.endswith(".key"))
    has_pos = any("position" in n for n in input_names)
    print(f"Layers: {num_layers}, position_ids: {has_pos}")

    kv_shape_meta = next(
        i for i in sess.get_inputs() if "past_key_values.0.key" in i.name
    )
    kv_shape = kv_shape_meta.shape
    num_heads = kv_shape[1] if isinstance(kv_shape[1], int) else 2
    head_dim = kv_shape[3] if isinstance(kv_shape[3], int) else 64
    kv_dtype = (
        np.float16 if "float16" in kv_shape_meta.type.lower() else np.float32
    )
    print(f"KV: heads={num_heads}, head_dim={head_dim}, dtype={kv_dtype}")

    return sess, tok, num_layers, has_pos, num_heads, head_dim, kv_dtype


def generate(sess, tok, prompt_text, num_layers, has_pos, num_heads, head_dim, kv_dtype):
    enc = tok.encode(prompt_text)
    ids = enc.ids

    input_ids = np.array([ids], dtype=np.int64)
    attn_mask = np.ones((1, len(ids)), dtype=np.int64)

    feeds = {"input_ids": input_ids, "attention_mask": attn_mask}
    if has_pos:
        feeds["position_ids"] = np.arange(len(ids), dtype=np.int64).reshape(1, -1)

    for i in range(num_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, num_heads, 0, head_dim), dtype=kv_dtype
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, num_heads, 0, head_dim), dtype=kv_dtype
        )

    outputs = sess.run(None, feeds)
    logits = outputs[0]
    kv = outputs[1:]

    eos_ids = set()
    for t in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
        tid = tok.token_to_id(t)
        if tid is not None:
            eos_ids.add(tid)

    generated = []
    total_len = len(ids)

    for step in range(MAX_NEW_TOKENS):
        next_token = int(np.argmax(logits[0, -1, :]))
        if next_token in eos_ids:
            break
        generated.append(next_token)
        total_len += 1

        feeds = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((1, total_len), dtype=np.int64),
        }
        if has_pos:
            feeds["position_ids"] = np.array([[total_len - 1]], dtype=np.int64)

        for i in range(num_layers):
            feeds[f"past_key_values.{i}.key"] = kv[i * 2]
            feeds[f"past_key_values.{i}.value"] = kv[i * 2 + 1]

        outputs = sess.run(None, feeds)
        logits = outputs[0]
        kv = outputs[1:]

    return tok.decode(generated, skip_special_tokens=True)


def main():
    sess, tok, num_layers, has_pos, num_heads, head_dim, kv_dtype = load_model()

    system = "You are a helpful assistant."

    for label, user_msg in PROMPTS:
        prompt = build_chatml_prompt(system, user_msg)
        print(f"\n{'='*60}")
        print(f"Test: {label}")
        print(f"{'='*60}")
        print(f"Prompt tokens: {len(tok.encode(prompt).ids)}")
        print("Generating...")

        result = generate(
            sess, tok, prompt, num_layers, has_pos, num_heads, head_dim, kv_dtype
        )
        print(f"\nOutput:\n{result}")

    # Test 3: actual pipeline prompt (JSON-edit format)
    pipeline_system = (
        "You are a conservative ASR transcript repair engine.\n"
        "\n"
        "Primary rule: when in doubt, copy the transcript exactly.\n"
        "\n"
        "Your job:\n"
        "- Repair only obvious ASR transcription errors.\n"
        "- Preserve the speaker's words, order, language, tone, dialect, "
        "grammar, tense, mood, repetitions, disfluencies, filler words, "
        "and fragments.\n"
        "- Add punctuation and casing only when it is clear from the text.\n"
        "\n"
        "Do not:\n"
        "- Rewrite, paraphrase, summarize, complete, translate, modernize, "
        "stylize, or improve the text.\n"
        "- Change valid wording just because a more common phrase exists.\n"
        "- Remove repetitions or filler words unless they are obvious ASR "
        "artifacts.\n"
        "- Guess named entities, numbers, technical terms, or foreign-language "
        "words without strong evidence.\n"
        "\n"
        "Output contract:\n"
        "- Return only the transcript text.\n"
        "- If no correction is certain, return the input transcript unchanged."
    )
    pipeline_user = (
        "Language: English\n"
        "\n"
        "Task: propose minimal ASR transcript repairs as JSON edits.\n"
        "Do not apply style instructions in this pass.\n"
        "\n"
        "Output format:\n"
        "- Return only a JSON array.\n"
        '- Use [] when no correction is certain.\n'
        '- Each edit: {"start_word":number,"end_word":number,'
        '"replacement":"text"}.\n'
        "- start_word inclusive; end_word exclusive.\n"
        "- Edit only suspicious word indices listed below.\n"
        "\n"
        "Edit standard:\n"
        "- Make the smallest possible change.\n"
        "- Replacements must sound like the original word "
        "(phonetic similarity).\n"
        "- Correct only words or punctuation clearly wrong from context.\n"
        "- Keep valid sentence fragments and spoken phrasing.\n"
        "- Keep transcript in English; do not translate.\n"
        "\n"
        "Forbidden edits:\n"
        "- paraphrasing\n- summarizing\n"
        "- completing unfinished thoughts\n"
        "- grammar cleanup unless word is clearly ASR error\n"
        "- style improvement\n- entity guessing\n"
        "- removing repetitions\n- removing filler words\n"
        "- changing valid tense/mood\n"
        "- changing valid dialogue fragments\n"
        "- inventing missing content\n"
        "\n"
        "If correction is not certain, copy the part exactly.\n"
        "If transcript already looks valid, return it unchanged.\n"
        "\n"
        "Suspicious word indices:\n"
        '- 2..3: "gona" confidence=0.30 fix_score=0.90\n'
        '- 9..10: "didnt" confidence=0.35 fix_score=0.80\n'
        '- 10..11: "no" confidence=0.25 fix_score=0.92\n'
        '- 11..12: "weather" confidence=0.20 fix_score=0.95\n'
        '- 18..19: "desided" confidence=0.28 fix_score=0.88\n'
        '- 22..23: "wach" confidence=0.22 fix_score=0.93\n'
        '- 24..25: "moovie" confidence=0.18 fix_score=0.96\n'
        '- 25..26: "insted" confidence=0.30 fix_score=0.85\n'
        '- 28..29: "frend" confidence=0.25 fix_score=0.90\n'
        '- 33..34: "sed" confidence=0.20 fix_score=0.95\n'
        '- 35..36: "wood" confidence=0.22 fix_score=0.94\n'
        '- 38..39: "latr" confidence=0.28 fix_score=0.88\n'
        '- 41..42: "cud" confidence=0.25 fix_score=0.92\n'
        '- 45..46: "togther" confidence=0.30 fix_score=0.85\n'
        "\n"
        "Transcript:\n"
        "<transcript>\n"
        + TRANSCRIPT + "\n"
        "</transcript>"
    )
    prompt = build_chatml_prompt(pipeline_system, pipeline_user)
    print(f"\n{'='*60}")
    print("Test: pipeline JSON-edit format")
    print(f"{'='*60}")
    print(f"Prompt tokens: {len(tok.encode(prompt).ids)}")
    print("Generating...")
    result = generate(
        sess, tok, prompt, num_layers, has_pos, num_heads, head_dim, kv_dtype
    )
    print(f"\nOutput:\n{result}")

    print(f"\n{'='*60}")
    print("Original:")
    print(TRANSCRIPT)


if __name__ == "__main__":
    main()
