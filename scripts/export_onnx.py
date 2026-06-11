#!/usr/bin/env python3
"""Custom ONNX exporter for Qwen3-ASR-0.6B and the wav2vec2 forced aligner.

Bypasses optimum-cli entirely. Uses torch.onnx.export with manual wrappers
that correctly handle Qwen3-ASR's architecture (GQA, q_norm/k_norm, RoPE,
SwiGLU, tied embeddings) and produce clean FP16 graphs with proper KV cache
I/O.

Usage:
    python scripts/export_onnx.py --asr --output-dir models/qwen3-asr-0.6b-onnx
    python scripts/export_onnx.py --asr --quant-policy balanced
    python scripts/export_onnx.py --asr --revision <commit-sha> --quant-policy safe
    python scripts/export_onnx.py --asr --quant-policy compact
    python scripts/export_onnx.py --asr --no-quantize

Requirements:
    pip install torch transformers onnx onnxscript onnxruntime
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import onnx
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message="Missing annotation for parameter")
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

try:
    import qwen_asr  # noqa: F401
except ImportError:
    pass

DTYPE = torch.float16


# ── Utilities ────────────────────────────────────────────────────────────────


def _save_consolidated(model: onnx.ModelProto, out_path: Path) -> None:
    """Save ONNX model with all large tensors in a single external data file.

    Serializes into a temp directory first, then moves both files into place,
    so a failed save never leaves a corrupt model/data pair behind.
    """
    data_name = out_path.stem + ".onnx_data"
    data_path = out_path.parent / data_name
    tmp_dir = out_path.parent / f".{out_path.stem}.tmp-save"
    if tmp_dir.exists():
        for stale in tmp_dir.iterdir():
            stale.unlink()
    else:
        tmp_dir.mkdir(parents=True)

    onnx.save_model(
        model,
        str(tmp_dir / out_path.name),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )
    (tmp_dir / out_path.name).replace(out_path)
    tmp_data = tmp_dir / data_name
    if tmp_data.exists():
        tmp_data.replace(data_path)
    elif data_path.exists():
        data_path.unlink()
    tmp_dir.rmdir()
    if data_path.exists():
        sz = data_path.stat().st_size / (1024 * 1024)
        print(f"    {data_name}: {sz:.1f} MB")


def _consolidate_external_data(directory: Path, onnx_name: str) -> None:
    onnx_path = directory / onnx_name
    if not onnx_path.exists():
        return
    print(f"  Consolidating {onnx_name}...")
    model = onnx.load(str(onnx_path), load_external_data=True)
    _save_consolidated(model, onnx_path)


def _list_files(directory: Path) -> None:
    total = 0
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        sz = f.stat().st_size
        total += sz
        if sz > 1_000_000:
            print(f"  {f.name}: {sz / (1024 * 1024):.1f} MB")
    print(f"  TOTAL: {total / (1024 * 1024):.1f} MB")


def _cleanup_temp_artifacts(output_dir: Path) -> None:
    """Remove exporter temp files while preserving canonical ONNX/data pairs."""
    keep = {
        "qwen3-asr-encoder.onnx",
        "qwen3-asr-encoder.onnx_data",
        "qwen3-asr-embed-tokens.onnx",
        "qwen3-asr-embed-tokens.onnx_data",
        "qwen3-asr-decoder.onnx",
        "qwen3-asr-decoder.onnx_data",
        "export_manifest.json",
    }
    for f in output_dir.iterdir():
        if f.is_dir() and f.name.endswith(".tmp-save"):
            for stale in f.iterdir():
                stale.unlink()
            f.rmdir()
        elif f.suffix in (".data", ".onnx.data") and f.name not in keep:
            f.unlink()
        elif f.name.startswith("onnx__"):
            f.unlink()


def _dtype_name(data_type: int) -> str:
    return onnx.TensorProto.DataType.Name(data_type) if data_type else "UNKNOWN"


def _write_export_manifest(
    output_dir: Path,
    *,
    model_id: str,
    revision: "str | None",
    quant_policy: str,
    decoder_int8_classes: "list[str]",
    int4_threshold: "float | None",
    int4_block: int,
    strict_quant: bool,
    allow_quality_drift: bool,
    min_quant_agreement: float,
    max_quant_mean_delta: float,
    verify_prompt_tokens: "int | None" = None,
    quant_notes: "list[dict] | None" = None,
) -> None:
    """Write a compact machine-readable export/quantization summary.

    The manifest is intentionally graph-derived after all rewrites so it can
    catch silent policy misses such as zero INT4 nodes, stale FP16 initializers,
    or unexpected op/domain drift.
    """
    graphs = []
    for name in [
        "qwen3-asr-encoder.onnx",
        "qwen3-asr-embed-tokens.onnx",
        "qwen3-asr-decoder.onnx",
    ]:
        path = output_dir / name
        if not path.exists():
            continue
        model = onnx.load(str(path), load_external_data=False)
        op_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        for node in model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
            dom = node.domain or "ai.onnx"
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
        init_counts: dict[str, int] = {}
        init_bytes: dict[str, int] = {}
        for init in model.graph.initializer:
            dt = _dtype_name(init.data_type)
            init_counts[dt] = init_counts.get(dt, 0) + 1
            elem_size = {
                "FLOAT16": 2,
                "FLOAT": 4,
                "INT8": 1,
                "UINT8": 1,
                "INT64": 8,
                "INT32": 4,
            }.get(dt, 0)
            n = 1
            for d in init.dims:
                n *= int(d)
            init_bytes[dt] = init_bytes.get(dt, 0) + n * elem_size
        graphs.append(
            {
                "file": name,
                "onnx_bytes": path.stat().st_size,
                "external_data_bytes": (
                    path.with_name(path.stem + ".onnx_data").stat().st_size
                    if path.with_name(path.stem + ".onnx_data").exists()
                    else 0
                ),
                "ops": dict(sorted(op_counts.items())),
                "domains": dict(sorted(domain_counts.items())),
                "initializer_count_by_dtype": dict(sorted(init_counts.items())),
                "initializer_bytes_by_dtype_estimate": dict(sorted(init_bytes.items())),
            }
        )
    manifest = {
        "model_id": model_id,
        "revision": revision,
        "quant_policy": quant_policy,
        "decoder_int8_classes": decoder_int8_classes,
        "int4_threshold": int4_threshold,
        "int4_block": int4_block,
        "strict_quant": strict_quant,
        "allow_quality_drift": allow_quality_drift,
        "min_quant_agreement": min_quant_agreement,
        "max_quant_mean_delta": max_quant_mean_delta,
        "verify_prompt_tokens": verify_prompt_tokens,
        "notes": quant_notes or [],
        "graphs": graphs,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ASR] Wrote export_manifest.json")


def _require_quantized(stage: str, count: int, *, strict: bool) -> None:
    """Fail on silent quantization misses unless explicitly allowed."""
    if count > 0:
        return
    msg = (
        f"{stage}: quantization matched 0 tensors. This usually means the graph "
        "layout, initializer names, or policy assumptions changed. Refusing to "
        "ship a silently under-compressed or misconfigured export."
    )
    if strict:
        raise RuntimeError(
            msg + " Pass --allow-quant-misses to downgrade this to a warning."
        )
    print(f"[ASR] WARNING: {msg}")


# Per-profile quality gates: (min top-1 agreement, max mean |Δlogit|) versus
# the torch reference on the verification prompt. safe is probe-lossless by
# construction; targeted profiles tolerate one-token-in-ten probe drift
# (measured balanced: 98.1% agreement, mean 0.195) before aborting toward
# the fallback.
_PROFILE_GATES = {
    "safe": (1.0, 0.25),
    "balanced": (0.9, 0.25),
    "compact": (0.9, 0.35),
}
_PROFILE_FALLBACK = {"balanced": "safe", "compact": "balanced"}


def _check_quant_quality(
    stats: "dict | None",
    *,
    profile: str,
    allow_quality_drift: bool,
    min_agreement: float,
    max_mean_delta: float,
) -> None:
    """Hard-gate quantized probe drift per profile; abort toward the fallback.

    The probe is text-only (no audio conditioning), so passing is necessary
    but not sufficient — fixture transcription via scripts/test_asr_reference.py
    remains the authoritative quality check before shipping.
    """
    if stats is None:
        return
    agreement = float(stats["top1_agreement"])
    mean_delta = float(stats["mean_abs_logit"])
    if agreement >= min_agreement and mean_delta <= max_mean_delta:
        print(
            f"[ASR] Quality gate passed ({profile}): "
            f"agreement={agreement:.2%}, mean|Δlogit|={mean_delta:.4f}"
        )
        return

    fallback = _PROFILE_FALLBACK.get(profile)
    msg = (
        f"{profile} build failed the quality gate: "
        f"top1_agreement={agreement:.2%} < {min_agreement:.2%} or "
        f"mean|Δlogit|={mean_delta:.4f} > {max_mean_delta:.4f}."
        + (f" Re-run with --quant-policy {fallback}." if fallback else "")
    )
    if allow_quality_drift:
        print(f"[ASR] WARNING: {msg}")
    else:
        raise RuntimeError(msg)


# ── Verification ─────────────────────────────────────────────────────────────

# Fallback verification prompt: chat-control tokens plus a few ordinary text
# ids. Only used when no tokenizer is available (e.g. --int4-requant smoke
# runs). Too short and too degenerate (token 0) to measure quantization drift
# reliably — export paths build a realistic prompt via _build_verify_ids.
_VERIFY_IDS = [151644, 8948, 198, 9707, 1879, 0]

# Dictation-style text for the export-time verification prompt: long and
# lexically varied so top-1 agreement is measured over ~64 realistic decode
# positions instead of 6 control tokens.
_VERIFY_TEXT = (
    "Please schedule a follow-up appointment for Tuesday at three thirty, "
    "then remind me to email the quarterly report to Sandra before noon. "
    "The patient described intermittent dizziness, mild nausea, and a "
    "persistent ringing in the left ear after the long flight from Denver."
)


def _build_verify_ids(processor) -> "list[int]":
    """Tokenize the realistic verification prompt with the model's tokenizer.

    Prefixed with the chat-control skeleton (im_start, "system", newline) so
    the prompt exercises both control and ordinary text embeddings; capped at
    64 tokens to keep the CPU smoke test fast.
    """
    tok = getattr(processor, "tokenizer", processor)
    ids = [151644, 8948, 198] + list(tok.encode(_VERIFY_TEXT))
    return ids[:64]


def verify_asr_export(
    output_dir: Path,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    ref_logits=None,
    label: str = "export",
    verify_ids: "list[int] | None" = None,
) -> None:
    """Smoke-test the three exported graphs end to end on CPU.

    Runs encoder → embed-tokens → decoder prefill → one cached decode step
    and raises on shape mismatches or non-finite outputs, so a broken export
    fails here instead of at runtime. When ``ref_logits`` (torch reference
    for the verification prompt) is given, reports max logit deviation and
    top-1 agreement of the prefill pass.
    """
    import numpy as np
    import onnxruntime as ort

    def session(name: str) -> ort.InferenceSession:
        return ort.InferenceSession(
            str(output_dir / name), providers=["CPUExecutionProvider"]
        )

    def check(name: str, arr, lead_shape: tuple) -> None:
        actual = tuple(int(d) for d in arr.shape[: len(lead_shape)])
        if actual != lead_shape:
            raise RuntimeError(
                f"[verify:{label}] {name}: shape {tuple(arr.shape)}, "
                f"expected leading dims {lead_shape}"
            )
        if not np.isfinite(arr.astype(np.float32)).all():
            raise RuntimeError(f"[verify:{label}] {name}: non-finite values")

    def to_numpy_logits(name: str, value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 3:
            raise RuntimeError(
                f"[verify:{label}] {name}: expected rank-3 logits [B, S, V], "
                f"got shape {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(f"[verify:{label}] {name}: non-finite values")
        return arr

    print(f"[ASR] Verifying {label} (CPU smoke test)...")

    # Encoder: deterministic pseudo-mel spanning multiple chunks.
    rng = np.random.default_rng(0)
    mel = (rng.standard_normal((1, 128, 300)) * 0.5).astype(np.float16)
    feats = session("qwen3-asr-encoder.onnx").run(["features"], {"mel": mel})[0]
    if feats.ndim != 3 or feats.shape[0] != 1 or feats.shape[2] != hidden_size:
        raise RuntimeError(f"[verify:{label}] encoder features shape {feats.shape}")
    if not np.isfinite(feats.astype(np.float32)).all():
        raise RuntimeError(f"[verify:{label}] encoder produced non-finite features")

    # Embed → decoder prefill.
    ids = np.array(
        [verify_ids if verify_ids is not None else _VERIFY_IDS], dtype=np.int64
    )
    seq = ids.shape[1]
    embeds = session("qwen3-asr-embed-tokens.onnx").run(["embeds"], {"input_ids": ids})[
        0
    ]
    check("embeds", embeds, (1, seq, hidden_size))

    dec = session("qwen3-asr-decoder.onnx")
    out_names = ["logits"] + [
        f"present_key_values.{i}.{kv}"
        for i in range(num_layers)
        for kv in ("key", "value")
    ]
    feed = {
        "inputs_embeds": embeds,
        "position_ids": np.broadcast_to(
            np.arange(seq, dtype=np.int64), (3, 1, seq)
        ).copy(),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
    }
    for i in range(num_layers):
        feed[f"past_key_values.{i}.key"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float16
        )
        feed[f"past_key_values.{i}.value"] = np.zeros(
            (1, num_kv_heads, 0, head_dim), dtype=np.float16
        )
    outs = dec.run(out_names, feed)
    logits, present = outs[0], outs[1:]
    check("prefill logits", logits, (1, seq))
    for i in range(num_layers):
        check(f"present.{i}.key", present[2 * i], (1, num_kv_heads, seq, head_dim))
        check(
            f"present.{i}.value", present[2 * i + 1], (1, num_kv_heads, seq, head_dim)
        )

    # One cached decode step: past from prefill, single new position.
    step_feed = {
        "inputs_embeds": embeds[:, -1:, :],
        "position_ids": np.full((3, 1, 1), seq, dtype=np.int64),
        "attention_mask": np.ones((1, seq + 1), dtype=np.int64),
    }
    for i in range(num_layers):
        step_feed[f"past_key_values.{i}.key"] = present[2 * i]
        step_feed[f"past_key_values.{i}.value"] = present[2 * i + 1]
    step_out = dec.run(out_names, step_feed)
    check("decode-step logits", step_out[0], (1, 1))
    step_present = step_out[1:]
    for i in range(num_layers):
        check(
            f"decode-step present.{i}.key",
            step_present[2 * i],
            (1, num_kv_heads, seq + 1, head_dim),
        )
        check(
            f"decode-step present.{i}.value",
            step_present[2 * i + 1],
            (1, num_kv_heads, seq + 1, head_dim),
        )

    stats = None
    if ref_logits is not None:
        onnx_logits = to_numpy_logits("onnx logits", logits)
        torch_logits = to_numpy_logits("torch reference logits", ref_logits)

        if onnx_logits.shape != torch_logits.shape:
            raise RuntimeError(
                f"[verify:{label}] logits shape mismatch: ONNX {onnx_logits.shape}, "
                f"torch reference {torch_logits.shape}"
            )

        diff = np.abs(onnx_logits - torch_logits)
        onnx_top1 = np.argmax(onnx_logits, axis=-1)
        torch_top1 = np.argmax(torch_logits, axis=-1)
        matches = onnx_top1 == torch_top1

        agree = float(matches.sum(dtype=np.float64) / matches.size)
        mismatches = int(matches.size - int(matches.sum()))

        stats = {
            "max_abs_logit": float(diff.max()),
            "mean_abs_logit": float(diff.mean()),
            "top1_agreement": float(agree),
            "top1_matches": int(matches.size - mismatches),
            "top1_total": int(matches.size),
            "top1_mismatches": int(mismatches),
        }
        print(
            f"[ASR]   vs torch reference: max|Δlogit|={stats['max_abs_logit']:.4f}, "
            f"mean|Δlogit|={stats['mean_abs_logit']:.4f}, "
            f"top-1 agreement={stats['top1_agreement']:.2%} "
            f"({stats['top1_matches']}/{stats['top1_total']})"
        )
        if label == "export" and mismatches:
            print(
                "[ASR]   WARNING: top-1 mismatch vs torch reference on the "
                "unquantized export — inspect before shipping"
            )

    print(f"[ASR] Verify {label}: OK")
    return stats


# ── AWQ Smoothing ────────────────────────────────────────────────────────────
#
# Activation-aware weight smoothing (Lin et al., AWQ), folded into the torch
# weights BEFORE export so the ONNX graphs need no post-hoc surgery and the
# fold is exactly output-equivalent in fp32:
#   gamma_new = gamma / s        (RMSNorm weight)
#   W_new     = W * s[None, :]   (input channels of every consumer linear)
# with s = (act_mean_abs / mean(act_mean_abs)) ** alpha per channel.
#
# Calibration runs REAL AUDIO through the full generate() path (encoder +
# autoregressive decode) over the repo's fixture corpus, matching the
# production activation distribution — text-only calibration misses the
# audio-token statistics that dominate dictation traffic.
#
# Groups (norm → linears sharing its output), Qwen3 decoder layout:
#   input_layernorm          → self_attn.{q,k,v}_proj
#   post_attention_layernorm → mlp.{gate,up}_proj
# o_proj / down_proj have no preceding norm and are not smoothed. The final
# norm → lm_head group is skipped: folding it would desync the tied
# embed_tokens table, which is exported as its own graph.

_AWQ_SYSTEM_PROMPT = (
    "You are an ASR engine. Transcribe spoken audio exactly in the spoken "
    "language. Do not translate. Do not add words not clearly present."
)


def _awq_groups(text_model) -> "dict[str, tuple]":
    groups: dict[str, tuple] = {}
    for i, layer in enumerate(text_model.layers):
        groups[f"layers.{i}.input_layernorm"] = (
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        groups[f"layers.{i}.post_attention_layernorm"] = (
            layer.post_attention_layernorm,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )
    return groups


def _awq_collect_activations(
    full_model,
    processor,
    wav_paths: "list[Path]",
    device: str,
    max_new_tokens: int = 64,
) -> "dict[str, torch.Tensor]":
    """Per-channel mean-abs activations at each smoothing norm, collected by
    running real audio through the full encoder + greedy-decode path."""
    text_model = full_model.thinker.model
    groups = _awq_groups(text_model)

    accum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []
    for name, (norm, _) in groups.items():

        def make_hook(n: str):
            def hook(_mod, _inp, output):
                stats = output.detach().float().abs().mean(dim=(0, 1)).cpu()
                if n in accum:
                    accum[n] += stats
                    counts[n] += 1
                else:
                    accum[n] = stats
                    counts[n] = 1

            return hook

        hooks.append(norm.register_forward_hook(make_hook(name)))

    done = 0
    try:
        for wav in wav_paths:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _AWQ_SYSTEM_PROMPT}],
                },
                {"role": "user", "content": [{"type": "audio", "audio": str(wav)}]},
            ]
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            inputs = {
                # Float tensors (mel features) must match the fp16 model.
                k: (
                    v.to(device=device, dtype=DTYPE)
                    if hasattr(v, "is_floating_point") and v.is_floating_point()
                    else v.to(device)
                    if hasattr(v, "to")
                    else v
                )
                for k, v in inputs.items()
            }
            with torch.no_grad():
                full_model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            done += 1
            if done % 8 == 0 or done == len(wav_paths):
                print(f"    calibrated {done}/{len(wav_paths)} audio samples")
    finally:
        for h in hooks:
            h.remove()

    missing = [n for n in groups if n not in accum]
    if missing:
        raise RuntimeError(
            f"AWQ calibration produced no activations for {len(missing)} norms "
            f"(e.g. {missing[0]}) — decoder layout changed?"
        )
    return {n: accum[n] / counts[n] for n in accum}


def _awq_search_alpha(
    act: "torch.Tensor",
    weights: "list[torch.Tensor]",
    alphas: "list[float]",
) -> "tuple[float, torch.Tensor]":
    """Pick the smoothing exponent minimizing activation-weighted INT8
    reconstruction error for this group (AWQ's salience objective, evaluated
    analytically against the same per-output-channel symmetric quantizer the
    export uses)."""
    a_norm = act / act.mean().clamp(min=1e-8)
    best_alpha, best_err, best_s = 0.0, None, torch.ones_like(act)
    for alpha in alphas:
        s = a_norm.pow(alpha).clamp(0.01, 100.0)
        err = 0.0
        for W in weights:
            ws = W * s
            scale = (ws.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
            rec = (ws / scale).round().clamp(-128, 127) * scale / s
            err += ((rec - W) * act).pow(2).sum().item()
        if best_err is None or err < best_err:
            best_alpha, best_err, best_s = alpha, err, s
    return best_alpha, best_s


def awq_smooth(
    full_model,
    processor,
    calib_dir: Path,
    calib_samples: "int | None",
    alpha: "float | None",
    device: str,
) -> None:
    """Calibrate on real audio and fold AWQ scales into the decoder weights.

    ``alpha=None`` grid-searches the exponent per group; a float pins it
    globally (0.0 disables smoothing). Output-equivalent in fp32; the fp16
    fold uses fp32 intermediates and scales clamped to [0.01, 100].
    """
    wavs = sorted(calib_dir.rglob("*.wav"))
    if calib_samples is not None:
        wavs = wavs[: max(1, calib_samples)]
    if not wavs:
        raise SystemExit(
            f"AWQ calibration: no .wav files under {calib_dir}. "
            "Pass --no-awq to skip smoothing or provide --calib-audio."
        )

    print(f"[ASR] AWQ smoothing: calibrating on {len(wavs)} audio files ({device})...")
    full_model.to(device)
    acts = _awq_collect_activations(full_model, processor, wavs, device)
    full_model.to("cpu")
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    text_model = full_model.thinker.model
    groups = _awq_groups(text_model)
    alphas = [i / 20.0 for i in range(21)] if alpha is None else [float(alpha)]

    chosen: list[float] = []
    for name, (norm, linears) in groups.items():
        act = acts[name]
        ws = [lin.weight.data.float() for lin in linears]
        a_star, s = _awq_search_alpha(act, ws, alphas)
        chosen.append(a_star)
        if a_star == 0.0:
            continue
        norm.weight.data = (norm.weight.data.float() / s).to(norm.weight.dtype)
        for lin in linears:
            lin.weight.data = (lin.weight.data.float() * s[None, :]).to(
                lin.weight.dtype
            )

    import collections

    hist = collections.Counter(f"{a:.2f}" for a in chosen)
    print(
        "[ASR] AWQ smoothing folded into "
        f"{sum(1 for a in chosen if a > 0)}/{len(chosen)} groups; "
        f"alpha histogram: {dict(sorted(hist.items()))}"
    )


# Output-channel count above which a projection is treated as a vocab head
# (lm_head): pinned to INT8 per-channel — never INT4 — as the most
# error-sensitive matmul in autoregressive decoding.
_VOCAB_HEAD_MIN_OUT = 32768


def _pack_matmulnbits_int4(W: "np.ndarray", block: int):
    """Pack W [K, N] (fp32) into MatMulNBits 4-bit blockwise layout.

    Symmetric around the op's implicit zero-point 8: q = round(W/s) + 8 in
    [0, 15], two nibbles per byte (low first). Scales are fp16-rounded before
    quantizing so the stored scale equals the dequant grid. Returns
    (packed [N, nb, block//2] uint8, scales [N*nb] fp16, frobenius_rel_err).
    """
    import numpy as np

    K, N = W.shape
    nb = K // block
    wt = W.T.reshape(N, nb, block)
    amax = np.abs(wt).max(-1, keepdims=True)
    # MSE clip search per block: rare outliers otherwise dictate the whole
    # block's grid (MinMax), wasting the 16 levels on values that occur once.
    best_scale = None
    best_err = None
    for ratio in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
        s = np.maximum(amax * (ratio / 7.0), 1e-8).astype(np.float16).astype(np.float32)
        q = np.clip(np.round(wt / s) + 8, 0, 15)
        err = (((q - 8.0) * s - wt) ** 2).sum(-1, keepdims=True)
        if best_err is None:
            best_scale, best_err = s, err
        else:
            better = err < best_err
            best_scale = np.where(better, s, best_scale)
            best_err = np.where(better, err, best_err)
    q = np.clip(np.round(wt / best_scale) + 8, 0, 15).astype(np.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    deq = (q.astype(np.float32) - 8.0) * best_scale
    rel = float(np.linalg.norm(deq - wt) / max(np.linalg.norm(wt), 1e-12))
    return packed, best_scale.reshape(N * nb).astype(np.float16), rel


def _pack_matmulnbits_int8(W: "np.ndarray", block: int):
    """Pack W [K, N] (fp32) into MatMulNBits 8-bit blockwise layout.

    Symmetric around the op's implicit zero-point 128: q = round(W/s) + 128
    in [0, 255], one byte per value, B is [N, K//block, block] uint8. Scales
    are fp16-rounded before quantizing so the stored scale equals the dequant
    grid. Per-block scales isolate rare outliers to their own block instead
    of letting them dictate an entire output channel's grid (the per-channel
    DequantizeLinear form). Returns (packed, scales [N*nb] fp16, rel_err).
    """
    import numpy as np

    K, N = W.shape
    nb = K // block
    wt = W.T.reshape(N, nb, block)
    amax = np.abs(wt).max(-1, keepdims=True)
    best_scale = None
    best_err = None
    for ratio in (1.0, 0.95, 0.9):
        s = (
            np.maximum(amax * (ratio / 127.0), 1e-8)
            .astype(np.float16)
            .astype(np.float32)
        )
        q = np.clip(np.round(wt / s) + 128, 0, 255)
        err = (((q - 128.0) * s - wt) ** 2).sum(-1, keepdims=True)
        if best_err is None:
            best_scale, best_err = s, err
        else:
            better = err < best_err
            best_scale = np.where(better, s, best_scale)
            best_err = np.where(better, err, best_err)
    q = np.clip(np.round(wt / best_scale) + 128, 0, 255).astype(np.uint8)
    deq = (q.astype(np.float32) - 128.0) * best_scale
    rel = float(np.linalg.norm(deq - wt) / max(np.linalg.norm(wt), 1e-12))
    return q, best_scale.reshape(N * nb).astype(np.float16), rel


def _quantize_matmul_weights_int8(
    model: onnx.ModelProto,
    max_out_channels: int | None = None,
    int4_threshold: "float | None" = None,
    int4_block: int = 64,
    select=None,
    int8_block: "int | None" = None,
) -> "tuple[int, int]":
    """Replace fp16 MatMul/Gemm weights with INT8 via DequantizeLinear.

    For each MatMul or Gemm with a constant 2-D weight (fp16), every output
    channel n gets its own scale, picked by MSE search over clip ratios:
      scale_n = argmin_r Σ_k (deq(quant(W[k,n]; r)) − W[k,n])²,
      r ∈ {1.0 … 0.7}, base scale = max|W[:,n]| / 127
    Then: op(x, W) → op(x, DequantizeLinear(W_i8, scales, axis=out_axis)).
    The output-channel axis follows the layout: 1 for MatMul / Gemm transB=0
    (W is [K, N]), 0 for Gemm transB=1 (W is [N, K]). DequantizeLinear
    outputs fp16 so the rest of the graph is unchanged. Per-channel scales +
    MSE-optimal clipping cut quantization error sharply versus per-tensor
    max scaling, at identical storage cost.

    Scales are rounded to fp16 before quantizing so the stored scales and
    the quantization grid agree exactly. Weights shared by several nodes are
    quantized once (provided all consumers agree on layout) and every
    consumer repointed to the same dequantized tensor; weights that also
    feed any other op type, and non-2-D weights, are left untouched.
    """
    import numpy as np
    import onnx.helper as oh
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}

    def _out_axis(n: onnx.NodeProto) -> int:
        if n.op_type == "Gemm":
            trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
            return 0 if trans_b else 1
        return 1

    # Weight name → consuming nodes, so shared weights are handled safely.
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp in inits:
                consumers.setdefault(inp, []).append(node)

    count = 0
    count4 = 0
    dq_for_weight: dict[str, str] = {}
    nbits_for_weight: dict[str, tuple] = {}

    for node in list(model.graph.node):
        if (
            node.op_type not in ("MatMul", "Gemm")
            or len(node.input) < 2
            or node.input[1] not in inits
        ):
            continue
        wt_name = node.input[1]

        if wt_name in dq_for_weight:
            node.input[1] = dq_for_weight[wt_name]
            count += 1
            continue
        if wt_name in nbits_for_weight:
            b_name, s_name, kk, nn, w_bits, w_block = nbits_for_weight[wt_name]
            act_in = node.input[0]
            del node.input[:]
            node.input.extend([act_in, b_name, s_name])
            node.op_type = "MatMulNBits"
            node.domain = "com.microsoft"
            node.attribute.extend(
                [
                    oh.make_attribute("K", kk),
                    oh.make_attribute("N", nn),
                    oh.make_attribute("bits", w_bits),
                    oh.make_attribute("block_size", w_block),
                ]
            )
            if w_bits == 4:
                count4 += 1
            else:
                count += 1
            continue
        wt_consumers = consumers.get(wt_name, [])
        if any(c.op_type not in ("MatMul", "Gemm") for c in wt_consumers):
            continue
        # The weight must be consumed as input[1] everywhere, with one layout.
        if any(list(c.input).index(wt_name) != 1 for c in wt_consumers):
            continue
        out_axis = _out_axis(node)
        if any(_out_axis(c) != out_axis for c in wt_consumers):
            continue

        init = inits[wt_name]
        W = numpy_helper.to_array(init).astype(np.float32)
        if W.ndim != 2:
            continue
        if select is not None and not select(wt_name, W, out_axis):
            continue
        # Blockwise 8-bit MatMulNBits: same byte count as per-channel INT8 DQ
        # but K//block scales per output channel instead of one, and the
        # weight ships in the packed layout ORT keeps resident instead of
        # relying on runtime DQ+MatMul fusion.
        if (
            int8_block is not None
            and node.op_type == "MatMul"
            and W.shape[0] % int8_block == 0
        ):
            packed, scales8, _rel = _pack_matmulnbits_int8(W, int8_block)
            b_name = f"{wt_name}_q8"
            s_name = f"{wt_name}_q8_scales"
            model.graph.initializer.append(numpy_helper.from_array(packed, b_name))
            model.graph.initializer.append(numpy_helper.from_array(scales8, s_name))
            kk, nn = W.shape
            act_in = node.input[0]
            del node.input[:]
            node.input.extend([act_in, b_name, s_name])
            node.op_type = "MatMulNBits"
            node.domain = "com.microsoft"
            node.attribute.extend(
                [
                    oh.make_attribute("K", kk),
                    oh.make_attribute("N", nn),
                    oh.make_attribute("bits", 8),
                    oh.make_attribute("block_size", int8_block),
                ]
            )
            nbits_for_weight[wt_name] = (b_name, s_name, kk, nn, 8, int8_block)
            for i, ini in enumerate(model.graph.initializer):
                if ini.name == wt_name:
                    model.graph.initializer.pop(i)
                    break
            count += 1
            continue
        # Sensitivity scan: prefer blockwise INT4 (emitted directly as
        # MatMulNBits — the packed form ORT keeps resident in VRAM) when the
        # post-AWQ reconstruction error stays under threshold. Vocab-sized
        # heads are pinned to INT8; Gemm weights use the INT8 DQ path.
        force_int8 = (
            max_out_channels is not None and W.shape[out_axis] > max_out_channels
        )
        if (
            int4_threshold is not None
            and not force_int8
            and node.op_type == "MatMul"
            and W.shape[0] % int4_block == 0
        ):
            packed, scales4, rel = _pack_matmulnbits_int4(W, int4_block)
            if rel <= int4_threshold:
                b_name = f"{wt_name}_q4"
                s_name = f"{wt_name}_q4_scales"
                model.graph.initializer.append(numpy_helper.from_array(packed, b_name))
                model.graph.initializer.append(numpy_helper.from_array(scales4, s_name))
                kk, nn = W.shape
                act_in = node.input[0]
                del node.input[:]
                node.input.extend([act_in, b_name, s_name])
                node.op_type = "MatMulNBits"
                node.domain = "com.microsoft"
                node.attribute.extend(
                    [
                        oh.make_attribute("K", kk),
                        oh.make_attribute("N", nn),
                        oh.make_attribute("bits", 4),
                        oh.make_attribute("block_size", int4_block),
                    ]
                )
                nbits_for_weight[wt_name] = (b_name, s_name, kk, nn, 4, int4_block)
                for i, ini in enumerate(model.graph.initializer):
                    if ini.name == wt_name:
                        model.graph.initializer.pop(i)
                        break
                count4 += 1
                continue

        red_axis = 1 - out_axis
        ch_max = np.maximum(np.abs(W).max(axis=red_axis), 1e-8)

        def _expand(s: "np.ndarray") -> "np.ndarray":
            return s[:, None] if out_axis == 0 else s[None, :]

        best_scale = None
        best_err = None
        for ratio in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
            s = (ch_max * (ratio / 127.0)).astype(np.float16).astype(np.float32)
            s = np.maximum(s, 1e-8)
            q = np.clip(np.round(W / _expand(s)), -128, 127)
            err = ((q * _expand(s) - W) ** 2).sum(axis=red_axis)
            if best_err is None:
                best_scale, best_err = s, err
            else:
                better = err < best_err
                best_scale = np.where(better, s, best_scale)
                best_err = np.where(better, err, best_err)
        W_i8 = np.clip(np.round(W / _expand(best_scale)), -128, 127).astype(np.int8)

        i8_name = f"{wt_name}_i8"
        scale_name = f"{wt_name}_scale"
        dq_out = f"{wt_name}_dq"

        i8_init = numpy_helper.from_array(W_i8, name=i8_name)
        scale_init = numpy_helper.from_array(
            best_scale.astype(np.float16), name=scale_name
        )
        model.graph.initializer.append(i8_init)
        model.graph.initializer.append(scale_init)

        dq_node = oh.make_node(
            "DequantizeLinear",
            inputs=[i8_name, scale_name],
            outputs=[dq_out],
            axis=out_axis,
        )

        # Insert DequantizeLinear before its first consumer.
        idx = list(model.graph.node).index(node)
        model.graph.node.insert(idx, dq_node)

        # Point MatMul at the dequantized output instead of raw weight.
        node.input[1] = dq_out
        dq_for_weight[wt_name] = dq_out

        # Remove old fp16 weight initializer.
        for i, ini in enumerate(model.graph.initializer):
            if ini.name == wt_name:
                model.graph.initializer.pop(i)
                break

        count += 1

    if nbits_for_weight and not any(
        o.domain == "com.microsoft" for o in model.opset_import
    ):
        model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    return count, count4


def _arch_from_decoder(path: Path) -> "tuple[int, int, int, int]":
    """Derive (num_layers, num_kv_heads, head_dim, hidden) from decoder I/O."""
    m = onnx.load(str(path), load_external_data=False)
    past = [i for i in m.graph.input if i.name.startswith("past_key_values.")]
    num_layers = len(past) // 2
    dims = past[0].type.tensor_type.shape.dim
    num_kv = dims[1].dim_value
    head_dim = dims[3].dim_value
    emb = next(i for i in m.graph.input if i.name == "inputs_embeds")
    hidden = emb.type.tensor_type.shape.dim[2].dim_value
    return num_layers, num_kv, head_dim, hidden


def _convert_int8_decoder_to_int4(
    path: Path,
    threshold: float,
    block: int,
    max_out_channels: "int | None",
    selection: str = "middle-ffn",
    quant_notes: "list[dict] | None" = None,
) -> int:
    """Requantize an already-INT8 decoder graph to blockwise INT4 in place.

    Reconstructs each weight from its DequantizeLinear pair and replaces DQ +
    MatMul with a packed MatMulNBits node. This materially reduces file size
    and ORT resident/prepacked VRAM for accepted weights. Vocab-sized heads stay
    INT8. ``selection`` controls the quality/size frontier:
      middle-ffn: only non-edge FFN-shaped weights (default, ASR-safe)
      all-ffn: every FFN-shaped weight passing the error threshold
      all: every MatMul weight passing the error threshold, except vocab head
    """
    import numpy as np
    import onnx.helper as oh
    from onnx import numpy_helper

    if selection not in {"middle-ffn", "all-ffn", "all"}:
        raise ValueError(f"unknown INT4 selection policy: {selection}")

    m = onnx.load(str(path))
    inits = {i.name: i for i in m.graph.initializer}
    consumers: dict[str, list] = {}
    for node in m.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    candidates = []
    for dq in [n for n in m.graph.node if n.op_type == "DequantizeLinear"]:
        if dq.input[0] not in inits:
            continue
        Wq = numpy_helper.to_array(inits[dq.input[0]])
        if Wq.dtype != np.int8 or Wq.ndim != 2:
            continue
        users = consumers.get(dq.output[0], [])
        if len(users) != 1 or users[0].op_type != "MatMul":
            continue
        axis = next((a.i for a in dq.attribute if a.name == "axis"), 1)
        if axis != 1:
            continue
        sc = numpy_helper.to_array(inits[dq.input[1]]).astype(np.float32)
        W = Wq.astype(np.float32) * sc[None, :]
        K, N = W.shape
        if max_out_channels is not None and N > max_out_channels:
            continue
        if K % block:
            continue
        candidates.append((dq, users[0], W, K, N))

    ffn = [c for c in candidates if max(c[3], c[4]) >= 3 * min(c[3], c[4])]
    if selection == "all":
        selected = {id(c[0]) for c in candidates}
    elif selection == "all-ffn":
        selected = {id(c[0]) for c in ffn}
    else:
        per_layer = 3  # gate, up, down
        selected = (
            {id(c[0]) for c in ffn[per_layer:-per_layer]}
            if len(ffn) > 2 * per_layer
            else set()
        )

    drop_nodes = []
    drop_inits: set[str] = set()
    n4 = 0
    rejected = 0
    considered = 0
    for dq, mm, W, K, N in candidates:
        if id(dq) not in selected:
            continue
        considered += 1
        packed, scales4, rel = _pack_matmulnbits_int4(W, block)
        if rel > threshold:
            rejected += 1
            continue

        base = dq.input[0]
        b_name, s_name = f"{base}_q4", f"{base}_q4s"
        m.graph.initializer.append(numpy_helper.from_array(packed, b_name))
        m.graph.initializer.append(numpy_helper.from_array(scales4, s_name))
        act_in = mm.input[0]
        del mm.input[:]
        mm.input.extend([act_in, b_name, s_name])
        mm.op_type = "MatMulNBits"
        mm.domain = "com.microsoft"
        del mm.attribute[:]
        mm.attribute.extend(
            [
                oh.make_attribute("K", K),
                oh.make_attribute("N", N),
                oh.make_attribute("bits", 4),
                oh.make_attribute("block_size", block),
            ]
        )
        drop_nodes.append(dq)
        drop_inits.add(dq.input[0])
        drop_inits.add(dq.input[1])
        n4 += 1
        if quant_notes is not None:
            quant_notes.append(
                {
                    "graph": path.name,
                    "op": "MatMulNBits",
                    "source_initializer": base,
                    "shape": [int(K), int(N)],
                    "block": int(block),
                    "relative_error": float(rel),
                    "selection": selection,
                }
            )

    for n in drop_nodes:
        m.graph.node.remove(n)
    for init in [i for i in m.graph.initializer if i.name in drop_inits]:
        m.graph.initializer.remove(init)
    if n4 and not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))

    _save_consolidated(m, path)
    print(
        f"    {path.name}: {n4}/{considered} selected weights INT8→INT4 "
        f"(selection={selection}, rejected={rejected}, block={block}, threshold={threshold})"
    )
    return n4


def _quantize_gather_embeddings_int8(model: onnx.ModelProto) -> int:
    """Quantize large fp16 Gather tables (token embeddings) to row-wise INT8.

    Gather(E, ids) becomes
        Cast(Gather(E_i8, ids), fp16) * Unsqueeze(Gather(scales, ids), -1)
    so only the gathered rows are dequantized at runtime (negligible compute)
    while the dominant initializer halves in size. Row-wise scales preserve
    per-token accuracy; a per-tensor scale would let outlier rows dominate.
    """
    import numpy as np
    import onnx.helper as oh
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    use_count: dict[str, int] = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp in inits:
                use_count[inp] = use_count.get(inp, 0) + 1

    count = 0
    for node in list(model.graph.node):
        if node.op_type != "Gather" or node.input[0] not in inits:
            continue
        base = node.input[0]
        if use_count.get(base, 0) != 1:
            continue
        E = numpy_helper.to_array(inits[base])
        if E.ndim != 2 or E.size < (1 << 20):
            continue

        Ef = E.astype(np.float32)
        scales = np.abs(Ef).max(axis=1) / 127.0
        scales = np.maximum(scales.astype(np.float16).astype(np.float32), 1e-8)
        E_i8 = np.clip(np.round(Ef / scales[:, None]), -128, 127).astype(np.int8)

        ids_name = node.input[1]
        out_name = node.output[0]
        model.graph.initializer.append(numpy_helper.from_array(E_i8, f"{base}_i8"))
        model.graph.initializer.append(
            numpy_helper.from_array(scales.astype(np.float16), f"{base}_scales")
        )
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([-1], dtype=np.int64), f"{base}_unsq_axes")
        )

        node.input[0] = f"{base}_i8"
        node.output[0] = f"{base}_rows_i8"
        idx = list(model.graph.node).index(node)
        new_nodes = [
            oh.make_node(
                "Cast",
                [f"{base}_rows_i8"],
                [f"{base}_rows_f16"],
                to=onnx.TensorProto.FLOAT16,
            ),
            oh.make_node(
                "Gather", [f"{base}_scales", ids_name], [f"{base}_row_scales"]
            ),
            oh.make_node(
                "Unsqueeze",
                [f"{base}_row_scales", f"{base}_unsq_axes"],
                [f"{base}_row_scales_u"],
            ),
            oh.make_node(
                "Mul", [f"{base}_rows_f16", f"{base}_row_scales_u"], [out_name]
            ),
        ]
        for off, nn in enumerate(new_nodes, 1):
            model.graph.node.insert(idx + off, nn)

        for i, ini in enumerate(model.graph.initializer):
            if ini.name == base:
                model.graph.initializer.pop(i)
                break
        count += 1

    return count


def _quantize_graph_file_int8(
    path: Path,
    gather: bool = False,
    max_out_channels: "int | None" = None,
    int4_threshold: "float | None" = None,
    int4_block: int = 64,
) -> "tuple[int, int]":
    """Load an exported graph, quantize its big tensors (INT8/INT4), save back."""
    print(f"  Quantizing {path.name}...")
    m = onnx.load(str(path))
    if gather:
        n8, n4 = _quantize_gather_embeddings_int8(m), 0
    else:
        target_opset = 19
        cur = next((o.version for o in m.opset_import if o.domain == ""), 17)
        if cur < target_opset:
            m = onnx.version_converter.convert_version(m, target_opset)
        n8, n4 = _quantize_matmul_weights_int8(
            m,
            max_out_channels=max_out_channels,
            int4_threshold=int4_threshold,
            int4_block=int4_block,
        )
    _save_consolidated(m, path)
    print(f"    {path.name}: {n8} tensors INT8, {n4} INT4")
    return n8, n4


# Decoder weight classes eligible for INT8 per profile. down_proj is NEVER
# quantized: it eats heavy-tailed SwiGLU products with no AWQ-smoothable norm,
# and INT8 there produced garbage real-world output twice (full-decoder INT8
# build, and a compact build whose only delta from the known-good balanced
# was 28 down_proj weights — 96% probe agreement, '!!!!!!' in dictation).
# The probe cannot catch this failure mode; only real transcription can.
_PROFILE_INT8_CLASSES = {
    "balanced": {"lm_head", "gate_up"},
    "compact": {"lm_head", "gate_up", "attention"},
}


def _classify_decoder_weight(k: int, n: int) -> str:
    """Classify a decoder MatMul weight [K, N] by shape.

    The legacy exporter renames Linear weights to anonymous onnx::MatMul_*
    initializers, so module names are unavailable; shape is unambiguous:
    vocab head has N >= 32768, FFN matrices have >= 3:1 aspect (gate/up grow
    K->N, down shrinks K->N), attention projections are square-ish.
    """
    if n >= _VOCAB_HEAD_MIN_OUT:
        return "lm_head"
    if n >= 3 * k:
        return "gate_up"
    if k >= 3 * n:
        return "down_proj"
    return "attention"


def _quantize_decoder_targeted(
    path: Path, profile: str, quant_notes: "list[dict] | None" = None
) -> dict:
    """Quantize only the profile's allowed decoder weight classes to INT8.

    Per-output-channel symmetric weight-only INT8 via DequantizeLinear;
    activations stay FP16. Returns {"int8": total, "classes": {cls: count}}.
    """
    allowed = _PROFILE_INT8_CLASSES[profile]
    decisions: dict[str, int] = {}

    def select(name, W, out_axis):
        if out_axis == 0:
            k, n = int(W.shape[1]), int(W.shape[0])
        else:
            k, n = int(W.shape[0]), int(W.shape[1])
        cls = _classify_decoder_weight(k, n)
        if cls not in allowed:
            return False
        decisions[cls] = decisions.get(cls, 0) + 1
        return True

    print(f"  Quantizing {path.name} (targeted: {', '.join(sorted(allowed))})...")
    m = onnx.load(str(path))
    target_opset = 19
    cur = next((o.version for o in m.opset_import if o.domain == ""), 17)
    if cur < target_opset:
        m = onnx.version_converter.convert_version(m, target_opset)
    # Ship per-channel DQ, not pre-emitted blockwise MatMulNBits(bits=8):
    # measured 2026-06: blockwise cut probe drift 31% (mean 0.283->0.194) at
    # +6.7MB and CUDA decode parity (4.94 vs 4.96 ms/step), but the CPU
    # MatMulNBits-8 kernel is 15-70x slower per node at batch-1 (decode step
    # 19ms -> 218ms), and CPU must stay viable as a fallback EP. ORT's CUDA
    # optimizer fuses DQ+MatMul into packed nbits8 at session load anyway.
    n8, _ = _quantize_matmul_weights_int8(m, select=select)
    _save_consolidated(m, path)
    print(f"    {path.name}: {n8} tensors INT8 by class {decisions}")
    if quant_notes is not None:
        quant_notes.append(
            {"graph": path.name, "int8_by_class": dict(decisions), "int8_total": n8}
        )
    return {"int8": n8, "classes": decisions}


# ── ASR Wrappers ──────────────────────────────────────────────────────────────


class EncoderWrapper(nn.Module):
    """ONNX-friendly audio encoder wrapper for batch=1 unpadded mel input.

    Mirrors Qwen3-ASR's chunked audio-tower path closely enough to preserve the
    exported encoder geometry. The upstream module uses Python list splits that
    do not survive ONNX export with dynamic mel lengths, so this wrapper keeps
    the same math but reformulates chunking/masking with tensor ops.
    """

    def __init__(self, thinker):
        super().__init__()
        at = thinker.audio_tower
        self.conv2d1 = at.conv2d1
        self.conv2d2 = at.conv2d2
        self.conv2d3 = at.conv2d3
        self.conv_out = at.conv_out
        self.register_buffer("pos_embed", at.positional_embedding.positional_embedding)
        self.layers = at.layers
        self.ln_post = at.ln_post
        self.proj1 = at.proj1
        self.act = at.act
        self.proj2 = at.proj2
        self.n_window = int(at.n_window)
        self.n_window_infer = int(at.n_window_infer)
        self.chunk_size = int(at.n_window * 2)
        self.chunk_aftercnn = self._feat_extract_output_lengths(
            torch.tensor(self.chunk_size, dtype=torch.long)
        ).item()
        self.window_aftercnn = self.chunk_aftercnn * (
            self.n_window_infer // self.chunk_size
        )
        self.max_encoder_frames = 6000
        self.max_aftercnn = self._feat_extract_output_lengths(
            torch.tensor(self.max_encoder_frames, dtype=torch.long)
        ).item()
        self.max_attention_windows = (
            self.max_aftercnn + self.window_aftercnn - 1
        ) // self.window_aftercnn

    def _feat_extract_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = (lengths - 1) // 2 + 1
        lengths = (lengths - 1) // 2 + 1
        lengths = (lengths - 1) // 2 + 1
        return lengths

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Runtime only uses a single unpadded utterance per encoder invocation.
        # Keep export faithful for that path instead of pretending to support
        # arbitrary padded batches.
        if mel.shape[0] != 1:
            raise ValueError("EncoderWrapper exports batch=1 only")

        total_frames = mel.shape[-1]
        feature_len = torch.ones(1, dtype=torch.long, device=mel.device) * total_frames
        chunk_size = self.chunk_size
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        padded_frames = num_chunks * chunk_size
        pad_frames = padded_frames - total_frames

        padded = torch.nn.functional.pad(mel, (0, pad_frames))
        chunks = (
            padded.reshape(1, mel.shape[1], num_chunks, chunk_size)
            .permute(2, 0, 1, 3)
            .squeeze(1)
        )

        chunk_lengths = torch.full(
            (num_chunks,),
            chunk_size,
            dtype=torch.long,
            device=mel.device,
        )
        chunk_lengths[-1] = torch.remainder(feature_len[0] - 1, chunk_size) + 1

        x = chunks.unsqueeze(1)  # [num_chunks, 1, 128, chunk_size]
        x = torch.nn.functional.gelu(self.conv2d1(x))
        x = torch.nn.functional.gelu(self.conv2d2(x))
        x = torch.nn.functional.gelu(self.conv2d3(x))
        b, c, f, t = x.size()
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        x = x + self.pos_embed[:t, :].unsqueeze(0).to(x.dtype)

        chunk_aftercnn_lengths = self._feat_extract_output_lengths(chunk_lengths)
        valid_mask = torch.arange(t, device=mel.device).unsqueeze(
            0
        ) < chunk_aftercnn_lengths.unsqueeze(1)
        hidden = x[valid_mask]

        aftercnn_len = self._feat_extract_output_lengths(feature_len)[0]
        step_ends = (
            torch.arange(
                1, self.max_attention_windows + 1, device=mel.device, dtype=torch.long
            )
            * self.window_aftercnn
        )
        step_ends = torch.minimum(
            step_ends,
            aftercnn_len.expand(self.max_attention_windows),
        )
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, device=mel.device, dtype=torch.long),
                step_ends,
            ]
        ).to(torch.int32)

        for layer in self.layers:
            hidden = layer(hidden, cu_seqlens)[0]
        hidden = self.ln_post(hidden)
        hidden = self.proj1(hidden)
        hidden = self.act(hidden)
        hidden = self.proj2(hidden)
        return hidden.unsqueeze(0)  # [1, S, D]


class EmbedTokensWrapper(nn.Module):
    def __init__(self, thinker):
        super().__init__()
        self.embed_tokens = thinker.model.embed_tokens

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class DecoderWrapper(nn.Module):
    """Wraps the Qwen3 decoder with KV cache for ONNX export.

    Accepts an explicit attention_mask so the graph supports batched inference
    (B>1 with padding). For B=1 single-item inference, pass all-ones mask.
    """

    def __init__(self, thinker, num_layers: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.text_model = thinker.model
        self.lm_head = thinker.lm_head
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def _build_causal_mask_4d(
        self,
        attention_mask: torch.Tensor,  # [B, total_len]
        seq_len: int,
        total_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a 4D causal attention mask from a 2D padding mask.

        Constructs the mask directly with triu + padding to avoid the
        non-traceable ``create_causal_mask`` in transformers ≥5.x.
        """
        past_len = total_len - seq_len
        # Upper-triangular causal mask: -inf above diagonal (relative to past_len)
        causal = torch.triu(
            torch.full(
                (seq_len, total_len),
                torch.finfo(dtype).min,
                dtype=dtype,
                device=attention_mask.device,
            ),
            diagonal=past_len + 1,
        )
        causal_4d = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]
        # Apply 2D padding mask: positions with 0 → -inf
        padding = (1.0 - attention_mask.to(dtype)).unsqueeze(1).unsqueeze(2)
        padding = padding * torch.finfo(dtype).min
        return causal_4d + padding  # [B, 1, seq_len, total_len]

    @torch.no_grad()
    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [B, L, H]
        position_ids: torch.Tensor,  # [3, B, L] (mRoPE)
        attention_mask: torch.Tensor,  # [B, past_len + L]
        past_key_values_flat: list[torch.Tensor],
    ):
        from transformers import DynamicCache

        past = DynamicCache()
        for i in range(self.num_layers):
            k = past_key_values_flat[i * 2]
            v = past_key_values_flat[i * 2 + 1]
            past.update(k, v, i)

        seq_len = inputs_embeds.shape[1]
        total_len = attention_mask.shape[1]
        # Build 4D causal mask ourselves to bypass transformers' non-traceable
        # create_causal_mask. Passing a 4D mask makes the model skip its own.
        causal_mask_4d = self._build_causal_mask_4d(
            attention_mask,
            seq_len,
            total_len,
            inputs_embeds.dtype,
        )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=causal_mask_4d,
            past_key_values=past,
            use_cache=True,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        new_cache = outputs.past_key_values
        flat_present: list[torch.Tensor] = []
        for i in range(self.num_layers):
            k, v = new_cache[i]
            flat_present.append(k)
            flat_present.append(v)

        return (logits, flat_present)


# ── ASR Export ────────────────────────────────────────────────────────────────


def export_asr(
    output_dir: Path,
    model_id: str = "Qwen/Qwen3-ASR-0.6B",
    revision: "str | None" = None,
    quantize: bool = True,
    calib_samples: int | None = None,
    alpha: "float | None" = None,
    calib_audio_dir: "Path | None" = None,
    int4_threshold: "float | None" = 0.035,
    int4_block: int = 32,
    quant_policy: str = "balanced",
    no_awq: bool = False,
    strict_quant: bool = True,
    allow_quality_drift: bool = False,
    min_quant_agreement: "float | None" = None,
    max_quant_mean_delta: "float | None" = None,
) -> None:
    """Export Qwen3-ASR-0.6B to encoder/embed/decoder ONNX graphs.

    Policy frontier (encoder stays FP16 everywhere; token embeddings are
    row-wise INT8 in every quantized profile):
      fp16: no quantization anywhere.
      safe: decoder fully FP16. Guaranteed-quality fallback.
      balanced (default): decoder targeted INT8 — lm_head + AWQ-smoothed
          gate/up per-output-channel; attention and down_proj stay FP16.
      compact: balanced plus attention q/k/v/o INT8 and encoder INT8;
          down_proj stays FP16 (INT8 there breaks real decoding); middle
          FFN INT4 is opportunistic under a strict threshold.

    Full-decoder INT8 and broad INT4 are intentionally absent: both were
    built and measurably degraded real dictation. Each quantized profile is
    hard-gated against the torch reference probe and aborts toward its
    fallback (compact -> balanced -> safe) on drift.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not quantize:
        quant_policy = "fp16"
    valid_policies = {"fp16", "safe", "balanced", "compact"}
    if quant_policy not in valid_policies:
        raise ValueError(
            f"unknown quant_policy={quant_policy!r}; expected one of {sorted(valid_policies)}"
        )
    if int4_block < 2 or int4_block % 2 != 0:
        raise ValueError(f"int4_block must be an even integer >= 2, got {int4_block}")
    if min_quant_agreement is not None and not 0.0 <= min_quant_agreement <= 1.0:
        raise ValueError(
            f"min_quant_agreement must be in [0, 1], got {min_quant_agreement}"
        )
    if max_quant_mean_delta is not None and max_quant_mean_delta < 0.0:
        raise ValueError(
            f"max_quant_mean_delta must be >= 0, got {max_quant_mean_delta}"
        )

    # Profile gate thresholds; explicit CLI overrides win.
    gate_agreement, gate_mean = _PROFILE_GATES.get(quant_policy, (1.0, 0.25))
    if min_quant_agreement is not None:
        gate_agreement = min_quant_agreement
    if max_quant_mean_delta is not None:
        gate_mean = max_quant_mean_delta

    decoder_int8_classes = sorted(_PROFILE_INT8_CLASSES.get(quant_policy, set()))
    awq_on = quant_policy in ("balanced", "compact") and not no_awq

    print(f"[ASR] Exporting {model_id} → {output_dir}")
    print(
        f"[ASR] Policy: {quant_policy}, revision={revision or 'default'}, "
        f"decoder_int8={decoder_int8_classes or 'none'}, "
        f"int4_threshold={int4_threshold}, int4_block={int4_block}, "
        f"gate=({gate_agreement:.2f}, {gate_mean:.2f}), "
        f"awq={'on' if awq_on else 'off'}"
    )

    # Seeded so dummy trace inputs and verification are reproducible.
    torch.manual_seed(0)

    from transformers import AutoProcessor

    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
    except ImportError as e:
        raise SystemExit(
            f"qwen_asr is required for ASR export but could not be imported: {e}\n"
            "Install it in this environment: uv add qwen-asr"
        ) from e

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision or "main",
        trust_remote_code=True,
        dtype=DTYPE,
        attn_implementation="eager",
    )
    full_model.eval()

    verify_prompt_ids = _build_verify_ids(processor)
    print(f"[ASR] Verification prompt: {len(verify_prompt_ids)} tokens")

    thinker = full_model.thinker
    text_config = thinker.config.text_config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = getattr(
        text_config, "num_key_value_heads", text_config.num_attention_heads
    )
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    hidden_size = text_config.hidden_size

    print(
        f"[ASR] Architecture: {num_layers}L, {hidden_size}H, "
        f"{num_kv_heads}KV, {head_dim}D, dtype={DTYPE}"
    )

    # AWQ smoothing folds gamma->W scales for the {q,k,v} and {gate,up}
    # groups only. Both quantized profiles target gate/up, so smoothing
    # directly conditions exactly the weights that drop to INT8; lm_head and
    # down_proj sit outside the smoothing groups regardless.
    if awq_on:
        if calib_audio_dir is None:
            calib_audio_dir = Path(__file__).resolve().parent.parent / "fixtures"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        awq_smooth(full_model, processor, calib_audio_dir, calib_samples, alpha, device)

    # ── 1. Encoder ───────────────────────────────────────────────────────
    print("[ASR]   Exporting qwen3-asr-encoder.onnx ...")
    enc = EncoderWrapper(thinker)
    enc.eval()
    # Use a multi-chunk dummy (500 frames = 3 chunks of 200) so torch.export
    # traces the chunking/reshape/windowing logic with symbolic shapes rather
    # than baking in num_chunks=1 from a single-chunk dummy.
    dummy_mel = torch.randn(1, 128, 500, dtype=DTYPE)
    mel_dim = torch.export.Dim("mel_frames", min=2, max=6000)

    torch.onnx.export(
        enc,
        (dummy_mel,),
        str(output_dir / "qwen3-asr-encoder.onnx"),
        input_names=["mel"],
        output_names=["features"],
        dynamic_shapes=({2: mel_dim},),
    )
    print("[ASR]   qwen3-asr-encoder.onnx exported")

    # ── 2. Embed tokens ──────────────────────────────────────────────────
    print("[ASR]   Exporting qwen3-asr-embed-tokens.onnx ...")
    emb = EmbedTokensWrapper(thinker)
    emb.eval()
    dummy_ids = torch.tensor([[151644, 8948, 198]], dtype=torch.long)
    seq_dim = torch.export.Dim("seq_len", min=1, max=4096)

    torch.onnx.export(
        emb,
        (dummy_ids,),
        str(output_dir / "qwen3-asr-embed-tokens.onnx"),
        input_names=["input_ids"],
        output_names=["embeds"],
        dynamic_shapes=({1: seq_dim},),
    )
    print("[ASR]   qwen3-asr-embed-tokens.onnx exported")

    # ── 3. Decoder (dynamic batch + attention_mask for B>1 support) ────
    print("[ASR]   Exporting qwen3-asr-decoder.onnx ...")
    dec = DecoderWrapper(thinker, num_layers, num_kv_heads, head_dim)
    dec.eval()

    batch_size = 1
    seq_len = 10
    past_len = 0
    dummy_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=DTYPE)
    dummy_position_ids = (
        torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
    )
    # attention_mask covers full sequence (past + current)
    dummy_attn_mask = torch.ones(batch_size, past_len + seq_len, dtype=torch.long)
    dummy_past = []
    for _ in range(num_layers):
        dummy_past.append(
            torch.zeros(batch_size, num_kv_heads, past_len, head_dim, dtype=DTYPE)
        )
        dummy_past.append(
            torch.zeros(batch_size, num_kv_heads, past_len, head_dim, dtype=DTYPE)
        )

    input_names = ["inputs_embeds", "position_ids", "attention_mask"]
    output_names = ["logits"]
    for i in range(num_layers):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present_key_values.{i}.key")
        output_names.append(f"present_key_values.{i}.value")
    # NOTE: Qwen3 decoder + DynamicCache is incompatible with torch.export
    # (dynamo path). Use legacy opset export with dynamic_axes instead.
    dynamic_axes: dict[str, dict[int, str]] = {
        "inputs_embeds": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "attention_mask": {1: "attn_len"},
        "logits": {1: "seq_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_values.{i}.key"] = {2: "past_seq_len"}
        dynamic_axes[f"past_key_values.{i}.value"] = {2: "past_seq_len"}
        dynamic_axes[f"present_key_values.{i}.key"] = {2: "total_seq_len"}
        dynamic_axes[f"present_key_values.{i}.value"] = {2: "total_seq_len"}

    torch.onnx.export(
        dec,
        (dummy_embeds, dummy_position_ids, dummy_attn_mask, dummy_past),
        str(output_dir / "qwen3-asr-decoder.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    print("[ASR]   qwen3-asr-decoder.onnx exported")

    # Torch reference logits for the verification prompt, captured before the
    # torch model is released.
    verify_ids = torch.tensor([verify_prompt_ids], dtype=torch.long)
    with torch.no_grad():
        v_embeds = emb(verify_ids)
        v_len = verify_ids.shape[1]
        v_pos = torch.arange(v_len).reshape(1, 1, v_len).expand(3, 1, v_len)
        v_mask = torch.ones(1, v_len, dtype=torch.long)
        v_past = [
            torch.zeros(1, num_kv_heads, 0, head_dim, dtype=DTYPE)
            for _ in range(2 * num_layers)
        ]
        ref_logits = dec(v_embeds, v_pos, v_mask, v_past)[0].float().numpy()

    # Release the torch model: quantization and verification reload the
    # graphs from disk and don't need ~1.2 GB of weights resident.
    del dec, emb, enc, thinker, full_model
    import gc

    gc.collect()

    # ── 4. Consolidate ───────────────────────────────────────────────────
    for name in [
        "qwen3-asr-encoder.onnx",
        "qwen3-asr-embed-tokens.onnx",
        "qwen3-asr-decoder.onnx",
    ]:
        _consolidate_external_data(output_dir, name)

    # ── 4.5 Verify unquantized export against the torch reference ───────
    verify_asr_export(
        output_dir,
        num_layers,
        num_kv_heads,
        head_dim,
        hidden_size,
        ref_logits=ref_logits,
        label="export",
        verify_ids=verify_prompt_ids,
    )

    # ── 5. Quantize ────────────────────────────────────────────────────
    # Encoder stays FP16 except in compact: it runs once per utterance (not
    # in the decode loop where errors compound), and encoder INT8 was part
    # of every fixture-validated earlier build.
    quant_notes: list[dict] = []
    if quant_policy == "fp16":
        print("[ASR] Skipping quantization; all graphs keep FP16 weights")
    else:
        # Embeddings: row-wise INT8 Gather is the proven-safe baseline (the
        # safe profile measures 100% probe agreement with only this applied).
        n8, _ = _quantize_graph_file_int8(
            output_dir / "qwen3-asr-embed-tokens.onnx", gather=True
        )
        _require_quantized("embed-tokens row-wise INT8", n8, strict=strict_quant)

        if quant_policy in ("balanced", "compact"):
            res = _quantize_decoder_targeted(
                output_dir / "qwen3-asr-decoder.onnx", quant_policy, quant_notes
            )
            _require_quantized(
                "decoder targeted INT8", res["int8"], strict=strict_quant
            )

        if quant_policy == "compact":
            n8, _ = _quantize_graph_file_int8(output_dir / "qwen3-asr-encoder.onnx")
            _require_quantized("encoder INT8", n8, strict=strict_quant)

        if quant_policy == "compact" and int4_threshold is not None:
            # Opportunistic: middle-layer FFN weights drop to INT4 only when
            # reconstruction error clears the strict threshold. Measured
            # block-32 errors are ~0.10-0.13, so zero acceptances is the
            # expected outcome until the INT4 quantizer improves — not a
            # failure, hence no _require_quantized here.
            _convert_int8_decoder_to_int4(
                output_dir / "qwen3-asr-decoder.onnx",
                threshold=int4_threshold,
                block=int4_block,
                max_out_channels=_VOCAB_HEAD_MIN_OUT,
                selection="middle-ffn",
                quant_notes=quant_notes,
            )

    # ── 5.5 Verify quantized graphs still run and stay finite ───────────
    if quant_policy != "fp16":
        quant_stats = verify_asr_export(
            output_dir,
            num_layers,
            num_kv_heads,
            head_dim,
            hidden_size,
            ref_logits=ref_logits,
            label="quantized",
            verify_ids=verify_prompt_ids,
        )
        _check_quant_quality(
            quant_stats,
            profile=quant_policy,
            allow_quality_drift=allow_quality_drift,
            min_agreement=gate_agreement,
            max_mean_delta=gate_mean,
        )

    # ── 6. Manifest + cleanup ──────────────────────────────────────────
    _write_export_manifest(
        output_dir,
        model_id=model_id,
        revision=revision,
        quant_policy=quant_policy,
        decoder_int8_classes=decoder_int8_classes,
        int4_threshold=int4_threshold,
        int4_block=int4_block,
        strict_quant=strict_quant,
        allow_quality_drift=allow_quality_drift,
        min_quant_agreement=gate_agreement,
        max_quant_mean_delta=gate_mean,
        verify_prompt_tokens=len(verify_prompt_ids),
        quant_notes=quant_notes,
    )
    _cleanup_temp_artifacts(output_dir)

    processor.save_pretrained(str(output_dir))

    print(f"[ASR] Export complete: {output_dir}")
    _list_files(output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen3 models to ONNX format")
    parser.add_argument("--asr", action="store_true", help="Export Qwen3-ASR-0.6B")
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization entirely (all graphs keep FP16 weights)",
    )
    parser.add_argument(
        "--quant-policy",
        choices=["fp16", "safe", "balanced", "compact"],
        default="balanced",
        help="Compression policy. balanced (default) = targeted decoder INT8 "
        "(lm_head + AWQ-smoothed gate/up only) + INT8 embeddings; safe = "
        "decoder FP16 fallback; compact = balanced + attention INT8 + "
        "encoder INT8 (down_proj always FP16 — INT8 there breaks decoding).",
    )
    parser.add_argument(
        "--no-awq",
        action="store_true",
        help="Skip AWQ smoothing before targeted decoder quantization. "
        "Faster export, lower INT8 quality on gate/up.",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=None,
        help="Max calibration audio files for AWQ (default: all in the dir)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Pin the AWQ smoothing exponent globally "
        "(default: per-group grid search; 0 disables smoothing)",
    )
    parser.add_argument(
        "--calib-audio",
        type=Path,
        default=None,
        help="Directory of .wav calibration audio (default: fixtures/)",
    )
    parser.add_argument(
        "--int4-threshold",
        type=float,
        default=0.035,
        help="Reconstruction relative-error ceiling for opportunistic "
        "middle-FFN INT4 in compact; <=0 disables INT4. Measured block-32 "
        "errors are ~0.10-0.13, so the default accepts nothing until the "
        "INT4 quantizer improves — zero acceptances is a valid outcome.",
    )
    parser.add_argument(
        "--int4-block",
        type=int,
        default=32,
        help="INT4 quantization block size along K (default: 32)",
    )
    parser.add_argument(
        "--int4-requant",
        action="store_true",
        help="Requantize the INT8 decoder in --output-dir to blockwise INT4 "
        "in place (no re-export), then re-run the CPU smoke test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: models/<model-name>-onnx)",
    )
    parser.add_argument(
        "--asr-model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="HuggingFace model ID for ASR",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Pinned HuggingFace model/code revision or commit SHA for reproducible export",
    )
    parser.add_argument(
        "--strict-quant",
        dest="strict_quant",
        action="store_true",
        default=True,
        help="Fail if a requested quantization stage matches zero tensors (default)",
    )
    parser.add_argument(
        "--allow-quant-misses",
        dest="strict_quant",
        action="store_false",
        help="Warn instead of failing when a requested quantization stage matches zero tensors",
    )
    parser.add_argument(
        "--allow-quality-drift",
        action="store_true",
        help="Continue with a warning when the quantized probe fails the "
        "profile quality gate instead of aborting toward the fallback profile",
    )
    parser.add_argument(
        "--min-quant-agreement",
        type=float,
        default=None,
        help="Override the profile's top-1 probe agreement gate "
        "(default: safe 1.0, balanced/compact 0.9)",
    )
    parser.add_argument(
        "--max-quant-mean-delta",
        type=float,
        default=None,
        help="Override the profile's mean |Δlogit| gate "
        "(default: safe/balanced 0.25, compact 0.35)",
    )
    args = parser.parse_args()

    if args.int4_block < 2 or args.int4_block % 2 != 0:
        parser.error("--int4-block must be an even integer >= 2")
    if args.min_quant_agreement is not None and not (
        0.0 <= args.min_quant_agreement <= 1.0
    ):
        parser.error("--min-quant-agreement must be in [0, 1]")
    if args.max_quant_mean_delta is not None and args.max_quant_mean_delta < 0.0:
        parser.error("--max-quant-mean-delta must be >= 0")

    if args.int4_requant:
        out = args.output_dir or Path("models/qwen3-asr-0.6b-onnx")
        dec = out / "qwen3-asr-decoder.onnx"
        if not dec.exists():
            parser.error(f"--int4-requant: {dec} not found")
        _convert_int8_decoder_to_int4(
            dec,
            threshold=args.int4_threshold,
            block=args.int4_block,
            max_out_channels=_VOCAB_HEAD_MIN_OUT,
            selection="middle-ffn",
        )
        num_layers, num_kv, head_dim, hidden = _arch_from_decoder(dec)
        verify_asr_export(out, num_layers, num_kv, head_dim, hidden, label="int4")
        _list_files(out)
        return

    if not args.asr:
        parser.error("Specify --asr")
    if args.calib_samples is not None and args.calib_samples < 1:
        parser.error("--calib-samples must be >= 1")

    out = args.output_dir or Path("models/qwen3-asr-0.6b-onnx")
    export_asr(
        out,
        args.asr_model,
        revision=args.revision,
        quantize=not args.no_quantize,
        calib_samples=args.calib_samples,
        alpha=args.alpha,
        calib_audio_dir=args.calib_audio,
        int4_threshold=args.int4_threshold if args.int4_threshold > 0 else None,
        int4_block=args.int4_block,
        quant_policy=args.quant_policy,
        no_awq=args.no_awq,
        strict_quant=args.strict_quant,
        allow_quality_drift=args.allow_quality_drift,
        min_quant_agreement=args.min_quant_agreement,
        max_quant_mean_delta=args.max_quant_mean_delta,
    )


if __name__ == "__main__":
    main()
