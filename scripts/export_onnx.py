#!/usr/bin/env python3
"""Custom ONNX exporter for Qwen3-0.6B and Qwen3-ASR-0.6B.

Bypasses optimum-cli entirely. Uses torch.onnx.export with manual wrappers
that correctly handle Qwen3's architecture (GQA, q_norm/k_norm, RoPE, SwiGLU,
tied embeddings) and produce clean FP16 graphs with proper KV cache I/O.

Usage:
    python scripts/export_onnx.py --llm --output-dir models/qwen3-0.6b-onnx
    python scripts/export_onnx.py --asr --output-dir models/qwen3-asr-0.6b-onnx
    python scripts/export_onnx.py --llm --asr
    python scripts/export_onnx.py --llm --no-quantize

Requirements:
    pip install torch transformers onnx onnxruntime
"""
from __future__ import annotations

import argparse
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
    """Save ONNX model with all large tensors in a single external data file."""
    data_name = out_path.stem + ".onnx_data"
    data_path = out_path.parent / data_name
    if data_path.exists():
        data_path.unlink()

    onnx.save_model(
        model,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )
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
        sz = f.stat().st_size
        total += sz
        if sz > 1_000_000:
            print(f"  {f.name}: {sz / (1024 * 1024):.1f} MB")
    print(f"  TOTAL: {total / (1024 * 1024):.1f} MB")


# ── INT4 Quantization ────────────────────────────────────────────────────────


def quantize_int4(model_path: Path) -> None:
    """Apply INT4 weight-only quantization (MatMulNBits) in-place."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    print(f"  Quantizing {model_path.name} to INT4...")
    model = onnx.load(str(model_path), load_external_data=True)

    quantizer = MatMulNBitsQuantizer(
        model,
        bits=4,
        block_size=128,
        is_symmetric=False,
        accuracy_level=0,
    )
    quantizer.process()

    qmodel = quantizer.model
    if hasattr(qmodel, "model"):
        qmodel = qmodel.model

    _save_consolidated(qmodel, model_path)
    print(f"  {model_path.name}: quantized to INT4")


# ── LLM Wrappers ─────────────────────────────────────────────────────────────


class Qwen3EmbedTokensWrapper(nn.Module):
    """Token embedding lookup (separate ONNX file for flexibility)."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids).to(DTYPE)


class Qwen3DecoderWrapper(nn.Module):
    """Qwen3 decoder with KV cache, using the model's own forward path.

    This preserves q_norm/k_norm, RoPE, GQA, and SwiGLU exactly as
    implemented in transformers, avoiding manual reimplementation bugs.
    """

    def __init__(self, model, num_layers: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.text_model = model.model
        self.lm_head = model.lm_head
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(
        self,
        inputs_embeds: torch.Tensor,       # [B, L, H]
        position_ids: torch.Tensor,         # [B, L]
        past_key_values_flat: list[torch.Tensor],
    ):
        from transformers import DynamicCache

        past = DynamicCache()
        for i in range(self.num_layers):
            past.update(
                past_key_values_flat[i * 2],
                past_key_values_flat[i * 2 + 1],
                i,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
        )

        logits = self.lm_head(outputs[0])

        new_cache = outputs.past_key_values
        flat_present: list[torch.Tensor] = []
        for i in range(self.num_layers):
            k, v = new_cache[i]
            flat_present.append(k)
            flat_present.append(v)

        return (logits, flat_present)


# ── LLM Export ────────────────────────────────────────────────────────────────


def export_llm(
    output_dir: Path,
    model_id: str = "Qwen/Qwen3-0.6B",
    quantize: bool = True,
) -> None:
    """Export Qwen3-0.6B to ONNX via direct torch.onnx.export."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LLM] Exporting {model_id} → {output_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        attn_implementation="eager",  # no flash-attn / SDPA for tracing
    )
    model.eval()

    cfg = model.config
    num_layers = cfg.num_hidden_layers       # 28
    num_kv_heads = cfg.num_key_value_heads   # 8
    head_dim = cfg.head_dim                  # 128
    hidden_size = cfg.hidden_size            # 1024
    vocab_size = cfg.vocab_size              # 151936

    print(
        f"[LLM] Architecture: {num_layers}L, {hidden_size}H, "
        f"{num_kv_heads}KV, {head_dim}D, vocab={vocab_size}, tied={cfg.tie_word_embeddings}"
    )

    # ── 1. Export embed_tokens ───────────────────────────────────────────
    print("[LLM]   Exporting qwen3-0.6b-embed-tokens.onnx ...")
    embed_wrapper = Qwen3EmbedTokensWrapper(model)
    embed_wrapper.eval()

    dummy_ids = torch.tensor([[151644, 8948, 198]], dtype=torch.long)
    seq_dim = torch.export.Dim("seq_len", min=1, max=4096)

    torch.onnx.export(
        embed_wrapper,
        (dummy_ids,),
        str(output_dir / "qwen3-0.6b-embed-tokens.onnx"),
        input_names=["input_ids"],
        output_names=["embeds"],
        dynamic_shapes=({1: seq_dim},),
    )
    print("[LLM]   qwen3-0.6b-embed-tokens.onnx exported")

    # ── 2. Export decoder ────────────────────────────────────────────────
    print("[LLM]   Exporting qwen3-0.6b-decoder.onnx ...")
    dec_wrapper = Qwen3DecoderWrapper(model, num_layers, num_kv_heads, head_dim)
    dec_wrapper.eval()

    seq_len = 4
    past_len = 0
    dummy_embeds = torch.randn(1, seq_len, hidden_size, dtype=DTYPE)
    dummy_position_ids = torch.arange(past_len, past_len + seq_len).unsqueeze(0)  # [1, L]

    dummy_past = []
    for _ in range(num_layers):
        dummy_past.append(torch.zeros(1, num_kv_heads, past_len, head_dim, dtype=DTYPE))  # key
        dummy_past.append(torch.zeros(1, num_kv_heads, past_len, head_dim, dtype=DTYPE))  # value

    input_names = ["inputs_embeds", "position_ids"]
    output_names = ["logits"]
    for i in range(num_layers):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present_key_values.{i}.key")
        output_names.append(f"present_key_values.{i}.value")

    seq_dim_d = torch.export.Dim("seq_len", min=1, max=4096)
    past_dim = torch.export.Dim("past_seq_len", min=0, max=8192)

    past_dyn = [{2: past_dim} for _ in range(num_layers * 2)]
    dyn_shapes = (
        {1: seq_dim_d},       # inputs_embeds
        {1: seq_dim_d},       # position_ids
        past_dyn,             # past KV
    )

    torch.onnx.export(
        dec_wrapper,
        (dummy_embeds, dummy_position_ids, dummy_past),
        str(output_dir / "qwen3-0.6b-decoder.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dyn_shapes,
    )
    print("[LLM]   qwen3-0.6b-decoder.onnx exported")

    # ── 3. Consolidate external data ─────────────────────────────────────
    for name in ["qwen3-0.6b-embed-tokens.onnx", "qwen3-0.6b-decoder.onnx"]:
        _consolidate_external_data(output_dir, name)

    # ── 4. Quantize decoder ──────────────────────────────────────────────
    if quantize:
        quantize_int4(output_dir / "qwen3-0.6b-decoder.onnx")

    # ── 5. Cleanup stale scattered external data ─────────────────────────
    keep = {
        "qwen3-0.6b-embed-tokens.onnx", "qwen3-0.6b-embed-tokens.onnx_data",
        "qwen3-0.6b-decoder.onnx", "qwen3-0.6b-decoder.onnx_data",
    }
    for f in output_dir.iterdir():
        if f.suffix == ".data" and f.name not in keep:
            f.unlink()
        elif f.name.startswith("onnx__"):
            f.unlink()

    # ── 6. Save tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(output_dir))

    print(f"[LLM] Export complete: {output_dir}")
    _list_files(output_dir)


# ── ASR Wrappers ──────────────────────────────────────────────────────────────


class EncoderWrapper(nn.Module):
    """ONNX-friendly audio encoder wrapper.

    Reuses original weights through standard ops (no cu_seqlens / Flash Attention).
    """

    def __init__(self, thinker):
        super().__init__()
        at = thinker.audio_tower
        self.conv2d1 = at.conv2d1
        self.conv2d2 = at.conv2d2
        self.conv2d3 = at.conv2d3
        self.conv_out = at.conv_out
        self.register_buffer(
            "pos_embed", at.positional_embedding.positional_embedding
        )
        self.layers = at.layers
        self.ln_post = at.ln_post
        self.proj1 = at.proj1
        self.act = at.act
        self.proj2 = at.proj2

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(1)  # [1, 1, 128, T]
        x = torch.nn.functional.gelu(self.conv2d1(x))
        x = torch.nn.functional.gelu(self.conv2d2(x))
        x = torch.nn.functional.gelu(self.conv2d3(x))
        b, c, f, t = x.size()
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        x = x + self.pos_embed[:t, :].unsqueeze(0).to(x.dtype)
        hidden = x.squeeze(0)
        cu_seqlens = torch.zeros(2, dtype=torch.int32, device=mel.device)
        cu_seqlens[1] = hidden.shape[0]
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
    """Wraps the Qwen3 decoder with KV cache for ONNX export."""

    def __init__(self, thinker, num_layers: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.text_model = thinker.model
        self.lm_head = thinker.lm_head
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values_flat: list[torch.Tensor],
    ):
        from transformers import DynamicCache

        past = DynamicCache()
        for i in range(self.num_layers):
            k = past_key_values_flat[i * 2]
            v = past_key_values_flat[i * 2 + 1]
            past.update(k, v, i)

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
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
    quantize: bool = True,
) -> None:
    """Export Qwen3-ASR-0.6B to three ONNX files (FP16), then INT4-quantize decoder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ASR] Exporting {model_id} → {output_dir}")

    from transformers import AutoProcessor
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=DTYPE,
        attn_implementation="eager",
    )
    full_model.eval()

    thinker = full_model.thinker
    text_config = thinker.config.text_config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
    hidden_size = text_config.hidden_size

    print(
        f"[ASR] Architecture: {num_layers}L, {hidden_size}H, "
        f"{num_kv_heads}KV, {head_dim}D, dtype={DTYPE}"
    )

    # ── 1. Encoder ───────────────────────────────────────────────────────
    print("[ASR]   Exporting qwen3-asr-encoder.onnx ...")
    enc = EncoderWrapper(thinker)
    enc.eval()
    dummy_mel = torch.randn(1, 128, 200, dtype=DTYPE)
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

    # ── 3. Decoder ───────────────────────────────────────────────────────
    print("[ASR]   Exporting qwen3-asr-decoder.onnx ...")
    dec = DecoderWrapper(thinker, num_layers, num_kv_heads, head_dim)
    dec.eval()

    seq_len = 10
    dummy_embeds = torch.randn(1, seq_len, hidden_size, dtype=DTYPE)
    dummy_position_ids = (
        torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
    )
    dummy_past = []
    for _ in range(num_layers):
        dummy_past.append(torch.zeros(1, num_kv_heads, 0, head_dim, dtype=DTYPE))
        dummy_past.append(torch.zeros(1, num_kv_heads, 0, head_dim, dtype=DTYPE))

    input_names = ["inputs_embeds", "position_ids"]
    output_names = ["logits"]
    for i in range(num_layers):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")
        output_names.append(f"present_key_values.{i}.key")
        output_names.append(f"present_key_values.{i}.value")

    seq_dim_a = torch.export.Dim("seq_len", min=1, max=4096)
    past_dim = torch.export.Dim("past_seq_len", min=0, max=8192)

    past_dyn = [{2: past_dim} for _ in range(num_layers * 2)]
    dyn_shapes = ({1: seq_dim_a}, {2: seq_dim_a}, past_dyn)

    torch.onnx.export(
        dec,
        (dummy_embeds, dummy_position_ids, dummy_past),
        str(output_dir / "qwen3-asr-decoder.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dyn_shapes,
    )
    print("[ASR]   qwen3-asr-decoder.onnx exported")

    # ── 4. Consolidate ───────────────────────────────────────────────────
    for name in [
        "qwen3-asr-encoder.onnx",
        "qwen3-asr-embed-tokens.onnx",
        "qwen3-asr-decoder.onnx",
    ]:
        _consolidate_external_data(output_dir, name)

    # ── 5. Quantize decoder ──────────────────────────────────────────────
    if quantize:
        quantize_int4(output_dir / "qwen3-asr-decoder.onnx")

    # ── 6. Cleanup ───────────────────────────────────────────────────────
    keep = {
        "qwen3-asr-encoder.onnx", "qwen3-asr-encoder.onnx_data",
        "qwen3-asr-embed-tokens.onnx", "qwen3-asr-embed-tokens.onnx_data",
        "qwen3-asr-decoder.onnx", "qwen3-asr-decoder.onnx_data",
    }
    for f in output_dir.iterdir():
        if f.suffix in (".data", ".onnx.data") and f.name not in keep:
            f.unlink()
        elif f.name.startswith("onnx__"):
            f.unlink()

    processor.save_pretrained(str(output_dir))

    print(f"[ASR] Export complete: {output_dir}")
    _list_files(output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 models to ONNX format"
    )
    parser.add_argument("--llm", action="store_true", help="Export Qwen3-0.6B LLM")
    parser.add_argument("--asr", action="store_true", help="Export Qwen3-ASR-0.6B")
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip INT4 quantization (keep FP16 weights)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: models/<model-name>-onnx)",
    )
    parser.add_argument(
        "--llm-model", default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID for LLM",
    )
    parser.add_argument(
        "--asr-model", default="Qwen/Qwen3-ASR-0.6B",
        help="HuggingFace model ID for ASR",
    )
    args = parser.parse_args()

    if not args.llm and not args.asr:
        parser.error("Specify --llm, --asr, or both")

    do_quant = not args.no_quantize

    if args.llm:
        out = args.output_dir or Path("models/qwen3-0.6b-onnx")
        export_llm(out, args.llm_model, quantize=do_quant)

    if args.asr:
        out = args.output_dir or Path("models/qwen3-asr-0.6b-onnx")
        export_asr(out, args.asr_model, quantize=do_quant)


if __name__ == "__main__":
    main()
