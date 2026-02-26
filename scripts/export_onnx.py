#!/usr/bin/env python3
"""Export Qwen3-0.6B (LLM) and Qwen3-ASR-0.6B to ONNX format.

Usage:
    # Export LLM only:
    python scripts/export_onnx.py --llm --output-dir models/qwen3-0.6b-onnx

    # Export ASR only:
    python scripts/export_onnx.py --asr --output-dir models/qwen3-asr-0.6b-onnx

    # Export both:
    python scripts/export_onnx.py --llm --asr

    # Skip INT4 quantization (keep FP16):
    python scripts/export_onnx.py --llm --asr --no-quantize

Requirements:
    pip install torch transformers optimum[onnxruntime] onnx onnxruntime
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import onnx
import torch
import torch.nn as nn

# Suppress noisy third-party warnings (torchvision custom-op annotations, etc.)
warnings.filterwarnings("ignore", message="Missing annotation for parameter")
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

# Register Qwen3-ASR model type with transformers (no-op if already native).
try:
    import qwen_asr  # noqa: F401
except ImportError:
    pass


DTYPE = torch.float16


# ── INT4 Quantization ───────────────────────────────────────────────────────


def quantize_int4(model_path: Path) -> None:
    """Apply INT4 weight quantization (MatMulNBits) to an ONNX model in-place."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    print(f"  Quantizing {model_path.name} to INT4...")
    model = onnx.load(str(model_path), load_external_data=True)

    quantizer = MatMulNBitsQuantizer(
        model,
        bits=4,
        block_size=128,
        is_symmetric=False,
        accuracy_level=4,
    )
    quantizer.process()

    qmodel = quantizer.model
    if hasattr(qmodel, "model"):
        qmodel = qmodel.model

    _save_consolidated(qmodel, model_path)
    print(f"  {model_path.name}: quantized to INT4")


def _save_consolidated(model: onnx.ModelProto, out_path: Path) -> None:
    """Save ONNX model with all large tensors in a single external data file."""
    data_name = out_path.stem + ".onnx_data"
    data_path = out_path.parent / data_name

    # Remove stale external data before saving.
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


# ── LLM Export ───────────────────────────────────────────────────────────────


def export_llm(
    output_dir: Path,
    model_id: str = "Qwen/Qwen3-0.6B",
    quantize: bool = True,
) -> None:
    """Export Qwen3-0.6B to ONNX via optimum-cli, then quantize to INT4.

    Produces qwen3-0.6b.onnx with input_ids + past_key_values
    → logits + present_key_values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LLM] Exporting {model_id} → {output_dir}")

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_id,
        "--task", "text-generation-with-past",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)

    # Copy tokenizer.json if present in the source model.
    tokenizer_src = output_dir / "tokenizer.json"
    if not tokenizer_src.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(str(output_dir))

    # Rename optimum's default filenames and update all external data refs.
    _rename_onnx_with_external_data(output_dir, "model.onnx", "qwen3-0.6b.onnx")

    if quantize:
        quantize_int4(output_dir / "qwen3-0.6b.onnx")

    print(f"[LLM] Export complete: {output_dir}")
    _list_files(output_dir)


# ── ASR Export ───────────────────────────────────────────────────────────────


def export_asr(
    output_dir: Path,
    model_id: str = "Qwen/Qwen3-ASR-0.6B",
    quantize: bool = True,
) -> None:
    """Export Qwen3-ASR-0.6B to three ONNX files (FP16), then INT4-quantize decoder.

    1. qwen3-asr-encoder.onnx: AuT audio encoder + projector
       Input: mel[1,128,T] → Output: features[1,S,1024]

    2. qwen3-asr-embed-tokens.onnx: Token embedding lookup
       Input: input_ids[1,L] → Output: embeds[1,L,1024]

    3. qwen3-asr-decoder.onnx: Qwen3 decoder with KV cache
       Input: inputs_embeds[1,L,1024] + past_key_values → Output: logits + present_key_values
    """
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
        attn_implementation="eager",  # Flash attention can't be traced by ONNX
    )
    full_model.eval()

    thinker = full_model.thinker

    text_config = thinker.config.text_config
    num_layers = text_config.num_hidden_layers
    num_kv_heads = getattr(
        text_config, "num_key_value_heads", text_config.num_attention_heads
    )
    head_dim = getattr(
        text_config, "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    hidden_size = text_config.hidden_size

    print(
        f"[ASR] Architecture: {num_layers} layers, {hidden_size} hidden, "
        f"{num_kv_heads} KV heads, {head_dim} head_dim, dtype={DTYPE}"
    )

    # ── 1. Export encoder ────────────────────────────────────────────────
    _export_encoder(thinker, output_dir, hidden_size)

    # ── 2. Export embed_tokens ───────────────────────────────────────────
    _export_embed_tokens(thinker, output_dir, hidden_size)

    # ── 3. Export decoder ────────────────────────────────────────────────
    _export_decoder(
        thinker, output_dir, num_layers, num_kv_heads, head_dim, hidden_size
    )

    # ── Consolidate external data ────────────────────────────────────────
    # torch.onnx.export scatters external data as per-tensor files.
    # Re-save each model with consolidated single-file external data.
    for name in [
        "qwen3-asr-encoder.onnx",
        "qwen3-asr-embed-tokens.onnx",
        "qwen3-asr-decoder.onnx",
    ]:
        _consolidate_external_data(output_dir, name)

    # ── INT4-quantize the decoder ────────────────────────────────────────
    if quantize:
        quantize_int4(output_dir / "qwen3-asr-decoder.onnx")

    # ── Clean up stale external data files ─────────────────────────────
    # torch.onnx.export creates *.onnx.data; consolidation saves *.onnx_data.
    # Remove the originals plus any scattered per-tensor files.
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

    # ── Copy tokenizer ──────────────────────────────────────────────────
    processor.save_pretrained(str(output_dir))

    print(f"[ASR] Export complete: {output_dir}")
    _list_files(output_dir)


def _consolidate_external_data(directory: Path, onnx_name: str) -> None:
    """Load an ONNX model with scattered external data and re-save consolidated."""
    onnx_path = directory / onnx_name
    if not onnx_path.exists():
        return

    print(f"  Consolidating {onnx_name}...")
    model = onnx.load(str(onnx_path), load_external_data=True)
    _save_consolidated(model, onnx_path)


class EncoderWrapper(nn.Module):
    """ONNX-friendly audio encoder wrapper.

    The original encoder uses cu_seqlens + Flash Attention and dynamic chunking
    that can't be traced. This wrapper reuses the same weights but processes a
    single mel spectrogram through standard ops:
      Conv2d×3 → Linear → PositionalEmbed → EncoderLayers → LayerNorm → Proj
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
        # mel: [1, 128, T]
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
        return hidden.unsqueeze(0)  # [1, S, output_dim]


def _export_encoder(thinker, output_dir: Path, hidden_size: int) -> None:
    print("[ASR]   Exporting qwen3-asr-encoder.onnx ...")
    wrapper = EncoderWrapper(thinker)
    wrapper.eval()

    dummy_mel = torch.randn(1, 128, 200, dtype=DTYPE)

    mel_dim = torch.export.Dim("mel_frames", min=2, max=6000)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(output_dir / "qwen3-asr-encoder.onnx"),
        input_names=["mel"],
        output_names=["features"],
        dynamic_shapes=({2: mel_dim},),
    )
    print("[ASR]   qwen3-asr-encoder.onnx exported")


class EmbedTokensWrapper(nn.Module):
    """Wraps the token embedding layer for ONNX export."""

    def __init__(self, thinker):
        super().__init__()
        self.embed_tokens = thinker.model.embed_tokens

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)  # [1, L, hidden_size]


def _export_embed_tokens(thinker, output_dir: Path, hidden_size: int) -> None:
    print("[ASR]   Exporting qwen3-asr-embed-tokens.onnx ...")
    wrapper = EmbedTokensWrapper(thinker)
    wrapper.eval()

    dummy_ids = torch.tensor([[151644, 8948, 198]], dtype=torch.long)

    seq_dim = torch.export.Dim("seq_len", min=1, max=4096)

    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        str(output_dir / "qwen3-asr-embed-tokens.onnx"),
        input_names=["input_ids"],
        output_names=["embeds"],
        dynamic_shapes=({1: seq_dim},),
    )
    print("[ASR]   qwen3-asr-embed-tokens.onnx exported")


class DecoderWrapper(nn.Module):
    """Wraps the Qwen3 decoder for ONNX export with KV cache.

    Uses the text model's forward + lm_head rather than manual layer iteration.
    This ensures RoPE (including mRoPE if present) is handled correctly.
    """

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


def _export_decoder(
    thinker,
    output_dir: Path,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
) -> None:
    print("[ASR]   Exporting qwen3-asr-decoder.onnx ...")
    wrapper = DecoderWrapper(thinker, num_layers, num_kv_heads, head_dim)
    wrapper.eval()

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

    seq_dim = torch.export.Dim("seq_len", min=1, max=4096)
    past_dim = torch.export.Dim("past_seq_len", min=0, max=8192)

    past_dyn = [{2: past_dim} for _ in range(num_layers * 2)]
    dyn_shapes = ({1: seq_dim}, {2: seq_dim}, past_dyn)

    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_position_ids, dummy_past),
        str(output_dir / "qwen3-asr-decoder.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dyn_shapes,
    )
    print("[ASR]   qwen3-asr-decoder.onnx exported")


# ── Utilities ────────────────────────────────────────────────────────────────


def _rename_onnx_with_external_data(
    directory: Path, old_name: str, new_name: str,
) -> None:
    """Rename an ONNX file and its external data, updating all internal refs."""
    old_onnx = directory / old_name
    if not old_onnx.exists():
        return

    old_data = old_name.removesuffix(".onnx") + ".onnx_data"
    new_data = new_name.removesuffix(".onnx") + ".onnx_data"

    model = onnx.load(str(old_onnx), load_external_data=False)

    for init in model.graph.initializer:
        for ext in init.external_data:
            if ext.key == "location" and ext.value == old_data:
                ext.value = new_data

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.t and attr.t.ByteSize() > 0:
                for ext in attr.t.external_data:
                    if ext.key == "location" and ext.value == old_data:
                        ext.value = new_data

    onnx.save(model, str(directory / new_name))
    if old_onnx.name != new_name:
        old_onnx.unlink()

    data_file = directory / old_data
    if data_file.exists() and old_data != new_data:
        data_file.rename(directory / new_data)

    print(f"  Renamed {old_name} → {new_name}")


def _list_files(directory: Path) -> None:
    total = 0
    for f in sorted(directory.iterdir()):
        sz = f.stat().st_size
        total += sz
        if sz > 1_000_000:
            print(f"  {f.name}: {sz / (1024 * 1024):.1f} MB")
    print(f"  TOTAL: {total / (1024 * 1024):.1f} MB")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 models to ONNX format for lune-voice"
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
