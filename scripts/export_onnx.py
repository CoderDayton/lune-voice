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
    pip install torch transformers onnx onnxruntime nncf
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

# Diverse calibration corpus covering ASR output styles: formal prose, technical
# content, coding dictation, meeting notes, and casual speech. 32 samples drives
# AWQ scale search and Scale Estimation without overfitting any single domain.
_CALIB_TEXTS: list[str] = [
    "The meeting is scheduled for Tuesday at three pm.",
    "Please open the configuration file and save it to the downloads folder.",
    "She finished the report ahead of schedule and her manager approved it.",
    "The thermal dissipation is roughly 450 watts and the substrate should not degrade.",
    "Install the dependency and then run the build command in the terminal.",
    "The quick brown fox jumps over the lazy dog.",
    "I wanted to follow up on our discussion last week regarding the project timeline.",
    "The package was delayed in transit and the supplier was notified of the issue.",
    "Let's make an async function called sync data that awaits a fetch to the API.",
    "The city thrived for years in the shadow of those ancient walls.",
    "Handle the case where the cache is empty or the key does not exist.",
    "So we gotta nuke the old user profile class and replace it with a new interface.",
    "Thank you for getting back to me. I will review the document and send my feedback by Friday.",
    "The build is failing on main and someone needs to fix that.",
    "Discussed Q3 roadmap. Action: John handle the API redesign by next Tuesday.",
    "The main blocker is the dependency on the auth service owned by the platform team.",
    "Returns a sorted copy of the slice without modifying the original.",
    "The letter arrived on Monday after three weeks of delays.",
    "Hey can you take a look at the PR when you get a chance?",
    "It's totally ridiculous that the wear is this bad after a month.",
    "From the respect paid her on all sides she seemed like a queen.",
    "The church was built in the fourteenth century and restored in eighteen ninety.",
    "Please confirm your attendance by responding to this email before Friday.",
    "The server is currently processing your request. Please wait a moment.",
    "We need to increase the timeout value and redeploy the service.",
    "The neural network was trained on a large corpus of annotated speech data.",
    "I'll review the pull request and merge it once the CI pipeline passes.",
    "The quarterly earnings report will be published on the company website.",
    "Set the environment variable before running the initialization script.",
    "She opened the door and walked into the conference room.",
    "The activation function introduces non-linearity into the model architecture.",
    "After three weeks of testing we finally shipped the feature to production.",
]


def _repack_matmulnbits_zero_points(model: onnx.ModelProto) -> None:
    """Repack MatMulNBits zero_points from byte-per-nibble to nibble-packed.

    NNCF stores INT4 zero points as one uint8 byte per value (shape [N, n_groups]).
    ORT's MatMulNBits requires two zero points packed per byte — lo nibble then hi
    nibble — giving shape [N, ceil(n_groups/2)].  Without this, ORT raises a shape
    mismatch error at model load time.
    """
    import numpy as np
    from onnx import numpy_helper

    init_idx = {init.name: i for i, init in enumerate(model.graph.initializer)}

    for node in model.graph.node:
        if node.op_type != "MatMulNBits" or len(node.input) < 4:
            continue
        zp_name = node.input[3]
        if not zp_name or zp_name not in init_idx:
            continue

        attrs = {a.name: a for a in node.attribute}
        K = attrs["K"].i
        block_size = attrs["block_size"].i
        n_groups = (K + block_size - 1) // block_size
        expected_cols = (n_groups + 1) // 2  # bytes per output row after packing

        init = model.graph.initializer[init_idx[zp_name]]
        arr = numpy_helper.to_array(init)

        if arr.ndim != 2 or arr.shape[-1] == expected_cols:
            continue  # already packed or unexpected layout — leave as-is

        if arr.shape[-1] == n_groups:
            N_rows = arr.shape[0]
            if n_groups % 2 != 0:
                arr = np.concatenate([arr, np.zeros((N_rows, 1), dtype=arr.dtype)], axis=1)
            lo = arr[:, 0::2].astype(np.uint8) & 0x0F
            hi = arr[:, 1::2].astype(np.uint8) & 0x0F
            packed = lo | (hi << 4)
            model.graph.initializer[init_idx[zp_name]].CopyFrom(
                numpy_helper.from_array(packed, name=zp_name)
            )


def _cast_matmulnbits_scales_fp16(model: onnx.ModelProto) -> None:
    """Cast float32 scales in MatMulNBits nodes to float16 in-place.

    NNCF stores quantization scales as float32 by default. ORT's MatMulNBits
    requires the scale type (T1) to match the activation type. Since our models
    are exported in FP16, mismatched scales cause a Type Error at load time.
    """
    import numpy as np
    from onnx import numpy_helper

    init_idx = {init.name: i for i, init in enumerate(model.graph.initializer)}

    for node in model.graph.node:
        if node.op_type == "MatMulNBits" and len(node.input) >= 3:
            scale_name = node.input[2]
            if scale_name in init_idx:
                init = model.graph.initializer[init_idx[scale_name]]
                if init.data_type == onnx.TensorProto.FLOAT:
                    arr = numpy_helper.to_array(init).astype(np.float16)
                    model.graph.initializer[init_idx[scale_name]].CopyFrom(
                        numpy_helper.from_array(arr, name=scale_name)
                    )


def _wrap_inputs_with_identity(model: onnx.ModelProto) -> None:
    """In-place: prepend Identity nodes after each graph input.

    NNCF 3.x assumes every tensor consumed by a node is produced by another
    node (i.e., present in parents_node_mapping). Graph inputs violate this,
    causing a KeyError in _get_bias_attr. Proxying them through Identity ops
    satisfies the invariant. ORT's graph optimizer elides these at load time.
    """
    import onnx.helper as oh

    input_names = {inp.name for inp in model.graph.input}
    name_remap = {name: f"_nncf_proxy_{name}" for name in input_names}

    # Re-point all node inputs that currently reference a raw graph input.
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in name_remap:
                node.input[i] = name_remap[inp]

    # Prepend Identity nodes: graph_input → proxy name.
    identity_nodes = [
        oh.make_node("Identity", inputs=[name], outputs=[proxy])
        for name, proxy in name_remap.items()
    ]
    for i, id_node in enumerate(identity_nodes):
        model.graph.node.insert(i, id_node)


def quantize_int4_nncf(
    model_path: Path,
    embed_path: Path,
    model_id: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    fa_mode: bool = False,
) -> None:
    """INT4 weight-only quantization via NNCF AWQ + Scale Estimation.

    AWQ (Activation-Aware Weight Quantization) identifies salient weight
    channels by observing activation distributions and rescales them before
    quantization — this is the W4A16 analogue of SmoothQuant's outlier
    protection. Scale Estimation then refines the per-group scales using
    calibration statistics to further minimise reconstruction error.

    Produces INT4_ASYM / MatMulNBits weights readable by OnnxRuntime.
    """
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    import nncf
    from nncf.onnx.quantization.backend_parameters import BackendParameters

    print(f"  Quantizing {model_path.name} to INT4 (NNCF AWQ+SE)...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    embed_sess = ort.InferenceSession(
        str(embed_path), providers=["CPUExecutionProvider"]
    )

    # Detect position_ids rank from the exported graph topology.
    # Qwen3-ASR uses 3-D mrope [3, 1, S]; Qwen3-LLM uses standard [1, S].
    meta = onnx.load(str(model_path), load_external_data=False)
    pos_ndim = next(
        (
            len(inp.type.tensor_type.shape.dim)
            for inp in meta.graph.input
            if inp.name == "position_ids"
        ),
        2,
    )
    del meta

    def _make_sample(text: str) -> dict:
        ids = tokenizer(text, return_tensors="np").input_ids  # [1, S]
        embeds = embed_sess.run(["embeds"], {"input_ids": ids})[0]  # [1, S, H]
        seq = embeds.shape[1]
        pos = np.arange(seq, dtype=np.int64)

        if fa_mode:
            # FA decoder: NAR, no KV cache, standard [1, S] position_ids.
            return {"inputs_embeds": embeds, "position_ids": pos.reshape(1, seq)}

        pos_ids = (
            np.broadcast_to(pos, (3, 1, seq)).copy()  # mrope: ASR
            if pos_ndim == 3
            else pos.reshape(1, seq)                   # standard: LLM
        )
        # Empty KV cache — prefill pass exercises all MatMul layers.
        kv_zero = np.zeros((1, num_kv_heads, 0, head_dim), dtype=np.float16)
        sample: dict = {"inputs_embeds": embeds, "position_ids": pos_ids}
        for i in range(num_layers):
            sample[f"past_key_values.{i}.key"] = kv_zero
            sample[f"past_key_values.{i}.value"] = kv_zero
        return sample

    dataset = nncf.Dataset(_CALIB_TEXTS, _make_sample)

    # Load topology only; NNCF streams weights lazily via EXTERNAL_DATA_DIR.
    onnx_model = onnx.load(str(model_path), load_external_data=False)

    # Workaround: NNCF 3.x nncf_graph_builder._get_bias_attr crashes with a
    # KeyError when a node's input is a raw graph input (not produced by any op).
    # Insert no-op Identity nodes so every tensor has an entry in
    # parents_node_mapping. ORT's graph optimizer removes these at runtime.
    _wrap_inputs_with_identity(onnx_model)

    # Find MatMul/Gemm nodes whose weight initializer has >2 dims.
    # These are positional-encoding constants (e.g. mrope [3,1,64,1]) that are
    # not linear-projection weights and cannot be group-quantized on any axis.
    init_shapes = {i.name: list(i.dims) for i in onnx_model.graph.initializer}
    skip_nodes: list[str] = []
    for node in onnx_model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            for inp in node.input:
                if inp in init_shapes and len(init_shapes[inp]) > 2:
                    if node.name:
                        skip_nodes.append(node.name)
                        print(f"  Skipping {node.name!r} (weight shape {init_shapes[inp]})")

    ignored = nncf.IgnoredScope(names=skip_nodes) if skip_nodes else None

    common_kwargs: dict = dict(
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        ratio=1.0,
        # Keep backup layers (lm_head) at original fp16 — NNCF's int8 dequantize
        # pattern emits float32 which causes a type mismatch with fp16 activations
        # in ORT. BackupMode.NONE avoids the int8 path entirely.
        backup_mode=nncf.BackupMode.NONE,
        awq=True,
        scale_estimation=True,
        subset_size=len(_CALIB_TEXTS),
        dataset=dataset,
        ignored_scope=ignored,
        advanced_parameters=nncf.AdvancedCompressionParameters(
            backend_params={BackendParameters.EXTERNAL_DATA_DIR: model_path.parent.resolve()}
        ),
    )

    try:
        compressed = nncf.compress_weights(onnx_model, group_size=128, **common_kwargs)
        group_desc = "group=128"
    except nncf.errors.UnsupportedModelError:
        # Residual 3-D weights still incompatible with group quantization — fall
        # back to per-channel. Also drop scale_estimation: it calls np.stack on
        # per-sample activation stats and fails when calibration sequences have
        # different lengths (variable-length text → variable hidden-state shapes).
        print("  group_size=128 unsupported, retrying per-channel (no SE)...")
        onnx_model = onnx.load(str(model_path), load_external_data=False)
        _wrap_inputs_with_identity(onnx_model)
        common_kwargs["scale_estimation"] = False
        compressed = nncf.compress_weights(onnx_model, group_size=-1, **common_kwargs)
        group_desc = "per-channel"

    # ORT requires MatMulNBits scale type (T1) to match activation dtype (fp16).
    # NNCF emits scales as float32 by default — cast them to fp16 before saving.
    _cast_matmulnbits_scales_fp16(compressed)
    # ORT requires zero_points nibble-packed (2 INT4 per byte).
    # NNCF stores one INT4 per byte — repack before saving.
    _repack_matmulnbits_zero_points(compressed)

    _save_consolidated(compressed, model_path)
    print(f"  {model_path.name}: INT4 AWQ+SE ({group_desc}) complete")


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
        quantize_int4_nncf(
            output_dir / "qwen3-0.6b-decoder.onnx",
            output_dir / "qwen3-0.6b-embed-tokens.onnx",
            model_id,
            num_layers,
            num_kv_heads,
            head_dim,
        )

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
        quantize_int4_nncf(
            output_dir / "qwen3-asr-decoder.onnx",
            output_dir / "qwen3-asr-embed-tokens.onnx",
            model_id,
            num_layers,
            num_kv_heads,
            head_dim,
        )

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


# ── FA Wrapper ────────────────────────────────────────────────────────────────


class ForcedAlignerDecoderWrapper(nn.Module):
    """NAR forced-aligner decoder: single forward pass, no KV cache.

    `thinker.lm_head` is the timestamp head (Linear(hidden_size, classify_num))
    when `model_type` includes "forced_aligner" — the model __init__ already
    replaces the vocabulary head with the timestamp classification head.
    """

    def __init__(self, thinker):
        super().__init__()
        self.text_model = thinker.model
        # For FA checkpoints, lm_head is Linear(hidden_size, classify_num).
        self.timestamp_head = thinker.lm_head

    @torch.no_grad()
    def forward(
        self,
        inputs_embeds: torch.Tensor,   # [1, S, H]  f16
        position_ids: torch.Tensor,    # [1, S]     i64
    ) -> torch.Tensor:                 # [1, S, classify_num]  f16
        out = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=False,
        )
        return self.timestamp_head(out[0]).to(DTYPE)


# ── FA Export ─────────────────────────────────────────────────────────────────


def export_forcedaligner(
    output_dir: Path,
    asr_dir: Path,
    model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    quantize: bool = True,
) -> None:
    """Export Qwen3-ForcedAligner-0.6B FA decoder to ONNX (INT4).

    The encoder and embed_tokens are REUSED from `asr_dir` — they share weights
    with the ASR model. Only the NAR FA decoder (no KV cache, timestamp head) is
    exported to `output_dir`.

    Writes `fa_config.json` with timestamp_token_id, timestamp_segment_ms,
    and timestamp_classes for the Rust `OrtForcedAligner`.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FA] Exporting {model_id} → {output_dir}")
    print(f"[FA] Reusing encoder/embed_tokens from {asr_dir}")

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

    # Timestamp head output size (classify_num / num_timestamps).
    timestamp_classes = thinker.lm_head.out_features
    # Timestamp segment duration — each class bucket is 80 ms (1 mel frame = 10 ms,
    # Qwen-ASR uses 8 mel frames per encoder step → 80 ms per timestamp class).
    timestamp_segment_ms = getattr(thinker.config, "timestamp_segment_ms", 80)
    # Timestamp token ID — the special <|ts|> token appended to each word.
    timestamp_token_id: int | None = getattr(thinker.config, "timestamp_token_id", None)
    if timestamp_token_id is None:
        # Fallback: search the tokenizer vocabulary for the TS token.
        for candidate in ("<|ts|>", "<ts>", "<|timestamp|>"):
            tid = processor.tokenizer.convert_tokens_to_ids(candidate)
            if tid != processor.tokenizer.unk_token_id:
                timestamp_token_id = int(tid)
                print(f"[FA]   Found timestamp token {candidate!r} → id={timestamp_token_id}")
                break
    if timestamp_token_id is None:
        raise RuntimeError(
            "Could not determine timestamp_token_id from model config or tokenizer. "
            "Set thinker.config.timestamp_token_id or add the token to the vocabulary."
        )

    print(
        f"[FA] Architecture: {num_layers}L, {hidden_size}H, "
        f"{num_kv_heads}KV, {head_dim}D, ts_classes={timestamp_classes}"
    )

    # ── 1. Export FA decoder ─────────────────────────────────────────────
    print("[FA]   Exporting qwen3-fa-decoder.onnx ...")
    fa_dec = ForcedAlignerDecoderWrapper(thinker)
    fa_dec.eval()

    seq_len = 20  # typical: prefix (4) + enc (~10) + words (~6)
    dummy_embeds = torch.randn(1, seq_len, hidden_size, dtype=DTYPE)
    dummy_pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, S]

    seq_dim = torch.export.Dim("seq_len", min=2, max=8192)
    dyn_shapes = ({1: seq_dim}, {1: seq_dim})

    torch.onnx.export(
        fa_dec,
        (dummy_embeds, dummy_pos),
        str(output_dir / "qwen3-fa-decoder.onnx"),
        input_names=["inputs_embeds", "position_ids"],
        output_names=["logits"],
        dynamic_shapes=dyn_shapes,
    )
    print("[FA]   qwen3-fa-decoder.onnx exported")

    # ── 2. Consolidate external data ─────────────────────────────────────
    _consolidate_external_data(output_dir, "qwen3-fa-decoder.onnx")

    # ── 3. INT4 quantize decoder ─────────────────────────────────────────
    # Reuse ASR embed_tokens for calibration (same weights, same vocabulary).
    asr_embed_path = asr_dir / "qwen3-asr-embed-tokens.onnx"
    if quantize:
        quantize_int4_nncf(
            output_dir / "qwen3-fa-decoder.onnx",
            asr_embed_path,
            model_id,
            num_layers,
            num_kv_heads,
            head_dim,
            fa_mode=True,
        )

    # ── 4. Cleanup ────────────────────────────────────────────────────────
    keep = {"qwen3-fa-decoder.onnx", "qwen3-fa-decoder.onnx_data", "fa_config.json"}
    for f in output_dir.iterdir():
        if f.suffix in (".data", ".onnx.data") and f.name not in keep:
            f.unlink()
        elif f.name.startswith("onnx__"):
            f.unlink()

    # ── 5. Write fa_config.json ───────────────────────────────────────────
    fa_config = {
        "timestamp_token_id": timestamp_token_id,
        "timestamp_segment_ms": timestamp_segment_ms,
        "timestamp_classes": timestamp_classes,
    }
    config_path = output_dir / "fa_config.json"
    config_path.write_text(json.dumps(fa_config, indent=2))
    print(f"[FA]   fa_config.json: {fa_config}")

    print(f"[FA] Export complete: {output_dir}")
    _list_files(output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 models to ONNX format"
    )
    parser.add_argument("--llm", action="store_true", help="Export Qwen3-0.6B LLM")
    parser.add_argument("--asr", action="store_true", help="Export Qwen3-ASR-0.6B")
    parser.add_argument(
        "--fa", action="store_true",
        help="Export Qwen3-ForcedAligner-0.6B FA decoder (requires --asr or existing ASR dir)",
    )
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
    parser.add_argument(
        "--fa-model", default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="HuggingFace model ID for the forced aligner",
    )
    parser.add_argument(
        "--asr-dir", type=Path, default=Path("models/qwen3-asr-0.6b-onnx"),
        help="Directory of exported ASR ONNX files (encoder/embed reused by FA)",
    )
    args = parser.parse_args()

    if not args.llm and not args.asr and not args.fa:
        parser.error("Specify --llm, --asr, --fa, or a combination")

    do_quant = not args.no_quantize

    if args.llm:
        out = args.output_dir or Path("models/qwen3-0.6b-onnx")
        export_llm(out, args.llm_model, quantize=do_quant)

    if args.asr:
        out = args.output_dir or Path("models/qwen3-asr-0.6b-onnx")
        export_asr(out, args.asr_model, quantize=do_quant)

    if args.fa:
        out = args.output_dir or Path("models/qwen3-fa-0.6b-onnx")
        export_forcedaligner(out, args.asr_dir, args.fa_model, quantize=do_quant)


if __name__ == "__main__":
    main()
