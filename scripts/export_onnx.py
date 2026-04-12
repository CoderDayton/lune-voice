#!/usr/bin/env python3
"""Custom ONNX exporter for Qwen3-ASR-0.6B and the wav2vec2 forced aligner.

Bypasses optimum-cli entirely. Uses torch.onnx.export with manual wrappers
that correctly handle Qwen3-ASR's architecture (GQA, q_norm/k_norm, RoPE,
SwiGLU, tied embeddings) and produce clean FP16 graphs with proper KV cache
I/O.

Usage:
    python scripts/export_onnx.py --asr --output-dir models/qwen3-asr-0.6b-onnx
    python scripts/export_onnx.py --fa
    python scripts/export_onnx.py --asr --no-quantize

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
        # attention_mask: all-ones for B=1 calibration (no padding).
        attn_mask = np.ones((1, seq), dtype=np.int64)
        sample: dict = {
            "inputs_embeds": embeds,
            "position_ids": pos_ids,
            "attention_mask": attn_mask,
        }
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
        self.register_buffer(
            "pos_embed", at.positional_embedding.positional_embedding
        )
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
        self.window_aftercnn = self.chunk_aftercnn * (self.n_window_infer // self.chunk_size)
        self.max_encoder_frames = 6000
        self.max_aftercnn = self._feat_extract_output_lengths(
            torch.tensor(self.max_encoder_frames, dtype=torch.long)
        ).item()
        self.max_attention_windows = (self.max_aftercnn + self.window_aftercnn - 1) // self.window_aftercnn

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
        chunks = padded.reshape(1, mel.shape[1], num_chunks, chunk_size).permute(2, 0, 1, 3).squeeze(1)

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
        valid_mask = (
            torch.arange(t, device=mel.device).unsqueeze(0) < chunk_aftercnn_lengths.unsqueeze(1)
        )
        hidden = x[valid_mask]

        aftercnn_len = self._feat_extract_output_lengths(feature_len)[0]
        step_ends = (
            torch.arange(1, self.max_attention_windows + 1, device=mel.device, dtype=torch.long)
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
            torch.full((seq_len, total_len), torch.finfo(dtype).min, dtype=dtype,
                        device=attention_mask.device),
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
        inputs_embeds: torch.Tensor,       # [B, L, H]
        position_ids: torch.Tensor,         # [3, B, L] (mRoPE)
        attention_mask: torch.Tensor,       # [B, past_len + L]
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
            attention_mask, seq_len, total_len, inputs_embeds.dtype,
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
        dummy_past.append(torch.zeros(batch_size, num_kv_heads, past_len, head_dim, dtype=DTYPE))
        dummy_past.append(torch.zeros(batch_size, num_kv_heads, past_len, head_dim, dtype=DTYPE))

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


# ── FA Export (removed) ───────────────────────────────────────────────────────
# Forced aligner (wav2vec2 CTC) was removed — Parakeet v3 provides native
# word-level timestamps. Keeping this comment as a tombstone.


def _export_forcedaligner_removed(
    output_dir: Path,
    asr_dir: Path,
    model_id: str = "WAV2VEC2_ASR_BASE_960H",
    quantize: bool = True,
) -> None:
    """Export wav2vec2 CTC model for forced alignment to ONNX.

    Uses torchaudio's pretrained wav2vec2 model (94M params, 360MB fp32).
    Single ONNX file: raw 16kHz audio → CTC logits [1, T/320, 29].
    CTC Viterbi alignment is done in Rust at runtime.

    Writes `fa_config.json` with labels, blank_id, stride_ms.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FA] Exporting wav2vec2 CTC → {output_dir}")

    import torchaudio

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    labels = list(bundle.get_labels())
    model = bundle.get_model()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[FA] Model: WAV2VEC2_ASR_BASE_960H ({total_params:,} params)")
    print(f"[FA] Labels ({len(labels)}): {labels}")

    # ── 1. Export ONNX ───────────────────────────────────────────────────
    print("[FA]   Exporting model.onnx ...")
    dummy = torch.randn(1, 16000 * 5, dtype=torch.float32)
    seq_dim = torch.export.Dim("seq_len", min=1600, max=16000 * 60)

    torch.onnx.export(
        model,
        (dummy,),
        str(output_dir / "model.onnx"),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_shapes=({1: seq_dim},),
    )
    print("[FA]   model.onnx exported")

    # ── 2. Consolidate external data ─────────────────────────────────────
    _consolidate_external_data(output_dir, "model.onnx")

    # ── 3. Write fa_config.json ──────────────────────────────────────────
    fa_config = {
        "model_type": "wav2vec2_ctc",
        "sample_rate": 16000,
        "labels": labels,
        "blank_id": 0,
        "word_delimiter": "|",
        "stride_ms": 20,
    }
    config_path = output_dir / "fa_config.json"
    config_path.write_text(json.dumps(fa_config, indent=2))
    print(f"[FA]   fa_config.json written")

    # ── 4. Cleanup ────────────────────────────────────────────────────────
    keep = {"model.onnx", "model.onnx_data", "fa_config.json"}
    for f in output_dir.iterdir():
        if f.suffix in (".data", ".onnx.data") and f.name not in keep:
            f.unlink()
        elif f.name.startswith("onnx__"):
            f.unlink()

    print(f"[FA] Export complete: {output_dir}")
    _list_files(output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 models to ONNX format"
    )
    parser.add_argument("--asr", action="store_true", help="Export Qwen3-ASR-0.6B")
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Skip INT4 quantization (keep FP16 weights)",
    )
    parser.add_argument(
        "--fp16-decoder", action="store_true",
        help="Keep decoder as FP16 (skip INT4 quantization on decoder only). "
             "Encoder stays FP16 by default. Use for mixed-precision export "
             "where INT4 autoregressive decoding degrades quality.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: models/<model-name>-onnx)",
    )
    parser.add_argument(
        "--asr-model", default="Qwen/Qwen3-ASR-0.6B",
        help="HuggingFace model ID for ASR",
    )
    args = parser.parse_args()

    if not args.asr:
        parser.error("Specify --asr")

    do_quant = not args.no_quantize
    fp16_decoder = args.fp16_decoder

    if args.asr:
        out = args.output_dir or Path("models/qwen3-asr-0.6b-onnx")
        export_asr(out, args.asr_model, quantize=do_quant and not fp16_decoder)


if __name__ == "__main__":
    main()
