# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_llm/direct_quantized_importer.py

"""
[EXPERIMENTAL R&D MODULE] Direct Quantized Model Importer
========================================================
This module provides experimental support for converting pre-quantized AWQ/GPTQ models.
For production 100% quality parity (0% loss), use `otf_llm.convert_model()`.
"""

import os
import time
import math
import gc
import warnings
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

try:
    from safetensors.torch import safe_open, save_file as safe_save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from .convert_global_universal import QuantizedEmbedding, GlobalSymmetricINT4Linear


def import_prequantized_hf_model(model_id: str, outlier_pct: float = 0.01) -> str:
    """
    [EXPERIMENTAL] Directly imports pre-quantized 4-bit AWQ/GPTQ safetensors models from Hugging Face.
    For production 100% quality parity (0% loss), use `otf_llm.convert_model()`.
    """
    warnings.warn(
        "`import_prequantized_hf_model` is an experimental R&D feature. "
        "For production 100% quality parity (0% loss), use `otf_llm.convert_model()`.",
        UserWarning,
        stacklevel=2
    )

    clean_name = model_id.split("/")[-1].lower().replace("-", "_")
    save_path = f"otf_{clean_name}_compressed.safetensors"

    print("=" * 75)
    print(f"📦 [EXPERIMENTAL] DIRECT QUANTIZED IMPORTER (AWQ / GPTQ -> OTF ENGINE)")
    print(f"🎯 Target Model ID: {model_id}")
    print(f"💾 Download Size: ~3.8 GB (80% bandwidth savings vs FP16!)")
    print("=" * 75)

    t0 = time.time()
    print("\n📥 Downloading pre-quantized safetensors shards (~3.8 GB)...")
    model_dir = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "*.json"])

    safetensors_files = sorted([
        os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")
    ])

    if not safetensors_files:
        raise FileNotFoundError("No .safetensors files found in pre-quantized model repository!")

    print("\n⚙️ Reconstructing AutoAWQ interleaved weights -> OTF Engine...")
    otf_state_dict = {}
    group_size = 64

    for sf_file in safetensors_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Process Linear Layers via AWQ qweight
                if key.endswith(".qweight"):
                    prefix = key.replace(".qweight", "")
                    qweight = f.get_tensor(key)

                    scale_key = f"{prefix}.scales"
                    scales = f.get_tensor(scale_key) if scale_key in f.keys() else None

                    qzeros_key = f"{prefix}.qzeros"
                    qzeros = f.get_tensor(qzeros_key) if qzeros_key in f.keys() else None

                    in_dim = qweight.shape[0] * 8 if qweight.shape[0] < qweight.shape[1] else qweight.shape[0]
                    out_dim = qweight.shape[1] if qweight.shape[0] < qweight.shape[1] else qweight.shape[1] * 8

                    qw = qweight.to(torch.int32)

                    # 1. Unpack AWQ interleaved bit-shifts [0, 16, 4, 20, 8, 24, 12, 28]
                    w0 = (qw >> 0) & 0x0F
                    w1 = (qw >> 16) & 0x0F
                    w2 = (qw >> 4) & 0x0F
                    w3 = (qw >> 20) & 0x0F
                    w4 = (qw >> 8) & 0x0F
                    w5 = (qw >> 24) & 0x0F
                    w6 = (qw >> 12) & 0x0F
                    w7 = (qw >> 28) & 0x0F

                    stacked_w = torch.stack([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)
                    unpacked_qweight = stacked_w.view(in_dim, out_dim).float()

                    # 2. Unpack AWQ interleaved qzeros
                    if qzeros is not None:
                        qz = qzeros.to(torch.int32)
                        qz0 = (qz >> 0) & 0x0F
                        qz1 = (qz >> 16) & 0x0F
                        qz2 = (qz >> 4) & 0x0F
                        qz3 = (qz >> 20) & 0x0F
                        qz4 = (qz >> 8) & 0x0F
                        qz5 = (qz >> 24) & 0x0F
                        qz6 = (qz >> 12) & 0x0F
                        qz7 = (qz >> 28) & 0x0F

                        stacked_qz = torch.stack([qz0, qz1, qz2, qz3, qz4, qz5, qz6, qz7], dim=2)
                        unpacked_qzeros = stacked_qz.view(scales.shape[0], out_dim).float()
                    else:
                        unpacked_qzeros = torch.full((scales.shape[0], out_dim), 8.0, dtype=torch.float32)

                    # 3. Dynamic group size calculation
                    awq_groups = scales.shape[0]
                    awq_group_size = in_dim // awq_groups if awq_groups > 0 else 128

                    scales_exp = scales.T.repeat_interleave(awq_group_size, dim=1).float()
                    qzeros_exp = unpacked_qzeros.T.repeat_interleave(awq_group_size, dim=1).float()

                    # 4. Calculate true FP16 matrix W_fp16
                    W_fp16 = ((unpacked_qweight.T - qzeros_exp) * scales_exp).half().contiguous()

                    # 5. Convert W_fp16 into OTF Symmetric INT4 Zero-Point Elimination format
                    num_k = math.ceil(int(in_dim * outlier_pct) / group_size) * group_size
                    num_k = max(num_k, group_size)

                    perm_idx = torch.arange(in_dim, dtype=torch.int32)
                    wrapped = GlobalSymmetricINT4Linear(in_dim, out_dim, bias=False, group_size=group_size)
                    wrapped.quantize_direct_weight(W_fp16, perm_idx, num_k)

                    otf_state_dict[f"{prefix}.perm_idx"] = wrapped.perm_idx
                    otf_state_dict[f"{prefix}.W_outliers_fp16"] = wrapped.W_outliers_fp16
                    otf_state_dict[f"{prefix}.packed_q_bg"] = wrapped.packed_q_bg
                    otf_state_dict[f"{prefix}.scale_bg"] = wrapped.scale_bg

                    del qweight, scales, qzeros, qw, w0, w1, w2, w3, w4, w5, w6, w7, stacked_w, unpacked_qweight, W_fp16, wrapped
                    gc.collect()

                # Process Embeddings
                elif "embed_tokens.weight" in key:
                    tensor = f.get_tensor(key)
                    emb_quant = QuantizedEmbedding(tensor.shape[0], tensor.shape[1], group_size=64,
                                                   weight_tensor=tensor)
                    otf_state_dict["model.embed_tokens.packed_q"] = emb_quant.packed_q
                    otf_state_dict["model.embed_tokens.scale"] = emb_quant.scale
                    del tensor, emb_quant
                    gc.collect()

                # Keep Norms and Biases in FP16
                elif (not key.endswith(".weight") or "norm" in key) and not key.endswith(
                        ".scales") and not key.endswith(".qzeros") and not key.endswith(".g_idx"):
                    tensor = f.get_tensor(key)
                    otf_state_dict[key] = tensor.half().cpu()
                    del tensor
                    gc.collect()

    print(f"\n💾 Saving converted OTF safetensors checkpoint to: {save_path}...")
    safe_save_file(otf_state_dict, save_path)
    del otf_state_dict
    gc.collect()

    elapsed = time.time() - t0
    file_size_gb = os.path.getsize(save_path) / (1024 ** 3)
    print(f"⚡ SUCCESS! Pre-quantized model imported & converted to {file_size_gb:.2f} GB in {elapsed:.2f} sec!\n")
    return save_path


if __name__ == "__main__":
    import_prequantized_hf_model("Qwen/Qwen2.5-7B-Instruct-AWQ")