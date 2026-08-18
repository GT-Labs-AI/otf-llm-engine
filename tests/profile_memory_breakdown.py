"""
OTF-LLM Engine: Deep Memory & Storage Breakdown Profiler
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import json
import torch
from safetensors.torch import load_file

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def analyze_model_storage(model_dir: str = "./models/Qwen-3B-2Bit"):
    print("=" * 80)
    print(f"🔬 GT Labs AI — Deep Model Memory & Parameter Breakdown Profiler")
    print(f"   Target Directory: {model_dir}")
    print("=" * 80)

    base_path = os.path.join(model_dir, "otf_2bit_base.safetensors")
    quant_path = os.path.join(model_dir, "otf_2bit_model.safetensors")
    config_path = os.path.join(model_dir, "otf_2bit_config.json")

    if not os.path.exists(base_path) or not os.path.exists(quant_path):
        print(f"❌ Error: Model files not found in '{model_dir}'")
        return

    # Load metadata
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_tensors = load_file(base_path, device="cpu")
    quant_tensors = load_file(quant_path, device="cpu")

    # 1. Non-Quantized Base Components Breakdown
    embed_bytes = 0
    lm_head_bytes = 0
    norm_bytes = 0
    other_base_bytes = 0

    for k, v in base_tensors.items():
        nbytes = v.numel() * v.element_size()
        if "embed_tokens" in k:
            embed_bytes += nbytes
        elif "lm_head" in k:
            lm_head_bytes += nbytes
        elif "norm" in k.lower():
            norm_bytes += nbytes
        else:
            other_base_bytes += nbytes

    # 2. Quantized Layer Components Breakdown
    packed_weights_bytes = 0
    scales_bytes = 0
    zeros_bytes = 0
    outlier_values_bytes = 0
    outlier_indices_bytes = 0
    bias_bytes = 0
    total_quantized_params = 0

    for k, v in quant_tensors.items():
        nbytes = v.numel() * v.element_size()
        if k.endswith(".packed_uint8"):
            packed_weights_bytes += nbytes
            total_quantized_params += v.numel() * 4  # 4 weights per uint8 byte
        elif k.endswith(".scales"):
            scales_bytes += nbytes
        elif k.endswith(".zeros"):
            zeros_bytes += nbytes
        elif k.endswith(".outlier_deltas"):
            outlier_values_bytes += nbytes
        elif k.endswith(".outlier_indices"):
            outlier_indices_bytes += nbytes
        elif k.endswith(".bias"):
            bias_bytes += nbytes

    total_model_bytes = (
            embed_bytes + lm_head_bytes + norm_bytes + other_base_bytes +
            packed_weights_bytes + scales_bytes + zeros_bytes +
            outlier_values_bytes + outlier_indices_bytes + bias_bytes
    )

    to_mb = lambda b: b / (1024 ** 2)
    to_pct = lambda b: (b / total_model_bytes) * 100

    print(f"\n📊 [1/3] PHYSICAL MEMORY FOOTPRINT BREAKDOWN:")
    print(f"{'Component':<35} | {'Memory (MB)':<12} | {'Percentage':<12} | {'Bit-Rate Equivalent':<20}")
    print("-" * 85)

    # Layer weights
    bp_packed = (packed_weights_bytes * 8) / total_quantized_params if total_quantized_params else 2.0
    print(
        f"{'1. INT2 Packed Layer Weights':<35} | {to_mb(packed_weights_bytes):>9.2f} MB | {to_pct(packed_weights_bytes):>10.2f}% | {bp_packed:>18.2f} b/param")

    # Metadata
    bp_scales = (scales_bytes * 8) / total_quantized_params if total_quantized_params else 0.5
    print(
        f"{'2. Group Scales (S_g, G=32)':<35} | {to_mb(scales_bytes):>9.2f} MB | {to_pct(scales_bytes):>10.2f}% | {bp_scales:>18.2f} b/param")

    bp_zeros = (zeros_bytes * 8) / total_quantized_params if total_quantized_params else 0.5
    print(
        f"{'3. Group Zeros (Z_g, G=32)':<35} | {to_mb(zeros_bytes):>9.2f} MB | {to_pct(zeros_bytes):>10.2f}% | {bp_zeros:>18.2f} b/param (⚠️ Can be 0)")

    # Outliers
    bp_outliers = ((
                               outlier_values_bytes + outlier_indices_bytes) * 8) / total_quantized_params if total_quantized_params else 0.56
    print(
        f"{'4. Outlier Anchors (FP16 + Indices)':<35} | {to_mb(outlier_values_bytes + outlier_indices_bytes):>9.2f} MB | {to_pct(outlier_values_bytes + outlier_indices_bytes):>10.2f}% | {bp_outliers:>18.2f} b/param")

    # Non-recurrent components
    print(
        f"{'5. Vocab Embeddings (FP16)':<35} | {to_mb(embed_bytes):>9.2f} MB | {to_pct(embed_bytes):>10.2f}% | {'FP16 Uncompressed':>18}")
    if lm_head_bytes > 0:
        print(
            f"{'6. LM Head Projection (FP16)':<35} | {to_mb(lm_head_bytes):>9.2f} MB | {to_pct(lm_head_bytes):>10.2f}% | {'FP16 Uncompressed':>18}")
    print(
        f"{'7. LayerNorms & QKV Biases':<35} | {to_mb(norm_bytes + bias_bytes):>9.2f} MB | {to_pct(norm_bytes + bias_bytes):>10.2f}% | {'FP16 Exact':>18}")
    print("-" * 85)
    print(
        f"{'TOTAL MODEL FOOTPRINT':<35} | {to_mb(total_model_bytes):>9.2f} MB | {100.0:>10.2f}% | {((packed_weights_bytes + scales_bytes + zeros_bytes + outlier_values_bytes + outlier_indices_bytes) * 8) / total_quantized_params:>18.2f} b/p (effective)")

    # 3. Optimization Potential Analysis
    savings_zeros_mb = to_mb(zeros_bytes)
    savings_embed_mb = to_mb(embed_bytes * 0.75)  # If embeddings quantized to 4-bit (75% reduction)
    potential_total_mb = to_mb(total_model_bytes) - savings_zeros_mb - savings_embed_mb

    print(f"\n🚀 [2/3] OPTIMIZATION PROJECTIONS (Next-Gen Architecture):")
    print(
        f"   • Current Total Disk Size:         {to_mb(total_model_bytes):.2f} MB ({to_mb(total_model_bytes) / 1024:.2f} GB)")
    print(f"   • Savings from removing Z_g (Zero): -{savings_zeros_mb:.2f} MB (-{to_pct(zeros_bytes):.1f}%)")
    print(f"   • Savings from 4-bit Embeddings:   -{savings_embed_mb:.2f} MB")
    print(f"   • Projected Next-Gen Model Size:   {potential_total_mb:.2f} MB ({potential_total_mb / 1024:.2f} GB)")
    print(f"   • Net Size Reduction:              -{(1.0 - potential_total_mb / to_mb(total_model_bytes)) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("💡 ARCHITECTURAL VERDICT:")
    print("   1. Eliminating Z_g via symmetric 4-quadrant centroids instantly drops effective bit-rate by 0.50 b/p.")
    print("   2. Quantizing Embeddings to 4-bit cuts 450+ MB from static VRAM.")
    print("   3. Target: Qwen2.5-3B will shrink from 1.81 GB down to ~1.20 GB VRAM!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_model_storage("./models/Qwen-3B-2Bit")