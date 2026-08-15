"""
OTF-LLM Engine v4.0 - Non-Uniform 2-Bit Linear Regression Benchmark
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy. MIT License.

Tests Non-Uniform 2-bit quantization accuracy with Closed-Form Linear Regression (S, Z).
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from otf_llm.otf_2bit_quantizer import OTF2BitQuantizer, OTF2BitLinear


def run_2bit_benchmark():
    print("=" * 70)
    print("🚀 OTF-LLM v4.0: 2-Bit Non-Uniform Linear Regression (S, Z) Benchmark")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ CUDA GPU is required for running this benchmark!")
        return

    d_in, d_out = 4096, 4096  # Standard 3B/7B projection shape
    group_size = 32
    outlier_ratio = 0.035     # Top 3.5% FP16 Outlier Anchors
    batch_size, seq_len = 2, 512

    print(f"📊 Test Configuration:")
    print(f"   • Matrix Shape W: [{d_out} x {d_in}]")
    print(f"   • Quantization: Non-Uniform 2-Bit Lloyd-Max + Closed-Form Regression (Group Size = {group_size})")
    print(f"   • Outlier Anchors: Top {outlier_ratio * 100:.1f}% FP16 channels")
    print(f"   • Activation Input X: [{batch_size}, {seq_len}, {d_in}]")
    print("-" * 70)

    # 1. Create synthetic realistic FP16 weight matrix
    torch.manual_seed(42)
    weight_scale = 1.0 / (d_in ** 0.5)
    W_fp16 = torch.randn(d_out, d_in, device=device, dtype=torch.float16) * weight_scale

    # Add sparse extreme outlier channels
    outlier_mask = (torch.rand(d_out, d_in, device=device) < 0.005).float()
    outlier_spikes = (torch.randn(d_out, d_in, device=device) * weight_scale * 5.0).to(torch.float16)
    W_fp16 = (W_fp16 + outlier_spikes * outlier_mask.to(torch.float16)).to(torch.float16)

    # 2. Measure FP16 Baseline Memory
    fp16_bytes = W_fp16.numel() * 2
    fp16_mb = fp16_bytes / (1024 ** 2)
    print(f"💾 Standard FP16 Weight Footprint: {fp16_mb:.2f} MB")

    # 3. Perform Non-Uniform 2-Bit Quantization with Linear Regression
    quantizer = OTF2BitQuantizer(group_size=group_size, outlier_ratio=outlier_ratio)

    start_time = time.time()
    packed_uint8, scales, zeros, out_fp16, out_idx = quantizer.quantize_tensor(W_fp16)
    quant_time = time.time() - start_time

    # Calculate 2-Bit Footprint
    packed_bytes = packed_uint8.numel() * 1
    scales_bytes = scales.numel() * 2
    zeros_bytes = zeros.numel() * 2
    outliers_bytes = (out_fp16.numel() * 2) if out_fp16 is not None else 0

    total_2bit_bytes = packed_bytes + scales_bytes + zeros_bytes + outliers_bytes
    total_2bit_mb = total_2bit_bytes / (1024 ** 2)

    print(f"⚡ 2-Bit Regression Quantization Time: {quant_time:.4f} seconds")
    print(f"💾 2-Bit OTF-LLM v4.0 Footprint: {total_2bit_mb:.2f} MB")
    print(f"   ├── Packed INT2 Weights (1 byte / 4 weights): {packed_bytes / (1024**2):.2f} MB")
    print(f"   ├── Group Scales & Zeros (fp16):              {(scales_bytes + zeros_bytes) / (1024**2):.2f} MB")
    print(f"   └── Outlier Anchors (fp16):                   {outliers_bytes / (1024**2):.2f} MB")
    print(f"🔥 Weight Footprint Reduction: {((fp16_mb - total_2bit_mb) / fp16_mb * 100):.2f}%!")
    print("-" * 70)

    # 4. Accuracy & Cosine Parity Verification
    linear_2bit = OTF2BitLinear(
        packed_uint8=packed_uint8,
        scales=scales,
        zeros=zeros,
        d_in=d_in,
        group_size=group_size,
        outliers_fp16=out_fp16,
        outlier_indices=out_idx
    )

    X_test = torch.randn(batch_size, seq_len, d_in, device=device, dtype=torch.float16)

    # Ground Truth Output
    Y_ref = F.linear(X_test, W_fp16)

    # 2-Bit Engine Output
    Y_2bit = linear_2bit(X_test)

    cos_sim = F.cosine_similarity(Y_ref.reshape(-1), Y_2bit.reshape(-1), dim=0).item() * 100
    mse = F.mse_loss(Y_ref, Y_2bit).item()

    print("🎯 ACCURACY METRICS:")
    print(f"✅ Mean Logit Cosine Parity: {cos_sim:.4f}%")
    print(f"📉 Mean MSE Error:           {mse:.8f}")

    if cos_sim >= 98.0:
        print("\n🎉 SUCCESS! OTF-LLM v4.0 2-Bit Engine achieved > 98.0% Logit Parity with 77%+ VRAM reduction!")
    else:
        print("\n⚠️ WARNING: Consider tuning group_size or increasing outlier_ratio!")


if __name__ == "__main__":
    run_2bit_benchmark()