"""
OTF-LLM Engine v4.0 - Triton INT2 GEMM Kernel Parity Test
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy. MIT License.

Compares outputs of PyTorch Reference Unpacker vs Custom Fused Triton INT2 Kernel.
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from otf_llm.otf_2bit_quantizer import OTF2BitQuantizer
from otf_llm.otf_triton_2bit_kernel import triton_2bit_gemm, HAS_TRITON


def test_triton_parity():
    print("=" * 70)
    print("🚀 OTF-LLM v4.0: Triton INT2 GEMM Kernel Parity Test")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" or not HAS_TRITON:
        print("❌ CUDA GPU and Triton package are required for this test!")
        return

    M, N, K = 1, 2048, 2048  # Single-token decoding shape
    group_size = 32

    torch.manual_seed(42)
    W = torch.randn(N, K, device=device, dtype=torch.float16) * 0.02
    X = torch.randn(1, M, K, device=device, dtype=torch.float16)

    quantizer = OTF2BitQuantizer(group_size=group_size, outlier_ratio=0.0)
    packed_uint8, scales, zeros, _, _ = quantizer.quantize_tensor(W)

    # 1. PyTorch Reference Output
    w_dequant = quantizer.unpack_2bit_uint8(packed_uint8, scales, zeros, K)
    Y_ref = F.linear(X, w_dequant)

    # 2. Triton Kernel Output
    Y_triton = triton_2bit_gemm(
        x=X,
        packed_uint8=packed_uint8,
        scales=scales,
        zeros=zeros,
        codebook=quantizer.codebook,
        group_size=group_size
    )

    cos_sim = F.cosine_similarity(Y_ref.reshape(-1), Y_triton.reshape(-1), dim=0).item() * 100
    mse = F.mse_loss(Y_ref, Y_triton).item()

    print(f"📊 TRITON KERNEL PARITY RESULTS:")
    print(f"   • Logit Cosine Parity vs PyTorch Reference: {cos_sim:.4f}%")
    print(f"   • MSE Reconstruction Error:               {mse:.8f}")
    print("=" * 70)


if __name__ == "__main__":
    test_triton_parity()