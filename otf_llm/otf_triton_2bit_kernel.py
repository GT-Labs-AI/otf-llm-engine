"""
OTF-LLM Engine v4.0 - Custom Fused OpenAI Triton INT2 GEMM Kernel
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy. MIT License.

Unpacks 4 2-bit weight indices per uint8 byte in GPU SRAM registers,
applies non-uniform codebook centroids, scales, zeros, and outlier channel masking.
Includes PyTorch C++ optimized fallback wrapper for universal cross-platform support.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _triton_2bit_gemm_kernel(
        x_ptr, packed_w_ptr, scales_ptr, zeros_ptr, codebook_ptr, outlier_mask_ptr, out_ptr,
        M, N, K,
        has_outlier_mask: tl.constexpr,
        group_size: tl.constexpr,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sg,
        stride_zn, stride_zg,
        stride_om, stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr
    ):
        """
        Fused 2-bit unpack + Outlier Masking + GEMM kernel executing inside GPU SRAM.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Load codebook centroids into SRAM registers in FP16 precision
        cb0 = tl.load(codebook_ptr + 0).to(tl.float16)
        cb1 = tl.load(codebook_ptr + 1).to(tl.float16)
        cb2 = tl.load(codebook_ptr + 2).to(tl.float16)
        cb3 = tl.load(codebook_ptr + 3).to(tl.float16)

        # Iterate over K dimension in chunks of BLOCK_SIZE_K
        for k_offset in range(0, K, BLOCK_SIZE_K):
            rk = k_offset + tl.arange(0, BLOCK_SIZE_K)

            # Load activation chunk X: [BLOCK_SIZE_M, BLOCK_SIZE_K]
            x_mask = (rm[:, None] < M) & (rk[None, :] < K)
            x_tile = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk, mask=x_mask, other=0.0)

            # Load packed 2-bit uint8 weights: [BLOCK_SIZE_N, BLOCK_SIZE_K // 4]
            rk_packed = rk // 4
            w_mask = (rn[:, None] < N) & (rk_packed[None, :] < (K // 4))
            packed_tile = tl.load(packed_w_ptr + rn[:, None] * stride_wn + rk_packed[None, :] * stride_wk, mask=w_mask, other=0)

            # Extract 2-bit index shift for current element
            bit_shift = (rk[None, :] % 4) * 2
            q_idx = (packed_tile >> bit_shift) & 0x03

            # Load group scale and zero
            g_idx = rk // group_size
            s_mask = (rn[:, None] < N) & (g_idx[None, :] < (K // group_size))
            scale_tile = tl.load(scales_ptr + rn[:, None] * stride_sn + g_idx[None, :] * stride_sg, mask=s_mask, other=1.0)
            zero_tile = tl.load(zeros_ptr + rn[:, None] * stride_zn + g_idx[None, :] * stride_zg, mask=s_mask, other=0.0)

            # Map 2-bit indices to centroids
            c_val = tl.where(q_idx == 0, cb0, tl.where(q_idx == 1, cb1, tl.where(q_idx == 2, cb2, cb3)))

            # Dequantize weight in SRAM registers: W = scale * centroid + zero
            w_tile = scale_tile * c_val + zero_tile

            # Zero out outlier columns in Triton if outlier mask is present
            if has_outlier_mask:
                out_m = tl.load(outlier_mask_ptr + rk[None, :], mask=(rk[None, :] < K), other=1.0)
                w_tile = w_tile * out_m

            # Matrix Multiply Accumulate: acc += X @ W^T
            acc += tl.dot(x_tile, tl.trans(w_tile))

        # Write result back to global VRAM
        out_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on, acc.to(tl.float16), mask=out_mask)


def triton_2bit_gemm(
    x: torch.Tensor,
    packed_uint8: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 32,
    outlier_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Python wrapper executing Fused Triton INT2 GEMM on CUDA with Outlier Masking.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).to(torch.float16)
    M, K = x_2d.shape
    N = packed_uint8.shape[0]

    device = x.device
    packed_uint8 = packed_uint8.to(device)
    scales = scales.to(device)
    zeros = zeros.to(device)
    codebook = codebook.to(device=device, dtype=torch.float32)

    has_mask = outlier_mask is not None
    mask_ptr = outlier_mask.to(device=device, dtype=torch.float16) if has_mask else packed_uint8

    # Fallback to PyTorch unpacked evaluation if Triton is absent
    if not HAS_TRITON or not x.is_cuda:
        codebook_fp16 = codebook.to(dtype=torch.float16)

        q0 = (packed_uint8 & 0x03)
        q1 = ((packed_uint8 >> 2) & 0x03)
        q2 = ((packed_uint8 >> 4) & 0x03)
        q3 = ((packed_uint8 >> 6) & 0x03)

        q_unpacked = torch.stack([q0, q1, q2, q3], dim=-1).view(N, K).to(torch.long)
        w_centroids = codebook_fp16[q_unpacked]
        w_groups = w_centroids.view(N, -1, group_size)

        w_dequant = (scales.unsqueeze(-1) * w_groups + zeros.unsqueeze(-1)).view(N, K).to(torch.float16)

        if has_mask:
            w_dequant = w_dequant * outlier_mask.unsqueeze(0)

        out_2d = F.linear(x_2d, w_dequant)
        return out_2d.view(*orig_shape[:-1], N)

    # Launch Triton Kernel
    out_2d = torch.empty((M, N), device=device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )

    _triton_2bit_gemm_kernel[grid](
        x_2d, packed_uint8, scales, zeros, codebook, mask_ptr, out_2d,
        M, N, K,
        has_outlier_mask=has_mask,
        group_size=group_size,
        stride_xm=x_2d.stride(0), stride_xk=x_2d.stride(1),
        stride_wn=packed_uint8.stride(0), stride_wk=packed_uint8.stride(1),
        stride_sn=scales.stride(0), stride_sg=scales.stride(1),
        stride_zn=zeros.stride(0), stride_zg=zeros.stride(1),
        stride_om=out_2d.stride(0), stride_on=out_2d.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32
    )

    return out_2d.view(*orig_shape[:-1], N)