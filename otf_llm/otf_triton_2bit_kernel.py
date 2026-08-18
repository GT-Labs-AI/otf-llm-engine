"""
OTF-LLM Engine: High-Performance Stream-Safe CUDA C++ Fused 2-Bit GEMV Engine
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

from typing import Optional
import torch

HAS_TRITON = True
LLOYD_MAX_SYMMETRIC_CENTROIDS = [-1.52, -0.45, 0.45, 1.52]

CUDA_GEMV_MODULE = None

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sym2bit_gemv_warp_kernel(
    const half* __restrict__ x,           // [K]
    const uint8_t* __restrict__ packed_w, // [N, stride_w]
    const half* __restrict__ scales,      // [N, stride_s]
    half* __restrict__ out,               // [N]
    const int N,
    const int K,
    const int stride_w,
    const int stride_s
) {
    const float c_lut[4] = {-1.52f, -0.45f, 0.45f, 1.52f};

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int row = blockIdx.x * blockDim.y + warp_id;

    if (row >= N) return;

    const int total_bytes = K / 4;
    const uint8_t* row_w = packed_w + (row * stride_w);
    const half* row_scales = scales + (row * stride_s);

    float sum = 0.0f;

    for (int k_byte = lane_id; k_byte < total_bytes; k_byte += 32) {
        const uint8_t b = row_w[k_byte];
        const int g_idx = k_byte / 8; // 8 bytes = 32 weights = 1 scale
        const float s = __half2float(row_scales[g_idx]);

        const int k_base = k_byte * 4;
        const float x0 = __half2float(x[k_base + 0]);
        const float x1 = __half2float(x[k_base + 1]);
        const float x2 = __half2float(x[k_base + 2]);
        const float x3 = __half2float(x[k_base + 3]);

        const int q0 = (b & 0x03);
        const int q1 = ((b >> 2) & 0x03);
        const int q2 = ((b >> 4) & 0x03);
        const int q3 = ((b >> 6) & 0x03);

        const float dot4 = (c_lut[q0] * x0) +
                           (c_lut[q1] * x1) +
                           (c_lut[q2] * x2) +
                           (c_lut[q3] * x3);

        sum += dot4 * s;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        sum = fminf(fmaxf(sum, -65504.0f), 65504.0f);
        out[row] = __float2half(sum);
    }
}

torch::Tensor sym2bit_gemv_cuda(
    torch::Tensor x,           // [1, K] fp16
    torch::Tensor packed_w,    // [N, K / 4] uint8
    torch::Tensor scales       // [N, K / 32] fp16
) {
    auto x_contig = x.contiguous();
    auto w_contig = packed_w.contiguous();
    auto s_contig = scales.contiguous();

    const int N = w_contig.size(0);
    const int K = x_contig.size(1);

    auto out = torch::empty({1, N}, x.options());

    dim3 block(32, 4);
    dim3 grid((N + block.y - 1) / block.y);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    sym2bit_gemv_warp_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(x_contig.data_ptr<at::Half>()),
        reinterpret_cast<const uint8_t*>(w_contig.data_ptr<uint8_t>()),
        reinterpret_cast<const half*>(s_contig.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        N,
        K,
        w_contig.stride(0),
        s_contig.stride(0)
    );

    return out;
}
"""

CPP_SRC = """
#include <torch/extension.h>
torch::Tensor sym2bit_gemv_cuda(torch::Tensor x, torch::Tensor packed_w, torch::Tensor scales);
"""


def _load_cuda_extension():
    global CUDA_GEMV_MODULE
    if CUDA_GEMV_MODULE is not None:
        return CUDA_GEMV_MODULE

    try:
        from torch.utils.cpp_extension import load_inline
        print("⚡ Compiling stream-safe CUDA Fused GEMV Warp Kernel...", flush=True)
        CUDA_GEMV_MODULE = load_inline(
            name="otf_cuda_sym2bit_gemv_core",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["sym2bit_gemv_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False
        )
        print("✅ Native CUDA 2-Bit Warp GEMV ready!", flush=True)
        return CUDA_GEMV_MODULE
    except Exception as e:
        print(f"⚠️ CUDA JIT compiler unavailable ({e}). Falling back to 256-Byte LUT executor.", flush=True)
        CUDA_GEMV_MODULE = False
        return False


_GLOBAL_BYTE_LUT: Optional[torch.Tensor] = None


def _get_byte_lut(device: torch.device) -> torch.Tensor:
    global _GLOBAL_BYTE_LUT
    if _GLOBAL_BYTE_LUT is None or _GLOBAL_BYTE_LUT.device != device:
        centroids = torch.tensor(LLOYD_MAX_SYMMETRIC_CENTROIDS, dtype=torch.float16)
        lut = torch.empty((256, 4), dtype=torch.float16)
        for b in range(256):
            q0 = b & 0x03
            q1 = (b >> 2) & 0x03
            q2 = (b >> 4) & 0x03
            q3 = (b >> 6) & 0x03
            lut[b] = torch.tensor([centroids[q0], centroids[q1], centroids[q2], centroids[q3]], dtype=torch.float16)
        _GLOBAL_BYTE_LUT = lut.to(device=device)
    return _GLOBAL_BYTE_LUT


def triton_symmetric_2bit_gemm(
    x: torch.Tensor,
    packed_uint8: torch.Tensor,
    scales: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    group_size: int = 32,
    outlier_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).to(dtype=torch.float16)
    M = x_2d.shape[0]
    N = packed_uint8.shape[0]
    d_in = packed_uint8.shape[1] * 4

    if outlier_mask is not None:
        x_2d = x_2d * outlier_mask

    # 🚀 1. Нативный CUDA Warp-Reduction Kernel для M=1 decode (21+ tok/s)
    if M == 1 and x.is_cuda:
        cuda_mod = _load_cuda_extension()
        if cuda_mod:
            out = cuda_mod.sym2bit_gemv_cuda(x_2d, packed_uint8, scales)
            return out.view(*orig_shape[:-1], N)

    # 2. Векторизованный LUT для Prefill (M > 1)
    lut = _get_byte_lut(x_2d.device)
    p_long = packed_uint8.to(torch.long)
    centroids_4d = lut[p_long]
    w_dequant = (centroids_4d.view(N, -1, group_size) * scales.unsqueeze(-1)).view(N, d_in)
    y = torch.matmul(x_2d, w_dequant.T)
    return torch.clamp(y, min=-65504.0, max=65504.0).view(*orig_shape[:-1], N)


def triton_2bit_gemm(
    x: torch.Tensor,
    packed_uint8: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    codebook: Optional[torch.Tensor] = None,
    group_size: int = 32,
    outlier_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return triton_symmetric_2bit_gemm(
        x=x,
        packed_uint8=packed_uint8,
        scales=scales,
        codebook=codebook,
        group_size=group_size,
        outlier_mask=outlier_mask
    )