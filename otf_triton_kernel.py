# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & GlebTikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_triton_kernel.py
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_int4_gemm_kernel(
        # Указатели на тензоры в VRAM
        a_ptr, b_ptr, scales_ptr, c_ptr,
        # Размеры
        M, N, K,
        group_size: tl.constexpr,
        # Strides (Шаги памяти)
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_scales_n, stride_scales_g,
        stride_cm, stride_cn,
        # Размер тайлов (блоков)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused INT4 GEMM Kernel для Triton.
    Распаковывает uint8 веса прямиком в регистрах GPU и умножает на активации.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Указатели на блоки
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 2) * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Цикл по блокам K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_indices = k * BLOCK_SIZE_K + offs_k
        mask_k = k_indices < K

        # 1. Загрузка блока активаций
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float16)

        # 2. Загрузка упакованных 4-битных весов (uint8)
        b_packed = tl.load(b_ptrs, mask=mask_k[None, :], other=0)

        # 3. РАСПАКОВКА В РЕГИСТРАХ GPU
        is_odd = (k_indices % 2) == 1
        w_int4 = tl.where(is_odd[None, :], (b_packed >> 4) & 0x0F, b_packed & 0x0F)
        w_fp = (w_int4.to(tl.float32) - 8.0)

        # 4. Загрузка масштабов групп
        group_idx = k_indices // group_size
        scale_ptrs = scales_ptr + (offs_bn[:, None] * stride_scales_n + group_idx[None, :] * stride_scales_g)
        scale = tl.load(scale_ptrs).to(tl.float32)

        # 5. Деквантование и перевод в float16 для Tensor Cores
        w_dequant = (w_fp * scale).to(tl.float16)

        # 6. Умножение матриц на Tensor Cores
        accumulator += tl.dot(a, tl.trans(w_dequant))

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_fused_int4_linear(x: torch.Tensor, packed_q_2d: torch.Tensor, scale_2d: torch.Tensor,
                             group_size: int = 64) -> torch.Tensor:
    """
    Python wrapper для запуска Triton GEMM ядра.
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])

    M, K = x_2d.shape
    N = packed_q_2d.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    _fused_int4_gemm_kernel[grid](
        x_2d, packed_q_2d, scale_2d, out,
        M, N, K,
        group_size=group_size,
        stride_am=x_2d.stride(0), stride_ak=x_2d.stride(1),
        stride_bn=packed_q_2d.stride(0), stride_bk=packed_q_2d.stride(1),
        stride_scales_n=scale_2d.stride(0), stride_scales_g=scale_2d.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
    )

    return out.view(*orig_shape[:-1], N)