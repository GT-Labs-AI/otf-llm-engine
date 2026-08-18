"""
OTF-LLM Engine: Clean Symmetric Zero-Free 2-Bit + INT8 Embedding Engine
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from otf_llm.otf_triton_2bit_kernel import triton_symmetric_2bit_gemm

LLOYD_MAX_SYMMETRIC_CENTROIDS = [-1.52, -0.45, 0.45, 1.52]


class QuantizedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight_int8", torch.zeros((num_embeddings, embedding_dim), dtype=torch.int8))
        self.register_buffer("scales", torch.ones((num_embeddings, 1), dtype=torch.float16))

    @torch.no_grad()
    def quantize_from_fp16(self, fp16_weight: torch.Tensor, chunk_size: int = 8192):
        num_emb, dim = fp16_weight.shape
        self.weight_int8 = torch.empty((num_emb, dim), dtype=torch.int8, device="cpu")
        self.scales = torch.empty((num_emb, 1), dtype=torch.float16, device="cpu")

        for start in range(0, num_emb, chunk_size):
            end = min(start + chunk_size, num_emb)
            w_chunk = fp16_weight[start:end].to(dtype=torch.float32)
            max_vals = torch.max(torch.abs(w_chunk), dim=1, keepdim=True).values
            scales = torch.clamp(max_vals / 127.0, min=1e-8)
            w_quant = torch.clamp(torch.round(w_chunk / scales), -127, 127).to(torch.int8)

            self.weight_int8[start:end] = w_quant.contiguous()
            self.scales[start:end] = scales.to(dtype=torch.float16).contiguous()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        dev = input_ids.device
        w = self.weight_int8.to(device=dev)
        s = self.scales.to(device=dev)
        int8_lookup = F.embedding(input_ids, w)
        scale_lookup = F.embedding(input_ids, s)
        return (int8_lookup.to(dtype=torch.float16) * scale_lookup).to(dtype=torch.float16)


class QuantizedLinearHead(nn.Module):
    """Zero-VRAM Chunked INT8 LM Head."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones((out_features, 1), dtype=torch.float16))

    @torch.no_grad()
    def quantize_from_fp16(self, fp16_weight: torch.Tensor, chunk_size: int = 8192):
        num_emb, dim = fp16_weight.shape
        self.weight_int8 = torch.empty((num_emb, dim), dtype=torch.int8, device="cpu")
        self.scales = torch.empty((num_emb, 1), dtype=torch.float16, device="cpu")

        for start in range(0, num_emb, chunk_size):
            end = min(start + chunk_size, num_emb)
            w_chunk = fp16_weight[start:end].to(dtype=torch.float32)
            max_vals = torch.max(torch.abs(w_chunk), dim=1, keepdim=True).values
            scales = torch.clamp(max_vals / 127.0, min=1e-8)
            w_quant = torch.clamp(torch.round(w_chunk / scales), -127, 127).to(torch.int8)

            self.weight_int8[start:end] = w_quant.contiguous()
            self.scales[start:end] = scales.to(dtype=torch.float16).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).to(dtype=torch.float16)
        dev = x.device
        M = x_2d.shape[0]

        CHUNK_V = 16384 if M <= 4 else self.out_features
        logits = torch.empty((M, self.out_features), dtype=torch.float32, device=dev)

        for start in range(0, self.out_features, CHUNK_V):
            end = min(start + CHUNK_V, self.out_features)
            w_chunk = self.weight_int8[start:end].to(dtype=torch.float16)
            s_chunk = self.scales[start:end].to(dtype=torch.float16).squeeze(-1)
            logits[:, start:end] = (torch.matmul(x_2d, w_chunk.T) * s_chunk).to(torch.float32)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return logits.view(*orig_shape[:-1], self.out_features)


class Symmetric2BitQuantizer:
    def __init__(self, group_size: int = 32, outlier_ratio: float = 0.035):
        self.group_size = group_size
        self.outlier_ratio = outlier_ratio
        self.codebook = torch.tensor(LLOYD_MAX_SYMMETRIC_CENTROIDS, dtype=torch.float32)

    @torch.no_grad()
    def pack_2bit_uint8(self, q_indices: torch.Tensor) -> torch.Tensor:
        q = q_indices.to(torch.uint8)
        q0 = q[:, 0::4]
        q1 = q[:, 1::4]
        q2 = q[:, 2::4]
        q3 = q[:, 3::4]
        return (q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)).contiguous()

    @torch.no_grad()
    def quantize_tensor(
        self,
        W: torch.Tensor,
        activation_profile: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        dev = W.device
        W_f32 = W.detach().to(dtype=torch.float32)
        d_out, d_in = W_f32.shape
        codebook_dev = self.codebook.to(device=dev)

        outliers_fp16 = None
        outlier_indices = None
        mask_active = torch.ones((1, d_in), dtype=torch.bool, device=dev)

        # 1. Profile-Guided Outlier Channel Isolation
        if self.outlier_ratio > 0.0:
            if activation_profile is not None:
                act = activation_profile.detach().to(device=dev, dtype=torch.float32)
                col_impact = torch.norm(W_f32, dim=0) * act
            else:
                col_impact = torch.norm(W_f32, dim=0)

            num_outliers = max(1, int(d_in * self.outlier_ratio))
            outlier_indices = torch.topk(col_impact, k=num_outliers).indices
            outliers_fp16 = W_f32[:, outlier_indices].to(torch.float16).contiguous()

            W_f32 = W_f32.clone()
            W_f32[:, outlier_indices] = 0.0
            mask_active[:, outlier_indices] = False

        # 2. Stable Masked Group OLS Optimization
        num_groups = d_in // self.group_size
        W_groups = W_f32.view(d_out, num_groups, self.group_size)
        mask_groups = mask_active.view(1, num_groups, self.group_size).expand(d_out, -1, -1)

        active_counts = torch.clamp(torch.sum(mask_groups.float(), dim=-1), min=1.0)
        group_rms = torch.sqrt(torch.sum(W_groups ** 2, dim=-1) / active_counts + 1e-8)
        scales = torch.clamp(group_rms / 0.95, min=1e-8)

        for _ in range(5):
            scales_exp = scales.unsqueeze(-1)
            W_norm = W_groups / torch.clamp(scales_exp, min=1e-8)
            dists = torch.abs(W_norm.unsqueeze(-1) - codebook_dev)
            q_groups = torch.argmin(dists, dim=-1)

            C_q = codebook_dev[q_groups] * mask_groups.float()
            dot_wc = torch.sum(W_groups * C_q, dim=-1)
            dot_cc = torch.sum(C_q * C_q, dim=-1)

            scales = torch.clamp(dot_wc / torch.clamp(dot_cc, min=1e-8), min=1e-8)

        q_indices = q_groups.view(d_out, d_in).to(torch.uint8)
        packed_uint8 = self.pack_2bit_uint8(q_indices)

        return (
            packed_uint8.contiguous(),
            scales.to(torch.float16).contiguous(),
            outliers_fp16,
            outlier_indices
        )


class SymmetricOTF2BitLinear(nn.Module):
    def __init__(
        self,
        packed_uint8: torch.Tensor,
        scales: torch.Tensor,
        d_in: int,
        group_size: int = 32,
        outliers_fp16: Optional[torch.Tensor] = None,
        outlier_indices: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.d_in = d_in
        self.group_size = group_size
        self.out_features = packed_uint8.shape[0]
        dev = packed_uint8.device

        self.register_buffer("packed_uint8", packed_uint8.contiguous())
        self.register_buffer("scales", scales.to(device=dev, dtype=torch.float16).contiguous())
        self.register_buffer("codebook", torch.tensor(LLOYD_MAX_SYMMETRIC_CENTROIDS, dtype=torch.float16, device=dev))

        if bias is not None:
            self.register_buffer("bias", bias.to(device=dev, dtype=torch.float16).contiguous())
            self.has_bias = True
        else:
            self.bias = None
            self.has_bias = False

        if outliers_fp16 is not None and outlier_indices is not None:
            self.register_buffer("outliers_fp16", outliers_fp16.to(device=dev, dtype=torch.float16).contiguous())
            self.register_buffer("outlier_indices", outlier_indices.to(device=dev, dtype=torch.long))

            out_mask = torch.ones(d_in, dtype=torch.float16, device=dev)
            out_mask[outlier_indices] = 0.0
            self.register_buffer("outlier_mask_vector", out_mask.contiguous())
            self.has_outliers = True
        else:
            self.outliers_fp16 = None
            self.outlier_indices = None
            self.outlier_mask_vector = None
            self.has_outliers = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).to(dtype=torch.float16)

        y = triton_symmetric_2bit_gemm(
            x=x_2d,
            packed_uint8=self.packed_uint8,
            scales=self.scales,
            codebook=self.codebook,
            group_size=self.group_size,
            outlier_mask=self.outlier_mask_vector if self.has_outliers else None
        )

        if self.has_bias:
            y = y + self.bias

        if self.has_outliers:
            x_out = x_2d[..., self.outlier_indices]
            y = y + torch.matmul(x_out, self.outliers_fp16.T)

        y = torch.clamp(y, min=-65504.0, max=65504.0)
        return y.view(*orig_shape[:-1], self.out_features).to(dtype=x.dtype)