# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_llm/otf_moe_offloader.py

import time
import gc
import os
import collections
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

try:
    from safetensors.torch import safe_open

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from .otf_triton_kernel import triton_fused_int4_linear


class Hierarchical3TierMoECache:
    """
    3-Tier Hierarchical Memory Manager for Massive MoE Models (v3.1).
    Tier 1: Hot Pool (GPU VRAM) -> Immediate Triton INT4 Execution.
    Tier 2: Warm Pool (CPU RAM) -> Fast PCIe Gen4 Transfers (20ms).
    Tier 3: Cold Pool (NVMe SSD Disk) -> Memory-mapped zero-copy streaming via safetensors.
    """

    def __init__(
            self,
            safetensors_filepath: Optional[str] = None,
            cpu_experts_fallback: Optional[List[Dict[str, torch.Tensor]]] = None,
            max_vram_experts: int = 3,
            max_ram_experts: int = 6,
            device: str = "cuda"
    ):
        self.safetensors_filepath = safetensors_filepath
        self.cpu_experts_fallback = cpu_experts_fallback
        self.max_vram_experts = max_vram_experts
        self.max_ram_experts = max_ram_experts
        self.device = device

        # Tier 1: VRAM Cache (Hot Pool)
        self.vram_cache: collections.OrderedDict = collections.OrderedDict()

        # Tier 2: CPU RAM Cache (Warm Pool)
        self.ram_cache: collections.OrderedDict = collections.OrderedDict()

        self.transfer_stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None

    def _load_expert_from_disk_mmap(self, expert_id: int) -> Dict[str, torch.Tensor]:
        """
        Tier 3 -> Tier 2: Loads cold expert directly from NVMe SSD into CPU RAM via mmap.
        """
        if self.cpu_experts_fallback is not None and expert_id < len(self.cpu_experts_fallback):
            return self.cpu_experts_fallback[expert_id]

        expert_dict = {}
        if self.safetensors_filepath and os.path.exists(self.safetensors_filepath) and HAS_SAFETENSORS:
            with safe_open(self.safetensors_filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if f"experts.{expert_id}." in key or f"expert_{expert_id}." in key:
                        expert_dict[key] = f.get_tensor(key)
        return expert_dict

    def get_expert_for_execution(self, expert_id: int) -> Dict[str, torch.Tensor]:
        """
        Fetches active expert through the 3-Tier Hierarchy:
        - Hit Tier 1 (VRAM): Immediate execution.
        - Hit Tier 2 (CPU RAM): Stream to VRAM via non-blocking PCIe CUDA stream.
        - Hit Tier 3 (NVMe SSD): Load to CPU RAM via mmap, then stream to VRAM.
        """
        # Tier 1 Hit (GPU VRAM)
        if expert_id in self.vram_cache:
            self.vram_cache.move_to_end(expert_id)
            return self.vram_cache[expert_id]

        # Tier 2 Hit (CPU RAM)
        if expert_id in self.ram_cache:
            cpu_dict = self.ram_cache[expert_id]
            self.ram_cache.move_to_end(expert_id)
        else:
            # Tier 3 Hit (Cold Disk Fetch)
            cpu_dict = self._load_expert_from_disk_mmap(expert_id)

            # Put into Tier 2 (CPU RAM Warm Pool)
            if len(self.ram_cache) >= self.max_ram_experts:
                self.ram_cache.popitem(last=False)  # Evict coldest from CPU RAM
            self.ram_cache[expert_id] = cpu_dict

        # Stream Tier 2 (CPU RAM) -> Tier 1 (GPU VRAM)
        vram_dict = {}
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                for k, tensor in cpu_dict.items():
                    vram_dict[k] = tensor.to(self.device, non_blocking=True)
            self.transfer_stream.synchronize()
        else:
            for k, tensor in cpu_dict.items():
                vram_dict[k] = tensor.to(self.device)

        # Put into Tier 1 (GPU VRAM Hot Pool)
        if len(self.vram_cache) >= self.max_vram_experts:
            evicted_id, evicted_tensors = self.vram_cache.popitem(last=False)  # Evict oldest from VRAM
            del evicted_tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.vram_cache[expert_id] = vram_dict
        return vram_dict


class OTFSparseMoeBlockWrapper(nn.Module):
    """
    Universal OTF-LLM 3-Tier Hybrid Sparse MoE Layer Wrapper (v3.1).
    Replaces default Hugging Face SparseMoeBlock (Mixtral / QwenMoE / DeepSeek).
    """

    def __init__(
            self,
            gate_layer: nn.Linear,
            cpu_experts: Optional[List[Dict[str, torch.Tensor]]] = None,
            safetensors_filepath: Optional[str] = None,
            num_experts: int = 8,
            num_experts_per_tok: int = 2,
            max_vram_experts: int = 3,
            max_ram_experts: int = 6,
            group_size: int = 64
    ):
        super().__init__()
        self.gate = gate_layer
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.group_size = group_size

        self.cache_manager = Hierarchical3TierMoECache(
            safetensors_filepath=safetensors_filepath,
            cpu_experts_fallback=cpu_experts,
            max_vram_experts=max_vram_experts,
            max_ram_experts=max_ram_experts,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def _execute_int4_expert_forward(
            self,
            x: torch.Tensor,
            expert_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Executes Forward Pass for 1 active MoE expert using Fused Triton INT4 GEMM kernel.
        """
        device = x.device
        dtype = x.dtype

        perm_idx = expert_tensors["w1.perm_idx"].to(device).long()
        w1_outliers = expert_tensors["w1.W_outliers_fp16"].to(device, dtype=dtype)
        w1_packed_q = expert_tensors["w1.packed_q_bg"].to(device)
        w1_scale = expert_tensors["w1.scale_bg"].to(device)

        # 1. Gate / Up Projection
        x_permuted = torch.index_select(x, dim=-1, index=perm_idx)
        k = w1_outliers.shape[1]
        x_outliers, x_bg = x_permuted[..., :k], x_permuted[..., k:]

        out_outliers = nn.functional.linear(x_outliers, w1_outliers)
        out_bg = triton_fused_int4_linear(x_bg, w1_packed_q, w1_scale, self.group_size)

        # SwiGLU Activation
        hidden_states = nn.functional.silu(out_outliers + out_bg)

        # 2. Down Projection
        w2_perm_idx = expert_tensors["w2.perm_idx"].to(device).long()
        w2_outliers = expert_tensors["w2.W_outliers_fp16"].to(device, dtype=dtype)
        w2_packed_q = expert_tensors["w2.packed_q_bg"].to(device)
        w2_scale = expert_tensors["w2.scale_bg"].to(device)

        h_permuted = torch.index_select(hidden_states, dim=-1, index=w2_perm_idx)
        k2 = w2_outliers.shape[1]
        h_outliers, h_bg = h_permuted[..., :k2], h_permuted[..., k2:]

        out2_outliers = nn.functional.linear(h_outliers, w2_outliers)
        out2_bg = triton_fused_int4_linear(h_bg, w2_packed_q, w2_scale, self.group_size)

        return out2_outliers + out2_bg

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Dynamic MoE Forward Pass with 3-Tier Hierarchy (Disk -> RAM -> VRAM).
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # 1. Router logits calculation
        router_logits = self.gate(hidden_states_flat)
        routing_weights = nn.functional.softmax(router_logits, dim=-1)

        # 2. Select Top-K Active Experts per Token
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(hidden_states_flat)

        # 3. Dynamic Expert Routing through 3-Tier Memory Hierarchy
        for expert_id in range(self.num_experts):
            token_mask = (topk_indices == expert_id)
            if not token_mask.any():
                continue

            token_indices, topk_pos = torch.where(token_mask)
            selected_tokens = hidden_states_flat[token_indices]

            # Fetch through Tier 3 (Disk) -> Tier 2 (RAM) -> Tier 1 (VRAM)
            expert_vram_tensors = self.cache_manager.get_expert_for_execution(expert_id)

            # Compute Fused Triton Forward Pass
            expert_out = self._execute_int4_expert_forward(selected_tokens, expert_vram_tensors)

            # Weight and aggregate output
            expert_weights = topk_weights[token_indices, topk_pos].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_out * expert_weights)

        return final_output.view(batch_size, seq_len, hidden_dim)


if __name__ == "__main__":
    print("=" * 75)
    print("🧪 TESTING OTF 3-TIER MOE EXPERT OFFLOADER (v3.1)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    num_experts = 8
    in_dim, out_dim = 2048, 2048
    group_size = 64

    print(f"📦 Simulating 3-Tier Expert Storage (Disk -> RAM -> VRAM)...")
    cpu_experts = []
    for exp_id in range(num_experts):
        exp_dict = {
            "w1.perm_idx": torch.arange(in_dim, dtype=torch.int32),
            "w1.W_outliers_fp16": torch.randn(out_dim, int(in_dim * 0.01), dtype=dtype),
            "w1.packed_q_bg": torch.randint(0, 255, (out_dim, (in_dim - int(in_dim * 0.01)) // 2), dtype=torch.uint8),
            "w1.scale_bg": torch.randn(out_dim, (in_dim - int(in_dim * 0.01)) // group_size, dtype=dtype),
            "w2.perm_idx": torch.arange(out_dim, dtype=torch.int32),
            "w2.W_outliers_fp16": torch.randn(in_dim, int(out_dim * 0.01), dtype=dtype),
            "w2.packed_q_bg": torch.randint(0, 255, (in_dim, (out_dim - int(out_dim * 0.01)) // 2), dtype=torch.uint8),
            "w2.scale_bg": torch.randn(in_dim, (out_dim - int(out_dim * 0.01)) // group_size, dtype=dtype),
        }
        cpu_experts.append(exp_dict)

    gate_layer = nn.Linear(in_dim, num_experts).to(device=device, dtype=dtype)
    moe_wrapper = OTFSparseMoeBlockWrapper(
        gate_layer=gate_layer,
        cpu_experts=cpu_experts,
        num_experts=8,
        num_experts_per_tok=2,
        max_vram_experts=3,  # Tier 1 VRAM Limit: 3 experts
        max_ram_experts=5  # Tier 2 RAM Limit: 5 experts
    )

    dummy_input = torch.randn(1, 16, in_dim, device=device, dtype=dtype)

    t0 = time.time()
    with torch.no_grad():
        output = moe_wrapper(dummy_input)
    t_elapsed = (time.time() - t0) * 1000

    print(f"⚡ 3-Tier MoE Forward Pass completed in: {t_elapsed:.2f} ms!")
    print(f"🔥 Tier 1 VRAM Hot Pool: {list(moe_wrapper.cache_manager.vram_cache.keys())}")
    print(f"♨️ Tier 2 CPU RAM Warm Pool: {list(moe_wrapper.cache_manager.ram_cache.keys())}")
    print(f"✅ Output Shape: {output.shape}")
    print("=" * 75)

# Backward compatibility aliases for v3.0 / v3.1 APIs
MoEExpertLRUCache = Hierarchical3TierMoECache
AsynchronousExpertStreamer = Hierarchical3TierMoECache