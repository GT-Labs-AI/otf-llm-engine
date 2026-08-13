# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_llm/otf_moe_offloader.py

import time
import gc
import collections
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

from .otf_triton_kernel import triton_fused_int4_linear


class MoEExpertLRUCache:
    """
    VRAM Least-Recently-Used (LRU) Cache Manager for INT4 Quantized MoE Experts.
    Keeps top-K active experts in VRAM while dynamic experts are swapped over PCIe.
    """

    def __init__(self, max_vram_experts: int = 4, device: str = "cuda"):
        self.max_vram_experts = max_vram_experts
        self.device = device
        # Dict storing expert_id -> dict of VRAM tensors
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.transfer_stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None

    def contains(self, expert_id: int) -> bool:
        """Checks if expert weight tensors are currently resident in VRAM."""
        return expert_id in self.cache

    def get(self, expert_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieves VRAM expert tensors and updates LRU position."""
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
            return self.cache[expert_id]
        return None

    def put(self, expert_id: int, expert_tensors_vram: Dict[str, torch.Tensor]) -> None:
        """Inserts expert tensors into VRAM LRU cache, evicting oldest if capacity reached."""
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
        else:
            if len(self.cache) >= self.max_vram_experts:
                evicted_id, evicted_tensors = self.cache.popitem(last=False)
                del evicted_tensors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.cache[expert_id] = expert_tensors_vram


class AsynchronousExpertStreamer:
    """
    Asynchronously streams INT4 Quantized Expert weights from CPU RAM to GPU SRAM/VRAM
    over PCIe bus using dedicated CUDA streams without stalling main compute execution.
    """

    def __init__(self, cache_manager: MoEExpertLRUCache):
        self.cache_manager = cache_manager

    def stream_expert_to_vram(
            self,
            expert_id: int,
            cpu_expert_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Streams INT4 expert weights from CPU RAM to GPU VRAM asynchronously.
        """
        if self.cache_manager.contains(expert_id):
            return self.cache_manager.get(expert_id)

        device = self.cache_manager.device
        vram_tensors = {}

        # Perform asynchronous transfer over dedicated CUDA stream
        if self.cache_manager.transfer_stream is not None:
            with torch.cuda.stream(self.cache_manager.transfer_stream):
                for key, tensor in cpu_expert_dict.items():
                    vram_tensors[key] = tensor.to(device, non_blocking=True)
            self.cache_manager.transfer_stream.synchronize()
        else:
            for key, tensor in cpu_expert_dict.items():
                vram_tensors[key] = tensor.to(device)

        self.cache_manager.put(expert_id, vram_tensors)
        return vram_tensors


class OTFSparseMoeBlockWrapper(nn.Module):
    """
    Universal OTF-LLM Hybrid Sparse MoE Layer Wrapper.
    Replaces default Hugging Face SparseMoeBlock (Mixtral / QwenMoE / DeepSeek).
    """

    def __init__(
            self,
            gate_layer: nn.Linear,
            cpu_experts: List[Dict[str, torch.Tensor]],
            num_experts: int = 8,
            num_experts_per_tok: int = 2,
            max_vram_experts: int = 4,
            group_size: int = 64
    ):
        super().__init__()
        self.gate = gate_layer
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.group_size = group_size
        self.cpu_experts = cpu_experts

        self.cache_manager = MoEExpertLRUCache(max_vram_experts=max_vram_experts)
        self.streamer = AsynchronousExpertStreamer(self.cache_manager)

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
        Dynamic MoE Forward Pass with Router Top-K selection and LRU PCIe Swapping.
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

        # 3. Dynamic Expert Routing with VRAM LRU PCIe Offload
        for expert_id in range(self.num_experts):
            # Find tokens assigned to current expert
            token_mask = (topk_indices == expert_id)
            if not token_mask.any():
                continue

            token_indices, topk_pos = torch.where(token_mask)
            selected_tokens = hidden_states_flat[token_indices]

            # Stream or Retrieve Expert Weights from VRAM LRU Cache
            cpu_dict = self.cpu_experts[expert_id]
            expert_vram_tensors = self.streamer.stream_expert_to_vram(expert_id, cpu_dict)

            # Compute Fused Triton Forward Pass
            expert_out = self._execute_int4_expert_forward(selected_tokens, expert_vram_tensors)

            # Weight and aggregate output
            expert_weights = topk_weights[token_indices, topk_pos].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_out * expert_weights)

        return final_output.view(batch_size, seq_len, hidden_dim)


if __name__ == "__main__":
    print("=" * 75)
    print("🧪 TESTING OTF MOE EXPERT OFFLOADER (LRU CACHE + TRITON INT4)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate 8 Experts in CPU RAM
    num_experts = 8
    in_dim, out_dim, hidden_dim = 2048, 2048, 5632
    group_size = 64

    print(f"📦 Generating synthetic INT4 expert weight structures on CPU...")
    cpu_experts = []
    for exp_id in range(num_experts):
        exp_dict = {
            "w1.perm_idx": torch.arange(in_dim, dtype=torch.int32),
            "w1.W_outliers_fp16": torch.randn(out_dim, int(in_dim * 0.01), dtype=torch.float16),
            "w1.packed_q_bg": torch.randint(0, 255, (out_dim, (in_dim - int(in_dim * 0.01)) // 2), dtype=torch.uint8),
            "w1.scale_bg": torch.randn(out_dim, (in_dim - int(in_dim * 0.01)) // group_size, dtype=torch.float16),
            "w2.perm_idx": torch.arange(out_dim, dtype=torch.int32),
            "w2.W_outliers_fp16": torch.randn(in_dim, int(out_dim * 0.01), dtype=torch.float16),
            "w2.packed_q_bg": torch.randint(0, 255, (in_dim, (out_dim - int(out_dim * 0.01)) // 2), dtype=torch.uint8),
            "w2.scale_bg": torch.randn(in_dim, (out_dim - int(out_dim * 0.01)) // group_size, dtype=torch.float16),
        }
        cpu_experts.append(exp_dict)

    gate_layer = nn.Linear(in_dim, num_experts).to(device)
    moe_wrapper = OTFSparseMoeBlockWrapper(
        gate_layer=gate_layer,
        cpu_experts=cpu_experts,
        num_experts=8,
        num_experts_per_tok=2,
        max_vram_experts=3
    )

    dummy_input = torch.randn(1, 16, in_dim, device=device, dtype=torch.float16)

    t0 = time.time()
    with torch.no_grad():
        output = moe_wrapper(dummy_input)
    t_elapsed = (time.time() - t0) * 1000

    print(f"⚡ MoE Forward Pass completed in: {t_elapsed:.2f} ms!")
    print(f"📊 Active VRAM Experts in LRU Cache: {list(moe_wrapper.cache_manager.cache.keys())}")
    print(f"✅ Output Shape: {output.shape}")
    print("=" * 75)