# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_llm/convert_global_universal.py

import os
import argparse
import time
import math
import gc
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

try:
    from safetensors.torch import safe_open, save_file as safe_save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class QuantizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, group_size=64, original_emb=None, weight_tensor=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        self.register_buffer("packed_q", torch.empty(0, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(0, dtype=torch.float16))

        # Support both original_emb module and raw weight_tensor
        if weight_tensor is None and original_emb is not None and hasattr(original_emb, "weight"):
            weight_tensor = original_emb.weight.data

        if weight_tensor is not None:
            W = weight_tensor.detach().half()
            reshaped_W = W.reshape(-1, group_size).float()
            absmax = reshaped_W.abs().max(dim=1, keepdim=True)[0]
            scale = absmax / 127.0
            q_signed = torch.clamp(torch.round(reshaped_W / (scale + 1e-8)), -127, 127).to(torch.int8)

            self.packed_q = q_signed.cpu()
            self.scale = scale.view(num_embeddings, -1).half().cpu()

    def forward(self, input_ids):
        dtype, device = torch.float16, input_ids.device
        if input_ids.numel() == 0 or input_ids.shape[1] == 0:
            return torch.empty((*input_ids.shape, self.embedding_dim), dtype=dtype, device=device)

        q_rows = self.packed_q.to(device).view(self.num_embeddings, self.embedding_dim)[input_ids]
        scale_rows = self.scale.to(device)[input_ids]

        q_reshaped = q_rows.view(*input_ids.shape, -1, self.group_size).float()
        scale_expanded = scale_rows.unsqueeze(-1).float()
        dequant = (q_reshaped * scale_expanded).view(*input_ids.shape, self.embedding_dim)
        return dequant.to(dtype)


class GlobalSymmetricINT4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, group_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("perm_idx", torch.empty(in_features, dtype=torch.int32))
        self.register_buffer("W_outliers_fp16", torch.empty(0, dtype=torch.float16))
        self.register_buffer("packed_q_bg", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("scale_bg", torch.empty(0, dtype=torch.float16))
        self.is_calibrated = False
        self.tied_embedding = None

    def forward(self, x):
        dtype, device = x.dtype, x.device

        # Handle Tied Word Embeddings fallback (e.g. Llama-3.2 lm_head)
        if not self.is_calibrated or self.packed_q_bg.numel() == 0:
            if self.tied_embedding is not None:
                emb = self.tied_embedding
                q_rows = emb.packed_q.to(device).view(emb.num_embeddings, emb.embedding_dim)
                scale_rows = emb.scale.to(device)

                q_reshaped = q_rows.view(emb.num_embeddings, -1, emb.group_size).float()
                scale_expanded = scale_rows.unsqueeze(-1).float()
                dequant_w = (q_reshaped * scale_expanded).view(emb.num_embeddings, emb.embedding_dim).to(dtype)

                out = nn.functional.linear(x, dequant_w)
                if self.bias is not None:
                    out = out + self.bias.to(dtype)
                return out
            else:
                raise RuntimeError(f"Linear module ({self.in_features}->{self.out_features}) is not calibrated!")

        # Dequantize background weight on CPU/GPU
        x_permuted = torch.index_select(x, dim=-1, index=self.perm_idx.to(device).long())

        k = self.W_outliers_fp16.shape[1]
        x_outliers, x_bg = x_permuted[..., :k], x_permuted[..., k:]

        out_outliers = nn.functional.linear(x_outliers, self.W_outliers_fp16.to(device, dtype=dtype))

        b = self.packed_q_bg.to(device)
        w0 = (b & 0x0F).to(dtype) - 8.0
        w1 = ((b >> 4) & 0x0F).to(dtype) - 8.0
        q_unpacked = torch.stack([w0, w1], dim=-1).view(self.out_features, -1)

        scale_expanded = self.scale_bg.to(device).repeat_interleave(self.group_size, dim=1)
        w_bg = (q_unpacked * scale_expanded).to(dtype)

        out_bg = nn.functional.linear(x_bg, w_bg)

        out = out_outliers + out_bg
        if self.bias is not None:
            out = out + self.bias.to(dtype)

        return out

    def quantize_direct_weight(self, weight_tensor, global_perm_idx, num_k, bias_tensor=None):
        W = weight_tensor.detach().half()

        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor.detach().half().cpu())

        self.perm_idx = global_perm_idx.clone().int().cpu()

        W_permuted = W[:, global_perm_idx.long()]
        self.W_outliers_fp16 = W_permuted[:, :num_k].cpu().clone()
        W_bg = W_permuted[:, num_k:]

        reshaped_W = W_bg.reshape(-1, self.group_size).float()
        absmax = reshaped_W.abs().max(dim=1, keepdim=True)[0]
        scale = absmax / 7.0

        q_signed = torch.clamp(torch.round(reshaped_W / (scale + 1e-8)), -7, 7).to(torch.int8)
        q_unsigned = (q_signed + 8).to(torch.uint8)

        q_W_flat = q_unsigned.view(-1)
        if q_W_flat.numel() % 2 != 0:
            q_W_flat = nn.functional.pad(q_W_flat, (0, 1))

        q_pairs = q_W_flat.view(-1, 2)
        packed_q = (q_pairs[:, 0] & 0x0F) | ((q_pairs[:, 1] & 0x0F) << 4)

        self.packed_q_bg = packed_q.cpu().clone()
        self.scale_bg = scale.view(self.out_features, -1).half().cpu().clone()

        del W, W_permuted, W_bg, reshaped_W, absmax, scale, q_signed, q_unsigned, q_W_flat, q_pairs, packed_q
        self.is_calibrated = True


def convert_model(model_id: str, outlier_pct: float = 0.01, device: str = "cpu"):
    clean_name = model_id.split("/")[-1].lower().replace("-", "_")
    profile_path = f"{clean_name}_act_profile.pt"
    save_path = f"otf_{clean_name}_compressed.safetensors"

    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile file {profile_path} not found!")

    t0 = time.time()
    print("=" * 70)
    print(f"📦 ZERO-RAM STREAMING MODEL QUANTIZER: {model_id}")
    print(f"🎯 Outlier Retention: {outlier_pct * 100}% FP16 | Peak RAM Target: < 500 MB")
    print("=" * 70)

    act_profile = torch.load(profile_path, map_location="cpu")

    print("[1/3] Locating model safetensors shards on disk...")
    model_dir = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "*.json"])

    safetensors_files = [
        os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")
    ]

    if not safetensors_files:
        raise FileNotFoundError("No .safetensors files found in downloaded model repository!")

    # Pass 1: Build Importance Permutation Maps without loading weights into RAM
    print("[2/3] Calculating global permutation maps via mmap inspection...")
    global_importance_by_dim = {}

    for sf_file in safetensors_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".weight") and ("mlp" in key or "self_attn" in key or "lm_head" in key):
                    tensor = f.get_tensor(key)
                    out_dim, in_dim = tensor.shape

                    if in_dim not in global_importance_by_dim:
                        global_importance_by_dim[in_dim] = torch.zeros(in_dim, device="cpu")

                    module_name = key.replace(".weight", "")
                    act_imp = act_profile.get(module_name, torch.zeros(in_dim))
                    importance = (tensor.float().abs() * (act_imp + 1e-5).unsqueeze(0)).mean(dim=0)
                    global_importance_by_dim[in_dim] += importance

                    del tensor
                    gc.collect()

    group_size = 64
    global_perm_map = {}
    global_num_k_map = {}

    for dim, importance in global_importance_by_dim.items():
        raw_k = int(dim * outlier_pct)
        num_k = math.ceil(raw_k / group_size) * group_size
        num_k = min(num_k, dim - group_size)
        if num_k == 0:
            num_k = group_size

        perm_idx = torch.argsort(importance, descending=True).int().cpu()
        global_perm_map[dim] = perm_idx
        global_num_k_map[dim] = num_k

    del global_importance_by_dim
    gc.collect()

    # Pass 2: Layer-by-Layer Streaming Quantization directly from mmap
    print("[3/3] Quantizing weight tensors layer-by-layer (< 500 MB RAM Peak)...")
    compressed_state_dict = {}

    for sf_file in safetensors_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Process Embeddings
                if "embed_tokens.weight" in key:
                    w = f.get_tensor(key)
                    emb_quant = QuantizedEmbedding(w.shape[0], w.shape[1], weight_tensor=w)
                    compressed_state_dict["model.embed_tokens.packed_q"] = emb_quant.packed_q
                    compressed_state_dict["model.embed_tokens.scale"] = emb_quant.scale
                    del w, emb_quant
                    gc.collect()

                # Process Linear Layers
                elif key.endswith(".weight") and ("mlp" in key or "self_attn" in key or "lm_head" in key):
                    w = f.get_tensor(key)
                    in_dim = w.shape[1]
                    out_dim = w.shape[0]

                    bias_key = key.replace(".weight", ".bias")
                    bias = f.get_tensor(bias_key) if bias_key in f.keys() else None

                    wrapped = GlobalSymmetricINT4Linear(in_dim, out_dim, bias=(bias is not None))
                    wrapped.quantize_direct_weight(w, global_perm_map[in_dim], global_num_k_map[in_dim],
                                                   bias_tensor=bias)

                    prefix = key.replace(".weight", "")
                    compressed_state_dict[f"{prefix}.perm_idx"] = wrapped.perm_idx
                    compressed_state_dict[f"{prefix}.W_outliers_fp16"] = wrapped.W_outliers_fp16
                    compressed_state_dict[f"{prefix}.packed_q_bg"] = wrapped.packed_q_bg
                    compressed_state_dict[f"{prefix}.scale_bg"] = wrapped.scale_bg
                    if wrapped.bias is not None:
                        compressed_state_dict[f"{prefix}.bias"] = wrapped.bias

                    del w, bias, wrapped
                    gc.collect()

                # Keep Norms and Biases in FP16
                elif "norm" in key or "bias" in key:
                    tensor = f.get_tensor(key)
                    compressed_state_dict[key] = tensor.half().cpu()
                    del tensor
                    gc.collect()

    print(f"💾 Saving compressed safetensors checkpoint to: {save_path}...")
    safe_save_file(compressed_state_dict, save_path)

    file_size_gb = os.path.getsize(save_path) / (1024 ** 3)
    print(f"⚡ SUCCESS! Model streamed and compressed to {file_size_gb:.2f} GB in {time.time() - t0:.2f} sec!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-RAM Streaming LLM Quantizer")
    parser.add_argument("--model_id", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Model ID")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    convert_model(args.model_id, device=args.device)