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
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors.torch import save_file as safe_save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class QuantizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, group_size=64, original_emb=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        self.register_buffer("packed_q", torch.empty(0, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(0, dtype=torch.float16))

        if original_emb is not None and hasattr(original_emb, "weight"):
            W = original_emb.weight.data.half()
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

    def quantize_direct(self, linear_module, global_perm_idx, num_k, device="cpu"):
        W = linear_module.weight.data.to(device, dtype=torch.float16)

        if linear_module.bias is not None:
            self.bias = nn.Parameter(linear_module.bias.data.half().cpu())

        self.perm_idx = global_perm_idx.clone().int().cpu()

        W_permuted = W[:, global_perm_idx.to(device).long()]
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

    if HAS_SAFETENSORS:
        save_path = f"otf_{clean_name}_compressed.safetensors"
    else:
        save_path = f"otf_{clean_name}_compressed.pt"

    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile file {profile_path} not found!")

    t0 = time.time()
    print("=" * 70)
    print(f"📦 LAYER-WISE MODEL COMPRESSION: {model_id}")
    print(f"🎯 Outlier Retention: {outlier_pct * 100}% FP16 | Checkpoint: {save_path}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    act_profile = torch.load(profile_path, map_location="cpu")

    print("[1/3] Loading base model skeleton into CPU RAM (FP32 Windows-Safe mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # <--- Fixes Windows 0xC0000005 crash!
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    print("[2/3] Calculating global permutation maps...")
    global_importance_by_dim = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("mlp" in name or "self_attn" in name or name == "lm_head"):
            dim = module.in_features
            if dim not in global_importance_by_dim:
                global_importance_by_dim[dim] = torch.zeros(dim, device="cpu")

            act_imp = act_profile.get(name, torch.zeros(dim))
            importance = (module.weight.data.float().abs() * (act_imp + 1e-5).unsqueeze(0)).mean(dim=0)
            global_importance_by_dim[dim] += importance

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

    print("[3/3] Compressing weight tensors layer-by-layer...")

    # Embeddings -> INT8
    old_emb = model.model.embed_tokens
    new_emb = QuantizedEmbedding(old_emb.num_embeddings, old_emb.embedding_dim, original_emb=old_emb)
    model.model.embed_tokens = new_emb

    # Transformer Layers -> INT4
    for name, module in model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    dim = child_module.in_features
                    wrapped = GlobalSymmetricINT4Linear(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bias=(child_module.bias is not None)
                    )
                    wrapped.quantize_direct(child_module, global_perm_map[dim], global_num_k_map[dim], device=device)
                    setattr(module, child_name, wrapped)

    # lm_head -> INT4
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        dim = model.lm_head.in_features
        wrapped_lm = GlobalSymmetricINT4Linear(
            in_features=model.lm_head.in_features,
            out_features=model.lm_head.out_features,
            bias=(model.lm_head.bias is not None)
        )
        wrapped_lm.quantize_direct(model.lm_head, global_perm_map[dim], global_num_k_map[dim], device=device)
        model.lm_head = wrapped_lm

    gc.collect()

    print(f"💾 Saving compressed safetensors checkpoint to: {save_path}...")
    if HAS_SAFETENSORS:
        safe_save_file(model.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)

    file_size_gb = os.path.getsize(save_path) / (1024 ** 3)
    print(f"⚡ SUCCESS! Model compressed to {file_size_gb:.2f} GB in {time.time() - t0:.2f} sec!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal LLM Quantizer")
    parser.add_argument("--model_id", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Model ID")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    convert_model(args.model_id, device=args.device)