"""
OTF-LLM Engine: Dual-Trajectory Layer Profiler (Code vs Text Gating Analysis)
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
"""

import os
import sys
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

from otf_llm.symmetric_2bit_engine import (
    SymmetricOTF2BitLinear,
    QuantizedEmbedding,
    QuantizedLinearHead
)
from otf_llm.run_symmetric_2bit import fix_rotary_embeddings

PROMPT_CODE = "Write a Python function to compute the longest common subsequence using dynamic programming."
PROMPT_TEXT = "Explain the concept of quantum entanglement and Bell's inequality in simple terms."


@torch.inference_mode()
def profile_trajectories(model_2bit_dir: str = "models/Qwen-7B-2Bit-Sym", original_model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)

    # 1. FP16 Baseline Hidden States
    print("📥 Capturing FP16 baseline trajectories...", flush=True)
    fp16_model = AutoModelForCausalLM.from_pretrained(
        original_model_id, dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    fp16_code_hidden = {}
    fp16_text_hidden = {}

    def make_hook(layer_idx, target_dict):
        def hook(m, inp, out):
            target_dict[layer_idx] = out[0].detach().float().cpu()
        return hook

    hooks = []
    for i, layer in enumerate(fp16_model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i, fp16_code_hidden)))
    _ = fp16_model(tokenizer(PROMPT_CODE, return_tensors="pt").input_ids.to(device))
    for h in hooks: h.remove()

    hooks = []
    for i, layer in enumerate(fp16_model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i, fp16_text_hidden)))
    _ = fp16_model(tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device))
    for h in hooks: h.remove()

    del fp16_model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 2-Bit Model Hidden States
    print("⚡ Capturing 2-Bit model trajectories...", flush=True)
    hf_config = AutoConfig.from_pretrained(original_model_id, trust_remote_code=True)
    with open(os.path.join(model_2bit_dir, "otf_2bit_config.json"), "r") as f:
        meta_cfg = json.load(f)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, dtype=torch.float16)

    model.model.embed_tokens = nn.Identity()
    model.lm_head = nn.Identity()
    for i in range(meta_cfg["num_layers"]):
        for p in meta_cfg["projections"]:
            setattr(getattr(model.model.layers[i], p.split(".")[0]), p.split(".")[1], nn.Identity())

    model = model.to_empty(device=device)
    model.requires_grad_(False)

    base_tensors = {}
    with safe_open(os.path.join(model_2bit_dir, "otf_2bit_base.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys(): base_tensors[k] = f.get_tensor(k)
    quant_tensors = {}
    with safe_open(os.path.join(model_2bit_dir, "otf_2bit_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys(): quant_tensors[k] = f.get_tensor(k)

    q_emb = QuantizedEmbedding(hf_config.vocab_size, hf_config.hidden_size).to(device)
    q_emb.weight_int8.copy_(base_tensors["model.embed_tokens.weight_int8"].to(device))
    q_emb.scales.copy_(base_tensors["model.embed_tokens.scales"].to(device))
    model.model.embed_tokens = q_emb

    q_head = QuantizedLinearHead(hf_config.hidden_size, hf_config.vocab_size).to(device)
    if "lm_head.weight_int8" in base_tensors:
        q_head.weight_int8.copy_(base_tensors["lm_head.weight_int8"].to(device))
        q_head.scales.copy_(base_tensors["lm_head.scales"].to(device))
    else:
        q_head.weight_int8 = q_emb.weight_int8
        q_head.scales = q_emb.scales
    model.lm_head = q_head

    if "model.norm.weight" in base_tensors:
        model.model.norm.weight.copy_(base_tensors["model.norm.weight"].to(device))

    for i in range(meta_cfg["num_layers"]):
        for n_suf in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            k = f"model.layers.{i}.{n_suf}"
            if k in base_tensors:
                getattr(model.model.layers[i], n_suf.split(".")[0]).weight.copy_(base_tensors[k].to(device))

    for i in range(meta_cfg["num_layers"]):
        for proj in meta_cfg["projections"]:
            pfx = f"model.layers.{i}.{proj}"
            packed = quant_tensors[f"{pfx}.packed_uint8"].to(device)
            scales = quant_tensors[f"{pfx}.scales"].to(device)
            bias = quant_tensors.get(f"{pfx}.bias", None)
            if bias is not None: bias = bias.to(device)
            out_d = quant_tensors.get(f"{pfx}.outlier_deltas", None)
            out_i = quant_tensors.get(f"{pfx}.outlier_indices", None)
            if out_d is not None: out_d = out_d.to(device)
            if out_i is not None: out_i = out_i.to(device)

            layer_sym = SymmetricOTF2BitLinear(
                packed, scales, packed.shape[1] * 4, meta_cfg["group_size"], out_d, out_i, bias
            )
            setattr(getattr(model.model.layers[i], proj.split(".")[0]), proj.split(".")[1], layer_sym)

    fix_rotary_embeddings(model, hf_config, device)

    quant_code_hidden = {}
    quant_text_hidden = {}

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i, quant_code_hidden)))
    _ = model(tokenizer(PROMPT_CODE, return_tensors="pt").input_ids.to(device))
    for h in hooks: h.remove()

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i, quant_text_hidden)))
    _ = model(tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device))
    for h in hooks: h.remove()

    print("\n" + "=" * 80)
    print("📊 DUAL-TRAJECTORY LAYER-BY-LAYER PARITY PROFILE (Qwen2.5-7B)")
    print("=" * 80)
    print(f"{'Layer':<8} | {'Code Prompt 01 Cosine':<22} | {'Physics Prompt 05 Cosine':<24} | {'Divergence Δ'}")
    print("-" * 80)

    for i in range(meta_cfg["num_layers"]):
        h_code_fp16 = fp16_code_hidden[i].reshape(-1, hf_config.hidden_size)
        h_code_quant = quant_code_hidden[i].reshape(-1, hf_config.hidden_size)
        cos_code = F.cosine_similarity(h_code_fp16, h_code_quant, dim=-1).mean().item() * 100.0

        h_text_fp16 = fp16_text_hidden[i].reshape(-1, hf_config.hidden_size)
        h_text_quant = quant_text_hidden[i].reshape(-1, hf_config.hidden_size)
        cos_text = F.cosine_similarity(h_text_fp16, h_text_quant, dim=-1).mean().item() * 100.0

        delta = cos_text - cos_code
        print(f"Layer {i:02d} | {cos_code:6.2f}%                 | {cos_text:6.2f}%                   | {delta:+6.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    profile_trajectories()