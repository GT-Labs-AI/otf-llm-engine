"""
OTF-LLM Engine: Temperature & Logit Scale Calibration Sweep
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

# Гарантируем видимость корня проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

from otf_llm.symmetric_2bit_engine import (
    SymmetricOTF2BitLinear,
    QuantizedEmbedding,
    QuantizedLinearHead
)
from otf_llm.run_symmetric_2bit import fix_rotary_embeddings
from tests.test_formal_parity import collect_baseline_logits, BENCHMARK_PROMPTS


@torch.inference_mode()
def run_temperature_sweep(model_2bit_dir: str, original_model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_logits = collect_baseline_logits(original_model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)
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
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)

    quant_tensors = {}
    with safe_open(os.path.join(model_2bit_dir, "otf_2bit_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            quant_tensors[k] = f.get_tensor(k)

    q_emb = QuantizedEmbedding(hf_config.vocab_size, hf_config.hidden_size).to(device)
    q_emb.weight_int8.copy_(base_tensors["model.embed_tokens.weight_int8"].to(device))
    q_emb.scales.copy_(base_tensors["model.embed_tokens.scales"].to(device))
    model.model.embed_tokens = q_emb

    q_head = QuantizedLinearHead(hf_config.hidden_size, hf_config.vocab_size).to(device)
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
            if bias is not None:
                bias = bias.to(device)
            out_d = quant_tensors.get(f"{pfx}.outlier_deltas", None)
            out_i = quant_tensors.get(f"{pfx}.outlier_indices", None)
            if out_d is not None:
                out_d = out_d.to(device)
            if out_i is not None:
                out_i = out_i.to(device)

            layer_sym = SymmetricOTF2BitLinear(
                packed, scales, packed.shape[1] * 4, meta_cfg["group_size"], out_d, out_i, bias
            )
            setattr(getattr(model.model.layers[i], proj.split(".")[0]), proj.split(".")[1], layer_sym)

    fix_rotary_embeddings(model, hf_config, device)
    del base_tensors, quant_tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Собираем сырые логиты 2-битной модели
    raw_quant_logits = {}
    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        raw_quant_logits[idx] = model(input_ids).logits[:, -1, :].detach().float().cpu()

    # Сетка температур (Logit Multiplier = 1 / T)
    temp_grid = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

    print("\n" + "=" * 80)
    print("🔬 TEMPERATURE / LOGIT SCALE SENSITIVITY SWEEP")
    print("=" * 80)
    print(f"{'Temp (T)':<10} | {'Multiplier (γ=1/T)':<20} | {'Prob Cos Sim':<16} | {'KL Divergence':<15}")
    print("-" * 80)

    best_t = 1.0
    best_sim = 0.0

    for t in temp_grid:
        sims = []
        kls = []
        for idx in range(len(BENCHMARK_PROMPTS)):
            base_p = F.softmax(baseline_logits[idx] / 1.0, dim=-1)
            quant_p = F.softmax(raw_quant_logits[idx] / t, dim=-1)

            cos = F.cosine_similarity(base_p, quant_p, dim=-1).item() * 100.0
            kl = F.kl_div(quant_p.log(), base_p, reduction="batchmean").item()
            sims.append(cos)
            kls.append(kl)

        mean_sim = sum(sims) / len(sims)
        mean_kl = sum(kls) / len(kls)
        if mean_sim > best_sim:
            best_sim = mean_sim
            best_t = t

        print(f"{t:<10.2f} | {1.0/t:<20.2f} | {mean_sim:6.2f}%         | {mean_kl:<15.4f}")

    raw_base_sim = sum(
        [F.cosine_similarity(F.softmax(baseline_logits[i], dim=-1), F.softmax(raw_quant_logits[i], dim=-1), dim=-1).item() * 100.0 for i in range(len(BENCHMARK_PROMPTS))]
    ) / len(BENCHMARK_PROMPTS)

    print("=" * 80)
    print(f"🎯 OPTIMAL CALIBRATION: Temperature T = {best_t:.2f} (Scale Multiplier γ = {1.0/best_t:.2f}x)")
    print(f"📈 Prob Cosine Jump: {raw_base_sim:.2f}% -> {best_sim:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_temperature_sweep("models/Qwen-3B-2Bit-Sym", "Qwen/Qwen2.5-3B-Instruct")