"""
OTF-LLM Engine: Systematic Layer & Module Ablation Benchmark
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
from tests.test_formal_parity import collect_baseline_logits, BENCHMARK_PROMPTS


@torch.inference_mode()
def build_ablation_model(
    model_2bit_dir: str,
    original_model_id: str,
    hf_config: AutoConfig,
    fp16_layers: set,
    fp16_embeds: bool,
    fp16_lm_head: bool,
    device: str
):
    with open(os.path.join(model_2bit_dir, "otf_2bit_config.json"), "r") as f:
        meta_cfg = json.load(f)

    # 1. Загрузка исходных FP16 весов при необходимости
    fp16_src_model = None
    if fp16_layers or fp16_embeds or fp16_lm_head:
        fp16_src_model = AutoModelForCausalLM.from_pretrained(
            original_model_id, dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
        )

    # 2. Скелет на META
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, dtype=torch.float16)

    model.model.embed_tokens = nn.Identity()
    model.lm_head = nn.Identity()

    for i in range(meta_cfg["num_layers"]):
        if i not in fp16_layers:
            layer_obj = model.model.layers[i]
            for proj in meta_cfg["projections"]:
                sub = layer_obj
                parts = proj.split(".")
                for p in parts[:-1]:
                    sub = getattr(sub, p)
                setattr(sub, parts[-1], nn.Identity())

    model = model.to_empty(device=device)
    model.requires_grad_(False)

    # 3. Загрузка 2-битных SafeTensors
    base_tensors = {}
    with safe_open(os.path.join(model_2bit_dir, "otf_2bit_base.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys(): base_tensors[k] = f.get_tensor(k)
    quant_tensors = {}
    with safe_open(os.path.join(model_2bit_dir, "otf_2bit_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys(): quant_tensors[k] = f.get_tensor(k)

    # 4. Встраивание Embeddings & LM Head
    if fp16_embeds:
        model.model.embed_tokens = fp16_src_model.model.embed_tokens.to(device)
    else:
        q_emb = QuantizedEmbedding(hf_config.vocab_size, hf_config.hidden_size).to(device)
        q_emb.weight_int8.copy_(base_tensors["model.embed_tokens.weight_int8"].to(device))
        q_emb.scales.copy_(base_tensors["model.embed_tokens.scales"].to(device))
        model.model.embed_tokens = q_emb

    if fp16_lm_head:
        model.lm_head = fp16_src_model.lm_head.to(device)
    else:
        q_head = QuantizedLinearHead(hf_config.hidden_size, hf_config.vocab_size).to(device)
        if "lm_head.weight_int8" in base_tensors:
            q_head.weight_int8.copy_(base_tensors["lm_head.weight_int8"].to(device))
            q_head.scales.copy_(base_tensors["lm_head.scales"].to(device))
        else:
            q_head.weight_int8 = q_emb.weight_int8
            q_head.scales = q_emb.scales
        model.lm_head = q_head

    # 5. LayerNorms
    if "model.norm.weight" in base_tensors:
        model.model.norm.weight.copy_(base_tensors["model.norm.weight"].to(device))

    for i in range(meta_cfg["num_layers"]):
        for n_suf in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            k = f"model.layers.{i}.{n_suf}"
            if k in base_tensors:
                getattr(model.model.layers[i], n_suf.split(".")[0]).weight.copy_(base_tensors[k].to(device))

    # 6. Встраивание слоев (FP16 vs 2-Bit)
    for i in range(meta_cfg["num_layers"]):
        layer = model.model.layers[i]
        if i in fp16_layers:
            # Восстанавливаем оригинальный FP16 слой
            src_layer = fp16_src_model.model.layers[i].to(device)
            model.model.layers[i] = src_layer
        else:
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
                sub = layer
                parts = proj.split(".")
                for p in parts[:-1]: sub = getattr(sub, p)
                setattr(sub, parts[-1], layer_sym)

    fix_rotary_embeddings(model, hf_config, device)

    if fp16_src_model is not None:
        del fp16_src_model
    del base_tensors, quant_tensors
    gc.collect()
    torch.cuda.empty_cache()

    return model


@torch.inference_mode()
def evaluate_config(model, tokenizer, baseline_logits, device):
    raw_sims = []
    prob_sims = []
    top64_sims = []
    top1_cnt = 0
    top5_cnt = 0

    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        quant_logit = model(input_ids).logits[:, -1, :].detach().float().cpu()
        base_logit = baseline_logits[idx]

        # Raw Cosine
        raw_sims.append(F.cosine_similarity(base_logit, quant_logit, dim=-1).item() * 100.0)

        # Prob Cosine
        base_p = F.softmax(base_logit, dim=-1)
        quant_p = F.softmax(quant_logit, dim=-1)
        prob_sims.append(F.cosine_similarity(base_p, quant_p, dim=-1).item() * 100.0)

        # Top-64 Subspace
        top64_idx = torch.topk(base_logit, k=64, dim=-1).indices
        base_top64 = torch.gather(base_logit, dim=-1, index=top64_idx)
        quant_top64 = torch.gather(quant_logit, dim=-1, index=top64_idx)
        top64_sims.append(F.cosine_similarity(base_top64, quant_top64, dim=-1).item() * 100.0)

        # Top-1 & Top-5
        top1_b = torch.argmax(base_logit, dim=-1).item()
        top1_q = torch.argmax(quant_logit, dim=-1).item()
        if top1_b == top1_q: top1_cnt += 1
        top5_b = set(torch.topk(base_logit, k=5, dim=-1).indices[0].tolist())
        if top1_q in top5_b: top5_cnt += 1

    return {
        "raw_cos": sum(raw_sims) / len(raw_sims),
        "prob_cos": sum(prob_sims) / len(prob_sims),
        "top64_cos": sum(top64_sims) / len(top64_sims),
        "top1": (top1_cnt / len(BENCHMARK_PROMPTS)) * 100.0,
        "top5": (top5_cnt / len(BENCHMARK_PROMPTS)) * 100.0,
        "vram_mb": torch.cuda.memory_allocated() / (1024 ** 2)
    }


def run_all_ablations(model_2bit_dir: str = "models/Qwen-3B-2Bit-Sym", model_id: str = "Qwen/Qwen2.5-3B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_logits = collect_baseline_logits(model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    experiments = [
        ("Exp 0: Full 2-Bit + INT8 Embed/Head", set(), False, False),
        ("Exp 1: FP16 Embeddings & LM Head", set(), True, True),
        ("Exp 2: FP16 Layer 35 Only", {35}, False, False),
        ("Exp 3: FP16 Tail Block (32-35)", {32, 33, 34, 35}, False, False),
        ("Exp 4: FP16 First Layer (00) Only", {0}, False, False),
        ("Exp 5: FP16 Layers (00 + 35) + FP16 Head", {0, 35}, True, True),
    ]

    print("\n" + "=" * 90)
    print("🔬 ABLATION SUITE: PINPOINTING DEGRADATION SOURCES")
    print("=" * 90)
    print(f"{'Experiment':<42} | {'VRAM':<8} | {'Prob Cos':<9} | {'Top-64':<8} | {'Top-1':<6} | {'Top-5'}")
    print("-" * 90)

    for name, fp16_l, fp16_emb, fp16_head in experiments:
        model = build_ablation_model(model_2bit_dir, model_id, hf_config, fp16_l, fp16_emb, fp16_head, device)
        res = evaluate_config(model, tokenizer, baseline_logits, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(
            f"{name:<42} | {res['vram_mb']:6.1f}MB | {res['prob_cos']:6.2f}%  | {res['top64_cos']:6.2f}% | {res['top1']:4.1f}% | {res['top5']:4.1f}%"
        )

    print("=" * 90)


if __name__ == "__main__":
    run_all_ablations()