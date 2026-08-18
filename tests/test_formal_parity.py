"""
OTF-LLM Engine: Official Conversational & Reasoning Parity Benchmark (v4.1.4)
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import json
import time
import gc
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from otf_llm.symmetric_2bit_engine import (
    SymmetricOTF2BitLinear,
    QuantizedEmbedding,
    QuantizedLinearHead
)
from otf_llm.run_symmetric_2bit import fix_rotary_embeddings

# 🚀 Официальный набор из 20 промптов: Логика, Рассуждения, Анализ документов и Диалог
BENCHMARK_PROMPTS = [
    # 1. Интуитивные объяснения сложных концепций
    "Explain the concept of quantum computing, superposition, and entanglement in simple terms.",
    "Describe the fundamental differences between human intuition and artificial neural networks in decision making.",
    "Explain how the Fermi Paradox challenges our understanding of extraterrestrial intelligence in the universe.",
    "Explain why optical illusions trick the human visual cortex despite our conscious understanding.",

    # 2. Логические рассуждения и доказательства (Chain-of-Thought)
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If all roses are flowers and some flowers fade quickly, can we logically conclude that some roses fade quickly? Analyze rigorously.",
    "Three people check into a hotel room that costs $30. Explain the resolution to the classic missing dollar riddle.",
    "Provide a step-by-step philosophical proof exploring whether free will can exist in a deterministic universe.",

    # 3. Аналитика, аппаратные технологии и RLM
    "Summarize the key trade-offs between memory bandwidth, computational latency, and model compression in modern AI hardware.",
    "Analyze how open-source artificial intelligence enables breakthroughs on consumer-grade hardware and democratizes research.",
    "How does the tragedy of the commons apply to global climate change and shared resource management?",
    "Analyze the economic impact of general-purpose automation on global labor markets over the next decade.",

    # 4. Глубокие рассуждения и диалог на русском языке
    "Объясни простыми словами, как работает квантование нейросетей до 2 бит и почему сохранение логики возможно при потере точности.",
    "Напиши структурированное аналитическое эссе о перспективах развития локальных ИИ-ассистентов, работающих без интернета.",
    "Сравни преимущества и недостатки открытых и закрытых языковых моделей для корпоративного сектора.",
    "В чем разница между индуктивным и дедуктивным методом рассуждений? Приведи наглядные примеры.",

    # 5. Психология, когнитивистика и системный синтез
    "Describe how cognitive biases such as confirmation bias and anchoring effect influence strategic investments.",
    "Explain the psychological concept of flow state and how deliberate environment design facilitates deep work.",
    "Synthesize the relationship between language, thought, and culture according to the Sapir-Whorf hypothesis.",
    "Provide a detailed, step-by-step strategic plan for optimizing team productivity during high-uncertainty R&D projects."
]


@torch.inference_mode()
def collect_baseline_logits(model_id: str, device: str) -> Dict[int, torch.Tensor]:
    print("=" * 80, flush=True)
    print(f"🔬 PHASE 1: Collecting FP16 Baseline Logits from '{model_id}'...", flush=True)
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    model.eval()

    baseline_logits = {}
    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        outputs = model(input_ids)
        last_logit = outputs.logits[:, -1, :].detach().float().cpu()
        baseline_logits[idx] = last_logit
        print(f"   [{idx+1:02d}/{len(BENCHMARK_PROMPTS):02d}] Computed baseline for: '{prompt[:45]}...'", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 FP16 Baseline Model completely unloaded from VRAM.\n", flush=True)
    return baseline_logits


@torch.inference_mode()
def test_formal_parity(
    model_2bit_dir: str,
    original_model_id: str,
    device: str = "cuda"
):
    baseline_logits = collect_baseline_logits(original_model_id, device)

    print("=" * 80, flush=True)
    print(f"⚡ PHASE 2: Loading OTF Symmetric Engine from '{model_2bit_dir}'...", flush=True)
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(original_model_id, trust_remote_code=True)

    with open(os.path.join(model_2bit_dir, "otf_2bit_config.json"), "r") as f:
        meta_cfg = json.load(f)

    num_layers = meta_cfg["num_layers"]
    group_size = meta_cfg["group_size"]
    projections = meta_cfg["projections"]

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, dtype=torch.float16)

    model.model.embed_tokens = nn.Identity()
    model.lm_head = nn.Identity()

    for i in range(num_layers):
        layer_obj = model.model.layers[i]
        for proj in projections:
            sub = layer_obj
            parts = proj.split(".")
            for p in parts[:-1]:
                sub = getattr(sub, p)
            setattr(sub, parts[-1], nn.Identity())

    model = model.to_empty(device=device)
    model.requires_grad_(False)

    base_path = os.path.join(model_2bit_dir, "otf_2bit_base.safetensors")
    quant_path = os.path.join(model_2bit_dir, "otf_2bit_model.safetensors")

    base_tensors = {}
    with safe_open(base_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)

    quant_tensors = {}
    with safe_open(quant_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            quant_tensors[k] = f.get_tensor(k)

    # Embeddings & LM Head
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

    # Norms
    if "model.norm.weight" in base_tensors:
        model.model.norm.weight.copy_(base_tensors["model.norm.weight"].to(device))

    for i in range(num_layers):
        for n_suf in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            k = f"model.layers.{i}.{n_suf}"
            if k in base_tensors:
                norm_module = getattr(model.model.layers[i], n_suf.split(".")[0])
                norm_module.weight.copy_(base_tensors[k].to(device))

    # Layers
    for i in range(num_layers):
        layer = model.model.layers[i]
        for proj in projections:
            pfx = f"model.layers.{i}.{proj}"
            if f"{pfx}.packed_uint8" not in quant_tensors:
                continue

            packed = quant_tensors[f"{pfx}.packed_uint8"].to(device)
            scales = quant_tensors[f"{pfx}.scales"].to(device)
            bias = quant_tensors.get(f"{pfx}.bias", None)
            if bias is not None:
                bias = bias.to(device)

            out_deltas = quant_tensors.get(f"{pfx}.outlier_deltas", None)
            out_idx = quant_tensors.get(f"{pfx}.outlier_indices", None)
            if out_deltas is not None:
                out_deltas = out_deltas.to(device)
            if out_idx is not None:
                out_idx = out_idx.to(device)

            d_in = packed.shape[1] * 4
            layer_sym = SymmetricOTF2BitLinear(
                packed_uint8=packed,
                scales=scales,
                d_in=d_in,
                group_size=group_size,
                outliers_fp16=out_deltas,
                outlier_indices=out_idx,
                bias=bias
            )
            sub = layer
            parts = proj.split(".")
            for p in parts[:-1]:
                sub = getattr(sub, p)
            setattr(sub, parts[-1], layer_sym)

    fix_rotary_embeddings(model, hf_config, device)
    del base_tensors, quant_tensors
    gc.collect()
    torch.cuda.empty_cache()

    model_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"✅ OTF 2-Bit Engine Loaded in VRAM: {model_vram:.2f} MB\n", flush=True)

    # 3. Расчет расширенных метрик четности
    print("=" * 85, flush=True)
    print("📊 PHASE 3: Mathematical Multi-Tier Parity Evaluation...", flush=True)
    print("=" * 85, flush=True)

    raw_similarities = []
    prob_similarities = []
    top64_similarities = []
    top1_matches = 0
    top5_matches = 0

    print(f"{'#':<3} | {'Prompt Snippet':<36} | {'Raw Cos':<8} | {'Prob Cos':<9} | {'Top-64 Cos':<10} | {'Top-1'}")
    print("-" * 85)

    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        outputs = model(input_ids)
        quant_logit = outputs.logits[:, -1, :].detach().float().cpu()

        base_logit = baseline_logits[idx]

        # 1. Raw Logit Cosine Similarity (All 152k tokens)
        raw_cos = F.cosine_similarity(base_logit, quant_logit, dim=-1).item() * 100.0
        raw_similarities.append(raw_cos)

        # 2. Softmax Probability Cosine Similarity (Actual generative distribution)
        base_prob = F.softmax(base_logit, dim=-1)
        quant_prob = F.softmax(quant_logit, dim=-1)
        prob_cos = F.cosine_similarity(base_prob, quant_prob, dim=-1).item() * 100.0
        prob_similarities.append(prob_cos)

        # 3. Top-64 Subspace Cosine Similarity
        top64_idx = torch.topk(base_logit, k=64, dim=-1).indices
        base_top64 = torch.gather(base_logit, dim=-1, index=top64_idx)
        quant_top64 = torch.gather(quant_logit, dim=-1, index=top64_idx)
        top64_cos = F.cosine_similarity(base_top64, quant_top64, dim=-1).item() * 100.0
        top64_similarities.append(top64_cos)

        # Top-1 Match
        top1_base = torch.argmax(base_logit, dim=-1).item()
        top1_quant = torch.argmax(quant_logit, dim=-1).item()
        is_top1 = (top1_base == top1_quant)
        if is_top1:
            top1_matches += 1

        # Top-5 Match
        top5_base = set(torch.topk(base_logit, k=5, dim=-1).indices[0].tolist())
        if top1_quant in top5_base:
            top5_matches += 1

        snippet = (prompt[:34] + "...") if len(prompt) > 34 else prompt
        print(f"{idx+1:02d}  | {snippet:<36} | {raw_cos:6.2f}% | {prob_cos:7.2f}% | {top64_cos:8.2f}%  | {'YES' if is_top1 else 'NO'}")

    mean_raw_cos = sum(raw_similarities) / len(raw_similarities)
    mean_prob_cos = sum(prob_similarities) / len(prob_similarities)
    mean_top64_cos = sum(top64_similarities) / len(top64_similarities)
    top1_acc = (top1_matches / len(BENCHMARK_PROMPTS)) * 100.0
    top5_acc = (top5_matches / len(BENCHMARK_PROMPTS)) * 100.0

    print("=" * 85)
    print("🏆 FINAL SCIENTIFIC PARITY REPORT:")
    print(f"   • Softmax Probability Cosine Sim:  {mean_prob_cos:.2f}% (Generative Parity)")
    print(f"   • Top-64 Logit Subspace Cosine:    {mean_top64_cos:.2f}% (Decision Boundary Parity)")
    print(f"   • Full-Vocab Raw Logit Cosine:     {mean_raw_cos:.2f}% (152k Tokens with Tail Noise)")
    print(f"   • Greedy Top-1 Token Agreement:    {top1_acc:.2f}%")
    print(f"   • Top-5 Confidence Inclusion:      {top5_acc:.2f}%")
    print(f"   • Static Model VRAM Footprint:     {model_vram:.2f} MB")
    print("=" * 85)


if __name__ == "__main__":
    m_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen-7B-2Bit-Sym"
    o_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"
    test_formal_parity(m_dir, o_id)