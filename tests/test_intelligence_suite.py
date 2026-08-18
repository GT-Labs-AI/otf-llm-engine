"""
OTF-LLM Engine: Task-Level Capability Retention Benchmark (Zero-RAM Leak Edition)
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import gc
import json
import time
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from otf_llm.symmetric_2bit_engine import (
    SymmetricOTF2BitLinear,
    QuantizedEmbedding,
    QuantizedLinearHead
)
from otf_llm.run_symmetric_2bit import fix_rotary_embeddings

try:
    import psutil
    def get_ram_mb():
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
except ImportError:
    def get_ram_mb():
        return 0.0


CAPABILITY_DATASET = [
    # --- DOMAIN 1: LOGIC & DEDUCTION ---
    {
        "id": "LOGIC-01",
        "domain": "Logic",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Output ONLY the final numeric number in cents.",
        "target_keywords": ["5", "0.05", "five"]
    },
    {
        "id": "LOGIC-02",
        "domain": "Logic",
        "prompt": "Premise 1: All roses are flowers.\nPremise 2: Some flowers fade quickly.\nQuestion: Can we logically conclude with 100% certainty that some roses fade quickly? Answer YES or NO and explain in one sentence.",
        "target_keywords": ["no", "cannot", "invalid", "not necessarily"]
    },
    {
        "id": "LOGIC-03",
        "domain": "Logic",
        "prompt": "If yesterday was Tuesday, what day of the week will it be 4 days after tomorrow?",
        "target_keywords": ["monday"]
    },
    {
        "id": "LOGIC-04",
        "domain": "Logic",
        "prompt": "Mary's father has 5 daughters: Nana, Nene, Nini, Nono. What is the name of the fifth daughter?",
        "target_keywords": ["mary"]
    },

    # --- DOMAIN 2: MATH & ARITHMETIC ---
    {
        "id": "MATH-01",
        "domain": "Math",
        "prompt": "Calculate: (15 * 4) + (120 / 6) - 18. Output the step-by-step calculation and the final number.",
        "target_keywords": ["62"]
    },
    {
        "id": "MATH-02",
        "domain": "Math",
        "prompt": "A store offers a 20% discount on a $150 jacket. After the discount, an 8% sales tax is added. What is the final price of the jacket? Provide the exact numerical dollar amount.",
        "target_keywords": ["129.6", "129.60"]
    },
    {
        "id": "MATH-03",
        "domain": "Math",
        "prompt": "Solve for x: 3x + 15 = 48. What is the value of x?",
        "target_keywords": ["11"]
    },
    {
        "id": "MATH-04",
        "domain": "Math",
        "prompt": "A car travels at 60 mph for 2.5 hours, then at 40 mph for 1.5 hours. What is the total distance traveled in miles?",
        "target_keywords": ["210"]
    },

    # --- DOMAIN 3: REASONING & ANALYSIS ---
    {
        "id": "REASON-01",
        "domain": "Reasoning",
        "prompt": "Explain why increasing memory bandwidth is often more effective than adding more compute TFLOPs for accelerating LLM autoregressive token generation.",
        "target_keywords": ["memory bound", "memory-bound", "bandwidth", "batch", "weights", "transfer", "loading"]
    },
    {
        "id": "REASON-02",
        "domain": "Reasoning",
        "prompt": "What is the core difference between inductive and deductive reasoning? Provide one brief concrete example for each.",
        "target_keywords": ["general", "specific", "premises", "observation", "deductive", "inductive"]
    },
    {
        "id": "REASON-03",
        "domain": "Reasoning",
        "prompt": "Explain why quantum computers cannot simply replace classical computers for every everyday task like word processing or web browsing.",
        "target_keywords": ["superposition", "decoherence", "overhead", "error correction", "classical", "specialized"]
    },
    {
        "id": "REASON-04",
        "domain": "Reasoning",
        "prompt": "Describe the concept of 'Tragedy of the Commons' and how it applies to public grazing land.",
        "target_keywords": ["depletion", "overuse", "individual", "collective", "shared resource", "self-interest"]
    },

    # --- DOMAIN 4: RUSSIAN & MULTILINGUAL ---
    {
        "id": "RU-01",
        "domain": "Russian",
        "prompt": "В комнате горело 7 свечей. Ветром задуло 2 свечи, а еще одну потушил человек. Сколько свечей останется в комнате в итоге?",
        "target_keywords": ["3", "три", "останется 3", "сгорели", "растаяли"]
    },
    {
        "id": "RU-02",
        "domain": "Russian",
        "prompt": "У фермера было 17 овец. Все, кроме 9, убежали. Сколько овец осталось у фермера?",
        "target_keywords": ["9", "девять", "осталось 9"]
    },
    {
        "id": "RU-03",
        "domain": "Russian",
        "prompt": "Объясни в двух предложениях, почему небо голубое днем, но красное на закате.",
        "target_keywords": ["рэлей", "рассеян", "длин", "волн", "атмосфер", "толщ"]
    },
    {
        "id": "RU-04",
        "domain": "Russian",
        "prompt": "Переведи на русский язык и объясни смысл пословицы: 'Actions speak louder than words.'",
        "target_keywords": ["поступк", "дела", "слов", "действи"]
    },

    # --- DOMAIN 5: INSTRUCTION & FORMATTING ---
    {
        "id": "FORMAT-01",
        "domain": "Instruction",
        "prompt": "Generate a valid JSON object with exactly two keys: 'status' (string 'success') and 'code' (integer 200). Do not include any other text outside the JSON.",
        "target_keywords": ['"status"', '"success"', '"code"', '200']
    },
    {
        "id": "FORMAT-02",
        "domain": "Instruction",
        "prompt": "List exactly 3 primary colors in bullet points. Do not write any introduction or conclusion.",
        "target_keywords": ["red", "blue", "yellow"]
    },
    {
        "id": "FORMAT-03",
        "domain": "Instruction",
        "prompt": "Summarize the water cycle in exactly 4 numbered steps (1, 2, 3, 4).",
        "target_keywords": ["evaporation", "condensation", "precipitation", "collection"]
    },
    {
        "id": "FORMAT-04",
        "domain": "Instruction",
        "prompt": "Write a 3-sentence micro-story about a robot discovering a plant, ending with the word 'hope'.",
        "target_keywords": ["hope"]
    }
]


def evaluate_response(response: str, target_keywords: List[str]) -> Tuple[bool, float]:
    resp_lower = response.lower()
    matches = sum(1 for kw in target_keywords if kw.lower() in resp_lower)
    score = matches / len(target_keywords)
    is_correct = score >= 0.5 or (len(target_keywords) == 1 and matches == 1)
    return is_correct, score


@torch.inference_mode()
def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_2bit_dir = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen-7B-2Bit-Sym"
    original_model_id = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)

    eos_token_ids = [tokenizer.eos_token_id]
    for special_tok in ["<|im_end|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id is not None and tok_id not in eos_token_ids:
            eos_token_ids.append(tok_id)

    # -------------------------------------------------------------
    # 1. Сбор результатов FP16 Baseline
    # -------------------------------------------------------------
    print("=" * 80, flush=True)
    print(f"🔬 PHASE 1: Running FP16 Baseline on Capability Benchmark ({original_model_id})...", flush=True)
    print(f"📊 Initial Memory: CPU RAM = {get_ram_mb():.1f} MB | GPU VRAM = {torch.cuda.memory_allocated() / (1024**2):.1f} MB", flush=True)
    print("=" * 80, flush=True)

    fp16_model = AutoModelForCausalLM.from_pretrained(
        original_model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    fp16_model.eval()

    fp16_vram = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"💾 FP16 Loaded: CPU RAM = {get_ram_mb():.1f} MB | GPU VRAM = {fp16_vram:.1f} MB\n", flush=True)

    fp16_results = {}
    t0 = time.time()

    for idx, item in enumerate(CAPABILITY_DATASET):
        prompt_text = item["prompt"]
        formatted = f"<|im_start|>system\nYou are a helpful and precise reasoning assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)

        out = fp16_model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_ids
        )
        gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_ok, score = evaluate_response(gen_text, item["target_keywords"])
        fp16_results[item["id"]] = {"correct": is_ok, "score": score, "text": gen_text}
        print(f"   [{idx+1:02d}/{len(CAPABILITY_DATASET):02d}] FP16 [{item['domain']:<11}]: {'PASSED' if is_ok else 'FAILED'} -> {gen_text[:55]}...", flush=True)

    # 🚀 АГРЕССИВНАЯ ОЧИСТКА FP16 ИЗ ОЗУ И VRAM
    del fp16_model
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    print(f"\n🧹 FP16 Unloaded: CPU RAM = {get_ram_mb():.1f} MB | GPU VRAM = {torch.cuda.memory_allocated() / (1024**2):.1f} MB\n", flush=True)

    # -------------------------------------------------------------
    # 2. Сбор результатов OTF-Engine (Потоковая загрузка)
    # -------------------------------------------------------------
    print("=" * 80, flush=True)
    print(f"⚡ PHASE 2: Running OTF-Engine on Capability Benchmark...", flush=True)
    print("=" * 80, flush=True)

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
        for p in projections:
            setattr(getattr(model.model.layers[i], p.split(".")[0]), p.split(".")[1], nn.Identity())

    model = model.to_empty(device=device)
    model.requires_grad_(False)

    base_path = os.path.join(model_2bit_dir, "otf_2bit_base.safetensors")
    quant_path = os.path.join(model_2bit_dir, "otf_2bit_model.safetensors")

    # 🚀 ПОТОКОВАЯ ИНЖЕКЦИЯ (0 МБ УТЕЧЕК В ОЗУ)
    with safe_open(base_path, framework="pt", device="cpu") as f:
        # Embeddings
        q_emb = QuantizedEmbedding(hf_config.vocab_size, hf_config.hidden_size).to(device)
        q_emb.weight_int8.copy_(f.get_tensor("model.embed_tokens.weight_int8").to(device))
        q_emb.scales.copy_(f.get_tensor("model.embed_tokens.scales").to(device))
        model.model.embed_tokens = q_emb

        # LM Head
        q_head = QuantizedLinearHead(hf_config.hidden_size, hf_config.vocab_size).to(device)
        if "lm_head.weight_int8" in f.keys():
            q_head.weight_int8.copy_(f.get_tensor("lm_head.weight_int8").to(device))
            q_head.scales.copy_(f.get_tensor("lm_head.scales").to(device))
        else:
            q_head.weight_int8 = q_emb.weight_int8
            q_head.scales = q_emb.scales
        model.lm_head = q_head

        # Norms
        if "model.norm.weight" in f.keys():
            model.model.norm.weight.copy_(f.get_tensor("model.norm.weight").to(device))

        for i in range(num_layers):
            for n_suf in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                k = f"model.layers.{i}.{n_suf}"
                if k in f.keys():
                    getattr(model.model.layers[i], n_suf.split(".")[0]).weight.copy_(f.get_tensor(k).to(device))

    # Квантованные слои потоком
    with safe_open(quant_path, framework="pt", device="cpu") as f:
        for i in range(num_layers):
            layer = model.model.layers[i]
            for proj in projections:
                pfx = f"model.layers.{i}.{proj}"
                if f"{pfx}.packed_uint8" not in f.keys():
                    continue

                packed = f.get_tensor(f"{pfx}.packed_uint8").to(device)
                scales = f.get_tensor(f"{pfx}.scales").to(device)

                bias = f.get_tensor(f"{pfx}.bias").to(device) if f"{pfx}.bias" in f.keys() else None
                out_d = f.get_tensor(f"{pfx}.outlier_deltas").to(device) if f"{pfx}.outlier_deltas" in f.keys() else None
                out_i = f.get_tensor(f"{pfx}.outlier_indices").to(device) if f"{pfx}.outlier_indices" in f.keys() else None

                d_in = packed.shape[1] * 4
                layer_sym = SymmetricOTF2BitLinear(
                    packed, scales, d_in, group_size, out_d, out_i, bias
                )
                setattr(getattr(layer, proj.split(".")[0]), proj.split(".")[1], layer_sym)

    fix_rotary_embeddings(model, hf_config, device)
    gc.collect()
    torch.cuda.empty_cache()

    otf_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"✅ OTF-Engine Loaded: CPU RAM = {get_ram_mb():.1f} MB | GPU VRAM = {otf_vram_mb:.1f} MB\n", flush=True)

    otf_results = {}
    for idx, item in enumerate(CAPABILITY_DATASET):
        prompt_text = item["prompt"]
        formatted = f"<|im_start|>system\nYou are a helpful and precise reasoning assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)

        out = model.generate(
            input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_ids
        )
        gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_ok, score = evaluate_response(gen_text, item["target_keywords"])
        otf_results[item["id"]] = {"correct": is_ok, "score": score, "text": gen_text}
        print(f"   [{idx+1:02d}/{len(CAPABILITY_DATASET):02d}] OTF  [{item['domain']:<11}]: {'PASSED' if is_ok else 'FAILED'} -> {gen_text[:55]}...", flush=True)

    # -------------------------------------------------------------
    # 3. Итоговый отчет сохранения способностей (Scorecard)
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("🏆 OFFICIAL TASK-LEVEL CAPABILITY RETENTION REPORT")
    print("=" * 90)
    print(f"{'Task ID':<10} | {'Domain':<12} | {'FP16 Baseline':<15} | {'OTF-Engine (3.46GB)':<20} | {'Status'}")
    print("-" * 90)

    domain_scores = {}

    for item in CAPABILITY_DATASET:
        t_id = item["id"]
        dom = item["domain"]
        fp16_ok = fp16_results[t_id]["correct"]
        otf_ok = otf_results[t_id]["correct"]

        if dom not in domain_scores:
            domain_scores[dom] = {"fp16": 0, "otf": 0, "total": 0}

        domain_scores[dom]["total"] += 1
        if fp16_ok: domain_scores[dom]["fp16"] += 1
        if otf_ok: domain_scores[dom]["otf"] += 1

        status_str = "PRESERVED" if (fp16_ok and otf_ok) else ("NEW PASS" if (not fp16_ok and otf_ok) else ("LOST" if (fp16_ok and not otf_ok) else "BOTH FAILED"))
        print(f"{t_id:<10} | {dom:<12} | {'PASS' if fp16_ok else 'FAIL':<15} | {'PASS' if otf_ok else 'FAIL':<20} | {status_str}")

    print("-" * 90)
    print(f"{'DOMAIN SUMMARY':<24} | {'FP16 Accuracy':<15} | {'OTF Accuracy':<20} | {'Capability Retention'}")
    print("-" * 90)

    total_fp16 = 0
    total_otf = 0
    total_tasks = len(CAPABILITY_DATASET)

    for dom, sc in domain_scores.items():
        fp16_acc = (sc["fp16"] / sc["total"]) * 100.0
        otf_acc = (sc["otf"] / sc["total"]) * 100.0
        retention = (sc["otf"] / sc["fp16"] * 100.0) if sc["fp16"] > 0 else 100.0
        total_fp16 += sc["fp16"]
        total_otf += sc["otf"]
        print(f"{dom:<24} | {fp16_acc:5.1f}%          | {otf_acc:5.1f}%               | {retention:6.1f}%")

    total_fp16_acc = (total_fp16 / total_tasks) * 100.0
    total_otf_acc = (total_otf / total_tasks) * 100.0
    total_retention = (total_otf / total_fp16 * 100.0) if total_fp16 > 0 else 0.0

    print("=" * 90)
    print(f"📊 OVERALL BENCHMARK METRICS:")
    print(f"   • FP16 Baseline Accuracy:       {total_fp16_acc:.1f}% (VRAM: ~15.5 GB)")
    print(f"   • OTF-Engine Accuracy:          {total_otf_acc:.1f}% (VRAM: {otf_vram_mb:.1f} MB)")
    print(f"   • TOTAL CAPABILITY RETENTION:   {total_retention:.1f}%")
    print(f"   • Memory Compression:           {15500.0 / otf_vram_mb:.2f}x less VRAM")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()