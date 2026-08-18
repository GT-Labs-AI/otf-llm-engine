"""
OTF-LLM Engine: Conversational & Reasoning Activation Profiler (v4.1.4)
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.

Focused 100% on Dialogue, Document Analysis, Logical Reasoning, and Conceptual Synthesis.
"""

import os
import argparse
import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 🚀 Чистый разговорный, логический и аналитический набор (0% кода)
CALIBRATION_SUITE = [
    # 1. Интуитивные объяснения сложных концепций
    "Explain the concept of quantum computing, superposition, and entanglement using intuitive real-world analogies.",
    "Describe the fundamental differences between human intuition and artificial neural networks in decision making.",

    # 2. Логические рассуждения и цепочки мыслей (Chain-of-Thought)
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. Provide a clear, step-by-step logical proof of the ball's cost.",
    "If all roses are flowers and some flowers fade quickly, can we logically conclude that some roses fade quickly? Analyze rigorously.",

    # 3. Анализ документов, синтез и саммаризация (RLM & Long Context)
    "Summarize the key trade-offs between memory bandwidth, computational latency, and model compression in modern AI hardware.",
    "Analyze how open-source artificial intelligence enables breakthroughs on consumer-grade hardware and democratizes research.",

    # 4. Глубокий русский диалог и рассуждения
    "Объясни простыми словами, как работает квантование нейросетей до 2 бит и почему сохранение логики возможно при потере точности.",
    "Напиши структурированное аналитическое эссе о перспективах развития локальных ИИ-ассистентов, работающих без интернета.",

    # 5. Структурированные форматы и многошаговое планирование
    "Provide a detailed, step-by-step strategic plan for optimizing team productivity during high-uncertainty R&D projects.",
    "Synthesize the historical evolution of scientific paradigms from Newtonian determinism to quantum probabilities."
]


def create_act_profile(model_id: str, device: str = "cuda", force_recreate: bool = False):
    clean_name = model_id.split("/")[-1].lower().replace("-", "_")
    profile_path = f"{clean_name}_act_profile.pt"

    if os.path.exists(profile_path) and not force_recreate:
        print(f"✅ Found existing conversational activation profile: {profile_path}", flush=True)
        return profile_path

    print("=" * 75, flush=True)
    print(f"🎯 CONVERSATIONAL & REASONING ACTIVATION PROFILER: {model_id}", flush=True)
    print(f"💻 Device: {device.upper()} | Calibration Prompts: {len(CALIBRATION_SUITE)}", flush=True)
    print("=" * 75, flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    model.eval()

    importance_dict = {}

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().abs().float()
            # Mean activation magnitude per channel
            mean_x = x.reshape(-1, x.shape[-1]).mean(dim=0).cpu()
            if name not in importance_dict:
                importance_dict[name] = mean_x
            else:
                importance_dict[name] += mean_x
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("mlp" in name or "self_attn" in name or name == "lm_head"):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print("📥 Profiling conversational & reasoning attention pathways...", flush=True)
    with torch.no_grad():
        for idx, p in enumerate(CALIBRATION_SUITE):
            inputs = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inputs)
            print(f"   [{idx+1:02d}/{len(CALIBRATION_SUITE):02d}] Profiled: '{p[:45]}...'", flush=True)

    for h in hooks:
        h.remove()

    # Normalize accumulated importance
    for k in importance_dict:
        importance_dict[k] = importance_dict[k] / len(CALIBRATION_SUITE)

    torch.save(importance_dict, profile_path)

    hooks.clear()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"⚡ Conversational Profile saved in {time.time() - t0:.2f}s -> {profile_path}\n", flush=True)
    return profile_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational LLM Activation Profiler")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--force", action="store_true", help="Force recreate profile")
    args = parser.parse_args()

    create_act_profile(args.model_id, args.device, force_recreate=args.force)