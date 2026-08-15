# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_llm/make_profile_universal.py

import os
import argparse
import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_act_profile(model_id: str, device: str = "cpu"):
    clean_name = model_id.split("/")[-1].lower().replace("-", "_")
    profile_path = f"{clean_name}_act_profile.pt"

    if os.path.exists(profile_path):
        print(f"✅ Found existing activation profile: {profile_path}")
        return profile_path

    print("=" * 70)
    print(f"🎯 ACTIVATION PROFILE GENERATOR: {model_id}")
    print(f"💻 Execution Device: {device.upper()}")
    print("=" * 70)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )

    importance_dict = {}

    def make_hook(name):
        def hook(module, input, output):
            x = input[0].detach().abs().float()
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

    print("📥 Running calibration prompt suite...")
    prompts = [
        "Write a complex Python function to solve the Traveling Salesperson Problem with dynamic programming.",
        "Explain the internal mechanics of Transformer self-attention and Rotary Position Embeddings (RoPE).",
        "Draft a detailed VRAM optimization guide for running large language models on consumer GPUs."
    ]

    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inputs)

    for h in hooks:
        h.remove()

    torch.save(importance_dict, profile_path)

    # Aggressive RAM & VRAM Purge
    hooks.clear()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(f"⚡ Profile generated & RAM purged in {time.time() - t0:.2f} sec! File: {profile_path}\n")
    return profile_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Activation Profiler")
    parser.add_argument("--model_id", type=str, default="unsloth/Llama-3.2-3B-Instruct", help="Model ID")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    create_act_profile(args.model_id, args.device)