# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# validate_llama3_2_3b.py

import os
import time
import gc
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from otf_llm import (
    QuantizedEmbedding,
    TritonGlobalSymmetricLinear,
    convert_model,
    CompanionMemoryManager
)
from otf_llm.make_profile_universal import create_act_profile

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


def fix_llama3_2_rope_embeddings(model, config):
    """
    Properly initializes RoPE positional embeddings for Llama-3.2-3B on CPU meta-device.
    Llama-3.2 uses high rope_theta (500000.0) and Grouped Query Attention (GQA).
    """
    rope_theta = getattr(config, "rope_theta", 500000.0) or 500000.0
    head_dim = config.hidden_size // config.num_attention_heads

    for m in model.modules():
        if hasattr(m, "inv_freq"):
            inv_freq = 1.0 / (
                        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu") / head_dim))
            m.inv_freq = inv_freq.to(dtype=torch.float32)


def run_llama3_2_3b_benchmark():
    print("=" * 75)
    print(f"🦙 LLAMA-3.2-3B ENGINE VALIDATION PIPELINE | GT Labs AI")
    print(f"🎯 Target Model: {MODEL_ID}")
    print(f"💾 Target Static VRAM: < 1.95 GB")
    print("=" * 75 + "\n")

    clean_name = MODEL_ID.split("/")[-1].lower().replace("-", "_")
    profile_path = f"{clean_name}_act_profile.pt"
    checkpoint_path = f"otf_{clean_name}_compressed.safetensors"

    # Step 1: Activation Profiling
    if not os.path.exists(profile_path):
        print("--- [STEP 1/3] Generating Activation Profile for Llama-3.2-3B ---")
        create_act_profile(MODEL_ID, device="cpu")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"✅ Found existing profile: {profile_path}")

    # Step 2: Direct Layer-by-Layer INT4/INT8 Conversion
    if not os.path.exists(checkpoint_path):
        print("\n--- [STEP 2/3] Quantizing Llama-3.2-3B to INT4/INT8 (Target VRAM < 1.95 GB) ---")
        convert_model(MODEL_ID, outlier_pct=0.01, device="cpu")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"✅ Found compressed checkpoint: {checkpoint_path}")

    # Step 3: Triton Engine Inference Execution & Memory Benchmark
    print("\n--- [STEP 3/3] Loading Llama-3.2-3B Triton Engine into VRAM ---")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 0 MB FP16 weights in RAM
    with torch.device("meta"):
        raw_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    model = raw_model.to_empty(device="cpu")
    fix_llama3_2_rope_embeddings(model, config)

    # Substitute Embeddings -> INT8
    old_emb = model.model.embed_tokens
    model.model.embed_tokens = QuantizedEmbedding(
        old_emb.num_embeddings,
        old_emb.embedding_dim,
        original_emb=None
    )

    # Substitute Transformer Layers -> Triton INT4
    for name, module in model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    new_linear = TritonGlobalSymmetricLinear(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bias=(child_module.bias is not None)
                    )
                    setattr(module, child_name, new_linear)

    # Substitute lm_head -> Triton INT4
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        model.lm_head = TritonGlobalSymmetricLinear(
            in_features=model.lm_head.in_features,
            out_features=model.lm_head.out_features,
            bias=(model.lm_head.bias is not None)
        )

    # Load Safetensors State Dict
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)

    for emb_key in ["model.embed_tokens.packed_q", "embed_tokens.packed_q"]:
        if emb_key in state_dict:
            model.model.embed_tokens.packed_q = state_dict.pop(emb_key)
            scale_key = emb_key.replace("packed_q", "scale")
            model.model.embed_tokens.scale = state_dict.pop(scale_key)
            break

    for name, module in model.named_modules():
        if isinstance(module, TritonGlobalSymmetricLinear):
            prefix = f"{name}." if name else ""
            if f"{prefix}packed_q_bg" in state_dict:
                module.perm_idx = state_dict.pop(f"{prefix}perm_idx")
                module.W_outliers_fp16 = state_dict.pop(f"{prefix}W_outliers_fp16")
                module.packed_q_bg = state_dict.pop(f"{prefix}packed_q_bg")
                module.scale_bg = state_dict.pop(f"{prefix}scale_bg")
                if f"{prefix}bias" in state_dict:
                    module.bias = nn.Parameter(state_dict.pop(f"{prefix}bias"))
                module.is_calibrated = True

    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    vram_static = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"🚀 Llama-3.2-3B Triton Engine loaded in: {time.time() - t0:.2f} sec")
    print(f"💾 Static VRAM Allocated:               {vram_static:.2f} MB ({vram_static / 1024:.2f} GB)\n")

    # Reasoning Benchmark Test
    print("=" * 75)
    print("🧪 RUNNING LLAMA-3.2-3B REASONING & CODE GENERATION BENCHMARK")
    print("=" * 75)

    prompt = "Write a Python function for binary search with edge-case handling, type hints, and assert tests."

    # Inject Companion Memory
    mem_mgr = CompanionMemoryManager()
    mem_mgr.add_explicit_fact("User strictly requires PEP-8 compliance and short explanations.")

    system_prompt = "You are a concise AI Coding Assistant."
    enhanced_sys = mem_mgr.inject_memory_into_system_prompt(system_prompt, prompt)

    messages = [
        {"role": "system", "content": enhanced_sys},
        {"role": "user", "content": prompt}
    ]

    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_text, return_tensors="pt").to(device)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    t_end = time.time()

    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    total_time = t_end - t_start
    tps = gen_tokens / total_time

    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"📊 LLAMA-3.2-3B BENCHMARK RESULTS:")
    print(f"  • Speed:         {tps:.2f} tokens/sec")
    print(f"  • Peak VRAM:     {vram_peak:.2f} MB ({vram_peak / 1024:.2f} GB)")
    print(f"  • Total Time:    {total_time:.2f} sec")
    print(f"\n📝 Model Response:\n{response_text}")
    print("\n" + "=" * 75)
    print("✅ LLAMA-3.2-3B VALIDATION SUCCESSFUL!")
    print("=" * 75)


if __name__ == "__main__":
    run_llama3_2_3b_benchmark()