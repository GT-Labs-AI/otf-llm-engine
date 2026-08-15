# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# tests/test_intelligence_suite.py

import os
import time
import gc
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from otf_llm import QuantizedEmbedding, TritonGlobalSymmetricLinear, fix_rope_position_embeddings

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


def run_benchmark_test(model, tokenizer, test_id, test_name, prompt, max_tokens=250):
    print("=" * 80)
    print(f"🧪 TEST {test_id}: {test_name.upper()}")
    print("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    t_end = time.time()

    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    total_time = t_end - t_start
    tps = gen_tokens / total_time

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"📊 Execution Metrics:")
    print(f"  • Time:          {total_time:.2f} sec ({tps:.2f} tokens/sec)")
    print(f"  • Peak VRAM:     {vram_peak:.2f} MB ({vram_peak / 1024:.2f} GB)")
    print(f"\n📝 Model Response:\n")
    print(response)
    print("\n" + "-" * 80 + "\n")

    return {
        "test_id": test_id,
        "name": test_name,
        "time": total_time,
        "tps": tps,
        "vram": vram_peak,
        "response": response
    }


def run_full_intelligence_suite():
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Empty skeleton on meta device
    with torch.device("meta"):
        raw_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    model = raw_model.to_empty(device="cpu")
    fix_rope_position_embeddings(model, config)

    # 1. Substitute Embeddings
    old_emb = model.model.embed_tokens
    model.model.embed_tokens = QuantizedEmbedding(old_emb.num_embeddings, old_emb.embedding_dim, original_emb=None)

    # 2. Substitute Transformer Layers
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

    # 3. Substitute lm_head
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        model.lm_head = TritonGlobalSymmetricLinear(
            in_features=model.lm_head.in_features,
            out_features=model.lm_head.out_features,
            bias=(model.lm_head.bias is not None)
        )

    clean_name = MODEL_ID.split("/")[-1].lower().replace("-", "_")
    save_path = f"otf_{clean_name}_compressed.safetensors"

    print(f"📥 Loading compressed safetensors checkpoint {save_path}...")
    from safetensors.torch import load_file
    state_dict = load_file(save_path)

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

    if hasattr(model, "lm_head") and isinstance(model.lm_head, TritonGlobalSymmetricLinear):
        if not model.lm_head.is_calibrated:
            model.lm_head.tied_embedding = model.model.embed_tokens

    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vram_stat = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    print(f"🚀 Triton Engine Loaded in: {time.time() - t0:.2f} sec")
    print(f"💾 Static Weight VRAM:      {vram_stat:.2f} MB ({vram_stat / 1024:.2f} GB)\n")

    # Warmup
    dummy = tokenizer("Test", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**dummy, max_new_tokens=5, do_sample=False)

    results = []

    # TEST 1: Multi-Step Spatial & Logic Reasoning
    p1 = """I have 5 boxes placed in a row from left to right (1, 2, 3, 4, 5).
1. I put a red ball in box 2 and a blue ball in box 4.
2. I swap the contents of box 1 and box 4.
3. Then I take whatever is in box 2 and move it to box 5.
4. Finally, I reverse the entire row of boxes (box 5 becomes box 1, box 4 becomes box 2, etc.).

Which box from left to right now contains the blue ball, and which contains the red ball? Answer step by step."""

    results.append(run_benchmark_test(model, tokenizer, 1, "Multi-Step Spatial & Logic Reasoning", p1, max_tokens=350))

    # TEST 2: Strict Constraints (Avoid letter 'e')
    p2 = """Write a short story (3-4 sentences) about an astronaut on Mars.
Rules:
1. It is strictly forbidden to use the letter 'e' (neither lowercase nor uppercase) in the entire text.
2. Each sentence must start on a new line.
3. Reply ONLY with the story text, without introductory or concluding remarks."""

    results.append(run_benchmark_test(model, tokenizer, 2, "Strict Constraint Handling (No Letter 'E')", p2, max_tokens=200))

    # TEST 3: Code Generation & Edge Cases
    p3 = """Write a Python function `compress_string(s: str) -> str` that performs RLE (Run-Length Encoding) compression (e.g., 'aabccca' -> 'a2b1c3a1').
Requirements:
1. The algorithm must correctly handle empty strings and single-character strings.
2. Time complexity must be O(N) and auxiliary space complexity must be O(1) (excluding output string).
3. Include type hints and a docstring with examples. Write 3 unit tests using `assert`."""

    results.append(run_benchmark_test(model, tokenizer, 3, "RLE Code Generation & Edge Cases", p3, max_tokens=400))

    # TEST 4: Shallow Reasoning Trap / Riddle
    p4 = """Solve the riddle:
A man points at a portrait and says: 'Brothers and sisters I have none, but this man's father is my father's son.' The man claims he is looking at himself.
Is this statement correct? If not, who is actually depicted in the portrait and why?"""

    results.append(run_benchmark_test(model, tokenizer, 4, "Shallow Reasoning Trap / Riddle", p4, max_tokens=250))

    print("=" * 80)
    print("🏆 ALL 4 INTELLIGENCE SUITE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_intelligence_suite()