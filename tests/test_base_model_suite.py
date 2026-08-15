# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# tests/test_base_model_suite.py

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


def run_benchmark_test(model, tokenizer, test_id, test_name, prompt, max_tokens=250):
    print("=" * 80)
    print(f"🧪 [BASELINE FP16] TEST {test_id}: {test_name.upper()}")
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

    print(f"📊 Baseline FP16 Execution Metrics:")
    print(f"  • Time:          {total_time:.2f} sec ({tps:.2f} tokens/sec)")
    print(f"  • Peak VRAM:     {vram_peak:.2f} MB ({vram_peak / 1024:.2f} GB)")
    print(f"\n📝 Baseline FP16 Response:\n")
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


def run_baseline_suite():
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"📥 Loading baseline raw model {MODEL_ID} (Pure FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device
    )

    vram_stat = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    print(f"🚀 Baseline FP16 Model Loaded in: {time.time() - t0:.2f} sec")
    print(f"💾 Baseline FP16 Static VRAM:     {vram_stat:.2f} MB ({vram_stat / 1024:.2f} GB)\n")

    # Warmup
    dummy = tokenizer("Test", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**dummy, max_new_tokens=5, do_sample=False)

    results = []

    # TEST 1
    p1 = """I have 5 boxes placed in a row from left to right (1, 2, 3, 4, 5).
1. I put a red ball in box 2 and a blue ball in box 4.
2. I swap the contents of box 1 and box 4.
3. Then I take whatever is in box 2 and move it to box 5.
4. Finally, I reverse the entire row of boxes (box 5 becomes box 1, box 4 becomes box 2, etc.).

Which box from left to right now contains the blue ball, and which contains the red ball? Answer step by step."""

    results.append(run_benchmark_test(model, tokenizer, 1, "Multi-Step Spatial & Logic Reasoning", p1, max_tokens=350))

    # TEST 2
    p2 = """Write a short story (3-4 sentences) about an astronaut on Mars.
Rules:
1. It is strictly forbidden to use the letter 'e' (neither lowercase nor uppercase) in the entire text.
2. Each sentence must start on a new line.
3. Reply ONLY with the story text, without introductory or concluding remarks."""

    results.append(run_benchmark_test(model, tokenizer, 2, "Strict Constraint Handling (No Letter 'E')", p2, max_tokens=200))

    # TEST 3
    p3 = """Write a Python function `compress_string(s: str) -> str` that performs RLE (Run-Length Encoding) compression (e.g., 'aabccca' -> 'a2b1c3a1').
Requirements:
1. The algorithm must correctly handle empty strings and single-character strings.
2. Time complexity must be O(N) and auxiliary space complexity must be O(1) (excluding output string).
3. Include type hints and a docstring with examples. Write 3 unit tests using `assert`."""

    results.append(run_benchmark_test(model, tokenizer, 3, "RLE Code Generation & Edge Cases", p3, max_tokens=400))

    # TEST 4
    p4 = """Solve the riddle:
A man points at a portrait and says: 'Brothers and sisters I have none, but this man's father is my father's son.' The man claims he is looking at himself.
Is this statement correct? If not, who is actually depicted in the portrait and why?"""

    results.append(run_benchmark_test(model, tokenizer, 4, "Shallow Reasoning Trap / Riddle", p4, max_tokens=250))

    print("=" * 80)
    print("🏆 BASELINE FP16 SUITE TESTING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_suite()