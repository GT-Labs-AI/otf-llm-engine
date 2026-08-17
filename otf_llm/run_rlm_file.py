"""
OTF-LLM Engine: Real-File RLM Runner (Context-as-a-Variable)
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import argparse
import re
import gc
import json
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from safetensors.torch import load_file

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from otf_llm.otf_2bit_quantizer import OTF2BitLinear
from otf_llm.rlm_agent import ContextContainer, PythonREPLExecutor


def load_model_2bit_engine(model_dir: str, base_model_id: str, device: str = "cuda"):
    config_file = os.path.join(model_dir, "otf_2bit_config.json")
    base_weights_file = os.path.join(model_dir, "otf_2bit_base.safetensors")
    quant_weights_file = os.path.join(model_dir, "otf_2bit_model.safetensors")

    with open(config_file, "r", encoding="utf-8") as f:
        quant_config = json.load(f)

    group_size = quant_config.get("group_size", 32)

    base_tensors = load_file(base_weights_file, device="cpu")
    quant_tensors = load_file(quant_weights_file, device="cpu")

    hf_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

    # Pre-inject 2-bit linear modules
    for name, module in list(model.named_modules()):
        packed_key = f"{name}.packed_uint8"
        if packed_key in quant_tensors:
            packed_weights = quant_tensors[packed_key]
            scales = quant_tensors[f"{name}.scales"]
            zeros = quant_tensors[f"{name}.zeros"]

            out_deltas = quant_tensors.get(f"{name}.outlier_deltas", None)
            out_idx = quant_tensors.get(f"{name}.outlier_indices", None)
            bias = quant_tensors.get(f"{name}.bias", None)

            layer = OTF2BitLinear(
                packed_uint8=packed_weights.to(device=device),
                scales=scales.to(device=device, dtype=torch.float16),
                zeros=zeros.to(device=device, dtype=torch.float16),
                d_in=module.in_features,
                group_size=group_size,
                outliers_fp16=out_deltas.to(device=device, dtype=torch.float16) if out_deltas is not None else None,
                outlier_indices=out_idx.to(device=device, dtype=torch.long) if out_idx is not None else None,
                bias=bias.to(device=device, dtype=torch.float16) if bias is not None else None
            )

            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr_name, layer)

    # Allocate non-quantized base layers (Embeddings, LayerNorms)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)) or "norm" in name.lower():
            module.to_empty(device=device)

    base_tensors_cuda = {k: v.to(device=device, dtype=torch.float16) for k, v in base_tensors.items()}
    model.load_state_dict(base_tensors_cuda, strict=False)
    del base_tensors, base_tensors_cuda, quant_tensors
    torch.cuda.empty_cache()
    gc.collect()

    # RoPE initialization
    for m in model.modules():
        if hasattr(m, "rotary_emb"):
            rot = m.rotary_emb
            if hasattr(rot, "_set_cos_sin_cache"):
                rot._set_cos_sin_cache(seq_len=2048, device=device, dtype=torch.float16)
            if hasattr(rot, "inv_freq") and rot.inv_freq is not None:
                dim = rot.inv_freq.shape[0] * 2
                base = getattr(rot, "base", 10000.0)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                rot.register_buffer("inv_freq", inv_freq, persistent=False)
            if hasattr(rot, "max_seq_len_cached"):
                rot.max_seq_len_cached = 0

    if getattr(hf_config, "tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight

    model.eval()
    return model


def run_rlm_on_file(
    model_dir: str,
    base_model_id: str,
    file_path: str,
    query: str,
    max_turns: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 75)
    print(f"🚀 GT Labs AI — Clean Architecture RLM Engine")
    print(f"   Model Directory: {model_dir}")
    print(f"   Target File:     {file_path}")
    print(f"   Query:           {query}")
    print("=" * 75)

    if not os.path.exists(file_path):
        print(f"❌ Error: Target file '{file_path}' does not exist!")
        return

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        file_content = f.read()

    ctx = ContextContainer(file_content, name="ctx")
    print(f"📦 Loaded Context: `{ctx.describe()}` ({len(file_content) / 1024:.1f} KB)")

    print("\n📦 Initializing 2-Bit Inference Engine...")
    model = load_model_2bit_engine(model_dir, base_model_id, device=device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 True Peak Static VRAM: {vram_gb:.2f} GB")

    def llm_generate(messages: list, max_tokens: int = 256) -> str:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    system_prompt = (
        "You are an expert AI Systems Engineer analyzing a codebase loaded in variable `ctx`.\n"
        "To inspect code or configs, use `print(ctx.grep('keyword'))`."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    env_vars = {"ctx": ctx}

    for turn in range(1, max_turns + 1):
        print(f"\n🧠 [Turn {turn}/{max_turns}] Model Planning Action...", flush=True)
        response = llm_generate(messages, max_tokens=160 if turn == 1 else 256)
        print(f"💬 LLM Output:\n{response}\n")

        messages.append({"role": "assistant", "content": response})

        # Dynamic Tool Execution with Stopword Filtering
        res = PythonREPLExecutor.execute(response, env_vars, fallback_query=query if turn == 1 else "")

        if turn >= max_turns or not res["executed_code"]:
            print("✅ Final Answer Synthesis Complete.")
            break

        print(f"⚡ Executed Clean Tool Actions:\n{res['executed_code']}")
        obs = res["stdout"] if res["stdout"] else "[No relevant matches]"
        print(f"📥 Environment Observation:\n{obs}")

        # Tightly structured synthesis prompt
        messages.append({
            "role": "user",
            "content": (
                f"Facts extracted from codebase:\n{obs}\n\n"
                f"Based strictly on the extracted facts above, answer the 3 questions directly:\n"
                f"1) Exact values of LLOYD_MAX_2BIT_CENTROIDS\n"
                f"2) Author's contact email\n"
                f"3) Taboo Rule #4 regarding to_empty()"
            )
        })

    print("\n" + "=" * 75)
    print("🏁 RLM CODEBASE AUDIT FINISHED")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clean RLM on a Local File")
    parser.add_argument("file_path", type=str, help="Path to input text/log/code file")
    parser.add_argument("query", type=str, help="Target question or task")
    parser.add_argument("--model_dir", type=str, default="./models/Qwen-3B-2Bit", help="Path to 2-Bit model directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base HF model repository")

    args = parser.parse_args()
    run_rlm_on_file(args.model_dir, args.base_model, args.file_path, args.query)