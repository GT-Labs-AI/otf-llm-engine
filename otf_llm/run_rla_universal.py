"""
OTF-LLM Engine v4.0 - Universal Grouped RLA Inference Runner
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy. MIT License.

Loads Grouped RLA quantized base layers and micro-adapters to execute ultra-low VRAM LLM generation.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from otf_llm.otf_recurrent_layer import RLALinear


def fix_rotary_embeddings(model: nn.Module, hf_config: AutoConfig, device: str):
    """Safely re-initializes RoPE inv_freq buffers on CUDA using hf_config.rope_theta."""
    rope_theta = getattr(hf_config, "rope_theta", 1000000.0)

    for module in model.modules():
        if "RotaryEmbedding" in type(module).__name__:
            dim = getattr(module, "dim", None)
            if dim is None and hasattr(module, "head_dim"):
                dim = module.head_dim
            if dim is None and hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
                dim = hf_config.hidden_size // hf_config.num_attention_heads

            if dim is not None:
                inv_freq = 1.0 / (
                    rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
                )
                module.register_buffer("inv_freq", inv_freq, persistent=False)


def run_rla_inference(
    rla_dir: str,
    original_model_id: str,
    prompt: str = "Explain quantum computing in simple terms:",
    max_new_tokens: int = 100
):
    print("=" * 70, flush=True)
    print(f"🚀 OTF-LLM v4.0: Grouped RLA Inference Runner (Directory: {rla_dir})", flush=True)
    print("=" * 70, flush=True)

    config_path = os.path.join(rla_dir, "rla_config.json")
    if not os.path.exists(config_path):
        print(f"❌ ERROR: RLA Config '{config_path}' not found!", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ CUDA GPU is required for RLA inference!", flush=True)
        return

    torch.cuda.reset_peak_memory_stats()

    # 1. Load Tokenizer & Config
    print(f"📥 Loading Tokenizer and Config from '{original_model_id}'...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(original_model_id, trust_remote_code=True)

    # 2. Load RLA Metadata Config
    with open(config_path, "r", encoding="utf-8") as f:
        rla_meta = json.load(f)

    num_layers = rla_meta["num_layers"]
    num_groups = rla_meta.get("num_groups", 3)
    layers_per_group = (num_layers + num_groups - 1) // num_groups
    projections = rla_meta["projections"]
    print(f"📊 RLA Config: {num_layers} layers across {num_groups} Groups, rank r={rla_meta['rank']}", flush=True)

    # 3. Instantiate Skeleton Model on META device
    print("🦴 Constructing skeleton model on meta device...", flush=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, dtype=torch.float16)

    # Prune heavy projection layers on META device before to_empty()
    for i in range(num_layers):
        layer_obj = model.model.layers[i]
        for proj in projections:
            proj_parts = proj.split(".")
            submodule = layer_obj
            for part in proj_parts[:-1]:
                submodule = getattr(submodule, part)
            setattr(submodule, proj_parts[-1], nn.Identity())

    # 4. Move skeleton to CUDA
    print("⚡ Converting non-recurrent skeleton to CUDA...", flush=True)
    model = model.to_empty(device=device)

    # 5. Load Base Weights and Adapters from Safetensors directly into CUDA
    base_path = os.path.join(rla_dir, "rla_model_base.safetensors")
    adapter_path = os.path.join(rla_dir, "rla_model_adapters.safetensors")

    base_tensors = {}
    with safe_open(base_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)

    adapter_tensors = {}
    with safe_open(adapter_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            adapter_tensors[k] = f.get_tensor(k)

    # Copy non-recurrent weights into CUDA model
    model_state = model.state_dict()
    for k, tensor in base_tensors.items():
        if not k.startswith("rla.base"):
            if k in model_state:
                model_state[k].copy_(tensor)

    # 6. Inject Grouped RLA Linear Layers on CUDA AFTER to_empty() call!
    print("⚡ Injecting pristine Grouped RLA Base Weights and Micro-Adapters...", flush=True)
    for proj in projections:
        for g in range(num_groups):
            g_start = g * layers_per_group
            g_end = min(num_layers, (g + 1) * layers_per_group)

            base_key = f"rla.base_g{g}.{proj}.weight"
            # Fallback for single base models
            if base_key not in base_tensors:
                base_key = f"rla.base.{proj}.weight"

            base_weight = base_tensors[base_key].to(device)

            d_out, d_in = base_weight.shape
            base_module = nn.Linear(d_in, d_out, bias=False).to(device)
            base_module.weight.data = base_weight.to(torch.float16)

            for i in range(g_start, g_end):
                A = adapter_tensors[f"model.layers.{i}.{proj}.adapter_A"].to(device)
                B = adapter_tensors[f"model.layers.{i}.{proj}.adapter_B"].to(device)

                bias_key = f"model.layers.{i}.{proj}.bias"
                bias_tensor = adapter_tensors.get(bias_key, None)
                if bias_tensor is not None:
                    bias_tensor = bias_tensor.to(device)

                out_deltas_key = f"model.layers.{i}.{proj}.outlier_deltas"
                out_idx_key = f"model.layers.{i}.{proj}.outlier_indices"

                out_deltas = adapter_tensors.get(out_deltas_key, None)
                out_idx = adapter_tensors.get(out_idx_key, None)

                if out_deltas is not None:
                    out_deltas = out_deltas.to(device)
                if out_idx is not None:
                    out_idx = out_idx.to(device)

                rla_layer = RLALinear(
                    base_layer=base_module,
                    adapter_A=A,
                    adapter_B=B,
                    outlier_deltas_fp16=out_deltas,
                    outlier_indices=out_idx,
                    bias=bias_tensor
                )

                layer_obj = model.model.layers[i]
                proj_parts = proj.split(".")
                submodule = layer_obj
                for part in proj_parts[:-1]:
                    submodule = getattr(submodule, part)

                setattr(submodule, proj_parts[-1], rla_layer)

    # Re-initialize RoPE positional embeddings
    fix_rotary_embeddings(model, hf_config, device)

    print("✅ Model successfully assembled with Grouped RLA Engine!", flush=True)

    loaded_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"💾 Total Model Static VRAM Footprint: {loaded_vram_mb:.2f} MB", flush=True)
    print("-" * 70, flush=True)

    # 7. Run Autoregressive Generation
    print(f"📝 Prompt: '{prompt}'", flush=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print("⚡ Generating tokens...", flush=True)
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_time = time.time() - start_time
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = new_tokens / gen_time if gen_time > 0 else 0

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("-" * 70, flush=True)
    print("🎯 GENERATED TEXT:", flush=True)
    print(generated_text, flush=True)
    print("-" * 70, flush=True)
    print(f"📊 PERFORMANCE METRICS:", flush=True)
    print(f"   • Generated Tokens: {new_tokens}")
    print(f"   • Generation Time:  {gen_time:.2f} seconds")
    print(f"   • Inference Speed:  {tokens_per_sec:.2f} tokens/sec")
    print(f"   • Peak VRAM Usage:  {peak_vram_mb:.2f} MB")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python otf_llm/run_rla_universal.py <rla_model_dir> <original_model_id_or_path>")
        sys.exit(1)

    rla_path = sys.argv[1]
    orig_id = sys.argv[2]

    run_rla_inference(rla_path, orig_id)