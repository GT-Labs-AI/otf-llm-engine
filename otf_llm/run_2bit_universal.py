"""
OTF-LLM Engine v4.0/v4.1 - Universal 2-Bit High-Speed Inference Runner
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy. MIT License.

Loads 2-bit quantized packed uint8 weights, group scales, zeros/symmetric centroids,
and outlier anchors to execute ultra-low VRAM (<1.30 GB) and high-speed LLM generation.
"""

import os
import sys
import json
import time
import gc
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from otf_llm.otf_2bit_quantizer import OTF2BitLinear

DEFAULT_PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
]


def fix_rotary_embeddings(model: nn.Module, hf_config: AutoConfig, device: str):
    """Safely re-initializes RoPE inv_freq buffers on CUDA using hf_config.rope_theta."""
    rope_theta = getattr(hf_config, "rope_theta", 1000000.0)

    for module in model.modules():
        if "RotaryEmbedding" in type(module).__name__:
            dim = getattr(module, "dim", None) or getattr(module, "head_dim", 128)
            if dim is None and hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
                dim = hf_config.hidden_size // hf_config.num_attention_heads

            if dim is not None:
                inv_freq = 1.0 / (
                    rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
                )
                module.register_buffer("inv_freq", inv_freq, persistent=False)
                if hasattr(module, "max_seq_len_cached"):
                    module.max_seq_len_cached = 0


def run_2bit_inference(
    model_2bit_dir: str,
    original_model_id: str,
    prompt: str = "Explain quantum computing in simple terms:",
    max_new_tokens: int = 120
):
    print("=" * 70, flush=True)
    print(f"🚀 OTF-LLM: 2-Bit High-Speed Inference Runner (Directory: {model_2bit_dir})", flush=True)
    print("=" * 70, flush=True)

    config_path = os.path.join(model_2bit_dir, "otf_2bit_config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ CUDA GPU is required for 2-bit inference!", flush=True)
        return

    # 1. Load Tokenizer & Config
    print(f"📥 Loading Tokenizer and Config from '{original_model_id}'...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(original_model_id, trust_remote_code=True)

    # 2. Load 2-Bit Metadata Config safely
    meta_cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            meta_cfg = json.load(f)

    num_layers = meta_cfg.get("num_layers", getattr(hf_config, "num_hidden_layers", 28))
    group_size = meta_cfg.get("group_size", 32)
    projections = meta_cfg.get("projections", meta_cfg.get("target_projections", DEFAULT_PROJECTIONS))

    print(f"📊 2-Bit Config: {num_layers} layers, Group Size = {group_size}", flush=True)
    print(f"🎯 Target Projections: {projections}", flush=True)

    # 3. Instantiate Skeleton Model on META device
    print("🦴 Constructing skeleton model on meta device...", flush=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, dtype=torch.float16)

    # Prune heavy projection layers on META device BEFORE to_empty()
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

    # 5. Load Base Weights and 2-Bit Quantized Tensors
    base_path = os.path.join(model_2bit_dir, "otf_2bit_base.safetensors")
    quant_path = os.path.join(model_2bit_dir, "otf_2bit_model.safetensors")

    base_tensors = {}
    with safe_open(base_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)

    quantized_tensors = {}
    with safe_open(quant_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            quantized_tensors[k] = f.get_tensor(k)

    # Copy non-recurrent weights into CUDA model
    model_state = model.state_dict()
    for k, tensor in base_tensors.items():
        if k in model_state:
            model_state[k].copy_(tensor)

    # Restore tied lm_head weights if uninitialized
    if getattr(hf_config, "tie_word_embeddings", True) or model.lm_head.weight.norm().item() == 0:
        print("🔗 Tying lm_head.weight -> model.embed_tokens.weight...", flush=True)
        model.lm_head.weight = model.model.embed_tokens.weight

    # 6. Inject OTF2BitLinear Layers on CUDA
    print("⚡ Injecting 2-Bit Packed Linear Layers...", flush=True)
    for i in range(num_layers):
        layer_obj = model.model.layers[i]
        for proj in projections:
            prefix = f"model.layers.{i}.{proj}"

            if f"{prefix}.packed_uint8" not in quantized_tensors:
                continue

            packed_uint8 = quantized_tensors[f"{prefix}.packed_uint8"].to(device)
            scales = quantized_tensors[f"{prefix}.scales"].to(device)

            # zeros can be None for Symmetric v4.1
            zeros = quantized_tensors.get(f"{prefix}.zeros", None)
            if zeros is not None:
                zeros = zeros.to(device)
            else:
                zeros = torch.zeros_like(scales)

            bias_tensor = quantized_tensors.get(f"{prefix}.bias", None)
            if bias_tensor is not None:
                bias_tensor = bias_tensor.to(device)

            out_deltas = quantized_tensors.get(f"{prefix}.outlier_deltas", None)
            out_idx = quantized_tensors.get(f"{prefix}.outlier_indices", None)

            if out_deltas is not None:
                out_deltas = out_deltas.to(device)
            if out_idx is not None:
                out_idx = out_idx.to(device)

            d_in = packed_uint8.shape[1] * 4

            linear_2bit = OTF2BitLinear(
                packed_uint8=packed_uint8,
                scales=scales,
                zeros=zeros,
                d_in=d_in,
                group_size=group_size,
                outliers_fp16=out_deltas,
                outlier_indices=out_idx,
                bias=bias_tensor
            )

            proj_parts = proj.split(".")
            submodule = layer_obj
            for part in proj_parts[:-1]:
                submodule = getattr(submodule, part)

            setattr(submodule, proj_parts[-1], linear_2bit)

    # Re-initialize RoPE positional embeddings
    fix_rotary_embeddings(model, hf_config, device)

    # Free temporary loading dictionaries
    del base_tensors, quantized_tensors, model_state
    gc.collect()
    torch.cuda.empty_cache()

    # Pre-warm forward pass
    print("🔥 Warm-up forward pass...", flush=True)
    dummy_input = torch.tensor([[1]], device=device, dtype=torch.long)
    with torch.no_grad():
        _ = model(dummy_input)

    print("✅ Model successfully assembled with 2-Bit Engine!", flush=True)

    loaded_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"💾 Total Model Static VRAM Footprint: {loaded_vram_mb:.2f} MB", flush=True)
    print("-" * 70, flush=True)

    torch.cuda.reset_peak_memory_stats()

    # 7. Run Autoregressive Generation
    print(f"📝 Prompt: '{prompt}'", flush=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    print("⚡ Generating tokens...", flush=True)
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.18,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_time = time.time() - start_time
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = new_tokens / gen_time if gen_time > 0 else 0

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("-" * 70, flush=True)
    print("🎯 GENERATED TEXT:")
    print(generated_text)
    print("-" * 70, flush=True)
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   • Generated Tokens: {new_tokens}")
    print(f"   • Generation Time:  {gen_time:.2f} seconds")
    print(f"   • Inference Speed:  {tokens_per_sec:.2f} tokens/sec")
    print(f"   • Peak VRAM Usage:  {peak_vram_mb:.2f} MB")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python otf_llm/run_2bit_universal.py <2bit_model_dir> <original_model_id_or_path>")
        sys.exit(1)

    dir_2bit = sys.argv[1]
    orig_id = sys.argv[2]

    run_2bit_inference(dir_2bit, orig_id)