"""
OTF-LLM Engine v4.1 - Production High-Speed Symmetric 2-Bit Inference Runner
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
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

from otf_llm.symmetric_2bit_engine import (
    SymmetricOTF2BitLinear,
    QuantizedEmbedding,
    QuantizedLinearHead
)

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
    rope_theta = getattr(hf_config, "rope_theta", 1000000.0)
    for module in model.modules():
        if "RotaryEmbedding" in type(module).__name__:
            dim = getattr(module, "dim", None) or getattr(module, "head_dim", 128)
            if dim is None and hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
                dim = hf_config.hidden_size // hf_config.num_attention_heads
            if dim is not None:
                inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                module.register_buffer("inv_freq", inv_freq, persistent=False)
                if hasattr(module, "max_seq_len_cached"):
                    module.max_seq_len_cached = 0
                if hasattr(module, "cos_cached"):
                    module.cos_cached = None
                if hasattr(module, "sin_cached"):
                    module.sin_cached = None


@torch.inference_mode()
def run_symmetric_inference(
    model_dir: str,
    original_model_id: str,
    prompt: str = "Explain quantum computing in simple terms:",
    max_new_tokens: int = 150
):
    print("=" * 75, flush=True)
    print(f"🚀 OTF-LLM v4.1: Production Symmetric 2-Bit Engine Runner ({model_dir})", flush=True)
    print("=" * 75, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("❌ CUDA GPU is required!", flush=True)
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. Tokenizer & HF Config
    print(f"📥 Loading Tokenizer and Config from '{original_model_id}'...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(original_model_id, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(original_model_id, trust_remote_code=True)

    config_path = os.path.join(model_dir, "otf_2bit_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        meta_cfg = json.load(f)

    num_layers = meta_cfg.get("num_layers", getattr(hf_config, "num_hidden_layers", 28))
    group_size = meta_cfg.get("group_size", 32)
    projections = meta_cfg.get("projections", DEFAULT_PROJECTIONS)

    print(f"📊 2-Bit Config: {num_layers} layers | Group Size = {group_size}", flush=True)

    # 2. Скелет на META (0 MB FP16 утечек)
    print("🦴 Constructing skeleton on meta device...", flush=True)
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

    # 3. Загрузка весов через CPU
    base_path = os.path.join(model_dir, "otf_2bit_base.safetensors")
    quant_path = os.path.join(model_dir, "otf_2bit_model.safetensors")

    base_tensors = {}
    with safe_open(base_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)

    quant_tensors = {}
    with safe_open(quant_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            quant_tensors[k] = f.get_tensor(k)

    # 4. Инжектируем INT8 Embeddings и LM Head
    print("⚡ Injecting INT8 Embeddings & Zero-Copy INT8 LM Head...", flush=True)
    q_emb = QuantizedEmbedding(hf_config.vocab_size, hf_config.hidden_size).to(device)
    q_emb.weight_int8.copy_(base_tensors["model.embed_tokens.weight_int8"].to(device))
    q_emb.scales.copy_(base_tensors["model.embed_tokens.scales"].to(device))
    model.model.embed_tokens = q_emb

    q_head = QuantizedLinearHead(hf_config.hidden_size, hf_config.vocab_size).to(device)
    if "lm_head.weight_int8" in base_tensors:
        print("🔗 Loading independent INT8 LM Head...", flush=True)
        q_head.weight_int8.copy_(base_tensors["lm_head.weight_int8"].to(device))
        q_head.scales.copy_(base_tensors["lm_head.scales"].to(device))
    else:
        print("🔗 Tying LM Head to INT8 Embeddings (Zero Overhead Pointer)...", flush=True)
        q_head.weight_int8 = q_emb.weight_int8
        q_head.scales = q_emb.scales

    model.lm_head = q_head

    if "model.norm.weight" in base_tensors:
        model.model.norm.weight.copy_(base_tensors["model.norm.weight"].to(device))

    for i in range(num_layers):
        for n_suf in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            k = f"model.layers.{i}.{n_suf}"
            if k in base_tensors:
                norm_module = getattr(model.model.layers[i], n_suf.split(".")[0])
                norm_module.weight.copy_(base_tensors[k].to(device))

    # 5. Инжектируем Symmetric 2-Bit слои
    print("⚡ Injecting Symmetric 2-Bit Linear Layers...", flush=True)
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

    # Pre-warm
    print("🔥 Warm-up forward pass...", flush=True)
    dummy_input = torch.tensor([[1]], device=device, dtype=torch.long)
    _ = model(dummy_input)

    vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"💾 Total Model Static VRAM Footprint: {vram_mb:.2f} MB", flush=True)
    print("-" * 75, flush=True)

    torch.cuda.reset_peak_memory_stats()

    # 6. Диалоговое форматирование
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    print(f"📝 Prompt: '{prompt}'", flush=True)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)

    print("⚡ Generating tokens...", flush=True)
    start_t = time.time()

    # Сбор всех возможных EOS токенов
    eos_token_ids = [tokenizer.eos_token_id]
    for special_tok in ["<|im_end|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if tok_id is not None and tok_id not in eos_token_ids:
            eos_token_ids.append(tok_id)

    # 🚀 Сбалансированный диалоговый семплер для 2-битных моделей
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.10,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_ids
    )

    gen_time = time.time() - start_t
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    tok_per_sec = new_tokens / gen_time if gen_time > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    print("-" * 75, flush=True)
    print("🎯 GENERATED TEXT:")
    print(generated_text)
    print("-" * 75, flush=True)
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   • Generated Tokens: {new_tokens}")
    print(f"   • Generation Time:  {gen_time:.2f} seconds")
    print(f"   • Inference Speed:  {tok_per_sec:.2f} tokens/sec")
    print(f"   • Peak VRAM Usage:  {peak_vram_mb:.2f} MB")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python otf_llm/run_symmetric_2bit.py <model_dir> <original_model_id> [prompt]")
        sys.exit(1)

    p = sys.argv[3] if len(sys.argv) > 3 else "Explain quantum computing in simple terms:"
    run_symmetric_inference(sys.argv[1], sys.argv[2], prompt=p)