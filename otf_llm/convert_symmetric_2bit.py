"""
OTF-LLM Engine v4.1 - Universal Symmetric 2-Bit + INT8 Embeddings Converter
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import json
import time
import gc
import traceback
import torch
from typing import Dict, List, Optional
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from otf_llm.symmetric_2bit_engine import (
    Symmetric2BitQuantizer,
    QuantizedEmbedding,
    QuantizedLinearHead
)
from otf_llm.make_profile_universal import create_act_profile

PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
]


def resolve_model_dir(model_input: str) -> str:
    if os.path.isdir(model_input):
        return model_input
    print(f"🌐 Directory '{model_input}' not found. Downloading from HuggingFace Hub...", flush=True)
    if snapshot_download is None:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    return snapshot_download(repo_id=model_input, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])


def convert_symmetric_model(
    model_input: str,
    output_dir: str,
    group_size: int = 32,
    outlier_ratio: float = 0.035,
    profile_path: Optional[str] = None
):
    print("=" * 75, flush=True)
    print(f"🚀 OTF-LLM v4.1: Universal Symmetric 2-Bit + INT8 Converter", flush=True)
    print(f"📦 Model Source: {model_input}", flush=True)
    print(f"📂 Output Dir:   {output_dir}", flush=True)
    print("=" * 75, flush=True)

    try:
        model_dir = resolve_model_dir(model_input)
        os.makedirs(output_dir, exist_ok=True)
        st_files = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")])

        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        clean_name = model_input.split("/")[-1].lower().replace("-", "_")
        auto_prof_path = f"{clean_name}_act_profile.pt"

        target_profile = profile_path or auto_prof_path
        if not os.path.exists(target_profile):
            print(f"⚡ Profile '{target_profile}' not found. Generating activation profile automatically...", flush=True)
            target_profile = create_act_profile(model_input, device="cuda" if torch.cuda.is_available() else "cpu")

        print(f"📊 Loading Profile-Guided Activation Profile: '{target_profile}'...", flush=True)
        act_profiles = torch.load(target_profile, map_location="cpu")

        print("🔍 Mapping safetensors keys...", flush=True)
        tensor_map = {}
        for st_path in st_files:
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensor_map[k] = st_path

        layer_indices = {int(k.split(".")[2]) for k in tensor_map if "model.layers." in k}
        num_layers = len(layer_indices)
        print(f"📊 Detected {num_layers} Transformer Layers.", flush=True)

        # -------------------------------------------------------------
        # 1. Сжатие Embeddings, LM Head и LayerNorms в INT8
        # -------------------------------------------------------------
        print("⚡ Quantizing Embeddings & LM Head to INT8...", flush=True)
        base_tensors: Dict[str, torch.Tensor] = {}

        # 1a. embed_tokens
        embed_key = "model.embed_tokens.weight"
        if embed_key in tensor_map:
            print(f"   • Quantizing '{embed_key}' to INT8...", flush=True)
            with safe_open(tensor_map[embed_key], framework="pt", device="cpu") as f:
                embed_fp16 = f.get_tensor(embed_key).clone()

            q_emb = QuantizedEmbedding(embed_fp16.shape[0], embed_fp16.shape[1])
            q_emb.quantize_from_fp16(embed_fp16)
            base_tensors["model.embed_tokens.weight_int8"] = q_emb.weight_int8.clone().contiguous()
            base_tensors["model.embed_tokens.scales"] = q_emb.scales.clone().contiguous()
            del embed_fp16, q_emb
            gc.collect()

        # 1b. lm_head (независимый)
        lm_head_key = "lm_head.weight"
        if lm_head_key in tensor_map:
            print(f"   • Quantizing independent '{lm_head_key}' to INT8...", flush=True)
            with safe_open(tensor_map[lm_head_key], framework="pt", device="cpu") as f:
                lm_head_fp16 = f.get_tensor(lm_head_key).clone()

            q_head = QuantizedLinearHead(lm_head_fp16.shape[1], lm_head_fp16.shape[0])
            q_head.quantize_from_fp16(lm_head_fp16)
            base_tensors["lm_head.weight_int8"] = q_head.weight_int8.clone().contiguous()
            base_tensors["lm_head.scales"] = q_head.scales.clone().contiguous()
            del lm_head_fp16, q_head
            gc.collect()

        # 1c. Финальная норма и LayerNorms
        if "model.norm.weight" in tensor_map:
            with safe_open(tensor_map["model.norm.weight"], framework="pt", device="cpu") as f:
                base_tensors["model.norm.weight"] = f.get_tensor("model.norm.weight").clone().contiguous()

        for i in range(num_layers):
            for n_suf in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                k = f"model.layers.{i}.{n_suf}"
                if k in tensor_map:
                    with safe_open(tensor_map[k], framework="pt", device="cpu") as f:
                        base_tensors[k] = f.get_tensor(k).clone().contiguous()

        base_file = os.path.join(output_dir, "otf_2bit_base.safetensors")
        print(f"💾 Saving Base INT8 & Norms -> {base_file}...", flush=True)
        save_file(base_tensors, base_file)
        base_mb = os.path.getsize(base_file) / (1024 ** 2)
        print(f"✅ Base saved successfully ({base_mb:.2f} MB). Freeing memory...", flush=True)

        del base_tensors
        gc.collect()

        # -------------------------------------------------------------
        # 2. 2-битное квантование всех слоев на GPU
        # -------------------------------------------------------------
        print("✅ Starting Profile-Guided 2-Bit Quantization on GPU...", flush=True)
        quantizer = Symmetric2BitQuantizer(group_size=group_size, outlier_ratio=outlier_ratio)
        quant_tensors: Dict[str, torch.Tensor] = {}
        start_t = time.time()

        for i in range(num_layers):
            print(f"⚡ Quantizing Layer {i+1:02d}/{num_layers:02d} Projections...", flush=True)

            for proj in PROJECTIONS:
                w_key = f"model.layers.{i}.{proj}.weight"
                b_key = f"model.layers.{i}.{proj}.bias"

                if w_key not in tensor_map:
                    continue

                with safe_open(tensor_map[w_key], framework="pt", device="cpu") as f:
                    W = f.get_tensor(w_key).clone().cuda().to(torch.float16)

                if b_key in tensor_map:
                    with safe_open(tensor_map[b_key], framework="pt", device="cpu") as f:
                        quant_tensors[b_key] = f.get_tensor(b_key).clone().contiguous()

                profile = None
                mod_name = f"model.layers.{i}.{proj}"
                if act_profiles and mod_name in act_profiles:
                    profile = act_profiles[mod_name].cuda()

                packed, scales, out_deltas, out_idx = quantizer.quantize_tensor(
                    W, activation_profile=profile
                )

                pfx = f"model.layers.{i}.{proj}"
                quant_tensors[f"{pfx}.packed_uint8"] = packed.cpu().clone().contiguous()
                quant_tensors[f"{pfx}.scales"] = scales.cpu().clone().contiguous()

                if out_deltas is not None and out_idx is not None:
                    quant_tensors[f"{pfx}.outlier_deltas"] = out_deltas.cpu().clone().contiguous()
                    quant_tensors[f"{pfx}.outlier_indices"] = out_idx.cpu().clone().contiguous()

                del W

            torch.cuda.empty_cache()

        quant_file = os.path.join(output_dir, "otf_2bit_model.safetensors")
        print(f"💾 Saving 2-Bit Quantized Layers -> {quant_file}...", flush=True)
        save_file(quant_tensors, quant_file)

        cfg = {
            "architecture": "OTF-Engine-v4.1-Symmetric-2Bit",
            "group_size": group_size,
            "outlier_ratio": outlier_ratio,
            "num_layers": num_layers,
            "projections": PROJECTIONS,
            "source_model": model_input,
            "created_by": "GT Labs AI (Gleb Tikhiy)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(output_dir, "otf_2bit_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        quant_mb = os.path.getsize(quant_file) / (1024 ** 2)
        total_mb = base_mb + quant_mb

        print("-" * 75, flush=True)
        print("🎉 SUCCESSFUL SYMMETRIC 2-BIT + INT8 CONVERSION!", flush=True)
        print(f"   • Base (INT8 Embeddings + LM Head + Norms): {base_mb:.2f} MB")
        print(f"   • 2-Bit Quantized Layers:                   {quant_mb:.2f} MB")
        print(f"   • Total Disk Footprint:                     {total_mb:.2f} MB")
        print(f"   • Elapsed Time:                             {time.time() - start_t:.2f}s")
        print("=" * 75, flush=True)

    except Exception as e:
        print(f"\n❌ CONVERSION ERROR: {str(e)}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python otf_llm/convert_symmetric_2bit.py <model_id> <output_dir> [group_size] [outlier_ratio] [profile_path]")
        sys.exit(1)

    in_model = sys.argv[1]
    out_dir = sys.argv[2]
    g_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    out_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.035
    prof = sys.argv[5] if len(sys.argv) > 5 else None

    convert_symmetric_model(in_model, out_dir, group_size=g_size, outlier_ratio=out_ratio, profile_path=prof)