"""
OTF-LLM Engine v4.0 - Universal 2-Bit Quantizer & Model Converter
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy. MIT License.

Converts HuggingFace LLM checkpoints (Llama-3.2-3B, Qwen2.5-3B/7B) into
Adaptive Non-Uniform 2-Bit format with Profile-Guided FP16 Outlier Anchors.
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

from otf_llm.otf_2bit_quantizer import OTF2BitQuantizer


PROJECTION_KEYS = [
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

    print(f"🌐 Local directory '{model_input}' not found. Downloading from HuggingFace Hub...", flush=True)
    if snapshot_download is None:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")

    download_dir = snapshot_download(
        repo_id=model_input,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"]
    )
    return download_dir


def find_safetensors_files(model_dir: str) -> List[str]:
    files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    files.sort()
    return files


def convert_model_to_2bit(
    model_input: str,
    output_dir: str,
    group_size: int = 32,
    outlier_ratio: float = 0.035,
    profile_path: Optional[str] = None,
    device: str = "cuda"
):
    print("=" * 70, flush=True)
    print(f"🚀 OTF-LLM v4.0: Universal 2-Bit Quantizer (Target: {model_input})", flush=True)
    print("=" * 70, flush=True)

    try:
        model_dir = resolve_model_dir(model_input)
        os.makedirs(output_dir, exist_ok=True)
        st_files = find_safetensors_files(model_dir)

        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        act_profiles = None
        if profile_path and os.path.exists(profile_path):
            print(f"📊 Loading Profile-Guided Activation Profile from '{profile_path}'...", flush=True)
            act_profiles = torch.load(profile_path, map_location="cpu")

        print(f"📦 Found {len(st_files)} safetensors shard(s). Indexing model tensors...", flush=True)

        tensor_file_map = {}
        for st_path in st_files:
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensor_file_map[k] = st_path

        layer_indices = set()
        for k in tensor_file_map.keys():
            if "model.layers." in k:
                parts = k.split(".")
                idx = int(parts[2])
                layer_indices.add(idx)

        num_layers = len(layer_indices)
        print(f"📊 Detected {num_layers} Transformer Layers.", flush=True)
        print(f"⚙️ Quantization Parameters: 2-Bit Group Size = {group_size}, Outliers = {outlier_ratio * 100:.1f}%", flush=True)
        print("-" * 70, flush=True)

        quantizer = OTF2BitQuantizer(group_size=group_size, outlier_ratio=outlier_ratio)

        base_tensors: Dict[str, torch.Tensor] = {}
        quantized_tensors: Dict[str, torch.Tensor] = {}

        # 1. Non-recurrent layer weights
        print("🔍 Preserving non-recurrent layers (Embeddings, Norms, LM Head)...", flush=True)
        non_recurrent_keys = [
            k for k in tensor_file_map.keys() if "model.layers." not in k
        ]

        for k in non_recurrent_keys:
            st_path = tensor_file_map[k]
            with safe_open(st_path, framework="pt", device="cpu") as f:
                base_tensors[k] = f.get_tensor(k).clone()

        # Preserve layer norms
        for i in range(num_layers):
            for norm_suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                k = f"model.layers.{i}.{norm_suffix}"
                if k in tensor_file_map:
                    st_path = tensor_file_map[k]
                    with safe_open(st_path, framework="pt", device="cpu") as f:
                        base_tensors[k] = f.get_tensor(k).clone()

        print("✅ Non-recurrent layers cached. Starting Profile-Guided 2-bit quantization...", flush=True)

        # 2. Process each layer projection weight
        start_time = time.time()

        for i in range(num_layers):
            print(f"⚡ Quantizing Layer {i+1}/{num_layers} projections...", flush=True)
            for proj_key in PROJECTION_KEYS:
                full_key_w = f"model.layers.{i}.{proj_key}.weight"
                full_key_b = f"model.layers.{i}.{proj_key}.bias"

                st_path = tensor_file_map[full_key_w]
                with safe_open(st_path, framework="pt", device="cpu") as f:
                    W_fp16 = f.get_tensor(full_key_w).clone().to(device=device, dtype=torch.float16)

                if full_key_b in tensor_file_map:
                    st_path_b = tensor_file_map[full_key_b]
                    with safe_open(st_path_b, framework="pt", device="cpu") as f:
                        quantized_tensors[f"model.layers.{i}.{proj_key}.bias"] = f.get_tensor(full_key_b).clone()

                # Extract profile for this layer module
                profile_tensor = None
                if act_profiles is not None:
                    mod_name = f"model.layers.{i}.{proj_key}"
                    if mod_name in act_profiles:
                        profile_tensor = act_profiles[mod_name].to(device)

                # Quantize matrix with activation profile guidance $|W| \times |X|$
                packed_uint8, scales, zeros, out_deltas, out_idx = quantizer.quantize_tensor(
                    W_fp16,
                    activation_profile=profile_tensor
                )

                prefix = f"model.layers.{i}.{proj_key}"
                quantized_tensors[f"{prefix}.packed_uint8"] = packed_uint8.cpu()
                quantized_tensors[f"{prefix}.scales"] = scales.cpu()
                quantized_tensors[f"{prefix}.zeros"] = zeros.cpu()

                if out_deltas is not None and out_idx is not None:
                    quantized_tensors[f"{prefix}.outlier_deltas"] = out_deltas.cpu()
                    quantized_tensors[f"{prefix}.outlier_indices"] = out_idx.cpu()

                del W_fp16
            torch.cuda.empty_cache()
            gc.collect()

        conversion_time = time.time() - start_time
        print("-" * 70, flush=True)
        print(f"✅ 2-Bit Quantization Completed in {conversion_time:.2f} seconds.", flush=True)

        base_file = os.path.join(output_dir, "otf_2bit_base.safetensors")
        quant_file = os.path.join(output_dir, "otf_2bit_model.safetensors")
        config_file = os.path.join(output_dir, "otf_2bit_config.json")

        print(f"💾 Saving Non-Recurrent Base Weights ({len(base_tensors)} tensors) -> {base_file}", flush=True)
        save_file(base_tensors, base_file)

        print(f"💾 Saving 2-Bit Quantized Layers ({len(quantized_tensors)} tensors) -> {quant_file}", flush=True)
        save_file(quantized_tensors, quant_file)

        config_data = {
            "architecture": "OTF-Engine-v4.0-2Bit",
            "group_size": group_size,
            "outlier_ratio": outlier_ratio,
            "num_layers": num_layers,
            "projections": PROJECTION_KEYS,
            "source_model": model_input,
            "created_by": "GT Labs AI (Gleb Tikhiy)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        base_size_mb = os.path.getsize(base_file) / (1024 ** 2)
        quant_size_mb = os.path.getsize(quant_file) / (1024 ** 2)
        total_mb = base_size_mb + quant_size_mb

        print("-" * 70, flush=True)
        print("🎉 2-BIT CONVERSION SUCCESSFUL!", flush=True)
        print(f"📦 Output Directory: {output_dir}")
        print(f"   ├── Base Weights File:      {base_size_mb:.2f} MB")
        print(f"   ├── 2-Bit Quantized Layers: {quant_size_mb:.2f} MB")
        print(f"   └── Total Disk Size:        {total_mb:.2f} MB")
        print("=" * 70, flush=True)

    except Exception as e:
        print(f"\n❌ CONVERSION ERROR: {str(e)}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python otf_llm/convert_2bit_universal.py <model_id_or_path> <output_dir> [group_size] [outlier_ratio] [profile_path]")
        sys.exit(1)

    in_model = sys.argv[1]
    out_dir = sys.argv[2]
    g_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    out_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.035
    prof_p = sys.argv[5] if len(sys.argv) > 5 else "qwen_profile.pt"

    convert_model_to_2bit(in_model, out_dir, group_size=g_size, outlier_ratio=out_ratio, profile_path=prof_p)