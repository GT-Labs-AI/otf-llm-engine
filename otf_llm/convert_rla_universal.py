"""
OTF-LLM Engine v4.0 - Universal Grouped RLA Quantizer & Model Converter
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy. MIT License.

Converts HuggingFace LLM checkpoints (Llama-3.2-3B, Qwen2.5-3B/7B) into
Grouped Recurrent Layer Adaptation (Grouped RLA) format.
Windows mmap-safe version.
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

from otf_llm.otf_recurrent_layer import RLAWeightSynthesizer


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
    """Resolves local directory or downloads checkpoint from HuggingFace Hub."""
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
    """Finds all .safetensors shard files inside model directory."""
    files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    files.sort()
    return files


def convert_model_to_rla(
    model_input: str,
    output_dir: str,
    rank: int = 48,
    outlier_ratio: float = 0.01,
    num_groups: int = 3,
    device: str = "cuda"
):
    """
    Grouped RLA conversion routine: Groups transformer layers into functional buckets
    (Early, Mid, Late) for high-precision SVD reconstruction.
    """
    print("=" * 70, flush=True)
    print(f"🚀 OTF-LLM v4.0: Grouped RLA Converter (Target: {model_input})", flush=True)
    print("=" * 70, flush=True)

    try:
        model_dir = resolve_model_dir(model_input)
        os.makedirs(output_dir, exist_ok=True)
        st_files = find_safetensors_files(model_dir)

        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

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
        layers_per_group = (num_layers + num_groups - 1) // num_groups

        print(f"📊 Detected {num_layers} Transformer Layers across {num_groups} Layer Groups ({layers_per_group} layers/group).", flush=True)
        print(f"⚙️ Conversion Parameters: Rank r = {rank}, Outlier Ratio = {outlier_ratio * 100:.1f}%", flush=True)
        print("-" * 70, flush=True)

        synthesizer = RLAWeightSynthesizer(rank=rank)

        base_tensors: Dict[str, torch.Tensor] = {}
        adapter_tensors: Dict[str, torch.Tensor] = {}

        # 1. Non-recurrent layer weights (embeddings, final norm, lm_head)
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

        print("✅ Non-recurrent layers cached. Starting Grouped RLA decomposition...", flush=True)

        # 2. Process each projection type across layer groups
        start_time = time.time()

        for proj_key in PROJECTION_KEYS:
            print(f"\n⚡ Processing Projection: '{proj_key}' across {num_groups} Groups...", flush=True)

            for g in range(num_groups):
                g_start = g * layers_per_group
                g_end = min(num_layers, (g + 1) * layers_per_group)
                group_layer_idxs = list(range(g_start, g_end))

                proj_weights = []
                for i in group_layer_idxs:
                    full_key_w = f"model.layers.{i}.{proj_key}.weight"
                    full_key_b = f"model.layers.{i}.{proj_key}.bias"

                    st_path = tensor_file_map[full_key_w]
                    with safe_open(st_path, framework="pt", device="cpu") as f:
                        tensor_fp16 = f.get_tensor(full_key_w).clone().to(torch.float16)
                        proj_weights.append(tensor_fp16)

                    if full_key_b in tensor_file_map:
                        st_path_b = tensor_file_map[full_key_b]
                        with safe_open(st_path_b, framework="pt", device="cpu") as f:
                            bias_fp16 = f.get_tensor(full_key_b).clone().to(torch.float16)
                            adapter_tensors[f"model.layers.{i}.{proj_key}.bias"] = bias_fp16

                # Compute group centroid W_base_g
                W_base_g = synthesizer.compute_centroid_base(proj_weights).to(device)

                base_key = f"rla.base_g{g}.{proj_key}.weight"
                base_tensors[base_key] = W_base_g.cpu().to(torch.float16)

                # Factorize layer residual deltas for layers in this group
                for idx_in_group, i in enumerate(group_layer_idxs):
                    W_layer = proj_weights[idx_in_group].to(device)

                    A, B, out_deltas, out_idx = synthesizer.decompose_residual(
                        W_layer=W_layer,
                        W_base=W_base_g,
                        activation_profile=None,
                        outlier_ratio=outlier_ratio
                    )

                    adapter_tensors[f"model.layers.{i}.{proj_key}.adapter_A"] = A.cpu()
                    adapter_tensors[f"model.layers.{i}.{proj_key}.adapter_B"] = B.cpu()

                    if out_deltas is not None and out_idx is not None:
                        adapter_tensors[f"model.layers.{i}.{proj_key}.outlier_deltas"] = out_deltas.cpu()
                        adapter_tensors[f"model.layers.{i}.{proj_key}.outlier_indices"] = out_idx.cpu()

                del proj_weights, W_base_g
                torch.cuda.empty_cache()
                gc.collect()

        conversion_time = time.time() - start_time
        print("-" * 70, flush=True)
        print(f"✅ Grouped RLA Decomposition Completed in {conversion_time:.2f} seconds.", flush=True)

        base_file = os.path.join(output_dir, "rla_model_base.safetensors")
        adapter_file = os.path.join(output_dir, "rla_model_adapters.safetensors")
        config_file = os.path.join(output_dir, "rla_config.json")

        print(f"💾 Saving Base Weights ({len(base_tensors)} tensors) -> {base_file}", flush=True)
        save_file(base_tensors, base_file)

        print(f"💾 Saving Micro-Adapters ({len(adapter_tensors)} tensors) -> {adapter_file}", flush=True)
        save_file(adapter_tensors, adapter_file)

        config_data = {
            "architecture": "Grouped-RLA-Engine-v4.0",
            "rank": rank,
            "outlier_ratio": outlier_ratio,
            "num_layers": num_layers,
            "num_groups": num_groups,
            "projections": PROJECTION_KEYS,
            "source_model": model_input,
            "created_by": "GT Labs AI (Gleb Tikhiy)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        base_size_mb = os.path.getsize(base_file) / (1024 ** 2)
        adapter_size_mb = os.path.getsize(adapter_file) / (1024 ** 2)
        total_mb = base_size_mb + adapter_size_mb

        print("-" * 70, flush=True)
        print("🎉 CONVERSION SUCCESSFUL!", flush=True)
        print(f"📦 Output Directory: {output_dir}")
        print(f"   ├── Base Weights File: {base_size_mb:.2f} MB")
        print(f"   ├── Adapters File:    {adapter_size_mb:.2f} MB")
        print(f"   └── Total Disk Size:  {total_mb:.2f} MB")
        print("=" * 70, flush=True)

    except Exception as e:
        print(f"\n❌ CONVERSION ERROR: {str(e)}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python otf_llm/convert_rla_universal.py <model_id_or_path> <output_dir> [rank] [outlier_ratio] [num_groups]")
        sys.exit(1)

    in_model = sys.argv[1]
    out_dir = sys.argv[2]
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    out_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    groups = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    convert_model_to_rla(in_model, out_dir, rank=r, outlier_ratio=out_ratio, num_groups=groups)