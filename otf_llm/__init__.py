# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.

"""
OTF-LLM Engine Package (v4.0.0)
===============================
High-performance hybrid LLM inference engine featuring Adaptive Non-Uniform 2-Bit Quantization
(Lloyd-Max Codebooks + Fused OpenAI Triton INT2 Kernel), Profile-Guided Outlier Anchors,
Zero-RAM Incremental Quantizer, 3-Tier Hierarchical MoE Offloading, and Gradio Interactive Web UI.
"""

__version__ = "4.0.0"
__author__ = "GT Labs AI & Gleb Tikhiy"
__email__ = "team.gtlabs@gmail.com"

# v4.0 2-Bit Non-Uniform Quantization & Triton INT2 Kernel Exports
from .otf_2bit_quantizer import OTF2BitQuantizer, OTF2BitLinear
from .otf_triton_2bit_kernel import triton_2bit_gemm
from .convert_2bit_universal import convert_model_to_2bit
from .run_2bit_universal import run_2bit_inference

# Legacy v3.2 INT4 Global Permutation Exports
from .convert_global_universal import QuantizedEmbedding, GlobalSymmetricINT4Linear, convert_model
from .otf_triton_kernel import triton_fused_int4_linear
from .run_triton_universal import TritonGlobalSymmetricLinear, run_inference, fix_rope_position_embeddings

# Profiler & Companion Memory
from .make_profile_universal import create_act_profile
from .companion_memory import CompanionMemoryManager

# MoE Offloader
from .otf_moe_offloader import (
    Hierarchical3TierMoECache,
    MoEExpertLRUCache,
    AsynchronousExpertStreamer,
    OTFSparseMoeBlockWrapper,
)
from .direct_quantized_importer import import_prequantized_hf_model

try:
    from .web_demo import launch_web_demo
except ImportError:
    def launch_web_demo(*args, **kwargs):
        raise ImportError("`launch_web_demo` requires `gradio`. Please install it with `pip install gradio`.")

__all__ = [
    # v4.0 2-Bit Engine Exports
    "OTF2BitQuantizer",
    "OTF2BitLinear",
    "triton_2bit_gemm",
    "convert_model_to_2bit",
    "run_2bit_inference",
    # Legacy INT4 Exports
    "QuantizedEmbedding",
    "GlobalSymmetricINT4Linear",
    "TritonGlobalSymmetricLinear",
    "triton_fused_int4_linear",
    "convert_model",
    "run_inference",
    "fix_rope_position_embeddings",
    # Profiler & Utilities
    "create_act_profile",
    "CompanionMemoryManager",
    "Hierarchical3TierMoECache",
    "MoEExpertLRUCache",
    "AsynchronousExpertStreamer",
    "OTFSparseMoeBlockWrapper",
    "launch_web_demo",
    "import_prequantized_hf_model",
]