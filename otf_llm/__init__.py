# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.

"""
OTF-LLM Engine Package
======================
High-performance hybrid LLM inference engine featuring custom Fused Triton INT4 GEMM kernels.
"""

__version__ = "2.0.4"
__author__ = "GT Labs AI & Gleb Tikhiy"
__email__ = "team.gtlabs@gmail.com"

from .convert_global_universal import QuantizedEmbedding, GlobalSymmetricINT4Linear, convert_model
from .make_profile_universal import create_act_profile
from .otf_triton_kernel import triton_fused_int4_linear
from .run_triton_universal import TritonGlobalSymmetricLinear, run_inference
from .companion_memory import CompanionMemoryManager

__all__ = [
    "QuantizedEmbedding",
    "GlobalSymmetricINT4Linear",
    "TritonGlobalSymmetricLinear",
    "triton_fused_int4_linear",
    "create_act_profile",
    "convert_model",
    "run_inference",
    "CompanionMemoryManager",
]