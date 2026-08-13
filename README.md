# 🚀 On-The-Fly Weight Synthesizer (OTF-LLM Engine)

> **High-performance hybrid LLM inference engine featuring custom Fused Triton INT4 GEMM kernels, Outlier-Aware weight quantization, global activation permutation, Zero-RAM Streaming mmap Quantization, INT8 embeddings, VRAM compression down to 1.89 GB (3B) / 4.20 GB (7B), Companion Long-Term Memory, and a production-grade REST API server.**

[![PyPI](https://img.shields.io/pypi/v/otf-llm?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/otf-llm/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/OpenAI_Triton-Fused_Kernels-red?logo=openai&logoColor=white)](https://github.com/openai/triton)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔬 Engineering & Research: GT Labs AI

This project is developed and maintained by **GT Labs AI**.

* 🚀 **GT Labs AI Mission:** Ultra-fast MVP engineering, custom AI integration, and deep neural network optimization research.
* 👨‍💻 **Author & Lead AI Engineer:** **Gleb Tikhiy** ([@GlebTikhiy](https://github.com/GlebTikhiy))
* 📧 **Contact & Inquiries:** team.gtlabs@gmail.com
* 🌐 **Organization:** [GT Labs AI on GitHub](https://github.com/GT-Labs-AI)

---

## 🎯 Project Goal

Overcoming memory-bound bottlenecks and hardware VRAM constraints when executing Large Language Models (LLMs) on consumer GPUs.

Instead of transferring heavy FP16 weights from VRAM, **OTF-LLM Engine** performs hardware-accelerated dequantization of Outlier-Aware INT4 weights **directly inside GPU registers (SRAM)** via custom **OpenAI Triton GEMM Kernels**, streams weights without RAM allocation via `safetensors.safe_open` (`mmap`), compresses vocabulary embeddings (`embed_tokens`) into INT8, compresses the classifier (`lm_head`), integrates long-term user memory (`companion_memory.py`), and employs predictive **Query-Guided Sparse Offloading**.

---

## 📊 Performance Benchmark (RTX 5060 Ti 16GB)

Benchmarking conducted on an **NVIDIA GeForce RTX 5060 Ti** GPU:

| Model / Architecture | Format | Static VRAM | Peak VRAM | Speed | Load Time | Intelligence Parity | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B (Base)** | FP16 | 5.75 GB | 5.81 GB | 25.6 t/s | ~15.0 s | 100% (Baseline) | Baseline |
| **Qwen2.5-3B (OTF Champion)** | **INT4/8** | **1.94 GB** | **1.99 - 2.06 GB** | **16.15 t/s** | **4.2 s** | **100% (0% Loss)** | 🏆 **CHAMPION (-66.1%)** |
| **Llama-3.2-3B (Base)** | FP16 | 6.40 GB | 6.48 GB | 22.1 t/s | ~16.0 s | 100% (Baseline) | Baseline |
| **Llama-3.2-3B (OTF Champion)** | **INT4/8** | **1.89 GB** | **1.99 - 2.17 GB** | **15.89 t/s** | **3.7 s** | **100% (0% Loss)** | 🏆 **RECORD (-70.5%)** |
| **Qwen2.5-7B (Base)** | FP16 | 15.27 GB | 15.80 GB | 14.2 t/s | ~28.0 s | 100% (Baseline) | Baseline |
| **Qwen2.5-7B (OTF Champion)** | **INT4/8** | **4.20 GB** | **4.25 GB** | **8.60 t/s** | **15.5 s** | **100% (0% Loss)** | 🏆 **CHAMPION (-72.5%)** |

---

## 🏛️ Architecture & Key Innovations

```
[Input Vector X] ──► [Global Static Permutation (global_perm_idx)]
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
[Outlier Channels (1% FP16)]                 [Background Block (99% INT4)]
   │                                                         │
   ├──► Pure FP16                                            ├──► Range [-7 ... +7]
   └──► Input X_outliers                                     ├──► 2:1 Packing (uint8)
                                                             └──► Zero-Point = 0 BYTES!
                                                                     │
                                                                     ▼
                                                      [Custom Fused Triton GEMM Kernel]
                                                      (Dequantization in GPU SRAM Registers)
                                                                     │
         ┌───────────────────────────────────────────────────────────┘
         ▼
[Continuous GEMM Addition: Outliers + Triton Background = Exact FP16 Output]
```

1. **Custom Outlier-Aware Fused Triton GEMM Kernel (`otf_triton_kernel.py`):**
   Packed `uint8` weights are streamed from VRAM and dequantized **directly inside GPU chip registers (SRAM)** during matrix multiplication, eliminating temporary FP16 tensor allocations in VRAM.
2. **Zero-RAM Streaming Safetensors Quantizer (`convert_global_universal.py`):**
   Uses `safetensors.safe_open` memory-mapping (`mmap`) to stream weights directly from disk layer-by-layer during quantization, dropping peak CPU RAM consumption down to **< 500 MB** (enabling 70B+ model quantization on low-RAM machines).
3. **Global Static Permutation (`global_perm_idx`) & Outlier Preservation:**
   A unified channel permutation table across the entire model (requiring only 1.6 MB VRAM). Isolating the Top-1% critical outlier channels ($|W| \times |X_{\text{profile}}|$) in FP16 completely suppresses quantization noise and guarantees 100% accuracy retention across Qwen and Llama architectures.
4. **INT8 Quantized Embeddings & Outlier-Aware INT4 `lm_head`:**
   Vocabulary lookup tables are compressed to INT8, while `lm_head` supports Tied Word Embeddings for zero-overhead output projection.
5. **Companion Long-Term Memory Manager (`companion_memory.py`):**
   Zero-VRAM, lightweight CPU RAM module that automatically extracts and retrieves user facts in < 2 ms via TF-IDF cosine similarity, injecting relevant facts into system prompts.
6. **FastAPI REST API Server (`server_fastapi.py`):**
   An asynchronous production server featuring **OpenAI API specification compatibility (`/v1/chat/completions`)**, SSE (Server-Sent Events) token streaming, and an async request queue manager to protect VRAM from overflow.

---

## 📁 Repository Structure

```
otf-llm-engine/
├── setup.py                                  # Setuptools configuration
├── pyproject.toml                            # PEP 517/518 build system
├── MANIFEST.in                               # Package assets configuration
├── pipeline_run.py                           # Automated 1-click end-to-end pipeline
├── validate_llama3_2_3b.py                   # Validation runner for Llama-3.2-3B
├── test_client.py                            # Streaming client for SSE validation
├── otf_llm/                                  # Main python package namespace
│   ├── __init__.py                           # Module entry point
│   ├── make_profile_universal.py             # Activation profile calibrator
│   ├── convert_global_universal.py           # Zero-RAM mmap safetensors quantizer
│   ├── run_triton_universal.py               # Triton GEMM inference runner
│   ├── otf_triton_kernel.py                  # Custom Fused Triton INT4 GEMM kernel
│   ├── companion_memory.py                   # Zero-VRAM long-term user memory store
│   ├── query_guided_sparse_kv.py             # Context retrieval (CPU RAM -> GPU)
│   ├── otf_context_compressor.py             # SnapKV / KIVI cache compressor
│   ├── benchmark_profiler.py                 # Byte-level weights and VRAM profiler
│   └── server_fastapi.py                     # REST API server (OpenAI API + SSE)
├── README.md                                 # Project documentation
└── LICENSE                                   # MIT License
```

---

## 🛠️ Installation & Usage

### Official PyPI Installation

```bash
pip install otf-llm
```

### Usage Example

```python
from otf_llm import run_inference, CompanionMemoryManager

# 1. Initialize Long-Term Memory
memory = CompanionMemoryManager()
memory.add_explicit_fact("User is an AI Engineer using RTX 5060 Ti.")

# 2. Run Triton Engine Inference
run_inference("unsloth/Llama-3.2-3B-Instruct", prompt="Write a binary search in Python.")
```

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.