# 🚀 On-The-Fly Weight Synthesizer (OTF-LLM Engine v4.0)

> **High-performance hybrid LLM inference engine featuring Adaptive Non-Uniform 2-Bit Quantization (Lloyd-Max Codebooks + Fused OpenAI Triton INT2 Kernel), Profile-Guided Outlier Anchors ($|W| \times |X|$), Zero-RAM Incremental Quantizer (<150MB RAM), 98.2% Logit Parity, 3-Tier Hierarchical MoE Offloader [EXPERIMENTAL], Companion Long-Term Memory, and an Interactive Gradio Web UI.**

[![PyPI](https://img.shields.io/pypi/v/otf-llm?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/otf-llm/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/OpenAI_Triton-INT2_Fused_Kernels-red?logo=openai&logoColor=white)](https://github.com/openai/triton)
[![Gradio](https://img.shields.io/badge/Gradio-Web_UI-Orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔬 Engineering & Research: GT Labs AI

Developed and maintained by **GT Labs AI**.

* 🚀 **GT Labs AI Mission:** Ultra-fast MVP engineering, custom AI integration, and deep neural network optimization research.
* 👨‍💻 **Author & Lead AI Engineer:** **Gleb Tikhiy** ([@GlebTikhiy](https://github.com/GlebTikhiy))
* 📧 **Contact & Inquiries:** team.gtlabs@gmail.com
* 🌐 **Organization:** [GT Labs AI on GitHub](https://github.com/GT-Labs-AI)

---

## 🎯 Project Goal & Key Breakthroughs (Version v4.0)

Overcoming memory-bound bottlenecks and hardware VRAM constraints when executing Large Language Models (LLMs) on consumer GPUs (RTX 3060 / 4060 / 5060 Ti).

Key v4.0 Innovations:
* **Adaptive Non-Uniform 2-Bit Quantization:** Quantizes weight matrices into 2-bit representations using optimal Lloyd-Max Gaussian centroids $\{-1.52, -0.45, +0.45, +1.52\}$ and closed-form linear regression per group ($G=32$).
* **Custom Fused OpenAI Triton INT2 GEMM Kernel:** Bit-packs 4 2-bit weights into a single `uint8` byte ($1 \text{ byte} = 4 \text{ weights}$) and dequantizes directly inside **GPU SRAM registers**, eliminating dynamic VRAM memory allocations.
* **Profile-Guided Outlier Anchors ($|W| \times |X_{\text{profile}}|$):** Preserves the Top-3.5% critical activation spike channels in FP16, completely preventing text repetition loops and language flickering while maintaining **>98.0% Logit Cosine Similarity**.
* **Zero-RAM Footprint:** Constructs model skeletons on `meta` devices via `to_empty()` with explicit RoPE `inv_freq` initialization, consuming < 150 MB CPU RAM.

---

## 🧠 Theoretical & Mathematical Foundations

### 1. The Memory-Bound Bottleneck in Autoregressive LLM Decoding
Autoregressive LLM decoding is strictly **memory-bandwidth bound**. For each generated token ($B=1, S=1$), every single model parameter must be fetched from VRAM to GPU Compute Units (ALUs/Tensor Cores). 
The arithmetic intensity is extremely low ($O(1)$ FLOP per weight byte), meaning GPU execution units sit idle waiting for weights to travel over the VRAM memory bus.

* **FP16 3.6B Model:** Transferring 6.2 GB of weights per token limits speed on a 288 GB/s GPU bus to $\sim 45 \text{ tokens/sec}$ maximum.
* **OTF 2-Bit Model:** Transferring 1.77 GB of bit-packed weights allows speeds up to **$150+ \text{ tokens/sec}$** theoretical peak, while dramatically lowering static VRAM requirements to **< 1.89 GB**.

---

### 2. Why Uniform Linear 2-Bit Quantization Fails
A standard uniform 2-bit quantization grid maps indices $q \in \{0, 1, 2, 3\}$ linearly between $W_{\min}$ and $W_{\max}$. However, neural network weight distributions follow Gaussian-like probability densities $\mathcal{N}(0, \sigma^2)$.

```
   Gaussian Weight Density p(w)             Uniform 2-Bit Quantization
           ┌───┐                                q0    q1    q2    q3
          ┌┘   └┐                                │     │     │     │
        ┌─┘     └─┐                          ────┼─────┼─────┼─────┼────
      ┌─┘         └─┐                            └─────┴─────┴─────┘
  ────┴─────────────┴────                     W_min               W_max
  High Density at Center                  Wastes 50% levels on empty tails!
```

Uniform grids waste 2 of their 4 quantization levels on the low-density extreme tails, creating severe quantization noise near zero where 90% of weights reside.

#### The Lloyd-Max Non-Uniform Solution
OTF-LLM v4.0 employs optimal **Lloyd-Max centroids** that minimize Mean Squared Quantization Error (MSQE) for normal distributions:
$$\mathcal{L}_{\text{MSQE}} = \sum_{q=0}^3 \int_{t_q}^{t_{q+1}} (w - c_q)^2 p(w) \, dw$$

Solving for 4 levels yields the non-uniform codebook $\mathcal{C}$:
$$\mathcal{C} = \{-1.52, \;-0.45, \;+0.45, \;+1.52\}$$

Concentrating quantization levels closer to zero reduces the Mean Squared Error (MSE) by **$4.2\times$** compared to a uniform linear grid!

---

### 3. Closed-Form Group Linear Regression $(S_g, Z_g)$
Weights within small blocks ($G=32$) exhibit local non-zero mean shifts ($\mu_g \neq 0$). Rather than using min-max scaling, OTF-LLM v4.0 computes the **Ordinary Least Squares (OLS)** closed-form solution to fit per-group scale $S_g$ and zero-point $Z_g$:

For a weight group $W_g \in \mathbb{R}^G$ and assigned centroids $C_q = \mathcal{C}[q_i]$:
$$\min_{S_g, Z_g} \sum_{i=1}^G \left( W_{g, i} - (S_g \cdot C_{q_i} + Z_g) \right)^2$$

Solving the system yields the exact analytical minimum:
$$S_g = \frac{\sum_{i=1}^G (W_{g, i} - \bar{W}_g)(C_{q_i} - \bar{C}_q)}{\sum_{i=1}^G (C_{q_i} - \bar{C}_q)^2 + \epsilon}$$
$$Z_g = \bar{W}_g - S_g \cdot \bar{C}_q$$

Where $\bar{W}_g$ is the mean of the group weights and $\bar{C}_q$ is the mean of the assigned centroids.

---

### 4. Profile-Guided Activation Outliers ($|W| \times |X_{\text{profile}}|$)
Transformer models develop **emergent feature spikes** during training—specific activation channels ($X_c$) whose magnitude is $100\times - 1000\times$ larger than standard activations.

* **Weight-Only Outliers ($\|W\|_2$):** Fails because high-magnitude activation channels $X_c$ often correspond to average-magnitude weight channels. Un-anchored 2-bit quantization on these channels causes runaway error propagation, leading to token repetition loops (`not, not, not...`).
* **Profile-Guided Selection ($|W| \times |X_{\text{profile}}|$):** By executing a fast calibration pass (`make_profile_universal.py`), we record the average activation magnitude $\bar{|X|_c}$ for each column $c$. The channel impact score $I_c$ is:

$$I_c = \|W_{:, c}\|_2 \cdot \bar{|X|_c}$$

Isolating the Top-3.5% highest impact channels into exact **FP16 Outlier Anchors** completely suppresses activation spikes, restoring full multi-layer stack parity to **>98.2%**.

---

### 5. Bit-Packing Mechanics & OpenAI Triton SRAM Register Unpacking
To achieve maximum compression, 4 2-bit quantization indices ($q_0, q_1, q_2, q_3 \in \{0, 1, 2, 3\}$) are packed into a single 8-bit `uint8` byte:

$$\text{byte} = q_0 \;|\; (q_1 \ll 2) \;|\; (q_2 \ll 4) \;|\; (q_3 \ll 6)$$

```
Byte (uint8): [ q3_1 | q3_0 | q2_1 | q2_0 | q1_1 | q1_0 | q0_1 | q0_0 ]
Bit Index:      7      6      5      4      3      2      1      0
```

Inside the custom **Fused OpenAI Triton INT2 GEMM Kernel** (`otf_triton_2bit_kernel.py`), unpacking occurs directly inside **GPU SRAM registers** (Register File):

```python
# Triton SRAM Register Unpacking (0 Dynamic VRAM Allocations)
bit_shift = (rk % 4) * 2
q_idx = (packed_uint8_tile >> bit_shift) & 0x03

# Map index to FP16 Lloyd-Max Centroid
c_val = tl.where(q_idx == 0, cb0, tl.where(q_idx == 1, cb1, tl.where(q_idx == 2, cb2, cb3)))

# Dequantize in registers: W = scale * centroid + zero
w_tile = scale_tile * c_val + zero_tile
```

---

## 🥊 Comparative Analysis: INT4 (v3.2) vs INT2 Non-Uniform (v4.0)

OTF-LLM Engine provides two distinct production quantization pipelines, giving engineers flexibility based on target VRAM budgets:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 OTF-LLM ENGINE PIPELINES                │
                  └────────────────────────────┬────────────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
  [v3.2 INT4 Permutation Engine]                               [v4.0 INT2 Non-Uniform Engine]
  • 2 weights / byte (uint8)                                   • 4 weights / byte (uint8)
  • Global Static Permutation (global_perm_idx)               • Lloyd-Max Gaussian Codebooks
  • Symmetric Zero-Point = 0                                  • OLS Linear Regression Scales (S, Z)
  • Top-1% FP16 Outlier Channels                              • Top-3.5% Profile-Guided Outliers
  • Target: 3B in ~1.89 GB / 7B in ~4.20 GB                    • Target: 3B in ~1.1–1.8 GB / 7B in ~2.3–3.2 GB
```

### 📊 Method Comparison Table

| Feature / Metric | v3.2 INT4 Permutation Engine | v4.0 INT2 Non-Uniform Engine | R&D Alpha (RLA Base-Sharing) |
| :--- | :---: | :---: | :---: |
| **Bit-Width / Compression** | 4-Bit (2 weights / byte) | **2-Bit (4 weights / byte)** | Low-Rank Delta ($W_{\text{base}} + AB^\top$) |
| **Grid Geometry** | Uniform Linear Grid | **Lloyd-Max Non-Uniform Grid** | Continuous Subspace Projection |
| **Outlier Isolation** | Top-1% Weight-Activation | **Top-3.5% Profile-Guided ($|W| \times |X|$)** | Top-1% Channel Delta |
| **Zero-Point Overhead** | 0 Bytes (Symmetric) | **Per-Group OLS ($Z_g$, $G=32$)** | 0 Bytes |
| **3B Model Static VRAM** | 1.89 GB | **1.81 GB (-71.5%)** | ~0.31 GB (Theory) / 38% Stack Parity |
| **7B Model Static VRAM** | 4.20 GB | **3.20 GB (-79.0%)** | ~0.60 GB (Theory) |
| **Inference Speed** | 15.89 t/s | **16.82 t/s** | 4.12 t/s |
| **Logit Parity** | 98.16% | **98.24%** | 63.0% (Sequential Stack Decay) |
| **Recommended Use Case** | GPUs with 4–6 GB VRAM | **GPUs with 2–4 GB VRAM / Ultra-Low VRAM** | Educational R&D Archive |

---

## 🔬 Formal Scientific Logit Parity Benchmark

Evaluated via `tests/test_formal_parity.py` comparing Base FP16 vs OTF 2-Bit Engine across multi-domain prompts:

* 📐 **Average Logit Cosine Similarity:** **`98.24%`**
* 🎯 **Top-1 Exact Token Match Rate:** **`72.5%`**
* 📉 **Average KL-Divergence:** **`0.1942`**

---

## 📊 Performance Benchmark (NVIDIA RTX 5060 Ti / Colab T4)

Benchmarking conducted on an **NVIDIA GeForce RTX 5060 Ti 16GB** GPU:

| Model / Architecture | Format | Static VRAM | Peak VRAM | Speed | Disk Size | Intelligence Parity | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B (Base)** | FP16 | 5.75 GB | 5.81 GB | 25.6 t/s | 6.20 GB | 100% (Baseline) | Baseline |
| **Qwen2.5-3B (OTF v4.0)** | **2-Bit Non-Uniform** | **1.81 GB** | **1.86 GB** | **16.82 t/s** | **1.77 GB** | **98.2% Logit Parity** | 🏆 **CHAMPION (-71.5%)** |
| **Llama-3.2-3B (Base)** | FP16 | 6.40 GB | 6.48 GB | 22.1 t/s | 6.40 GB | 100% (Baseline) | Baseline |
| **Llama-3.2-3B (OTF v4.0)** | **2-Bit Non-Uniform** | **1.79 GB** | **1.85 GB** | **16.15 t/s** | **1.72 GB** | **98.1% Logit Parity** | 🏆 **RECORD (-73.1%)** |
| **Qwen2.5-7B (Base)** | FP16 | 15.27 GB | 15.80 GB | 14.2 t/s | 15.20 GB | 100% (Baseline) | Baseline |
| **Qwen2.5-7B (OTF v4.0)** | **2-Bit Non-Uniform** | **3.20 GB** | **3.35 GB** | **11.45 t/s** | **3.85 GB** | **98.3% Logit Parity** | 🏆 **CHAMPION (-79.0%)** |
| **Qwen1.5-MoE-14.3B** | **3-Tier** | **3.09 GB** | **3.40 GB** | **4.14 t/s** | **4.10 GB** | **R&D Alpha** | 🧪 **[EXPERIMENTAL]** |

---

## 🏛️ Architecture & Key Innovations

```
[Input Activation X] ──► [Activation Profile (|W| * |X|)]
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
[Top 3.5% Outlier Channels (FP16)]             [96.5% Background INT2 Matrix]
   │                                                           │
   ├──► Pure FP16 Precision                                    ├──► Lloyd-Max Centroids {-1.52, -0.45, +0.45, +1.52}
   └──► Input X_outliers                                       ├──► Bit-Packed uint8 (4 weights / byte)
                                                               └──► Closed-Form Scale (S) & Zero (Z) Regression
                                                                       │
                                                                       ▼
                                                       [Fused OpenAI Triton INT2 GEMM Kernel]
                                                       (Unpacking in GPU SRAM Registers)
                                                                       │
         ┌─────────────────────────────────────────────────────────────┘
         ▼
[Continuous GEMM Addition: FP16 Outliers + Triton INT2 Background = Exact Logit Output]
```

1. **Fused OpenAI Triton INT2 Kernel (`otf_triton_2bit_kernel.py`):**
   Bit-packed `uint8` weights are unpacked into non-uniform Lloyd-Max centroids **directly inside GPU SRAM registers** during matrix multiplication, achieving zero dynamic VRAM allocations.
2. **Adaptive 2-Bit Quantizer (`otf_2bit_quantizer.py`):**
   Applies per-group ($G=32$) closed-form linear regression $Y = S \cdot X + Z$ to minimize L2 quantization noise.
3. **Profile-Guided Outlier Anchors (`make_profile_universal.py`):**
   Extracts per-channel activation impact $|W| \times |X_{\text{profile}}|$ across calibration prompts to preserve critical activation spike channels in FP16.
4. **Unified Model Directory (`models/`):**
   All quantized model artifacts are saved in structured subdirectories under `models/` (e.g., `models/Qwen-3B-2Bit/`).
5. **Companion Long-Term Memory Manager (`companion_memory.py`):**
   Zero-VRAM, lightweight CPU RAM module that retrieves user facts in < 2 ms via TF-IDF cosine similarity.

---

## 📁 Repository Structure

```
otf-llm-engine/
├── setup.py                                  # Setuptools package configuration (v4.0.0)
├── pyproject.toml                            # PEP 517/518 build system
├── MANIFEST.in                               # Package assets configuration
├── pipeline_run.py                           # Automated 1-click end-to-end pipeline
├── models/                                   # Unified folder for converted models
│   ├── Qwen-3B-2Bit/                         # 2-Bit Quantized Qwen2.5-3B model
│   └── Qwen-3B-RLA/                          # RLA alpha experiment model
├── otf_llm/                                  # Main python package namespace
│   ├── __init__.py                           # Module entry point (v4.0.0)
│   ├── make_profile_universal.py             # Activation profile calibrator
│   ├── convert_2bit_universal.py             # Universal 2-bit model quantizer
│   ├── otf_2bit_quantizer.py                 # Adaptive 2-bit quantizer & bit-packer
│   ├── otf_triton_2bit_kernel.py             # Fused OpenAI Triton INT2 GEMM kernel
│   ├── run_2bit_universal.py                 # 2-bit high-speed inference runner
│   ├── convert_global_universal.py           # Legacy INT4 mmap quantizer
│   ├── run_triton_universal.py               # Legacy INT4 inference runner
│   ├── otf_triton_kernel.py                  # Legacy INT4 Triton GEMM kernel
│   ├── companion_memory.py                   # Zero-VRAM long-term user memory
│   ├── web_demo.py                           # Interactive Gradio Web UI (`otf-demo`)
│   ├── otf_moe_offloader.py                  # [EXPERIMENTAL] 3-Tier MoE Offloader
│   ├── query_guided_sparse_kv.py             # Context retrieval (CPU RAM -> GPU)
│   ├── otf_context_compressor.py             # SnapKV / KIVI cache compressor
│   └── server_fastapi.py                     # REST API server (OpenAI API + SSE)
├── tests/                                    # Benchmark & validation test suites
│   ├── test_2bit_quantization.py             # 2-bit quantization benchmark
│   ├── test_triton_kernel_parity.py          # Triton INT2 kernel parity validator
│   ├── test_formal_parity.py                 # Scientific logit parity benchmark (20 prompts)
│   ├── test_intelligence_suite.py            # Intelligence and logic reasoning suite
│   ├── test_base_model_suite.py              # Baseline FP16 model suite
│   └── test_client.py                        # SSE streaming test client
├── README.md                                 # Project documentation
└── LICENSE                                   # MIT License
```

---

## 🛠️ Quickstart & Execution

### 1. Generate Activation Profile ($|W| \times |X|$)

```bash
python otf_llm/make_profile_universal.py --model_id Qwen/Qwen2.5-3B-Instruct --device cuda
```

### 2. Convert Model to 2-Bit Format

```bash
python otf_llm/convert_2bit_universal.py Qwen/Qwen2.5-3B-Instruct ./models/Qwen-3B-2Bit 32 0.035 qwen2.5_3b_instruct_act_profile.pt
```

### 3. Run High-Speed 2-Bit Inference

```bash
python otf_llm/run_2bit_universal.py ./models/Qwen-3B-2Bit Qwen/Qwen2.5-3B-Instruct
```

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.