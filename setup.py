# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.

import os
from setuptools import setup, find_packages

readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="otf-llm",
    version="4.1.0",
    author="GT Labs AI & Gleb Tikhiy",
    author_email="team.gtlabs@gmail.com",
    description="High-performance hybrid LLM inference engine with Adaptive Non-Uniform 2-Bit Quantization (Lloyd-Max + Fused Triton INT2 GEMM), RLM Context-as-a-Variable engine, 98.2% logit parity, and Zero-RAM quantizer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GT-Labs-AI/otf-llm-engine",
    project_urls={
        "Bug Tracker": "https://github.com/GT-Labs-AI/otf-llm-engine/issues",
        "Source Code": "https://github.com/GT-Labs-AI/otf-llm-engine",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "triton>=2.1.0; sys_platform != 'win32'",
        "triton-windows; sys_platform == 'win32'",
        "transformers>=4.38.0",
        "safetensors>=0.4.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "gradio>=4.0.0",
        "huggingface_hub>=0.20.0",
        "accelerate>=0.26.0",
    ],
    entry_points={
        "console_scripts": [
            "otf-server=otf_llm.server_fastapi:main",
            "otf-quantize=otf_llm.convert_2bit_universal:main",
            "otf-run=otf_llm.run_2bit_universal:main",
            "otf-rlm=otf_llm.run_rlm_file:main",
            "otf-quantize-int4=otf_llm.convert_global_universal:main",
            "otf-run-int4=otf_llm.run_triton_universal:main",
            "otf-demo=otf_llm.web_demo:launch_web_demo",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)