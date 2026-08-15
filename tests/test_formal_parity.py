# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# tests/test_formal_parity.py

import time
import math
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from otf_llm import QuantizedEmbedding, TritonGlobalSymmetricLinear, fix_rope_position_embeddings

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


def measure_formal_logits_parity():
    print("=" * 85)
    print("🔬 EXTENDED FORMAL SCIENTIFIC PARITY BENCHMARK (20 DIVERSE PROMPTS)")
    print("🎯 Metrics: Logit Cosine Similarity | KL-Divergence | Top-1 Token Agreement %")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. Load Baseline FP16 Model on CUDA
    print("\n[1/3] Loading Baseline FP16 Model on CUDA...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    ).to(device)

    # 2. Build OTF INT4 Champion Model
    print("[2/3] Loading OTF INT4 Champion Model on CUDA...")
    clean_name = MODEL_ID.split("/")[-1].lower().replace("-", "_")
    save_path = f"otf_{clean_name}_compressed.safetensors"

    with torch.device("meta"):
        raw_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    model_otf = raw_model.to_empty(device="cpu")
    fix_rope_position_embeddings(model_otf, config)

    old_emb = model_otf.model.embed_tokens
    model_otf.model.embed_tokens = QuantizedEmbedding(old_emb.num_embeddings, old_emb.embedding_dim, original_emb=None)

    for name, module in model_otf.named_modules():
        if "mlp" in name or "self_attn" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    new_linear = TritonGlobalSymmetricLinear(child_module.in_features, child_module.out_features, bias=(child_module.bias is not None))
                    setattr(module, child_name, new_linear)

    if hasattr(model_otf, "lm_head") and isinstance(model_otf.lm_head, nn.Linear):
        model_otf.lm_head = TritonGlobalSymmetricLinear(model_otf.lm_head.in_features, model_otf.lm_head.out_features, bias=(model_otf.lm_head.bias is not None))

    from safetensors.torch import load_file
    state_dict = load_file(save_path)

    for emb_key in ["model.embed_tokens.packed_q", "embed_tokens.packed_q"]:
        if emb_key in state_dict:
            model_otf.model.embed_tokens.packed_q = state_dict.pop(emb_key)
            scale_key = emb_key.replace("packed_q", "scale")
            model_otf.model.embed_tokens.scale = state_dict.pop(scale_key)
            break

    for name, module in model_otf.named_modules():
        if isinstance(module, TritonGlobalSymmetricLinear):
            prefix = f"{name}." if name else ""
            if f"{prefix}packed_q_bg" in state_dict:
                module.perm_idx = state_dict.pop(f"{prefix}perm_idx")
                module.W_outliers_fp16 = state_dict.pop(f"{prefix}W_outliers_fp16")
                module.packed_q_bg = state_dict.pop(f"{prefix}packed_q_bg")
                module.scale_bg = state_dict.pop(f"{prefix}scale_bg")
                if f"{prefix}bias" in state_dict:
                    module.bias = nn.Parameter(state_dict.pop(f"{prefix}bias"))
                module.is_calibrated = True

    if hasattr(model_otf, "lm_head") and isinstance(model_otf.lm_head, TritonGlobalSymmetricLinear):
        if not model_otf.lm_head.is_calibrated:
            model_otf.lm_head.tied_embedding = model_otf.model.embed_tokens

    model_otf.load_state_dict(state_dict, strict=False)
    model_otf.to(device)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 20 Extended Multi-Domain Prompts
    prompts = [
        # Coding & Data Structures
        "Write a Python function for quicksort algorithm with docstrings and type hints.",
        "Explain the difference between TCP and UDP protocols in computer networking.",
        "Write an SQL query to find the second highest salary from an Employee table.",
        "Implement a thread-safe Singleton pattern in C++.",

        # Logic & Math
        "Solve for x: 5x + 3(x - 2) = 42.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "I have 3 boxes: red, blue, and green. The red box is to the left of blue. Where is green if it is right of blue?",

        # Science & Physics
        "Explain quantum entanglement and how it relates to quantum computing.",
        "What is the function of mitochondria in eukaryotic cells?",
        "Explain Newton's second law of motion and give a practical real-world example.",

        # Multi-Language & Translation
        "Переведи на русский язык: 'Artificial intelligence is reshaping software engineering.'",
        "Объясни принцип работы алгоритма бинарного поиска простыми словами.",
        "Übersetze ins Deutsche: 'The weather today is very pleasant for a walk in the park.'",

        # Constraints & Reasoning Traps
        "Write a 3-sentence short story where the letter 'e' is never used.",
        "Solve: What has keys but no locks, space but no room, and you can enter but not go in?",

        # Creative & Summarization
        "Write a short 4-line poem about space exploration.",
        "Summarize the main principles of clean code design in 3 bullet points.",
        "Write a polite email requesting a deadline extension for a software project.",
        "Explain the concept of inflation in economics and its primary causes."
    ]

    print(f"\n[3/3] Evaluating Logit Parity across {len(prompts)} Multi-Domain Prompts...\n")
    cosine_sims = []
    kl_divs = []
    token_agreements = []

    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits_fp16 = model_fp16(**inputs).logits[:, -1, :].float()
            logits_otf = model_otf(**inputs).logits[:, -1, :].float()

        cos_sim = F.cosine_similarity(logits_fp16, logits_otf, dim=-1).item()
        cosine_sims.append(cos_sim)

        p_fp16 = F.softmax(logits_fp16, dim=-1)
        log_p_otf = F.log_softmax(logits_otf, dim=-1)
        kl_div = F.kl_div(log_p_otf, p_fp16, reduction="batchmean").item()
        kl_divs.append(kl_div)

        token_fp16 = torch.argmax(logits_fp16, dim=-1).item()
        token_otf = torch.argmax(logits_otf, dim=-1).item()
        token_agreements.append(token_fp16 == token_otf)

        match_str = "MATCH" if token_fp16 == token_otf else "DIFF"
        print(f"  [{idx + 1:02d}/{len(prompts)}] Cosine Sim = {cos_sim * 100:.4f}% | KL Div = {kl_div:.6f} | Token: {match_str}")

    avg_cos = (sum(cosine_sims) / len(cosine_sims)) * 100
    avg_kl = sum(kl_divs) / len(kl_divs)
    match_pct = (sum(token_agreements) / len(token_agreements)) * 100

    print("\n" + "=" * 85)
    print("📊 EXTENDED SCIENTIFIC PARITY REPORT (20 PROMPTS):")
    print(f"  • Average Logit Cosine Similarity: {avg_cos:.4f}% (Goal: > 98.0%)")
    print(f"  • Average KL-Divergence:           {avg_kl:.6f} (Goal: < 0.15)")
    print(f"  • Top-1 Token Agreement Rate:      {match_pct:.1f}%")
    print("=" * 85)


if __name__ == "__main__":
    measure_formal_logits_parity()