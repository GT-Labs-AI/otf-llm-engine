# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_llm/web_demo.py

import os
import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from .convert_global_universal import QuantizedEmbedding
from .run_triton_universal import TritonGlobalSymmetricLinear, fix_rope_position_embeddings
from .companion_memory import CompanionMemoryManager

MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None
memory_manager = None


def load_engine_for_demo(model_id_param: str = MODEL_ID):
    global model, tokenizer, memory_manager
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id_param)
    config = AutoConfig.from_pretrained(model_id_param)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    memory_manager = CompanionMemoryManager()

    clean_name = model_id_param.split("/")[-1].lower().replace("-", "_")
    save_path = f"otf_{clean_name}_compressed.safetensors"

    if not os.path.exists(save_path):
        from .convert_global_universal import convert_model
        convert_model(model_id_param, device="cpu")

    with torch.device("meta"):
        raw_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    model = raw_model.to_empty(device="cpu")
    fix_rope_position_embeddings(model, config)

    old_emb = model.model.embed_tokens
    model.model.embed_tokens = QuantizedEmbedding(old_emb.num_embeddings, old_emb.embedding_dim, original_emb=None)

    for name, module in model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear):
                    new_linear = TritonGlobalSymmetricLinear(child_module.in_features, child_module.out_features,
                                                             bias=(child_module.bias is not None))
                    setattr(module, child_name, new_linear)

    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        model.lm_head = TritonGlobalSymmetricLinear(model.lm_head.in_features, model.lm_head.out_features,
                                                    bias=(model.lm_head.bias is not None))

    from safetensors.torch import safe_open
    state_dict = {}
    with safe_open(save_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    for emb_key in ["model.embed_tokens.packed_q", "embed_tokens.packed_q"]:
        if emb_key in state_dict:
            model.model.embed_tokens.packed_q = state_dict.pop(emb_key)
            scale_key = emb_key.replace("packed_q", "scale")
            model.model.embed_tokens.scale = state_dict.pop(scale_key)
            break

    for name, module in model.named_modules():
        if isinstance(module, TritonGlobalSymmetricLinear):
            prefix = f"{name}." if name else ""
            if f"{prefix}packed_q_bg" in state_dict:
                module.perm_idx = state_dict.pop(f"{prefix}perm_idx")
                module.W_outliers_fp16 = state_dict.pop(f"{prefix}W_outliers_fp16")
                module.packed_q_bg = state_dict.pop(f"{prefix}packed_q_bg")
                module.scale_bg = state_dict.pop(f"{prefix}scale_bg")
                if f"{prefix}bias" in state_dict:
                    module.bias = torch.nn.Parameter(state_dict.pop(f"{prefix}bias"))
                module.is_calibrated = True

    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    return f"⚡ Loaded in {time.time() - t0:.2f}s | Static VRAM: {vram_mb:.2f} MB ({vram_mb / 1024:.2f} GB)"


def chat_response(message: str, history):
    global model, tokenizer, memory_manager
    if model is None:
        status = load_engine_for_demo()

    # Extract user personal facts automatically
    memory_manager.auto_extract_and_store(message)

    # Inject Memory into System Prompt
    system_prompt = "You are an advanced AI Assistant powered by GT Labs AI OTF-LLM Engine."
    enhanced_system = memory_manager.inject_memory_into_system_prompt(system_prompt, message)

    messages = [{"role": "system", "content": enhanced_system}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    t_gen = time.time() - t0

    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tps = gen_tokens / t_gen
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    info_footer = f"\n\n---\n📊 **GT Labs AI Metrics:** Speed: `{tps:.2f} t/s` | Peak VRAM: `{vram_peak:.2f} MB ({vram_peak / 1024:.2f} GB)`"
    return response + info_footer


def launch_web_demo():
    print("🚀 Launching GT Labs AI Web Demo...")
    load_engine_for_demo()

    with gr.Blocks(title="OTF-LLM Engine Demo | GT Labs AI") as demo:
        gr.Markdown(
            """
            # 🚀 OTF-LLM Engine (On-The-Fly Weight Synthesizer)
            ### Official Interactive Demo by **GT Labs AI** & **Gleb Tikhiy**
            *Ultra-fast, Outlier-Aware INT4 Inference Engine powered by Custom OpenAI Triton Kernels.*
            """
        )

        gr.Chatinterface(
            fn=chat_response,
            examples=[
                "Write a high-performance binary search in Python.",
                "Explain quantum computing in 2 simple sentences.",
                "My name is Alex and I am an AI researcher."
            ]
        )

    demo.queue().launch(share=True)


if __name__ == "__main__":
    launch_web_demo()