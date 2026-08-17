"""
OTF-LLM Engine: Interactive Gradio Web UI & RLM File Analyzer
Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
Distributed under the terms of the MIT License.
"""

import os
import sys
import time
import torch
import gradio as gr
from transformers import AutoTokenizer

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from otf_llm.run_rlm_file import load_model_2bit_engine
from otf_llm.rlm_agent import ContextContainer, PythonREPLExecutor

# Global model state
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_DIR = "./models/Qwen-3B-2Bit"
DEFAULT_BASE_ID = "Qwen/Qwen2.5-3B-Instruct"


def initialize_engine_if_needed(model_dir: str = DEFAULT_MODEL_DIR, base_model: str = DEFAULT_BASE_ID):
    """Loads 2-bit model into VRAM on first startup."""
    global GLOBAL_MODEL, GLOBAL_TOKENIZER
    if GLOBAL_MODEL is None:
        print(f"📦 [Web UI] Loading 2-Bit Model from '{model_dir}'...", flush=True)
        GLOBAL_MODEL = load_model_2bit_engine(model_dir, base_model, device=GLOBAL_DEVICE)
        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


def chat_generate(message: str, history: list, max_tokens: int, temperature: float, repetition_penalty: float):
    """Standard conversational streaming inference."""
    initialize_engine_if_needed()

    messages = [{"role": "system",
                 "content": "You are an intelligent, helpful AI assistant powered by GT Labs AI OTF-LLM Engine."}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    prompt_text = GLOBAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = GLOBAL_TOKENIZER(prompt_text, return_tensors="pt").to(GLOBAL_DEVICE)

    with torch.no_grad():
        outputs = GLOBAL_MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 0.01) if temperature > 0.0 else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=GLOBAL_TOKENIZER.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = GLOBAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def rlm_analyze_file(file_obj, query: str):
    """RLM Context-as-a-Variable file analyzer."""
    if file_obj is None:
        return "❌ Error: Please upload a text/code/log file first.", ""

    initialize_engine_if_needed()

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    ctx = ContextContainer(content, name="ctx")
    status_log = [f"📦 Context Loaded: `{ctx.describe()}` ({len(content) / 1024:.1f} KB)"]

    system_prompt = (
        "You are an expert AI Systems Engineer analyzing a file loaded in variable `ctx`.\n"
        "To search the document, output `print(ctx.grep('keyword'))`."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Find database configuration and server port."},
        {"role": "assistant", "content": "```python\nprint(ctx.grep(\"database\"))\nprint(ctx.grep(\"port\"))\n```"},
        {"role": "user", "content": query}
    ]

    env_vars = {"ctx": ctx}
    final_answer = ""

    for turn in range(1, 3):
        prompt_text = GLOBAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = GLOBAL_TOKENIZER(prompt_text, return_tensors="pt").to(GLOBAL_DEVICE)

        with torch.no_grad():
            outputs = GLOBAL_MODEL.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=GLOBAL_TOKENIZER.eos_token_id
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = GLOBAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

        messages.append({"role": "assistant", "content": response})

        res = PythonREPLExecutor.execute(response, env_vars, fallback_query=query if turn == 1 else "")

        if turn == 1:
            status_log.append(f"⚡ [Turn 1 Action]:\n{res['executed_code']}")
            obs = res["stdout"] if res["stdout"] else "[No matching lines found]"
            status_log.append(f"📥 [Observation]:\n{obs[:500]}")

            messages.append({
                "role": "user",
                "content": f"Facts extracted from document:\n{obs}\n\nAnswer the user request directly based strictly on these facts: '{query}'"
            })
        else:
            final_answer = response
            break

    vram_stat = f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB" if torch.cuda.is_available() else "N/A"
    status_log.append(f"\n📊 Peak VRAM Usage: {vram_stat}")

    return final_answer, "\n\n".join(status_log)


def get_system_vram_stats():
    """Returns live VRAM telemetry."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        device_name = torch.cuda.get_device_name(0)
        return f"🖥️ **Device:** {device_name}\n\n- **Active VRAM:** {allocated:.2f} GB\n- **Reserved VRAM:** {reserved:.2f} GB\n- **Peak VRAM:** {max_alloc:.2f} GB"
    return "Running on CPU (CUDA not active)"


def launch_web_demo(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """Launches the Gradio interface."""
    with gr.Blocks(title="OTF-LLM Engine v4.0", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🚀 OTF-LLM Engine (v4.0.0) — High-Performance 2-Bit LLM & RLM Suite
            **Developed by GT Labs AI & Gleb Tikhiy** • *Adaptive Non-Uniform 2-Bit Quantization + Recursive Context Engine (<2.4 GB VRAM)*
            """
        )

        with gr.Tabs():
            # Tab 1: Chat
            with gr.TabItem("⚡ 2-Bit High-Speed Chat"):
                chatbot = gr.Chatbot(height=480)
                msg = gr.Textbox(label="Your Message", placeholder="Ask anything...", lines=2)
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")
                    submit_btn = gr.Button("Send Message", variant="primary")

                with gr.Accordion("⚙️ Inference Hyperparameters", open=False):
                    max_tokens_slider = gr.Slider(64, 1024, value=256, step=32, label="Max New Tokens")
                    temp_slider = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    rep_slider = gr.Slider(1.0, 1.5, value=1.15, step=0.05, label="Repetition Penalty")

                def user_input(user_message, history):
                    return "", history + [[user_message, None]]

                def bot_response(history, max_tokens, temp, rep):
                    user_msg = history[-1][0]
                    bot_msg = chat_generate(user_msg, history[:-1], max_tokens, temp, rep)
                    history[-1][1] = bot_msg
                    return history

                msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot_response, [chatbot, max_tokens_slider, temp_slider, rep_slider], chatbot
                )
                submit_btn.click(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot_response, [chatbot, max_tokens_slider, temp_slider, rep_slider], chatbot
                )
                clear_btn.click(lambda: None, None, chatbot, queue=False)

            # Tab 2: RLM File Analyzer
            with gr.TabItem("🧠 RLM Infinite Context File Analyzer"):
                gr.Markdown("### 📄 Analyze Massive Codebases, Logs or Documents (Context-as-a-Variable)")
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="Upload Document / Log / Code (.txt, .py, .md, .log, .json)")
                        query_input = gr.Textbox(label="Question about the file",
                                                 placeholder="e.g. Find error logs and server IP...", lines=3)
                        analyze_btn = gr.Button("🚀 Run RLM Codebase Audit", variant="primary")

                    with gr.Column(scale=1):
                        output_answer = gr.Textbox(label="🎯 Synthesized Answer", lines=6)
                        execution_trace = gr.Textbox(label="🔍 Autonomous RLM Execution Log", lines=8)

                analyze_btn.click(rlm_analyze_file, [file_input, query_input], [output_answer, execution_trace])

            # Tab 3: Diagnostics
            with gr.TabItem("📊 Hardware & Engine Diagnostics"):
                diag_box = gr.Markdown(get_system_vram_stats())
                refresh_btn = gr.Button("🔄 Refresh Hardware Telemetry")
                refresh_btn.click(get_system_vram_stats, None, diag_box)

    print(f"\n🌐 Launching OTF-LLM Gradio Web UI on http://{host}:{port}...", flush=True)
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch_web_demo()