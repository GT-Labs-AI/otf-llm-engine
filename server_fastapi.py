# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & GlebTikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# server_fastapi.py
import os
import time
import json
import gc
import asyncio
import threading
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

try:
    from safetensors.torch import load_file as safe_load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from convert_global_universal import QuantizedEmbedding, GlobalSymmetricINT4Linear
from otf_triton_kernel import triton_fused_int4_linear

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


class TritonGlobalSymmetricLinear(GlobalSymmetricINT4Linear):
    def forward(self, x):
        if not self.is_calibrated or self.packed_q_bg.numel() == 0:
            return super().forward(x)

        dtype, device = x.dtype, x.device
        x_permuted = torch.index_select(x, dim=-1, index=self.perm_idx.to(device).long())

        k = self.W_outliers_fp16.shape[1]
        x_outliers, x_bg = x_permuted[..., :k], x_permuted[..., k:]

        out_outliers = nn.functional.linear(x_outliers, self.W_outliers_fp16.to(device, dtype=dtype))

        bg_in_features = self.in_features - k
        packed_q_2d = self.packed_q_bg.to(device).view(self.out_features, bg_in_features // 2)

        out_bg = triton_fused_int4_linear(x_bg, packed_q_2d, self.scale_bg.to(device), self.group_size)

        out = out_outliers + out_bg
        if self.bias is not None:
            out = out + self.bias.to(dtype)

        return out


model = None
tokenizer = None
inference_lock = asyncio.Lock()


# --- QUERY-GUIDED CONTEXT COMPRESSOR FOR VS CODE ---
def compress_large_context_if_needed(prompt_text: str, max_allowed_tokens: int = 2500) -> str:
    """Предиктивный отбор самых релевантных блоков кода на CPU за 0.05 сек при длинных промптах VS Code"""
    input_ids = tokenizer.encode(prompt_text)
    total_tokens = len(input_ids)

    if total_tokens <= max_allowed_tokens:
        return prompt_text

    print(f"📄 [CONTEXT COMPRESSOR] Обнаружен длинный промпт ({total_tokens} токенов). Включение Query-Guided отжима...")
    t0 = time.time()

    # Делим длинный текст на блоки по 512 токенов
    block_size = 512
    num_blocks = (total_tokens + block_size - 1) // block_size
    blocks = []

    for i in range(num_blocks):
        b_ids = input_ids[i * block_size : (i + 1) * block_size]
        blocks.append(tokenizer.decode(b_ids, skip_special_tokens=True))

    # Первый блок (системные инструкции) + Последний блок (самый свежий запрос пользователя)
    selected_indices = {0, num_blocks - 1}

    # Для промежуточных блоков выполняем весовой отбор
    user_query = blocks[-1]
    query_words = set(user_query.lower().split())

    scores = []
    for idx, block in enumerate(blocks):
        if idx in selected_indices:
            scores.append((1000.0, idx))
            continue
        block_words = block.lower().split()
        overlap = sum(1 for w in block_words if w in query_words and len(w) > 2)
        scores.append((float(overlap), idx))

    scores.sort(key=lambda x: x[0], reverse=True)

    # Берем лучшие блоки до лимита в 2500 токенов
    budget_blocks = (max_allowed_tokens // block_size)
    for score, idx in scores[:budget_blocks]:
        selected_indices.add(idx)

    sorted_indices = sorted(list(selected_indices))
    stitched_blocks = [blocks[idx] for idx in sorted_indices]

    compressed_text = "\n\n[... Отфильтрованные фоновые файлы проекта ...]\n\n".join(stitched_blocks)
    new_tokens = len(tokenizer.encode(compressed_text))

    print(f"⚡ [CONTEXT COMPRESSOR] Промпт сжат с {total_tokens} до {new_tokens} токенов за {time.time() - t0:.3f} сек!")
    return compressed_text


def fix_rope_position_embeddings(model, config):
    for m in model.modules():
        if hasattr(m, "inv_freq"):
            dim = m.inv_freq.shape[0] * 2 if m.inv_freq.numel() > 0 else (config.hidden_size // config.num_attention_heads)
            base = getattr(m, "base", getattr(config, "rope_theta", 1000000.0)) or 1000000.0
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
            m.inv_freq = inv_freq.to(dtype=torch.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    t0 = time.time()
    print("=" * 70)
    print(f"🚀 [LIFESPAN STARTUP] Запуск VS Code AI Engine ({MODEL_ID})...")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)

    with torch.device("meta"):
        raw_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    model = raw_model.to_empty(device="cpu")
    fix_rope_position_embeddings(model, config)

    # 1. Embeddings -> INT8
    old_emb = model.model.embed_tokens
    model.model.embed_tokens = QuantizedEmbedding(
        old_emb.num_embeddings,
        old_emb.embedding_dim,
        original_emb=None
    )

    # 2. Transformer Layers -> Triton INT4
    for name, module in model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    new_linear = TritonGlobalSymmetricLinear(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bias=(child_module.bias is not None)
                    )
                    setattr(module, child_name, new_linear)

    # 3. lm_head -> Triton INT4
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        model.lm_head = TritonGlobalSymmetricLinear(
            in_features=model.lm_head.in_features,
            out_features=model.lm_head.out_features,
            bias=(model.lm_head.bias is not None)
        )

    clean_name = MODEL_ID.split("/")[-1].lower().replace("-", "_")
    save_path_safetensors = f"otf_{clean_name}_compressed.safetensors"
    save_path_pt = f"otf_{clean_name}_compressed.pt"

    if os.path.exists(save_path_safetensors) and HAS_SAFETENSORS:
        save_path = save_path_safetensors
        is_safetensors = True
    elif os.path.exists(save_path_pt):
        save_path = save_path_pt
        is_safetensors = False
    else:
        raise FileNotFoundError(f"Сжатый чекпоинт для {MODEL_ID} не найден!")

    print(f"📥 Загрузка сжатого чекпоинта {save_path}...")
    if is_safetensors:
        state_dict = safe_load_file(save_path)
    else:
        state_dict = torch.load(save_path, map_location="cpu")

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
                    module.bias = nn.Parameter(state_dict.pop(f"{prefix}bias"))
                module.is_calibrated = True

    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    vram_stat = torch.cuda.memory_allocated() / (1024 ** 2)
    print("=" * 70)
    print(f"✅ OTF Triton Engine готов к работе за {time.time() - t0:.2f} сек!")
    print(f"💾 Занято VRAM весами: {vram_stat:.2f} МБ ({vram_stat / 1024:.2f} ГБ)")
    print("=" * 70 + "\n")

    yield

    print("🛑 Сервер останавливается...")
    torch.cuda.empty_cache()


app = FastAPI(
    title="OTF-LLM Engine API Server for VS Code",
    description="Высокоэффективный REST API сервер для ИИ-ассистентов разработки в VS Code",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = True


class CompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


@app.get("/health")
async def health_check():
    vram_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    return {
        "status": "online",
        "engine": "OTF Triton Champion v2.0",
        "model_id": MODEL_ID,
        "vram_allocated_mb": round(vram_allocated_mb, 2),
        "device": device
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "otf-llm-engine"
            }
        ]
    }


async def generate_stream_tokens(inputs, max_new_tokens, temperature):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    created_timestamp = int(time.time())

    for new_text in streamer:
        chunk_data = {
            "id": f"chatcmpl-{created_timestamp}",
            "object": "chat.completion.chunk",
            "created": created_timestamp,
            "model": MODEL_ID,
            "choices": [{
                "delta": {"content": new_text},
                "index": 0,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.001)

    done_data = {
        "id": f"chatcmpl-{created_timestamp}",
        "object": "chat.completion.chunk",
        "created": created_timestamp,
        "model": MODEL_ID,
        "choices": [{
            "delta": {},
            "index": 0,
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    async with inference_lock:
        messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
        raw_prompt = tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)

        # Применяем умный фильтр контекста для длинных файлов VS Code
        compressed_prompt = compress_large_context_if_needed(raw_prompt, max_allowed_tokens=2500)

        inputs = tokenizer(compressed_prompt, return_tensors="pt").to(device)

        if req.stream:
            return StreamingResponse(
                generate_stream_tokens(inputs, req.max_tokens, req.temperature),
                media_type="text/event-stream"
            )
        else:
            t_gen_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=(req.temperature > 0.0),
                    temperature=max(req.temperature, 0.01) if req.temperature > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "message": {"role": "assistant", "content": generated_text.strip()},
                    "finish_reason": "stop",
                    "index": 0
                }]
            }


# Эндпоинт автодополнения кода (Autocomplete / FIM)
@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    async with inference_lock:
        prompt_str = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=(req.temperature > 0.0),
                temperature=max(req.temperature, 0.01) if req.temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        completion_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "text": completion_text,
                "index": 0,
                "finish_reason": "stop"
            }]
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_fastapi:app", host="0.0.0.0", port=8000, reload=False)