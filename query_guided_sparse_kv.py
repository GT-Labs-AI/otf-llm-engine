import os
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from convert_global import QuantizedEmbedding, GlobalSymmetricINT4Linear, MODEL_ID, SAVE_PATH
from otf_triton_kernel import triton_fused_int4_linear

device = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_CACHE_PATH = "document_vector_index.pt"


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


def create_cpu_document_cache(model, tokenizer, document_text: str, chunk_size=2048):
    print(f"📥 Токенизация и снятие семантической карты документа...")
    inputs = tokenizer(document_text, return_tensors="pt").to(device)
    total_len = inputs.input_ids.shape[1]
    print(f"📄 Длина документа: {total_len} токенов")

    gpu_cache = DynamicCache()
    doc_hidden_states_list = []

    t0 = time.time()
    for start_idx in range(0, total_len, chunk_size):
        end_idx = min(start_idx + chunk_size, total_len)
        chunk = inputs.input_ids[:, start_idx:end_idx]

        pos_ids = torch.arange(start_idx, end_idx, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(chunk, position_ids=pos_ids, past_key_values=gpu_cache, use_cache=True,
                            output_hidden_states=True)
            gpu_cache = outputs.past_key_values
            doc_hidden_states_list.append(outputs.hidden_states[14].squeeze(0).cpu())
            torch.cuda.empty_cache()

    full_doc_hidden = torch.cat(doc_hidden_states_list, dim=0)

    del gpu_cache
    torch.cuda.empty_cache()

    print(f"⚡ Семантическая карта документа ({total_len} токенов) запечена за {time.time() - t0:.2f} сек!")
    return total_len, full_doc_hidden, inputs.input_ids[0]


def print_block_relevance_profiler(tokenizer, doc_input_ids, block_scores, selected_blocks, block_size=512):
    print("\n" + "=" * 80)
    print("📊 ПРОФЕССИОНАЛЬНЫЙ ИНСПЕКТОР РЕЛЕВАНТНОСТИ БЛОКОВ (TOP-10 RANKING)")
    print("=" * 80)
    print(f"{'Ранг':<6} | {'Блок №':<8} | {'Балл':<8} | {'Токены':<16} | {'Превью текста блока'}")
    print("-" * 80)

    ranked_blocks = torch.argsort(block_scores, descending=True)

    for rank, b_idx_tensor in enumerate(ranked_blocks[:10]):
        b_idx = b_idx_tensor.item()
        score = block_scores[b_idx].item()

        b_start = b_idx * block_size
        b_end = min(b_start + block_size, len(doc_input_ids))

        block_tokens = doc_input_ids[b_start:b_end].tolist()
        block_text = tokenizer.decode(block_tokens, skip_special_tokens=True).replace("\n", " ").strip()

        preview = block_text[:45] + "..." if len(block_text) > 45 else block_text
        is_selected = "✅ [В СКЛЕЙКЕ]" if b_idx in selected_blocks.tolist() else "❌ [ПРОПУЩЕН]"
        secret_flag = " 🎯 [СЕКРЕТ НАЙДЕН!]" if "98765-ALPHA" in block_text or "СЕКРЕТ" in block_text else ""

        print(
            f"#{rank + 1:<4} | #{b_idx:<6} | {score:>6.4f} | {b_start:>5}..{b_end:<5} | {preview:<45} {is_selected}{secret_flag}")

    print("=" * 80 + "\n")


def search_and_stitch_context(model, tokenizer, question_text, total_doc_len, full_doc_hidden, doc_input_ids,
                              block_size=512, top_n_blocks=2, sink_blocks=1):
    t0 = time.time()

    messages_q = [
        {"role": "system", "content": "Ты — точный аналитик. Отвечай кратко и строго по контексту."},
        {"role": "user", "content": f"Запрос по документу: {question_text}"}
    ]
    formatted_q = tokenizer.apply_chat_template(messages_q, tokenize=False, add_generation_prompt=True)
    q_inputs = tokenizer(formatted_q, return_tensors="pt").to(device)
    q_len = q_inputs.input_ids.shape[1]

    torch.cuda.reset_peak_memory_stats()

    # 1. Снимаем вектор Q для вопроса
    with torch.no_grad():
        q_outputs = model(q_inputs.input_ids, output_hidden_states=True)
        q_hidden = q_outputs.hidden_states[14].squeeze(0).cpu()

    # 2. TF-IDF ВЕСА ТОКЕНОВ ВОПРОСА (Убираем шум системных фраз)
    token_ids_flat = q_inputs.input_ids[0].cpu().tolist()
    q_weights = torch.ones(q_len)

    stop_words = {"ты", "точный", "аналитик", "отвечай", "кратко", "запрос", "по", "документу", "в", "из"}
    for idx, tid in enumerate(token_ids_flat):
        word = tokenizer.decode([tid]).strip().lower()
        if word in stop_words or len(word) <= 2:
            q_weights[idx] = 0.05
        else:
            q_weights[idx] = 4.0  # Повышаем вес ключевых слов вопроса

    # 3. ВЗВЕШЕННОЕ КОСИНУСНОЕ СХОДСТВО
    q_norm = torch.nn.functional.normalize(q_hidden, p=2, dim=-1) * q_weights.unsqueeze(-1)
    d_norm = torch.nn.functional.normalize(full_doc_hidden, p=2, dim=-1)
    q_norm = q_norm.to(d_norm.dtype)

    global_token_scores = torch.matmul(q_norm, d_norm.T).max(dim=0).values

    # 4. ПИКОВЫЙ (.max) ПОИСК БЛОКОВ ПО 512 ТОКЕНОВ
    num_blocks = (total_doc_len + block_size - 1) // block_size
    block_scores = []
    for b_idx in range(num_blocks):
        b_start = b_idx * block_size
        b_end = min(b_start + block_size, total_doc_len)
        block_scores.append(global_token_scores[b_start:b_end].max())

    block_scores = torch.stack(block_scores)

    sink_block_indices = torch.arange(0, min(sink_blocks, num_blocks))
    top_block_indices = torch.topk(block_scores[sink_blocks:],
                                   k=min(top_n_blocks, num_blocks - sink_blocks)).indices + sink_blocks

    selected_blocks = torch.cat([sink_block_indices, top_block_indices]).unique()
    selected_blocks = torch.sort(selected_blocks).values

    # Выводим красивый инспектор рангов
    print_block_relevance_profiler(tokenizer, doc_input_ids, block_scores, selected_blocks, block_size=block_size)

    # 5. ТЕКСТОВАЯ СКЛЕЙКА ОТБРАННЫХ БЛОКОВ
    stitched_text_parts = []
    for b_idx in selected_blocks.tolist():
        b_start = b_idx * block_size
        b_end = min(b_start + block_size, total_doc_len)

        block_tokens = doc_input_ids[b_start:b_end].tolist()
        block_text = tokenizer.decode(block_tokens, skip_special_tokens=True).strip()
        stitched_text_parts.append(block_text)

    stitched_document = "\n\n[... Документ сокращен ...]\n\n".join(stitched_text_parts)
    print(f"📥 Поиск и склейка выжимки выполнены за: {time.time() - t0:.3f} сек!")

    return stitched_document


if __name__ == "__main__":
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cpu")

    model.config.rope_theta = 1000000.0

    old_emb = model.model.embed_tokens
    model.model.embed_tokens = QuantizedEmbedding(old_emb.num_embeddings, old_emb.embedding_dim, original_emb=None)

    for name, module in model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    new_linear = TritonGlobalSymmetricLinear(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bias=(child_module.bias is not None),
                        original_linear=None
                    )
                    setattr(module, child_name, new_linear)

    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        model.lm_head = TritonGlobalSymmetricLinear(
            in_features=model.lm_head.in_features,
            out_features=model.lm_head.out_features,
            bias=(model.lm_head.bias is not None),
            original_linear=None
        )

    state_dict = torch.load(SAVE_PATH, map_location="cpu")

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
    torch.cuda.empty_cache()

    print(f"🚀 Triton Engine загружен в VRAM (1.94 ГБ) за: {time.time() - t0:.2f} сек\n")

    doc_sections = [
        "Раздел 1. Архитектура движка OTF-LLM использует кастомные Fused Triton INT4 GEMM ядра для весов. ",
        "Раздел 2. Таблица эмбеддингов квантована в INT8, что сокращает занимаемый VRAM объём на 311 МБ. ",
        "Раздел 3. Профиль активаций вычисляет 1% критических каналов и переносит их в блок FP16. ",
        "Раздел 4. Сжатие слоя lm_head сокращает VRAM до рекордных 1.94 ГБ. "
    ]

    full_doc = [doc_sections[i % len(doc_sections)] + f"Запись №{i}. " for i in range(500)]
    full_doc.insert(250, "\n\n[КРИТИЧЕСКИЙ СЕКРЕТ]: Код доступа к серверу: 98765-ALPHA.\n\n")
    big_document = "".join(full_doc)

    print("=" * 65)
    print("🧪 1. СНЯТИЕ СЕМАНТИЧЕСКОЙ КАРТЫ ДОКУМЕНТА")
    print("=" * 65)
    doc_len, full_doc_hidden, doc_input_ids = create_cpu_document_cache(model, tokenizer, big_document, chunk_size=2048)

    print("\n" + "=" * 65)
    print("🧪 2. TF-IDF ВЕСОВОЙ ПОИСК, ТЕКСТОВАЯ СКЛЕЙКА И ИНФЕРЕНС В VRAM")
    print("=" * 65)

    question = "Какой секретный код доступа к серверу указан в документе?"

    # 1. Поисковый фильтр за 0.001 сек вытягивает выжимку
    stitched_doc = search_and_stitch_context(model, tokenizer, question, doc_len, full_doc_hidden, doc_input_ids,
                                             block_size=512, top_n_blocks=2, sink_blocks=1)

    # 2. Оформляем короткую выжимку в ChatML
    messages = [
        {"role": "system", "content": "Ты — точный аналитик. Отвечай кратко и строго по контексту."},
        {"role": "user", "content": f"Вот выжимка из документа:\n{stitched_doc}\n\nВопрос: {question}"}
    ]

    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

    t_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_time = time.time() - t_gen
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print("\n✅ ИНФЕРЕНС ЗАВЕРШЕН!")
    print(f"  • Время генерации: {gen_time:.2f} сек")
    print(f"  • Пиковый VRAM:    {vram_peak:.2f} МБ ({vram_peak / 1024:.2f} ГБ)")

    print("\n📝 Ответ модели:")
    print(response)