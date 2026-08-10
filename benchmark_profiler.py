import os, time, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from convert_global import GlobalSymmetricINT4Linear, QuantizedEmbedding, MODEL_ID, SAVE_PATH
from run_triton import TritonGlobalSymmetricLinear

device = "cuda" if torch.cuda.is_available() else "cpu"


def profile_weights_breakdown(model, state_dict_path):
    print("=" * 65)
    print("📊 ПОБАЙТОВЫЙ РАЗБОР ВЕСОВ И МЕТАДАННЫХ МОДЕЛИ")
    print("=" * 65)

    file_size_mb = os.path.getsize(state_dict_path) / (1024 ** 2)
    state_dict = torch.load(state_dict_path, map_location="cpu")

    bytes_breakdown = {
        "Embeddings (INT8)": 0,
        "LM_Head (FP16)": 0,
        "INT4 Quantized Background": 0,
        "FP16 Outliers (1% Brain)": 0,
        "Scales (scale_bg / scale)": 0,
        "Permutation Tables (perm_idx)": 0,
        "LayerNorms & Biases": 0,
        "Other Metadata": 0
    }

    for key, tensor in state_dict.items():
        size_bytes = tensor.numel() * tensor.element_size()

        if "embed_tokens" in key and "packed_q" in key:
            bytes_breakdown["Embeddings (INT8)"] += size_bytes
        elif "lm_head" in key:
            bytes_breakdown["LM_Head (FP16)"] += size_bytes
        elif "packed_q_bg" in key:
            bytes_breakdown["INT4 Quantized Background"] += size_bytes
        elif "W_outliers_fp16" in key:
            bytes_breakdown["FP16 Outliers (1% Brain)"] += size_bytes
        elif "scale" in key:
            bytes_breakdown["Scales (scale_bg / scale)"] += size_bytes
        elif "perm_idx" in key:
            bytes_breakdown["Permutation Tables (perm_idx)"] += size_bytes
        elif "norm" in key or "bias" in key:
            bytes_breakdown["LayerNorms & Biases"] += size_bytes
        else:
            bytes_breakdown["Other Metadata"] += size_bytes

    print(f"📁 Общий размер файла на диске: {file_size_mb:.2f} МБ\n")

    total_bytes = sum(bytes_breakdown.values())
    for category, b_size in bytes_breakdown.items():
        mb_size = b_size / (1024 ** 2)
        pct = (b_size / total_bytes) * 100 if total_bytes > 0 else 0
        bar = "█" * int(pct / 4)
        print(f"  • {category:<32}: {mb_size:>7.2f} МБ ({pct:>5.1f}%) | {bar}")
    print("=" * 65 + "\n")


def run_single_task_benchmark(model, tokenizer, task_name, prompt, max_tokens=120):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    vram_before = torch.cuda.memory_allocated() / (1024 ** 2)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=2, do_sample=False)

    t_start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    t_end = time.time()

    vram_after = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    total_time = t_end - t_start
    tps = gen_tokens / total_time

    decoded_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"🎯 ЗАДАЧА: {task_name}")
    print(f"  • Сгенерировано токенов: {gen_tokens}")
    print(f"  • Время генерации:       {total_time:.2f} сек")
    print(f"  • Скорость (Speed):      {tps:.2f} токенов/сек")
    print(f"  • Использование VRAM:    {vram_after:.2f} МБ (Пик: {vram_peak:.2f} МБ)")
    print(f"  • Кэш контекста (KV):   {vram_after - vram_before:+.2f} МБ")
    print(f"\n📝 Ответ модели:\n{decoded_text[:200]}...")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    profile_weights_breakdown(None, SAVE_PATH)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cpu")

    # Подмена ТОЛЬКО Входных Эмбеддингов
    model.model.embed_tokens = QuantizedEmbedding(model.model.embed_tokens.num_embeddings,
                                                  model.model.embed_tokens.embedding_dim)

    # Подмена Слоев на Triton Engine
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

    state_dict = torch.load(SAVE_PATH, map_location="cpu")

    # Загрузка INT8 Эмбеддингов
    model.model.embed_tokens.packed_q = state_dict.pop("model.embed_tokens.packed_q")
    model.model.embed_tokens.scale = state_dict.pop("model.embed_tokens.scale")

    # Загрузка Трансформерных Слоев
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

    vram_init = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"🚀 Движок Triton загружен за: {time.time() - t0:.2f} сек")
    print(f"💾 Базовый объем VRAM весов:   {vram_init:.2f} МБ ({vram_init / 1024:.2f} ГБ)\n")

    print("=" * 65)
    print("🧪 ЗАПУСК СЕРИИ БЕНЧМАРК-ТЕСТОВ РАЗНЫХ ЗАДАЧ")
    print("=" * 65)

    task_code = "Python Code Generation"
    prompt_code = "Напиши функцию на Python для алгоритма бинарного поиска (binary search) в отсортированном массиве."
    run_single_task_benchmark(model, tokenizer, task_code, prompt_code, max_tokens=100)

    task_math = "Math & Logic Reasoning"
    prompt_math = "У фермера было 15 овец. Все, кроме 9, разбежались. Сколько овец осталось у фермера? Объясни логику шагов коротко."
    run_single_task_benchmark(model, tokenizer, task_math, prompt_math, max_tokens=80)

    task_text = "Text Summarization & Grammar"
    prompt_text = "Сократи следующий текст до одного предложения: Искусственный интеллект продолжает развиваться быстрыми темпами. Алгоритмы машинного обучения становятся всё более эффективными, позволяя запускать сложные большие языковые модели даже на обычных домашних видеокартах благодаря современным технологиям квантования весов."
    run_single_task_benchmark(model, tokenizer, task_text, prompt_text, max_tokens=80)