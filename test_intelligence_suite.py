import os
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert_global import QuantizedEmbedding, GlobalSymmetricINT4Linear, MODEL_ID, SAVE_PATH
from otf_triton_kernel import triton_fused_int4_linear

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


def run_benchmark_test(model, tokenizer, test_id, test_name, prompt, max_tokens=250):
    print("=" * 80)
    print(f"🧪 ТЕСТ {test_id}: {test_name.upper()}")
    print("=" * 80)

    torch.cuda.reset_peak_memory_stats()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    t_end = time.time()

    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    gen_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    total_time = t_end - t_start
    tps = gen_tokens / total_time

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"📊 Метрики выполнения:")
    print(f"  • Время:         {total_time:.2f} сек ({tps:.2f} токенов/сек)")
    print(f"  • Пиковый VRAM:  {vram_peak:.2f} МБ ({vram_peak / 1024:.2f} ГБ)")
    print(f"\n📝 Ответ модели:\n")
    print(response)
    print("\n" + "-" * 80 + "\n")

    return {
        "test_id": test_id,
        "name": test_name,
        "time": total_time,
        "tps": tps,
        "vram": vram_peak,
        "response": response
    }


if __name__ == "__main__":
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cpu")

    model.config.rope_theta = 1000000.0

    # 1. Замена Входных Эмбеддингов
    old_emb = model.model.embed_tokens
    model.model.embed_tokens = QuantizedEmbedding(old_emb.num_embeddings, old_emb.embedding_dim, original_emb=None)

    # 2. Замена Слоев Transformer
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

    # 3. Замена lm_head
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

    vram_stat = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"🚀 Triton Champion Engine (v2.0) загружен за: {time.time() - t0:.2f} сек")
    print(f"💾 Статичный VRAM весов модели: {vram_stat:.2f} МБ ({vram_stat / 1024:.2f} ГБ)\n")

    # Прогрев
    dummy = tokenizer("Тест", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**dummy, max_new_tokens=5, do_sample=False)

    results = []

    # ТЕСТ 1
    p1 = """У меня есть 5 коробок, стоящих в ряд слева направо (1, 2, 3, 4, 5).
1. В коробку 2 я кладу красный шар, а в коробку 4 — синий.
2. Я меняю местами содержимое коробки 1 и коробки 4.
3. Затем я беру то, что лежит в коробке 2, и перекладываю в коробку 5.
4. В конце я переворачиваю весь ряд коробок задом наперед (теперь коробка 5 становится первой, 4 — второй и т.д.).

В какой по счету коробке слева направо теперь лежит синий шар, а в какой — красный? Отвечай пошагово."""

    results.append(run_benchmark_test(model, tokenizer, 1, "Многошаговая логика и пространство", p1, max_tokens=350))

    # ТЕСТ 2
    p2 = """Напиши короткий рассказ (3-4 предложения) про космонавта на Марсе.
Правила:
1. Запрещено использовать букву "о" (ни строчную, ни заглавную) во всем тексте.
2. Каждое предложение должно начинаться с новой строки.
3. Ответь ТОЛЬКО текстом рассказа, без предсловий и комментариев."""

    results.append(run_benchmark_test(model, tokenizer, 2, "Жесткие ограничения (Без буквы 'о')", p2, max_tokens=200))

    # ТЕСТ 3
    p3 = """Напиши на Python функцию `compress_string(s: str) -> str`, которая выполняет RLE-сжатие строки (например, "aabccca" -> "a2b1c3a1").
Требования:
1. Алгоритм должен корректно обрабатывать пустую строку и строки из одного символа.
2. Сложность должна быть O(N) по времени и O(1) по дополнительной памяти (не считая выходную строку).
3. Добавь type hints и docstring с примерами. Напиши 3 unit-теста с помощью `assert`."""

    results.append(run_benchmark_test(model, tokenizer, 3, "Генерация кода RLE и краевые случаи", p3, max_tokens=400))

    # ТЕСТ 4
    p4 = """Реши задачу:
Человек смотрит на портрет и говорит: «Братьев и сестер у меня нет, но отец этого человека — сын моего отца». На портрете изображен сам этот человек.
Правильно ли это утверждение? Если нет, то кто на самом деле изображен на портрете и почему?"""

    results.append(run_benchmark_test(model, tokenizer, 4, "Ловушка на поверхностное мышление", p4, max_tokens=250))

    print("=" * 80)
    print("🏆 ВСЕ 4 БЕНЧМАРК-ТЕСТА УСПЕШНО ЗАВЕРШЕНЫ!")
    print("=" * 80)