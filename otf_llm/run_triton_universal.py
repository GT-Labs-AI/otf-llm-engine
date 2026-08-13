# OTF-LLM Engine (On-The-Fly Weight Synthesizer)
# Copyright (c) 2026 GT Labs AI & Gleb Tikhiy <team.gtlabs@gmail.com>
# Distributed under the terms of the MIT License.
# otf_llm/run_triton_universal.py

import os
import time
import gc
import argparse
import traceback
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

try:
    from safetensors.torch import load_file as safe_load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from .convert_global_universal import QuantizedEmbedding, GlobalSymmetricINT4Linear
from .otf_triton_kernel import triton_fused_int4_linear

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


def fix_rope_position_embeddings(model, config):
    """Точная математическая инициализация RoPE таблиц без загрузки FP16 тяжелых весов"""
    for m in model.modules():
        if hasattr(m, "inv_freq"):
            dim = m.inv_freq.shape[0] * 2 if m.inv_freq.numel() > 0 else (
                config.hidden_size // config.num_attention_heads
            )
            base = getattr(m, "base", getattr(config, "rope_theta", 1000000.0))
            if base is None:
                base = 1000000.0

            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
            m.inv_freq = inv_freq.to(dtype=torch.float32)


def run_inference(model_id: str, prompt: str = None):
    try:
        gc.collect()
        torch.cuda.empty_cache()

        clean_name = model_id.split("/")[-1].lower().replace("-", "_")

        save_path_safetensors = f"otf_{clean_name}_compressed.safetensors"
        save_path_pt = f"otf_{clean_name}_compressed.pt"

        if os.path.exists(save_path_safetensors) and HAS_SAFETENSORS:
            save_path = save_path_safetensors
            is_safetensors = True
        elif os.path.exists(save_path_pt):
            save_path = save_path_pt
            is_safetensors = False
        else:
            raise FileNotFoundError(f"Сжатый чекпоинт не найден! Сначала запустите convert_global_universal.py")

        t0 = time.time()
        print("=" * 70)
        print(f"🚀 МГНОВЕННЫЙ СТАРТ TRITON ENGINE ДЛЯ МОДЕЛИ: {model_id}")
        print(f"📦 Файл чекпоинта: {save_path} (0 МБ FP16 в ОЗУ!)")
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # МГНОВЕННОЕ СОЗДАНИЕ ПУСТОГО СКЕЛЕТА СЕТИ (0 МБ ВЕСОВ В ОЗУ!)
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

        # СВЯЗЫВАНИЕ TIED EMBEDDINGS ДЛЯ МОДЕЛЕЙ ТИПА LLAMA-3.2
        if hasattr(model, "lm_head") and isinstance(model.lm_head, TritonGlobalSymmetricLinear):
            if not model.lm_head.is_calibrated:
                model.lm_head.tied_embedding = model.model.embed_tokens

        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        vram_stat = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"⚡ Движок успешно загружен ВСЕГО за {time.time() - t0:.2f} сек!")
        print(f"💾 Занято VRAM весами модели: {vram_stat:.2f} МБ ({vram_stat / 1024:.2f} ГБ)\n")

        if not prompt:
            prompt = "Напиши короткую функцию на Python для алгоритма бинарного поиска."

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        torch.cuda.reset_peak_memory_stats()
        t_gen_start = time.time()
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        t_gen_end = time.time()

        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        gen_tokens = tokens.shape[1] - inputs.input_ids.shape[1]
        tps = gen_tokens / (t_gen_end - t_gen_start)

        print("=" * 70)
        print(f"📊 МЕТРИКИ ГЕНЕРАЦИИ:")
        print(f"  • Скорость:      {tps:.2f} токенов/сек")
        print(f"  • Пиковый VRAM:  {vram_peak:.2f} МБ ({vram_peak / 1024:.2f} ГБ)")
        print("=" * 70)
        print(f"\n📝 Ответ модели:\n")
        print(tokenizer.decode(tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())

    except Exception as e:
        print("\n❌ ОШИБКА ИНФЕРЕНСА:")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный Triton инференс-раннер")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Имя модели")
    parser.add_argument("--prompt", type=str, default=None, help="Промпт для генерации")
    args = parser.parse_args()

    run_inference(args.model_id, args.prompt)