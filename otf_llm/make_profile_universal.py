# make_profile_universal.py
import os, argparse, time, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_act_profile(model_id: str, device: str = "cpu"):
    clean_name = model_id.split("/")[-1].lower().replace("-", "_")
    profile_path = f"{clean_name}_act_profile.pt"

    print("=" * 70)
    print(f"🎯 УНИВЕРСАЛЬНЫЙ СНЯТЕЛЬ ПРОФИЛЯ ДЛЯ МОДЕЛИ: {model_id}")
    print(f"💻 Запуск на устройстве: {device.upper()}")
    print("=" * 70)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )

    importance_dict = {}

    def make_hook(name):
        def hook(module, input, output):
            x = input[0].detach().abs().float()
            mean_x = x.reshape(-1, x.shape[-1]).mean(dim=0).cpu()
            if name not in importance_dict:
                importance_dict[name] = mean_x
            else:
                importance_dict[name] += mean_x

        return hook

    hooks = []
    # Находим все целевые линейные слои трансформера и lm_head
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("mlp" in name or "self_attn" in name or name == "lm_head"):
            hooks.append(module.register_forward_hook(make_hook(name)))

    print("📥 Прогон универсального набора калибровочных промптов...")
    prompts = [
        "Write a complex Python function to solve the Traveling Salesperson Problem with dynamic programming.",
        "Explain the internal mechanics of Transformer self-attention and Rotary Position Embeddings (RoPE).",
        "Составь подробный план оптимизации VRAM при работе с крупными языковыми моделями."
    ]

    with torch.no_grad():
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inputs)

    for h in hooks:
        h.remove()

    torch.save(importance_dict, profile_path)

    # Гарантированное высвобождение ОЗУ перед запуском конвертера
    del model, tokenizer, hooks
    gc.collect()
    torch.cuda.empty_cache()

    print(f"✅ Профиль успешно сохранен в: {profile_path} за {time.time() - t0:.2f} сек!\n")
    return profile_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный профилировщик активаций LLM")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Имя модели")
    parser.add_argument("--device", type=str, default="cpu", help="cpu или cuda")
    args = parser.parse_args()

    create_act_profile(args.model_id, args.device)