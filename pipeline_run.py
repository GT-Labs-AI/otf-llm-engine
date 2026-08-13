# pipeline_run.py
import gc, torch, argparse
from otf_llm.make_profile_universal import create_act_profile
from otf_llm.convert_global_universal import convert_model
from otf_llm.run_triton_universal import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Единый пайплайн OTF-LLM Engine")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Имя модели")
    parser.add_argument("--device_quant", type=str, default="cpu", help="Где выполнять сжатие: cpu или cuda")
    args = parser.parse_args()

    print("\n" + "🚀" * 35)
    print(f"   ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА СЖАТИЯ И ИНФЕРЕНСА: {args.model_id}")
    print("🚀" * 35 + "\n")

    # Шаг 1: Профиль
    print("--- [ШАГ 1/3] Снятие калибровочного профиля активаций ---")
    create_act_profile(args.model_id, device=args.device_quant)
    gc.collect()
    torch.cuda.empty_cache()

    # Шаг 2: Конвертация
    print("--- [ШАГ 2/3] Послойное сжатие весов в INT4/INT8 ---")
    convert_model(args.model_id, device=args.device_quant)
    gc.collect()
    torch.cuda.empty_cache()

    # Шаг 3: Запуск Triton Движка
    print("--- [ШАГ 3/3] Запуск Triton Champion Engine в VRAM ---")
    run_inference(args.model_id)