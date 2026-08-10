# 🚀 On-The-Fly Weight Synthesizer (OTF-LLM Engine)

> **Высокоэффективный гибридный движок инференса LLM с кастомными Fused Triton INT4 GEMM ядрами, Outlier-Aware квантованием весов, глобальной перестановкой активаций, INT8-словарем, сжатием VRAM до 1.94 ГБ (3B) / 4.20 ГБ (7B) и продуктовым REST API сервером.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Triton](https://img.shields.io/badge/OpenAI_Triton-Fused_Kernels-red?logo=openai&logoColor=white)](https://github.com/openai/triton)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔬 Engineering & Research: GT Labs AI

This project is developed and maintained by **GT Labs AI**.

* 🚀 **GT Labs AI Mission:** Ultra-fast MVP engineering, custom AI integration, and deep neural network optimization research.
* 👨‍💻 **Author & Lead AI Engineer:** **Gleb Tikhiy** ([@GlebTikhiy](https://github.com/GlebTikhiy))
* 📧 **Contact & Inquiries:** team.gtlabs@gmail.com
* 🌐 **Organization:** [GT Labs AI on GitHub](https://github.com/GT-Labs-AI)

---

## 🎯 Цель Проекта

Преодоление барьера **Memory-Bound** и аппаратных ограничений VRAM при работе с большими языковыми моделями (LLM) и длинным контекстом на потребительских видеокартах.

Вместо перекачки тяжелых FP16 весов из VRAM, **OTF-LLM Engine** использует аппаратную деквантовку Outlier-Aware INT4 весов **прямиком в регистрах GPU (SRAM)** через кастомные **OpenAI Triton GEMM Kernels**, сжимает словарь (`embed_tokens`) и классификатор (`lm_head`), а также использует предиктивную выгрузку контекста **Query-Guided Sparse Offloading**.

---

## 📊 Бенчмарк Производительности (RTX 5060 Ti 16GB)

Тестирование проводилось на видеокарте **NVIDIA GeForce RTX 5060 Ti**:

| Модель / Архитектура | Формат | Статичный VRAM | Пиковый VRAM | Скорость | Время Загрузки | Паритет Интеллекта | Статус |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B (Base)** | FP16 | 5.75 ГБ | 5.81 ГБ | 25.6 т/с | ~15.0 сек | 100% (База) | База |
| **Qwen2.5-3B (OTF Champion)** | **INT4/8** | **1.94 ГБ** | **1.99 - 2.06 ГБ** | **16.15 т/с** | **4.2 сек** | **100% (0% Потерь)** | 🏆 **ЧЕМПИОН (-66.1%)** |
| **Qwen2.5-7B (Base)** | FP16 | 15.27 ГБ | 15.80 ГБ | 14.2 т/с | ~28.0 сек | 100% (База) | База |
| **Qwen2.5-7B (OTF Champion)** | **INT4/8** | **4.20 ГБ** | **4.25 ГБ** | **8.60 т/с** | **15.5 сек** | **100% (0% Потерь)** | 🏆 **ЧЕМПИОН (-72.5%)** |

---

## 🧠 Сравнительный Бенчмарк Интеллекта (A/B Test Suite)

Прямое A/B-тестирование на сложных наборах многошаговой логики, написания кода Python и текстовых ограничений подтверждает **0% деградации качества**:

| Тест / Задача | Результат OTF Triton Engine | Результат Base FP16 | Сохранение Качества |
| :--- | :---: | :---: | :---: |
| **1. Многошаговая пространственная логика** | Идентичные шаги перемещения | Идентичные шаги перемещения | **100% Паритет** 🧠 |
| **2. Строгие ограничения (Без буквы "о")** | Идеальное соблюдение правила | Соблюдение правила | **100% Паритет** 🎯 |
| **3. Генерация Python RLE Кода O(N)** | Точный алгоритм + Unit-тесты | Точный алгоритм + Unit-тесты | **100% Паритет** 💻 |
| **4. Загадки поверхностного мышления** | Распознана структура ловушки | Распознана структура ловушки | **100% Паритет** 🔍 |

---

## 🏛️ Архитектура и Ключевые Инновации

```
[Входной Вектор X] ──► [Global Static Permutation (global_perm_idx)]
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
[Аномальные Каналы (1% FP16)]                 [Фоновый Блок (99% INT4)]
   │                                                           │
   ├──► Чистый FP16                                            ├──► Диапазон [-7 ... +7]
   └──► Вход X_outliers                                        ├──► Упаковка 2:1 (uint8)
                                                               └──► Zero-Point = 0 БАЙТ!
                                                                       │
                                                                       ▼
                                                        [Custom Fused Triton GEMM Kernel]
                                                        (Распаковка в регистрах GPU SRAM)
                                                                       │
         ┌─────────────────────────────────────────────────────────────┘
         ▼
[Непрерывное сложение GEMM: Outliers + Triton Background = Точный Ответ FP16]
```

1. **Custom Outlier-Aware Fused Triton GEMM Kernel (`otf_triton_kernel.py`):**
   Упакованные `uint8` веса считываются из VRAM и деквантуются **прямиком в регистрах чипа GPU (SRAM)** во время матричного умножения. Это исключает выделение временных FP16 тензоров в памяти VRAM.
2. **Global Static Permutation (`global_perm_idx`) & Outlier Preservation:**
   Единая таблица перестановки каналов на всю модель (всего 1.6 МБ VRAM). Выделение Топ-1% критических аномальных каналов ($|W| \times |X_{\text{profile}}|$) в FP16 полностью блокирует квантовый шум и сохраняет 100% точности.
3. **INT8 Quantized Embeddings & Outlier-Aware INT4 `lm_head`:**
   Входной словарь сжат в INT8, а гигантский классификатор `lm_head` ($152\,064 \times 3584$) сжат из 1.09 ГБ до 280 МБ.
4. **Query-Guided Sparse Offload & Text-Stitching:**
   Документы выносятся в системную ОЗУ (CPU RAM, 0 МБ VRAM). За 0.07 сек предиктивный косинусный фильтр с весами TF-IDF отбирает наиболее релевантные текстовые блоки и подгружает их в GPU.
5. **FastAPI REST API Server (`server_fastapi.py`):**
   Асинхронный продуктовый сервер с поддержкой спецификации **OpenAI API (`/v1/chat/completions`)**, SSE (Server-Sent Events) стримингом токенов и асинхронным очередизатором для защиты VRAM от переполнения.

---

## 📁 Структура Репозитория

```
weight_synthesizer/
├── make_profile.py              # Калибратор профиля активаций для 3B моделей (50 КБ)
├── make_profile_7b.py           # Калибратор профиля активаций для 7B/8B моделей (120 КБ)
├── convert_global.py            # Конвертер 3B модели в формат INT4/INT8
├── convert_global_7b.py         # Конвертер 7B/8B моделей в формат INT4/INT8
├── otf_triton_kernel.py         # Кастомное Fused Triton INT4 GEMM ядро
├── run_triton.py                # Инференс-движок 3B модели (1.94 ГБ Static VRAM)
├── run_triton_7b.py             # Инференс-движок 7B модели (4.20 ГБ Static VRAM)
├── server_fastapi.py            # Продуктовый REST API Сервер (FastAPI + SSE Стриминг)
├── test_client.py               # Потоковый клиент для проверки SSE-стриминга
├── query_guided_sparse_kv.py    # Предиктивный вызов контекста (CPU RAM -> GPU)
├── otf_context_compressor.py    # Модуль SnapKV / KIVI асимметричного сжатия кэша
├── benchmark_profiler.py        # Побайтовый профилировщик весов и VRAM
├── test_intelligence_suite.py   # Бенчмарк интеллекта и логики на сжатом движке
├── test_base_model_suite.py     # A/B бенчмарк базовой FP16 модели
├── otf_qwen2.5_3b_global_symmetric.pt # Чекпоинт сжатой 3B модели (~2.08 ГБ)
├── otf_7b_global_symmetric.pt   # Чекпоинт сжатой 7B модели (~4.18 ГБ)
├── README.md                    # Документация проекта
└── LICENSE                      # Лицензия MIT
```

---

## 🛠️ Быстрый Запуск

### 1. Установка зависимостей

```bash
# Активация виртуального окружения
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux

# Установка PyTorch, Transformers и FastAPI
pip install torch transformers fastapi uvicorn pydantic requests

# Установка OpenAI Triton
pip install triton-windows  # Для Windows
# pip install triton        # Для Linux
```

### 2. Запуск 7B Модели (Qwen2.5-7B-Instruct в 4.20 ГБ VRAM)

```bash
# 1. Снятие профиля активаций (3 сек)
python make_profile_7b.py

# 2. Квантование модели в INT4/INT8 (10 сек)
python convert_global_7b.py

# 3. Автономный запуск инференса
python run_triton_7b.py
```

### 3. Запуск REST API Сервера (OpenAI API Compatible + SSE Streaming)

```bash
# Запуск асинхронного FastAPI сервера
python server_fastapi.py
```
*Сервер поднимется на `http://localhost:8000`. Эндпоинт проверки здоровья: `http://localhost:8000/health`.*

### 4. Проверка потокового SSE-стриминга

В отдельном терминале запустите клиент:
```bash
python test_client.py
```

---

## 🚫 Опровергнутые Гипотезы (Strict Disproved Paths)

1. **❌ Чистый PRNG шум / SVD / 2D DCT синтез 99% весов:** Разрушают векторное пространство (`lifylify...`).
2. **❌ 3-Tier с `torch.bool` масками:** Динамические маски раздувают VRAM на +340 МБ и замедляют инференс до 2.6 т/с.
3. **❌ Cross-Layer Background Sharing:** Усреднение фона ломает индивидуальные повороты слоев (`the the...`).
4. **❌ Безобучаемый SVD-синтез весов $\Delta W$ из KV-кэша:** Требует 10 минут вычислений на CPU и приводит к коллапсу логитов (`!!!!!!`).
5. **❌ Weight-Only отбор аномалий без $|X|$:** Приводит к зацикливанию текста (`korotak korotak...`).

---

## 🗺️ Выполненная Дорожная Карта (Roadmap)

- [x] Разработка Fused Triton INT4 GEMM ядра для деквантования в регистрах GPU SRAM.
- [x] Перевод таблицы слов `embed_tokens` в INT8.
- [x] Единый глобальный трафарет перестановок `global_perm_idx`.
- [x] Квантование слоя `lm_head` в Outlier-Aware INT4.
- [x] Query-Guided Sparse Offloading с предиктивным отбором контекста за 0.07 сек.
- [x] Масштабирование движка на модели **Qwen2.5-7B** и **Llama-3.1-8B** (**4.20 ГБ VRAM**).
- [x] Продуктовый REST API сервер (FastAPI) с поддержкой OpenAI API и SSE-стриминга.

---

## 📜 Лицензия

Проект распространяется под свободной лицензией **MIT**. Подробности в файле [LICENSE](LICENSE).