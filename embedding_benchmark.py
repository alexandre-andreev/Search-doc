"""
Диагностический скрипт для замера скорости embedding-моделей.
Цель: понять, какая модель оптимальна на вашей GPU для индексации ~3500 книг.

Тестирует:
  - intfloat/multilingual-e5-small  (118M, быстрая, baseline)
  - intfloat/multilingual-e5-base   (280M, разумный компромисс)
  - intfloat/multilingual-e5-large  (560M, лучшее качество, может не влезть)

Что меряет:
  - время загрузки модели
  - скорость на CPU (контрольный замер)
  - скорость на GPU при разных размерах батча
  - пиковое потребление VRAM
  - проекцию: сколько часов займёт индексация 1.75 млн чанков

Использование:
  python embedding_benchmark.py
  python embedding_benchmark.py --models small base       # пропустить large
  python embedding_benchmark.py --skip-cpu                # пропустить замер на CPU
  python embedding_benchmark.py --chunks 500              # размер тестового набора
"""

import argparse
import gc
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# --- Реалистичные тестовые чанки (смесь русского и английского, ~70/30) ---
SAMPLE_CHUNKS_RU = [
    "Модульное тестирование — это процесс проверки отдельных компонентов программы в изоляции от остальной системы. "
    "Главная цель — убедиться, что каждая единица кода работает корректно сама по себе. "
    "В Python для этого обычно используют pytest или unittest, в Java — JUnit, в C# — NUnit или xUnit.",

    "Принципы SOLID были сформулированы Робертом Мартином и стали фундаментом объектно-ориентированного проектирования. "
    "Single Responsibility Principle гласит, что каждый класс должен иметь только одну причину для изменения. "
    "Open/Closed Principle требует, чтобы классы были открыты для расширения, но закрыты для модификации.",

    "Чистая архитектура предполагает разделение приложения на слои: сущности, варианты использования, "
    "адаптеры интерфейсов и фреймворки с драйверами. Зависимости направлены строго внутрь — внешние слои "
    "знают о внутренних, но не наоборот. Это даёт независимость от UI, базы данных и внешних сервисов.",

    "Реактивное программирование основано на потоках данных и распространении изменений. "
    "Когда значение в источнике меняется, все зависимые вычисления автоматически пересчитываются. "
    "Библиотеки вроде RxJS, Reactor или RxJava предоставляют операторы для трансформации этих потоков.",

    "Криптографические хеш-функции должны обладать свойствами устойчивости к коллизиям и необратимости. "
    "SHA-256, используемая в Bitcoin, выдаёт 256-битный результат и считается криптографически стойкой. "
    "MD5 и SHA-1 сегодня считаются устаревшими и не должны применяться в системах безопасности.",

    "Контейнеризация с помощью Docker позволяет упаковать приложение со всеми его зависимостями в один образ. "
    "Это решает классическую проблему 'работает на моей машине' и упрощает развёртывание. "
    "Kubernetes идёт дальше и оркестрирует множество контейнеров, обеспечивая масштабирование и отказоустойчивость.",

    "Машинное обучение делится на обучение с учителем, без учителя и с подкреплением. "
    "В задачах с учителем модель обучается на размеченных данных, где для каждого входа известен правильный ответ. "
    "Кластеризация и понижение размерности — типичные задачи обучения без учителя.",
]

SAMPLE_CHUNKS_EN = [
    "Test-driven development is a software development process where tests are written before the actual code. "
    "The cycle consists of three steps: write a failing test, write minimal code to pass it, then refactor. "
    "This approach forces developers to think about the interface and requirements before implementation.",

    "Database indexing dramatically improves query performance by creating data structures that allow fast lookups. "
    "B-tree indexes are the most common and work well for equality and range queries on ordered data. "
    "Hash indexes excel at equality lookups but cannot handle range queries efficiently.",

    "Microservices architecture decomposes applications into small, independently deployable services. "
    "Each service owns its data and communicates with others through well-defined APIs, usually REST or gRPC. "
    "This style enables independent scaling and technology choices but introduces complexity in distributed systems.",
]


def make_chunks(n: int) -> list[str]:
    """Собирает n тестовых чанков из заготовок, перемешивая русский и английский 70/30."""
    chunks = []
    ru_pool = SAMPLE_CHUNKS_RU
    en_pool = SAMPLE_CHUNKS_EN
    for i in range(n):
        if i % 10 < 7:
            chunks.append(ru_pool[i % len(ru_pool)])
        else:
            chunks.append(en_pool[i % len(en_pool)])
    return chunks


# --- Конфигурация моделей ---
@dataclass
class ModelConfig:
    name: str
    repo: str
    params_millions: int
    expected_vram_fp32_gb: float
    note: str


MODELS = {
    "small": ModelConfig(
        name="e5-small",
        repo="intfloat/multilingual-e5-small",
        params_millions=118,
        expected_vram_fp32_gb=0.5,
        note="Самая быстрая, для baseline. Качество ниже, но на 1650Ti реалистична.",
    ),
    "base": ModelConfig(
        name="e5-base",
        repo="intfloat/multilingual-e5-base",
        params_millions=280,
        expected_vram_fp32_gb=1.1,
        note="Хороший компромисс скорость/качество. Скорее всего оптимальный выбор.",
    ),
    "large": ModelConfig(
        name="e5-large",
        repo="intfloat/multilingual-e5-large",
        params_millions=560,
        expected_vram_fp32_gb=2.2,
        note="Лучшее качество. На 4GB VRAM впритык — может потребоваться маленький батч.",
    ),
}


@dataclass
class BenchResult:
    model: str
    device: str
    batch_size: int
    chunks_per_sec: float
    seconds_total: float
    peak_vram_mb: Optional[float] = None
    ok: bool = True
    error: str = ""


# --- Утилиты ---
def print_header(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    print()
    print(f"--- {text} ---")


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} сек"
    if seconds < 3600:
        return f"{seconds/60:.1f} мин"
    return f"{seconds/3600:.2f} ч"


def get_system_info() -> dict:
    """Собирает информацию о системе."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 1)
        info["ram_available_gb"] = round(vm.available / (1024**3), 1)
    except ImportError:
        info["ram_total_gb"] = "psutil не установлен"
    return info


def check_torch_and_cuda() -> tuple[bool, dict]:
    """Проверяет установку PyTorch и доступность CUDA. Это критичный шаг."""
    try:
        import torch
    except ImportError:
        return False, {
            "error": "PyTorch не установлен.",
            "fix": "pip install torch --index-url https://download.pytorch.org/whl/cu121",
        }

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    else:
        info["error"] = "CUDA недоступна. PyTorch установлен в CPU-варианте."
        info["fix"] = (
            "Переустановите PyTorch с CUDA-сборкой:\n"
            "  pip uninstall torch -y\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    return True, info


def check_sentence_transformers() -> tuple[bool, str]:
    try:
        import sentence_transformers
        return True, sentence_transformers.__version__
    except ImportError:
        return False, "pip install sentence-transformers"


# --- Основной бенчмарк ---
def benchmark_model(
    model_key: str,
    chunks: list[str],
    device: str,
    batch_sizes: list[int],
    warmup_chunks: int = 16,
) -> list[BenchResult]:
    """
    Загружает модель и гоняет батчи разных размеров.
    Возвращает результаты для каждого batch_size.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    cfg = MODELS[model_key]
    results: list[BenchResult] = []

    print_section(f"Модель: {cfg.repo}  ({cfg.params_millions}M параметров)")
    print(f"  Ожидаемый VRAM в FP32: ~{cfg.expected_vram_fp32_gb} GB")
    print(f"  Заметка: {cfg.note}")

    # Загрузка
    print(f"  Загружаем модель на {device}...", end=" ", flush=True)
    t0 = time.time()
    try:
        model = SentenceTransformer(cfg.repo, device=device)
    except Exception as e:
        print("ОШИБКА")
        print(f"    {e}")
        return [BenchResult(
            model=cfg.name, device=device, batch_size=0,
            chunks_per_sec=0, seconds_total=0, ok=False, error=str(e),
        )]
    load_time = time.time() - t0
    print(f"готово за {load_time:.1f} сек")

    # Прогрев (первый прогон всегда медленнее: компиляция кернелов, выделение памяти)
    print(f"  Прогрев ({warmup_chunks} чанков)...", end=" ", flush=True)
    try:
        model.encode(chunks[:warmup_chunks], batch_size=8, show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        print("ок")
    except Exception as e:
        print(f"ОШИБКА: {e}")
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        return [BenchResult(
            model=cfg.name, device=device, batch_size=0,
            chunks_per_sec=0, seconds_total=0, ok=False, error=str(e),
        )]

    # Замеры по разным размерам батча
    for bs in batch_sizes:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        print(f"  batch_size={bs:>3}:", end=" ", flush=True)
        try:
            t0 = time.time()
            model.encode(chunks, batch_size=bs, show_progress_bar=False)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            cps = len(chunks) / elapsed

            peak_vram = None
            if device == "cuda":
                peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

            vram_str = f", peak VRAM {peak_vram:.0f} MB" if peak_vram else ""
            print(f"{cps:>6.1f} чанков/сек  ({elapsed:.1f} сек на {len(chunks)} чанков{vram_str})")

            results.append(BenchResult(
                model=cfg.name, device=device, batch_size=bs,
                chunks_per_sec=cps, seconds_total=elapsed, peak_vram_mb=peak_vram,
            ))
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM (не хватило VRAM)")
            results.append(BenchResult(
                model=cfg.name, device=device, batch_size=bs,
                chunks_per_sec=0, seconds_total=0, ok=False,
                error="CUDA out of memory",
            ))
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"ОШИБКА: {e}")
            results.append(BenchResult(
                model=cfg.name, device=device, batch_size=bs,
                chunks_per_sec=0, seconds_total=0, ok=False, error=str(e),
            ))

    # Освобождаем ресурсы перед следующей моделью
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def print_recommendations(all_results: list[BenchResult], total_chunks_estimate: int):
    """По результатам выдаёт рекомендацию: какую модель брать и какой батч."""
    print_header("РЕКОМЕНДАЦИИ")

    # Фильтруем только успешные GPU-замеры
    gpu_ok = [r for r in all_results if r.device == "cuda" and r.ok]

    if not gpu_ok:
        print("  Нет успешных замеров на GPU.")
        print("  Проверьте установку CUDA-сборки PyTorch (см. выше).")
        return

    # Для каждой модели выбираем самый быстрый успешный батч
    by_model: dict[str, BenchResult] = {}
    for r in gpu_ok:
        if r.model not in by_model or r.chunks_per_sec > by_model[r.model].chunks_per_sec:
            by_model[r.model] = r

    print(f"\n  Лучший результат для каждой модели на GPU:\n")
    print(f"  {'Модель':<12} {'Батч':>5} {'Чанков/сек':>12} {'VRAM peak':>11}   Индексация 1.75M чанков")
    print(f"  {'-'*12} {'-'*5} {'-'*12} {'-'*11}   {'-'*25}")
    for model_name, r in by_model.items():
        eta = total_chunks_estimate / r.chunks_per_sec
        vram = f"{r.peak_vram_mb:.0f} MB" if r.peak_vram_mb else "—"
        print(f"  {model_name:<12} {r.batch_size:>5} {r.chunks_per_sec:>12.1f} {vram:>11}   {fmt_time(eta)}")

    # Финальная рекомендация
    print()
    if "e5-base" in by_model:
        base = by_model["e5-base"]
        print(f"  ⇒ Для библиотеки 3500 книг рекомендуется e5-base с batch_size={base.batch_size}.")
        print(f"    Это разумный компромисс между качеством и скоростью на GTX 1650 Ti.")
        if "e5-large" in by_model:
            large = by_model["e5-large"]
            speedup = large.chunks_per_sec / base.chunks_per_sec if base.chunks_per_sec > 0 else 0
            print(f"    e5-large тоже работает, но в {1/speedup:.1f}x медленнее. "
                  f"Берите её, только если выдача e5-base окажется слабой.")
    elif "e5-small" in by_model:
        print(f"  ⇒ Используйте e5-small как baseline. Это самая лёгкая модель.")
        print(f"    Если качество поиска окажется недостаточным, попробуйте установить больше RAM/VRAM.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=["small", "base", "large"],
                        default=["small", "base", "large"],
                        help="Какие модели тестировать (по умолчанию: все три).")
    parser.add_argument("--chunks", type=int, default=500,
                        help="Размер тестового набора чанков (по умолчанию: 500).")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Пропустить контрольный замер на CPU (он медленный).")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Размеры батча для тестирования. По умолчанию: 8, 16, 32 для GPU.")
    args = parser.parse_args()

    print_header("ДИАГНОСТИКА EMBEDDING-СТЕКА")
    print(f"  Целевое железо: GTX 1650 Ti (4 GB VRAM)")
    print(f"  Целевая задача: индексация ~3500 книг ≈ 1.75M чанков")
    print(f"  Размер тестового набора: {args.chunks} чанков")

    # 1. Системная информация
    print_section("Система")
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k}: {v}")

    # 2. PyTorch и CUDA
    print_section("PyTorch / CUDA")
    torch_ok, torch_info = check_torch_and_cuda()
    if not torch_ok:
        print(f"  ✗ {torch_info['error']}")
        print(f"\n  Как исправить:\n    {torch_info['fix']}")
        sys.exit(1)
    for k, v in torch_info.items():
        marker = "✓" if k == "cuda_available" and v else ("✗" if k == "cuda_available" else " ")
        print(f"  {marker} {k}: {v}")

    cuda_ok = torch_info.get("cuda_available", False)
    if not cuda_ok:
        print(f"\n  ⚠ CUDA недоступна. Будет тестироваться только CPU — это будет МЕДЛЕННО.")
        print(f"    Как исправить:\n    {torch_info['fix']}")

    # 3. sentence-transformers
    print_section("sentence-transformers")
    st_ok, st_info = check_sentence_transformers()
    if not st_ok:
        print(f"  ✗ Не установлен.")
        print(f"    Установите: {st_info}")
        sys.exit(1)
    print(f"  ✓ версия: {st_info}")

    # 4. Готовим тестовые данные
    chunks = make_chunks(args.chunks)
    # e5 требует префикс "passage: " для документов (это часть протокола модели)
    chunks = [f"passage: {c}" for c in chunks]

    # 5. Бенчмарки
    all_results: list[BenchResult] = []
    batch_sizes_gpu = args.batch_sizes or [8, 16, 32]
    batch_sizes_cpu = [8]  # на CPU только маленький батч, иначе ждать вечно

    if cuda_ok:
        print_header("GPU БЕНЧМАРК")
        for model_key in args.models:
            results = benchmark_model(model_key, chunks, "cuda", batch_sizes_gpu)
            all_results.extend(results)

    if not args.skip_cpu:
        print_header("CPU БЕНЧМАРК (контрольный, для сравнения)")
        # На CPU гоняем только маленькую модель и маленький батч, чтобы не ждать час
        cpu_models = ["small"] if "small" in args.models else args.models[:1]
        cpu_chunks = chunks[:100]  # 100 хватит для оценки
        for model_key in cpu_models:
            print(f"\n  (на CPU гоним только {len(cpu_chunks)} чанков для экономии времени)")
            results = benchmark_model(model_key, cpu_chunks, "cpu", batch_sizes_cpu)
            all_results.extend(results)

    # 6. Итоги
    print_recommendations(all_results, total_chunks_estimate=1_750_000)

    print()
    print("=" * 70)
    print("  Готово. Сохраните вывод этого скрипта — он понадобится для дизайна индекса.")
    print("=" * 70)


if __name__ == "__main__":
    main()
