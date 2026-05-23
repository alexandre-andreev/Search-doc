"""
Pre-MVP: семантический поиск по саммари каталога с прогоном eval-набора.

Что делает:
1. Читает data/catalog.xlsx — все книги с саммари и метаданными.
2. Для каждой книги строит 1-2 псевдочанка:
   - title_chunk: "<название>. <автор>"
   - summary_chunk: AI-саммари (если есть)
3. Считает embedding'и через intfloat/multilingual-e5-small на GPU.
4. Хранит всё в NumPy-массиве в памяти (без SQLite).
5. Кэширует embedding'и в .npy файл — повторные запуски мгновенны.
6. Загружает eval/eval_queries.yaml и прогоняет все запросы.
7. Считает метрики: Recall@5, Recall@10, MRR.
8. Выводит отчёт с конкретными "провалами" — запросы, где ожидаемые книги
   не нашлись.

Дополнительно: режим --repl для интерактивного исследования выдачи.

Использование:
    # Первый запуск (займёт ~30 сек: загрузка модели + индексация саммари)
    python pre_mvp_eval.py

    # Перепрогон eval (мгновенно — embedding'и уже в кэше)
    python pre_mvp_eval.py

    # Принудительная переиндексация (например, если поменялся catalog.xlsx)
    python pre_mvp_eval.py --reindex

    # Интерактивный режим — вводи свои запросы и смотри выдачу
    python pre_mvp_eval.py --repl

    # Изменить расположение файлов
    python pre_mvp_eval.py --catalog data/catalog.xlsx --eval eval/eval_queries.yaml
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "intfloat/multilingual-e5-small"
DEVICE = "cuda"
BATCH_SIZE = 32

DEFAULT_CATALOG = "data/catalog.xlsx"
DEFAULT_EVAL = "eval/eval_queries.yaml"
CACHE_DIR = Path("cache")
CACHE_EMBEDDINGS = CACHE_DIR / "pre_mvp_embeddings.npy"
CACHE_META = CACHE_DIR / "pre_mvp_meta.npz"     # хранит чанк-метаданные

# Веса разных типов чанков при ранжировании
KIND_BONUS = {
    "title": 1.5,
    "summary": 1.3,
}


# ─────────────────────────────────────────────────────────────────────
# Безопасные парсеры — каталог "живой", в нём встречаются любые форматы
# ─────────────────────────────────────────────────────────────────────

def safe_int(value, field_name: str, row_idx: int, warnings: list[str]) -> int | None:
    """
    Пытается распарсить int. Принимает int, float, строку с числом,
    строку с диапазоном вида '2016-2017' (берёт первое число).
    Все остальные случаи → None с предупреждением.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value) if not np.isnan(value) else None
    s = str(value).strip()
    if not s:
        return None
    # Простой int
    try:
        return int(s)
    except ValueError:
        pass
    # Диапазон вида "2016-2017" или "2016/2017" или "2016, 2017"
    import re
    match = re.search(r"\d{4}", s)
    if match:
        return int(match.group())
    warnings.append(f"  ⚠ строка {row_idx}: поле '{field_name}' = {value!r} не распознано как число")
    return None


def safe_str(value) -> str | None:
    """Чистит строку, возвращает None если пусто."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


# ─────────────────────────────────────────────────────────────────────
# Чтение каталога
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CatalogBook:
    """Одна книга из catalog.xlsx."""
    catalog_id: int
    title: str
    author: str | None
    year: int | None
    category: str | None
    section: str | None
    subsection: str | None
    filename: str
    folder: str
    summary: str | None


@dataclass
class IndexChunk:
    """Один индексируемый псевдочанк."""
    text: str
    book_idx: int        # индекс в списке books
    kind: str            # 'title' | 'summary'


def load_catalog(path: Path) -> list[CatalogBook]:
    """Загружает книги из catalog.xlsx."""
    if not path.exists():
        print(f"  ✗ Каталог не найден: {path}", file=sys.stderr)
        print(f"    Скопируйте catalog.xlsx в {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(path, sheet_name="Каталог")

    # В разных версиях каталога названия колонок могут чуть разниться.
    # Проверим, что нужные есть.
    required = ["№", "Название", "Имя файла", "Папка"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ✗ В каталоге отсутствуют обязательные колонки: {missing}",
              file=sys.stderr)
        sys.exit(1)

    books = []
    warnings: list[str] = []
    skipped = 0

    for row_idx, row in df.iterrows():
        # Обязательные поля. Без них книгу не индексируем.
        catalog_id = safe_int(row.get("№"), "№", row_idx, warnings)
        title = safe_str(row.get("Название"))
        filename = safe_str(row.get("Имя файла"))

        if catalog_id is None or not title or not filename:
            skipped += 1
            continue

        # Парсим раздел/подраздел из колонки "Категория" (формата "Раздел/Подраздел")
        category = safe_str(row.get("Категория"))
        section = subsection = None
        if category and "/" in category:
            section, subsection = category.split("/", 1)
        elif category:
            section = category

        books.append(CatalogBook(
            catalog_id=catalog_id,
            title=title,
            author=safe_str(row.get("Автор")),
            year=safe_int(row.get("Год"), "Год", row_idx, warnings),
            category=category,
            section=section,
            subsection=subsection,
            filename=filename,
            folder=safe_str(row.get("Папка")) or "",
            summary=safe_str(row.get("Саммари")),
        ))

    # Печатаем предупреждения — но не больше 10, чтобы не засорять вывод
    if warnings:
        print(f"  ⚠ Предупреждений при разборе: {len(warnings)}")
        for w in warnings[:10]:
            print(w)
        if len(warnings) > 10:
            print(f"  ⚠ ... и ещё {len(warnings) - 10} (показаны первые 10)")
    if skipped:
        print(f"  ⚠ Пропущено строк без обязательных полей (id/название/имя файла): {skipped}")

    return books


def build_chunks(books: list[CatalogBook]) -> list[IndexChunk]:
    """Из каждой книги делаем 1-2 псевдочанка для индексации."""
    chunks = []
    for idx, book in enumerate(books):
        # 1. Title-чанк (всегда)
        author_part = f". {book.author}" if book.author else ""
        title_text = f"{book.title}{author_part}"
        chunks.append(IndexChunk(text=title_text, book_idx=idx, kind="title"))

        # 2. Summary-чанк (если есть и не слишком короткий)
        if book.summary and len(book.summary) >= 100:
            # Обрезаем очень длинные саммари до ~2000 симв — e5-small не умеет в длинный контекст
            text = book.summary[:2000]
            chunks.append(IndexChunk(text=text, book_idx=idx, kind="summary"))

    return chunks


# ─────────────────────────────────────────────────────────────────────
# Индексация: считаем embedding'и и кэшируем
# ─────────────────────────────────────────────────────────────────────

def build_or_load_index(
    chunks: list[IndexChunk],
    force_reindex: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает:
      embeddings: shape=(N_chunks, 384), float32, L2-normalized
      book_idx_per_chunk: shape=(N_chunks,) — индекс книги для каждого чанка
      kind_per_chunk:     shape=(N_chunks,) — 0 для title, 1 для summary
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Пробуем загрузить из кэша
    if not force_reindex and CACHE_EMBEDDINGS.exists() and CACHE_META.exists():
        embeddings = np.load(CACHE_EMBEDDINGS)
        meta = np.load(CACHE_META)
        book_idx = meta["book_idx"]
        kind = meta["kind"]
        if embeddings.shape[0] == len(chunks):
            print(f"  ✓ Загружено из кэша: {embeddings.shape[0]} embeddings")
            return embeddings, book_idx, kind
        print(f"  ⚠ Кэш не совпадает по размеру ({embeddings.shape[0]} vs {len(chunks)}), пересчитываем")

    # Считаем с нуля
    print(f"  Загружаем модель {MODEL_NAME} на {DEVICE}...", end=" ", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"готово за {time.time()-t0:.1f} сек")

    # e5 требует префикс "passage: " для индексируемых текстов
    texts = [f"passage: {c.text}" for c in chunks]

    print(f"  Считаем embedding'и для {len(chunks)} чанков...", end=" ", flush=True)
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,    # критично для e5: нормализация → cosine = dot product
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.time() - t0
    print(f"готово за {elapsed:.1f} сек ({len(chunks)/elapsed:.0f} чанков/сек)")

    book_idx = np.array([c.book_idx for c in chunks], dtype=np.int32)
    kind = np.array([0 if c.kind == "title" else 1 for c in chunks], dtype=np.int8)

    # Кэшируем
    np.save(CACHE_EMBEDDINGS, embeddings)
    np.savez(CACHE_META, book_idx=book_idx, kind=kind)
    print(f"  ✓ Кэш сохранён: {CACHE_EMBEDDINGS}")

    return embeddings, book_idx, kind


# ─────────────────────────────────────────────────────────────────────
# Поиск
# ─────────────────────────────────────────────────────────────────────

def search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    book_idx_per_chunk: np.ndarray,
    kind_per_chunk: np.ndarray,
    books: list[CatalogBook],
    top_k: int = 10,
) -> list[dict]:
    """
    Семантический поиск:
    1. Считаем embedding запроса (с префиксом "query: ").
    2. Cosine similarity со всеми чанками (one matrix multiply).
    3. Применяем бонусы за тип чанка (title=1.5, summary=1.3).
    4. Агрегируем по книгам: book_score = max(chunk_scores) + log(1 + N) * 0.1
    5. Возвращаем top_k книг.
    """
    # Embedding запроса
    q_text = f"query: {query}"
    q_vec = model.encode(
        [q_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    # Cosine similarity (нормализованные → просто dot product)
    scores = embeddings @ q_vec       # shape: (N_chunks,)

    # Бонусы за тип чанка
    bonus = np.where(kind_per_chunk == 0, KIND_BONUS["title"], KIND_BONUS["summary"])
    scores = scores * bonus

    # Агрегация по книгам
    # Для каждой книги: max score её чанков + log(1+N_matched) * 0.1
    # Считаем кандидатов: берём top-50 чанков, агрегируем
    n_candidates = min(50, len(scores))
    top_chunk_indices = np.argpartition(-scores, n_candidates - 1)[:n_candidates]

    # Группируем по book_idx
    book_scores: dict[int, list[tuple[float, str]]] = {}  # book_idx -> [(score, kind), ...]
    for ci in top_chunk_indices:
        bi = int(book_idx_per_chunk[ci])
        kind = "title" if kind_per_chunk[ci] == 0 else "summary"
        book_scores.setdefault(bi, []).append((float(scores[ci]), kind))

    # Финальный score книги
    results = []
    for bi, chunk_scores in book_scores.items():
        max_score = max(s for s, _ in chunk_scores)
        n_matched = len(chunk_scores)
        final_score = max_score + np.log(1 + n_matched) * 0.1
        kinds_matched = sorted(set(k for _, k in chunk_scores))

        book = books[bi]
        results.append({
            "score": float(final_score),
            "book_idx": bi,
            "catalog_id": book.catalog_id,
            "title": book.title,
            "author": book.author,
            "year": book.year,
            "category": book.category,
            "filename": book.filename,
            "matched_in": kinds_matched,
        })

    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


# ─────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────

@dataclass
class QueryEvalResult:
    id: str
    query: str
    notes: str
    expected_top_5: list[str]
    expected_top_10: list[str]
    found_top_5: list[str]
    found_top_10: list[str]
    missed_top_5: list[str]
    missed_top_10: list[str]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    results: list[dict]


def match_expectation(expected_substring: str, book_titles_and_filenames: list[str]) -> bool:
    """Проверяет, найдена ли ожидаемая книга. Совпадение — substring без регистра."""
    needle = expected_substring.lower()
    return any(needle in s.lower() for s in book_titles_and_filenames)


def find_position_of_expectation(expected_substring: str, results: list[dict]) -> int | None:
    """Возвращает 1-based позицию первого результата, удовлетворяющего ожиданию. None если нет."""
    needle = expected_substring.lower()
    for i, r in enumerate(results, start=1):
        if needle in (r["title"] or "").lower() or needle in (r["filename"] or "").lower():
            return i
    return None


def evaluate_query(query_def: dict, results: list[dict]) -> QueryEvalResult:
    """
    Считает метрики для одного запроса.

    Поддерживает два формата:
    v1: expected_in_top_5 / expected_in_top_10 — строгие списки.
    v2: relevant_books + min_in_top_5 / min_in_top_10 — "найти N из M возможных".
    Определяется по наличию ключа 'relevant_books'.
    """
    if "relevant_books" in query_def:
        return _evaluate_v2(query_def, results)
    return _evaluate_v1(query_def, results)


def _evaluate_v2(query_def: dict, results: list[dict]) -> QueryEvalResult:
    """
    v2: для широкого запроса достаточно найти min_in_top_K из relevant_books.

    Recall@K = min(found_in_top_K, min_in_top_K) / min_in_top_K
    То есть нашли требуемое число — Recall=1.0; нашли половину — 0.5.
    """
    relevant = query_def.get("relevant_books", []) or []
    min_5 = int(query_def.get("min_in_top_5", 1))
    min_10 = int(query_def.get("min_in_top_10", 2))

    titles_top_5 = [f"{r['title']}|{r['filename']}" for r in results[:5]]
    titles_top_10 = [f"{r['title']}|{r['filename']}" for r in results[:10]]

    found_5 = [b for b in relevant if match_expectation(b, titles_top_5)]
    found_10 = [b for b in relevant if match_expectation(b, titles_top_10)]
    missed_5 = [b for b in relevant if b not in found_5]
    missed_10 = [b for b in relevant if b not in found_10]

    recall_5 = min(len(found_5), min_5) / min_5 if min_5 > 0 else 1.0
    recall_10 = min(len(found_10), min_10) / min_10 if min_10 > 0 else 1.0

    # MRR: позиция первого попадания любой relevant_book
    positions = []
    for b in relevant:
        pos = find_position_of_expectation(b, results)
        if pos is not None:
            positions.append(pos)
    mrr = 1.0 / min(positions) if positions else 0.0

    return QueryEvalResult(
        id=query_def["id"],
        query=query_def["query"],
        notes=query_def.get("notes", "") or "",
        expected_top_5=relevant,    # просто чтобы было что показать в отчёте
        expected_top_10=[f"min {min_5}/{min_10}"],   # маркер
        found_top_5=found_5,
        found_top_10=found_10,
        missed_top_5=missed_5,
        missed_top_10=missed_10,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        mrr=mrr,
        results=results,
    )


def _evaluate_v1(query_def: dict, results: list[dict]) -> QueryEvalResult:
    """v1 формат (старый): expected_in_top_5 / expected_in_top_10."""
    expected_5 = query_def.get("expected_in_top_5", []) or []
    expected_10 = query_def.get("expected_in_top_10", []) or []

    titles_filenames_top_5 = [
        f"{r['title']}|{r['filename']}" for r in results[:5]
    ]
    titles_filenames_top_10 = [
        f"{r['title']}|{r['filename']}" for r in results[:10]
    ]

    found_5 = [e for e in expected_5 if match_expectation(e, titles_filenames_top_5)]
    missed_5 = [e for e in expected_5 if e not in found_5]

    all_expected_10 = list(set(expected_5) | set(expected_10))
    found_10 = [e for e in all_expected_10 if match_expectation(e, titles_filenames_top_10)]
    missed_10 = [e for e in all_expected_10 if e not in found_10]

    recall_5 = len(found_5) / len(expected_5) if expected_5 else 1.0
    recall_10 = len(found_10) / len(all_expected_10) if all_expected_10 else 1.0

    positions = []
    for e in expected_5 + expected_10:
        pos = find_position_of_expectation(e, results)
        if pos is not None:
            positions.append(pos)
    mrr = 1.0 / min(positions) if positions else 0.0

    return QueryEvalResult(
        id=query_def["id"],
        query=query_def["query"],
        notes=query_def.get("notes", "") or "",
        expected_top_5=expected_5,
        expected_top_10=expected_10,
        found_top_5=found_5,
        found_top_10=found_10,
        missed_top_5=missed_5,
        missed_top_10=missed_10,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        mrr=mrr,
        results=results,
    )


def print_eval_report(eval_results: list[QueryEvalResult], verbose: bool = False):
    """Печатает читаемый отчёт по eval."""
    print()
    print("=" * 78)
    print("  EVAL ОТЧЁТ")
    print("=" * 78)

    # По каждому запросу
    print(f"\n  {'ID':<30} {'R@5':>6} {'R@10':>6} {'MRR':>6}   Статус")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6}   {'-'*15}")

    for r in eval_results:
        status = "✓" if r.recall_at_10 >= 0.5 else ("△" if r.recall_at_10 > 0 else "✗")
        print(f"  {r.id:<30} {r.recall_at_5:>6.2f} {r.recall_at_10:>6.2f} "
              f"{r.mrr:>6.2f}   {status}")

    # Сводка
    n = len(eval_results)
    avg_r5 = np.mean([r.recall_at_5 for r in eval_results])
    avg_r10 = np.mean([r.recall_at_10 for r in eval_results])
    avg_mrr = np.mean([r.mrr for r in eval_results])
    perfect = sum(1 for r in eval_results if r.recall_at_10 >= 1.0)
    good = sum(1 for r in eval_results if r.recall_at_10 >= 0.5)
    bad = sum(1 for r in eval_results if r.recall_at_10 == 0)

    print()
    print("  СВОДКА:")
    print(f"    Запросов:        {n}")
    print(f"    Средний R@5:     {avg_r5:.3f}")
    print(f"    Средний R@10:    {avg_r10:.3f}")
    print(f"    Средний MRR:     {avg_mrr:.3f}")
    print(f"    Идеальных (R@10=1.0):    {perfect}/{n}")
    print(f"    Удовлетворительных (R@10≥0.5): {good}/{n}")
    print(f"    Полностью провалившихся: {bad}/{n}")

    # Критерии MVP
    print()
    print("  КРИТЕРИИ MVP:")
    crit_r10 = "✓" if avg_r10 >= 0.75 else "✗"
    crit_mrr = "✓" if avg_mrr >= 0.40 else "✗"
    print(f"    Recall@10 ≥ 0.75:  {avg_r10:.3f}  {crit_r10}")
    print(f"    MRR ≥ 0.40:        {avg_mrr:.3f}  {crit_mrr}")

    if avg_r10 >= 0.75 and avg_mrr >= 0.40:
        print()
        print("  → Гипотеза подтверждена: саммари + названия достаточны для MVP-качества.")
        print("    Body-индексация PDF/EPUB может быть отложена или сделана позже.")
    elif avg_r10 >= 0.5:
        print()
        print("  → Промежуточный результат: для большинства запросов работает,")
        print("    но body-индексация улучшит recall для сложных кейсов.")
    else:
        print()
        print("  → Результат слабый: либо eval-набор плохой, либо нужен body-extractor,")
        print("    либо нужна более сильная модель (e5-base).")

    # Детали по провалившимся запросам
    failed = [r for r in eval_results if r.recall_at_10 < 0.5]
    if failed:
        print()
        print("  " + "─" * 76)
        print("  ДЕТАЛИ ПРОВАЛИВШИХСЯ ЗАПРОСОВ (R@10 < 0.5):")
        print("  " + "─" * 76)
        for r in failed:
            print(f"\n  ▼ [{r.id}]  query: \"{r.query}\"")
            print(f"      R@5={r.recall_at_5:.2f}  R@10={r.recall_at_10:.2f}  MRR={r.mrr:.2f}")
            if r.missed_top_10:
                print(f"      НЕ НАЙДЕНО (ожидалось):")
                for m in r.missed_top_10:
                    print(f"         · {m}")
            print(f"      ТОП-5 ВЫДАЧИ:")
            for i, res in enumerate(r.results[:5], 1):
                marker = ""
                for e in r.expected_top_5 + r.expected_top_10:
                    if e.lower() in (res["title"] or "").lower() or \
                       e.lower() in (res["filename"] or "").lower():
                        marker = " ← ожидалось"
                        break
                print(f"         {i}. [{res['score']:.3f}] {res['title'][:60]}{marker}")

    if verbose:
        # Все запросы, не только провалившиеся
        print()
        print("  " + "─" * 76)
        print("  ВСЕ ЗАПРОСЫ — ТОП-5 ВЫДАЧИ:")
        print("  " + "─" * 76)
        for r in eval_results:
            print(f"\n  [{r.id}]  \"{r.query}\"")
            for i, res in enumerate(r.results[:5], 1):
                print(f"    {i}. [{res['score']:.3f}] {res['title'][:60]}")


# ─────────────────────────────────────────────────────────────────────
# REPL — интерактивный режим
# ─────────────────────────────────────────────────────────────────────

def repl(model, embeddings, book_idx, kind, books):
    """Интерактивный поиск."""
    print()
    print("=" * 60)
    print("  REPL — интерактивный поиск")
    print("  Введите запрос или 'exit' для выхода.")
    print("=" * 60)
    while True:
        try:
            query = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("exit", "quit", "q"):
            break

        t0 = time.time()
        results = search(query, model, embeddings, book_idx, kind, books, top_k=10)
        elapsed = (time.time() - t0) * 1000

        print(f"\n  Найдено топ-{len(results)} за {elapsed:.0f} мс:\n")
        for i, r in enumerate(results, 1):
            year = f", {r['year']}" if r["year"] else ""
            author = f" — {r['author']}" if r["author"] else ""
            cat = f"  [{r['category']}]" if r["category"] else ""
            print(f"  {i:>2}. [{r['score']:.3f}]  {r['title']}{author}{year}{cat}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, type=Path)
    parser.add_argument("--eval", default=DEFAULT_EVAL, type=Path,
                        dest="eval_path")
    parser.add_argument("--reindex", action="store_true",
                        help="Принудительно пересчитать embedding'и")
    parser.add_argument("--repl", action="store_true",
                        help="После eval запустить интерактивный режим")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Показать топ-5 выдачи по ВСЕМ запросам, не только провалившимся")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Только индексация + REPL, без прогона eval")
    args = parser.parse_args()

    print("=" * 78)
    print("  PRE-MVP: семантический поиск по саммари каталога")
    print("=" * 78)

    # 1. Каталог
    print(f"\n--- Загрузка каталога ---")
    print(f"  Файл: {args.catalog}")
    books = load_catalog(args.catalog)
    n_with_summary = sum(1 for b in books if b.summary)
    n_with_author = sum(1 for b in books if b.author)
    print(f"  Книг загружено: {len(books)}")
    print(f"  Из них с саммари: {n_with_summary} ({100*n_with_summary/len(books):.1f}%)")
    print(f"  Из них с автором: {n_with_author} ({100*n_with_author/len(books):.1f}%)")

    # 2. Чанки
    print(f"\n--- Построение чанков ---")
    chunks = build_chunks(books)
    n_title = sum(1 for c in chunks if c.kind == "title")
    n_summary = sum(1 for c in chunks if c.kind == "summary")
    print(f"  Всего чанков: {len(chunks)}")
    print(f"    title-чанков:   {n_title}")
    print(f"    summary-чанков: {n_summary}")

    # 3. Embeddings
    print(f"\n--- Индексация (embedding'и) ---")
    embeddings, book_idx, kind = build_or_load_index(chunks, force_reindex=args.reindex)
    print(f"  Размер индекса: {embeddings.nbytes / 1024 / 1024:.1f} MB в памяти")

    # 4. Модель для запросов
    # (если уже загружена при индексации, используем её; иначе загружаем)
    print(f"\n--- Подготовка модели для запросов ---")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"  Готово за {time.time()-t0:.1f} сек")

    # 5. Eval
    if not args.skip_eval:
        print(f"\n--- Прогон eval ---")
        print(f"  Файл: {args.eval_path}")
        if not args.eval_path.exists():
            print(f"  ✗ Eval-файл не найден: {args.eval_path}", file=sys.stderr)
            sys.exit(1)
        with open(args.eval_path, encoding="utf-8") as f:
            eval_data = yaml.safe_load(f)

        queries_defs = eval_data.get("queries", [])
        print(f"  Запросов в наборе: {len(queries_defs)}")

        eval_results = []
        for qd in queries_defs:
            results = search(qd["query"], model, embeddings, book_idx, kind, books, top_k=10)
            er = evaluate_query(qd, results)
            eval_results.append(er)

        print_eval_report(eval_results, verbose=args.verbose)

    # 6. REPL
    if args.repl:
        repl(model, embeddings, book_idx, kind, books)

    print()
    print("=" * 78)
    print("  Готово.")
    print("=" * 78)


if __name__ == "__main__":
    main()
