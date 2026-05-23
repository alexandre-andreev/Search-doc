"""Поиск похожих книг через embedding-сходство."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def find_related(
    conn: sqlite3.Connection,
    book_id: int,
    k: int = 10,
    exclude_book_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Находит k книг, семантически близких к book_id.

    Алгоритм:
    1. Берёт summary-эмбеддинг книги (fallback на title, если summary нет).
    2. Выполняет KNN через sqlite-vec, используя сохранённые байты напрямую.
    3. Маппит chunk_id → book_id, берёт max similarity на книгу.
    4. Исключает anchor-книгу, exclude_book_ids и их дубликаты.

    Возвращает [(book_id, similarity_score)], отсортированные по убыванию.
    """
    exclude = {book_id}
    if exclude_book_ids:
        exclude.update(exclude_book_ids)

    # Получаем байты эмбеддинга summary (или title как fallback)
    embedding_bytes = _get_book_embedding_bytes(conn, book_id)
    if embedding_bytes is None:
        return []

    # KNN: берём с запасом, чтобы после фильтрации осталось достаточно
    search_k = max(k * 6, 80)
    rows = conn.execute(
        """
        SELECT chunk_id, distance
        FROM chunk_vectors
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (embedding_bytes, search_k),
    ).fetchall()

    if not rows:
        return []

    chunk_ids = [r[0] for r in rows]
    # cosine_sim = 1 - L2² / 2 (для нормализованных векторов)
    score_map = {r[0]: 1.0 - (float(r[1]) ** 2) / 2.0 for r in rows}

    # Маппинг chunk_id → book_id + duplicate_of
    placeholders = ",".join("?" * len(chunk_ids))
    chunk_rows = conn.execute(
        f"""
        SELECT c.id, c.book_id, b.duplicate_of
        FROM chunks c
        JOIN books b ON b.id = c.book_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()

    book_best: dict[int, float] = {}
    for chunk_id, bid, dup_of in chunk_rows:
        # Пропускаем саму книгу, дубликаты и запрошенные исключения
        if bid in exclude:
            continue
        if dup_of is not None and dup_of in exclude:
            continue
        score = score_map.get(chunk_id, 0.0)
        if bid not in book_best or score > book_best[bid]:
            book_best[bid] = score

    result = sorted(book_best.items(), key=lambda x: -x[1])
    return result[:k]


def build_related_payload(
    conn: sqlite3.Connection,
    related_hits: list[tuple[int, float]],
    max_results: int = 5,
) -> list[dict]:
    """
    Строит список словарей для поля related_books в JSON-ответе.
    """
    if not related_hits:
        return []

    hits = related_hits[:max_results]
    book_ids = [bid for bid, _ in hits]
    score_by_bid = {bid: sim for bid, sim in hits}

    placeholders = ",".join("?" * len(book_ids))
    rows = conn.execute(
        f"""
        SELECT id, title, author, year, category, file_format, filename, folder, summary
        FROM books
        WHERE id IN ({placeholders})
        """,
        book_ids,
    ).fetchall()

    # Сохраняем порядок hits
    order = {bid: i for i, bid in enumerate(book_ids)}
    rows_sorted = sorted(rows, key=lambda r: order.get(r[0], 999))

    result = []
    for rank, row in enumerate(rows_sorted, start=1):
        bid, title, author, year, category, fmt, filename, folder, summary = row
        folder = folder or ""
        filename = filename or ""
        file_path = str(Path(folder) / filename) if folder and filename else filename or folder

        result.append({
            "rank": rank,
            "book_id": bid,
            "title": title,
            "author": author,
            "year": year,
            "category": category,
            "file_format": fmt,
            "file_path": file_path,
            "similarity_to_top_result": round(score_by_bid[bid], 3),
            "summary": summary,
        })

    return result


# ─── internal ────────────────────────────────────────────────────────────────

def _get_book_embedding_bytes(conn: sqlite3.Connection, book_id: int) -> bytes | None:
    """
    Возвращает raw float32 bytes эмбеддинга для книги.
    Предпочитает summary-чанк, fallback на title.
    """
    for kind in ("summary", "title"):
        row = conn.execute(
            """
            SELECT cv.embedding
            FROM chunks c
            JOIN chunk_vectors cv ON cv.chunk_id = c.id
            WHERE c.book_id = ? AND c.chunk_kind = ?
            LIMIT 1
            """,
            (book_id, kind),
        ).fetchone()
        if row is not None:
            return row[0]
    return None
