from __future__ import annotations

import re
import sqlite3


def _build_fts_query(text: str) -> str | None:
    """
    Строит безопасный FTS5-запрос из пользовательского текста.
    Убирает спецсимволы FTS5, квотирует каждый токен, объединяет через OR.
    Возвращает None, если токенов не осталось.
    """
    clean = re.sub(r'["\'*^(){}|\[\]~]', ' ', text)
    tokens = [t for t in re.split(r'\s+', clean.strip()) if len(t) >= 2]
    if not tokens:
        return None
    return ' OR '.join(f'"{t}"' for t in tokens)


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 50,
) -> list[tuple[int, float]]:
    """
    BM25-поиск через FTS5.
    Возвращает список (chunk_id, bm25_score) отсортированных по убыванию score.

    FTS5 rank отрицательный (чем меньше — тем релевантнее).
    Мы его негируем → положительный score, больше = лучше.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE text MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, k),
        ).fetchall()
    except Exception:
        return []

    results: list[tuple[int, float]] = []
    for row in rows:
        chunk_id = int(row[0])
        bm25 = -float(row[1])
        results.append((chunk_id, bm25))

    return results
