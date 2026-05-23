from __future__ import annotations

import sqlite3

import numpy as np


def semantic_search(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    k: int = 50,
) -> list[tuple[int, float]]:
    """
    KNN-поиск через sqlite-vec.
    Возвращает список (chunk_id, cosine_similarity), отсортированный по убыванию score.

    sqlite-vec возвращает L2-расстояние. Для нормализованных векторов:
        cosine_sim = 1 - L2² / 2
    """
    query_bytes = query_vec.astype(np.float32).tobytes()

    rows = conn.execute(
        """
        SELECT chunk_id, distance
        FROM chunk_vectors
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (query_bytes, k),
    ).fetchall()

    results: list[tuple[int, float]] = []
    for row in rows:
        chunk_id = row[0]
        distance = float(row[1])
        cosine_sim = 1.0 - (distance ** 2) / 2.0
        results.append((chunk_id, cosine_sim))

    return results
