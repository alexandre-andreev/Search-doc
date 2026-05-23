"""64-битный simhash и дедупликация книг по расстоянию Хэмминга."""
from __future__ import annotations

import re
import sqlite3
from hashlib import md5
from pathlib import Path

SIMHASH_BITS = 64
HAMMING_THRESHOLD = 3

# Приоритет форматов: ниже индекс → предпочтительнее как canonical
FORMAT_RANK: dict[str, int] = {
    fmt: i for i, fmt in enumerate(["epub", "fb2", "docx", "pdf", "mobi", "djvu", "txt"])
}
_MAX_RANK = len(FORMAT_RANK)


# ─── Simhash ─────────────────────────────────────────────────────────────────

def compute_simhash(text: str) -> int:
    """
    Вычисляет 64-битный simhash текста (первые 50 000 символов).
    Алгоритм: weighted sum of per-token md5 bits → sign → бит simhash.
    """
    tokens = re.findall(r'\b\w+\b', text[:50_000].lower())
    if not tokens:
        return 0

    v = [0] * SIMHASH_BITS
    for token in tokens:
        h = int(md5(token.encode("utf-8")).hexdigest()[:16], 16)  # 64 бит
        for i in range(SIMHASH_BITS):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    result = 0
    for i in range(SIMHASH_BITS):
        if v[i] > 0:
            result |= 1 << i
    # SQLite INTEGER signed 64-bit: конвертируем unsigned → signed
    if result >= (1 << 63):
        result -= (1 << 64)
    return result


def hamming_distance(h1: int, h2: int) -> int:
    """Количество различающихся бит между двумя 64-битными хешами (signed или unsigned)."""
    xor = (h1 ^ h2) & 0xFFFF_FFFF_FFFF_FFFF
    count = 0
    while xor:
        xor &= xor - 1
        count += 1
    return count


# ─── Union-Find ───────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py

    def groups(self) -> dict[int, list[int]]:
        """Возвращает {root: [member_ids]}."""
        result: dict[int, list[int]] = {}
        for node in self._parent:
            root = self.find(node)
            result.setdefault(root, []).append(node)
        return result


def _canonical_sort_key(book: dict) -> tuple:
    """
    Ключ сортировки для выбора canonical: меньше = предпочтительнее.
    Критерии: формат → размер (больше лучше) → id (меньше лучше).
    """
    fmt = (book.get("file_format") or "").lower()
    rank = FORMAT_RANK.get(fmt, _MAX_RANK)
    size = -(book.get("file_size_mb") or 0.0)
    return (rank, size, book["id"])


# ─── Main dedup ───────────────────────────────────────────────────────────────

def run_dedup(conn: sqlite3.Connection) -> dict:
    """
    1. Backfill: вычисляет simhash для книг с text_simhash IS NULL.
    2. Clustering: union-find по Hamming-расстоянию ≤ HAMMING_THRESHOLD.
    3. Marking: для каждого кластера помечает non-canonical как duplicate_of.

    Возвращает статистику.
    """
    # ── Шаг 1: backfill ──────────────────────────────────────────────────────
    nulls = conn.execute(
        "SELECT id, summary, title FROM books WHERE text_simhash IS NULL"
    ).fetchall()
    for book_id, summary, title in nulls:
        text = (summary or title or "")
        sh = compute_simhash(text)
        conn.execute("UPDATE books SET text_simhash = ? WHERE id = ?", (sh, book_id))
    if nulls:
        conn.commit()

    # ── Шаг 2: сброс старых меток (idempotent) ───────────────────────────────
    conn.execute("UPDATE books SET duplicate_of = NULL")
    conn.commit()

    # ── Шаг 3: загрузка книг ─────────────────────────────────────────────────
    rows = conn.execute(
        "SELECT id, file_format, file_size_mb, text_simhash FROM books WHERE text_simhash IS NOT NULL"
    ).fetchall()

    books = [{"id": r[0], "file_format": r[1], "file_size_mb": r[2], "simhash": r[3]} for r in rows]
    ids = [b["id"] for b in books]
    book_by_id = {b["id"]: b for b in books}

    # ── Шаг 4: union-find ─────────────────────────────────────────────────────
    uf = _UnionFind(ids)
    n = len(books)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(books[i]["simhash"], books[j]["simhash"]) <= HAMMING_THRESHOLD:
                uf.union(books[i]["id"], books[j]["id"])

    # ── Шаг 5: выбор canonical и разметка ────────────────────────────────────
    groups = uf.groups()
    dup_groups = {root: members for root, members in groups.items() if len(members) > 1}

    duplicates_marked = 0
    groups_found = len(dup_groups)

    for members in dup_groups.values():
        group_books = sorted([book_by_id[mid] for mid in members], key=_canonical_sort_key)
        canonical = group_books[0]
        for dup in group_books[1:]:
            conn.execute(
                "UPDATE books SET duplicate_of = ? WHERE id = ?",
                (canonical["id"], dup["id"]),
            )
            duplicates_marked += 1

    conn.commit()

    return {
        "backfilled": len(nulls),
        "groups_found": groups_found,
        "duplicates_marked": duplicates_marked,
    }
