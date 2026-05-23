"""Unit-тесты для src/search/related.py — find_related и build_related_payload."""
import sqlite3
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sqlite_vec

from src.search.related import _get_book_embedding_bytes, build_related_payload, find_related


# ─── Fixtures ────────────────────────────────────────────────────────────────

DIM = 8  # маленькая размерность для тестов


def _make_vec(values: list[float]) -> bytes:
    arr = np.array(values, dtype=np.float32)
    arr /= np.linalg.norm(arr) + 1e-9
    return arr.tobytes()


@pytest.fixture
def db():
    """Минимальная in-memory БД с несколькими книгами и эмбеддингами."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(f"""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            catalog_id INTEGER,
            title TEXT NOT NULL,
            author TEXT,
            year INTEGER,
            category TEXT,
            file_format TEXT,
            filename TEXT,
            folder TEXT,
            summary TEXT,
            duplicate_of INTEGER
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            book_id INTEGER NOT NULL,
            chunk_kind TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            char_count INTEGER NOT NULL
        );

        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{DIM}]
        );
    """)

    # Книги: 5 штук
    books = [
        (1, 1, "Книга A", "Автор 1", 2020, "Программирование/Python", "epub", "a.epub", "/books", "Саммари А"),
        (2, 2, "Книга B", "Автор 2", 2021, "Программирование/Python", "pdf",  "b.pdf",  "/books", "Саммари Б"),
        (3, 3, "Книга C", "Автор 3", 2022, "Алгоритмы/Общие",        "epub", "c.epub", "/books", "Саммари В"),
        (4, 4, "Книга D", "Автор 4", 2019, "Алгоритмы/Общие",        "pdf",  "d.pdf",  "/books", None),
        (5, 5, "Книга E", "Автор 5", 2023, "DevOps/Docker",           "epub", "e.epub", "/books", "Саммари Д"),
    ]
    conn.executemany(
        "INSERT INTO books (id,catalog_id,title,author,year,category,file_format,filename,folder,summary,duplicate_of)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
        books,
    )

    # Эмбеддинги: близкие друг к другу попарно
    # A и B близки (Python), C и D близки (Алгоритмы), E отдалена
    vecs = {
        1: [1.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # A
        2: [0.9, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # B (близко к A)
        3: [0.0, 0.0, 0.0, 1.0, 0.9, 0.1, 0.0, 0.0],  # C
        4: [0.0, 0.0, 0.0, 0.9, 1.0, 0.1, 0.0, 0.0],  # D (близко к C)
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # E (отдалена)
    }

    chunk_id = 1
    for book_id, vals in vecs.items():
        conn.execute(
            "INSERT INTO chunks (id, book_id, chunk_kind, chunk_index, text, text_hash, char_count)"
            " VALUES (?, ?, 'summary', 0, 'text', 'hash', 4)",
            (chunk_id, book_id),
        )
        conn.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, _make_vec(vals)),
        )
        chunk_id += 1

    conn.commit()
    yield conn
    conn.close()


# ─── _get_book_embedding_bytes ────────────────────────────────────────────────

class TestGetBookEmbeddingBytes:

    def test_returns_bytes_for_existing_book(self, db):
        result = _get_book_embedding_bytes(db, 1)
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) == DIM * 4  # float32

    def test_returns_none_for_nonexistent_book(self, db):
        result = _get_book_embedding_bytes(db, 999)
        assert result is None


# ─── find_related ─────────────────────────────────────────────────────────────

class TestFindRelated:

    def test_excludes_anchor_book(self, db):
        results = find_related(db, 1, k=10)
        ids = [bid for bid, _ in results]
        assert 1 not in ids

    def test_returns_similar_books_first(self, db):
        """Книга B должна быть первой в related для книги A."""
        results = find_related(db, 1, k=4)
        assert results[0][0] == 2  # B — ближайшая к A

    def test_excludes_specified_book_ids(self, db):
        """Исключение {2} → B не появляется."""
        results = find_related(db, 1, k=10, exclude_book_ids={2})
        ids = [bid for bid, _ in results]
        assert 2 not in ids

    def test_excludes_anchor_even_without_explicit_exclude(self, db):
        results = find_related(db, 3, k=10)
        assert 3 not in [bid for bid, _ in results]

    def test_scores_are_between_minus1_and_1(self, db):
        results = find_related(db, 1, k=4)
        for bid, score in results:
            assert -1.0 <= score <= 1.0001

    def test_sorted_descending(self, db):
        results = find_related(db, 1, k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_empty_for_nonexistent_book(self, db):
        results = find_related(db, 999, k=5)
        assert results == []

    def test_k_limits_results(self, db):
        results = find_related(db, 1, k=2)
        assert len(results) <= 2

    def test_excludes_duplicate_books(self, db):
        """Книга помеченная как дубликат исключённой не должна попасть."""
        # Помечаем книгу B (2) как дубликат книги A (1)
        db.execute("UPDATE books SET duplicate_of = 1 WHERE id = 2")
        db.commit()
        # Ищем related для книги C, исключая {1}
        results = find_related(db, 3, k=10, exclude_book_ids={1})
        ids = [bid for bid, _ in results]
        # Книга 2 — дубликат 1 (которая в exclude) → тоже должна быть исключена
        assert 2 not in ids


# ─── build_related_payload ───────────────────────────────────────────────────

class TestBuildRelatedPayload:

    def test_empty_hits_returns_empty(self, db):
        assert build_related_payload(db, []) == []

    def test_returns_correct_structure(self, db):
        hits = [(2, 0.95), (3, 0.80)]
        payload = build_related_payload(db, hits, max_results=5)
        assert len(payload) == 2
        item = payload[0]
        required_keys = {"rank", "book_id", "title", "author", "year",
                         "category", "file_format", "file_path",
                         "similarity_to_top_result", "summary"}
        assert required_keys.issubset(item.keys())

    def test_rank_is_sequential(self, db):
        hits = [(1, 0.9), (2, 0.8), (3, 0.7)]
        payload = build_related_payload(db, hits, max_results=5)
        ranks = [p["rank"] for p in payload]
        assert ranks == [1, 2, 3]

    def test_similarity_is_rounded(self, db):
        hits = [(1, 0.912345678)]
        payload = build_related_payload(db, hits)
        assert payload[0]["similarity_to_top_result"] == round(0.912345678, 3)

    def test_max_results_limit(self, db):
        hits = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)]
        payload = build_related_payload(db, hits, max_results=3)
        assert len(payload) == 3

    def test_order_matches_hits(self, db):
        """Порядок в payload соответствует порядку hits."""
        hits = [(3, 0.9), (1, 0.8)]
        payload = build_related_payload(db, hits)
        assert payload[0]["book_id"] == 3
        assert payload[1]["book_id"] == 1

    def test_file_path_combines_folder_and_filename(self, db):
        hits = [(1, 0.9)]
        payload = build_related_payload(db, hits)
        fp = payload[0]["file_path"]
        assert "a.epub" in fp
        assert "books" in fp

    def test_none_summary_is_preserved(self, db):
        """Книга D (id=4) без саммари → None в payload."""
        hits = [(4, 0.75)]
        payload = build_related_payload(db, hits)
        assert payload[0]["summary"] is None
