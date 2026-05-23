"""Логика полного импорта каталога в SQLite."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from tqdm import tqdm

from src.catalog_import.xlsx_loader import CatalogBook, IndexChunk, build_chunks, load_catalog
from src.embedder.e5_small import E5SmallEmbedder
from src.index.db import apply_schema, open_db, set_meta
from src.index.dedup import compute_simhash


def _row_hash(book: CatalogBook) -> str:
    """sha256 от ключевых полей — для инкрементального переиндекса (Этап 6)."""
    parts = "|".join([
        str(book.title or ""),
        str(book.author or ""),
        str(book.summary or ""),
        str(book.year or ""),
    ])
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _insert_book(conn: sqlite3.Connection, book: CatalogBook, row_hash: str) -> int:
    text_for_hash = (book.summary or book.title or "")
    simhash_val = compute_simhash(text_for_hash)
    cur = conn.execute(
        """
        INSERT INTO books (
            catalog_id, title, author, year, publisher,
            category, section, subsection,
            file_format, file_size_mb, filename, folder, summary,
            xlsx_row_hash, text_simhash, status, indexed_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            book.catalog_id, book.title, book.author, book.year, book.publisher,
            book.category, book.section, book.subsection,
            book.file_format, book.file_size_mb, book.filename, book.folder, book.summary,
            row_hash, simhash_val, "imported", time.time(),
        ),
    )
    return cur.lastrowid


def _insert_chunks(conn: sqlite3.Connection, book_id: int, chunks_for_book: list[IndexChunk]) -> list[int]:
    chunk_ids: list[int] = []
    for chunk_index, chunk in enumerate(chunks_for_book):
        cur = conn.execute(
            """
            INSERT INTO chunks (book_id, chunk_kind, chunk_index, text, text_hash, char_count)
            VALUES (?,?,?,?,?,?)
            """,
            (book_id, chunk.kind, chunk_index, chunk.text, _chunk_hash(chunk.text), len(chunk.text)),
        )
        chunk_ids.append(cur.lastrowid)
    return chunk_ids


def _insert_vectors(conn: sqlite3.Connection, chunk_ids: list[int], embeddings) -> None:
    import numpy as np
    for cid, emb in zip(chunk_ids, embeddings):
        conn.execute(
            "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)",
            (cid, emb.astype(np.float32).tobytes()),
        )


def run_import(
    catalog_path: Path,
    db_path: Path,
    rebuild: bool = False,
) -> dict:
    """
    Полный импорт catalog.xlsx в SQLite.
    Возвращает статистику: books_added, chunks_created, elapsed_sec.
    """
    t_start = time.time()

    # Открыть / создать БД
    conn = open_db(db_path)
    apply_schema(conn)

    if rebuild:
        conn.executescript("""
            DELETE FROM chunk_vectors;
            DELETE FROM chunks;
            DELETE FROM books;
            DELETE FROM import_runs;
        """)
        conn.commit()

    # Запись о начале импорта
    cur = conn.execute(
        "INSERT INTO import_runs(started_at, catalog_path, status) VALUES(?,?,?)",
        (t_start, str(catalog_path), "interrupted"),
    )
    run_id = cur.lastrowid
    conn.commit()

    # Загрузить каталог
    print(f"Загружаю каталог: {catalog_path}")
    books = load_catalog(catalog_path)
    print(f"  Книг в каталоге: {len(books)}")

    # Построить чанки
    all_chunks = build_chunks(books)
    n_title = sum(1 for c in all_chunks if c.kind == "title")
    n_summary = sum(1 for c in all_chunks if c.kind == "summary")
    print(f"  Чанков: {len(all_chunks)} (title: {n_title}, summary: {n_summary})")

    # Инициализировать embedder
    print("Загружаю модель e5-small на CUDA...")
    embedder = E5SmallEmbedder()

    # Подготовить тексты для embedding'ов
    texts = [c.text for c in all_chunks]

    print(f"Считаю embedding'и для {len(texts)} чанков...")
    embeddings = embedder.encode_passages(texts, show_progress=True)

    # Записать в БД
    print("Записываю в БД...")
    books_added = 0
    chunks_created = 0
    books_failed = 0

    # Группируем чанки по book_idx
    from collections import defaultdict
    book_chunks: dict[int, list[tuple[IndexChunk, int]]] = defaultdict(list)
    for global_idx, chunk in enumerate(all_chunks):
        book_chunks[chunk.book_idx].append((chunk, global_idx))

    with tqdm(total=len(books), desc="Записываю книги", unit="книг") as pbar:
        for book_idx, book in enumerate(books):
            try:
                row_hash = _row_hash(book)
                with conn:
                    book_id = _insert_book(conn, book, row_hash)
                    chunks_for_book = book_chunks[book_idx]
                    local_chunks = [c for c, _ in chunks_for_book]
                    global_indices = [gi for _, gi in chunks_for_book]

                    chunk_ids = _insert_chunks(conn, book_id, local_chunks)

                    chunk_embs = embeddings[global_indices]
                    _insert_vectors(conn, chunk_ids, chunk_embs)

                books_added += 1
                chunks_created += len(chunk_ids)
            except Exception as exc:
                books_failed += 1
                print(f"\n  ОШИБКА книга catalog_id={book.catalog_id}: {exc}")
            pbar.update(1)

    # Метаданные индекса
    with conn:
        set_meta(conn, "schema_version", "1.0")
        set_meta(conn, "embedding_model", embedder.model_name)
        set_meta(conn, "embedding_dim", str(embedder.dim))
        set_meta(conn, "created_at", str(t_start))
        set_meta(conn, "last_indexed_at", str(time.time()))
        set_meta(conn, "catalog_imported_at", str(time.time()))

        elapsed = time.time() - t_start
        conn.execute(
            """UPDATE import_runs SET
                finished_at=?, books_added=?, books_failed=?,
                chunks_created=?, status=?
               WHERE id=?""",
            (time.time(), books_added, books_failed, chunks_created, "completed", run_id),
        )

    conn.close()

    return {
        "books_added": books_added,
        "books_failed": books_failed,
        "chunks_created": chunks_created,
        "elapsed_sec": elapsed,
        "n_title": n_title,
        "n_summary": n_summary,
    }
