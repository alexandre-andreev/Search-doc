"""Логика импорта каталога в SQLite: полный и инкрементальный режимы."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.catalog_import.xlsx_loader import CatalogBook, IndexChunk, build_chunks, load_catalog
from src.embedder.e5_small import E5SmallEmbedder
from src.index.db import apply_schema, open_db, set_meta
from src.index.dedup import compute_simhash


def _row_hash(book: CatalogBook) -> str:
    """sha256 от ключевых полей — изменение хеша = необходим переиндекс."""
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
    simhash_val = compute_simhash(book.summary or book.title or "")
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


def _update_book(conn: sqlite3.Connection, book_id: int, book: CatalogBook, row_hash: str) -> None:
    """Обновляет поля книги и сбрасывает duplicate_of (dedup нужно перезапустить)."""
    simhash_val = compute_simhash(book.summary or book.title or "")
    conn.execute(
        """
        UPDATE books SET
            title=?, author=?, year=?, publisher=?,
            category=?, section=?, subsection=?,
            file_format=?, file_size_mb=?, filename=?, folder=?, summary=?,
            xlsx_row_hash=?, text_simhash=?, duplicate_of=NULL,
            status='imported', indexed_at=?
        WHERE id=?
        """,
        (
            book.title, book.author, book.year, book.publisher,
            book.category, book.section, book.subsection,
            book.file_format, book.file_size_mb, book.filename, book.folder, book.summary,
            row_hash, simhash_val, time.time(),
            book_id,
        ),
    )


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


def _delete_book_chunks(conn: sqlite3.Connection, book_id: int) -> None:
    """Удаляет векторы и чанки книги (vec0 не поддерживает CASCADE)."""
    old_cids = [
        r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE book_id=?", (book_id,)
        ).fetchall()
    ]
    if old_cids:
        ph = ",".join("?" * len(old_cids))
        conn.execute(f"DELETE FROM chunk_vectors WHERE chunk_id IN ({ph})", old_cids)
        conn.execute("DELETE FROM chunks WHERE book_id=?", (book_id,))


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
    Импортирует catalog.xlsx в SQLite.

    Incremental mode (rebuild=False):
    - Книги с неизменённым xlsx_row_hash → пропускаются.
    - Новые книги → добавляются; изменившиеся → переиндексируются.
    - Если изменений нет, E5SmallEmbedder не загружается (<5 сек).
    - Книги, удалённые из каталога, помечаются status='removed'.

    Возвращает: books_added, books_updated, books_skipped, books_removed,
                books_failed, chunks_created, elapsed_sec, n_title, n_summary.
    """
    t_start = time.time()
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

    # ── Запись о начале импорта ──────────────────────────────────────────
    cur = conn.execute(
        "INSERT INTO import_runs(started_at, catalog_path, status) VALUES(?,?,?)",
        (t_start, str(catalog_path), "interrupted"),
    )
    run_id = cur.lastrowid
    conn.commit()

    # ── Загрузить каталог ────────────────────────────────────────────────
    print(f"Загружаю каталог: {catalog_path}")
    books = load_catalog(catalog_path)
    print(f"  Книг в каталоге: {len(books)}")

    # ── Загрузить существующие книги (все, включая 'removed') ────────────
    # catalog_id → (db_book_id, row_hash, status)
    existing: dict[int, tuple[int, str, str]] = {}
    for row in conn.execute(
        "SELECT catalog_id, id, xlsx_row_hash, status FROM books"
    ).fetchall():
        existing[row[0]] = (row[1], row[2], row[3])

    # ── Классификация книг ───────────────────────────────────────────────
    to_insert: list[tuple[CatalogBook, str]] = []       # (book, row_hash)
    to_update: list[tuple[CatalogBook, str, int]] = []  # (book, row_hash, db_book_id)
    skipped = 0

    for book in books:
        rh = _row_hash(book)
        if book.catalog_id not in existing:
            to_insert.append((book, rh))
        else:
            db_id, db_hash, db_status = existing[book.catalog_id]
            if db_hash != rh or db_status == "removed":
                to_update.append((book, rh, db_id))
            else:
                skipped += 1

    # ── Книги, удалённые из каталога ─────────────────────────────────────
    catalog_ids = {b.catalog_id for b in books}
    removed_count = 0
    for cid, (bid, _, db_status) in existing.items():
        if cid not in catalog_ids and db_status != "removed":
            conn.execute("UPDATE books SET status='removed' WHERE id=?", (bid,))
            removed_count += 1
    if removed_count:
        conn.commit()
        print(f"  Помечено удалёнными: {removed_count}")

    if to_insert:
        print(f"  Новых книг: {len(to_insert)}")
    if to_update:
        print(f"  Изменившихся книг: {len(to_update)}")
    if skipped:
        print(f"  Пропущено (без изменений): {skipped}")

    # ── Early exit: ничего не изменилось ─────────────────────────────────
    if not to_insert and not to_update:
        elapsed = time.time() - t_start
        with conn:
            set_meta(conn, "last_indexed_at", str(time.time()))
            conn.execute(
                """UPDATE import_runs SET
                    finished_at=?, books_added=0, books_updated=0,
                    books_skipped=?, books_failed=0, chunks_created=0, status='completed'
                   WHERE id=?""",
                (time.time(), skipped, run_id),
            )
        conn.close()
        return {
            "books_added": 0,
            "books_updated": 0,
            "books_skipped": skipped,
            "books_removed": removed_count,
            "books_failed": 0,
            "chunks_created": 0,
            "elapsed_sec": elapsed,
            "n_title": 0,
            "n_summary": 0,
        }

    # ── Построить чанки только для новых/изменившихся книг ───────────────
    books_to_process: list[CatalogBook] = (
        [b for b, _ in to_insert] + [b for b, _, _ in to_update]
    )
    all_new_chunks = build_chunks(books_to_process)
    n_title = sum(1 for c in all_new_chunks if c.kind == "title")
    n_summary = sum(1 for c in all_new_chunks if c.kind == "summary")
    print(
        f"  Чанков для индексации: {len(all_new_chunks)} "
        f"(title: {n_title}, summary: {n_summary})"
    )

    # ── Загружаем embedder только если есть что индексировать ────────────
    print("Загружаю модель e5-small на CUDA...")
    embedder = E5SmallEmbedder()

    print(f"Считаю embedding'и для {len(all_new_chunks)} чанков...")
    embeddings = embedder.encode_passages(
        [c.text for c in all_new_chunks], show_progress=True
    )

    # Группируем чанки по индексу в books_to_process
    book_chunks_map: dict[int, list[tuple[IndexChunk, int]]] = defaultdict(list)
    for gi, chunk in enumerate(all_new_chunks):
        book_chunks_map[chunk.book_idx].append((chunk, gi))

    # ── INSERT новых книг ─────────────────────────────────────────────────
    print("Записываю в БД...")
    books_added = books_updated = books_failed = chunks_created = 0

    n_ops = len(to_insert) + len(to_update)
    with tqdm(total=n_ops, desc="Записываю", unit="книг") as pbar:

        for embed_idx, (book, rh) in enumerate(to_insert):
            try:
                with conn:
                    book_id = _insert_book(conn, book, rh)
                    c_and_i = book_chunks_map.get(embed_idx, [])
                    local_chunks = [c for c, _ in c_and_i]
                    global_indices = [gi for _, gi in c_and_i]
                    chunk_ids = _insert_chunks(conn, book_id, local_chunks)
                    if chunk_ids:
                        _insert_vectors(conn, chunk_ids, embeddings[global_indices])
                books_added += 1
                chunks_created += len(chunk_ids)
            except Exception as exc:
                books_failed += 1
                print(f"\n  ОШИБКА книга catalog_id={book.catalog_id}: {exc}")
            pbar.update(1)

        # ── UPDATE изменившихся книг ──────────────────────────────────────
        for i, (book, rh, db_book_id) in enumerate(to_update):
            embed_idx = len(to_insert) + i
            try:
                with conn:
                    _delete_book_chunks(conn, db_book_id)
                    _update_book(conn, db_book_id, book, rh)
                    c_and_i = book_chunks_map.get(embed_idx, [])
                    local_chunks = [c for c, _ in c_and_i]
                    global_indices = [gi for _, gi in c_and_i]
                    chunk_ids = _insert_chunks(conn, db_book_id, local_chunks)
                    if chunk_ids:
                        _insert_vectors(conn, chunk_ids, embeddings[global_indices])
                books_updated += 1
                chunks_created += len(chunk_ids)
            except Exception as exc:
                books_failed += 1
                print(f"\n  ОШИБКА книга catalog_id={book.catalog_id}: {exc}")
            pbar.update(1)

    # ── Метаданные ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    with conn:
        set_meta(conn, "schema_version", "1.0")
        set_meta(conn, "embedding_model", embedder.model_name)
        set_meta(conn, "embedding_dim", str(embedder.dim))
        if not conn.execute("SELECT value FROM meta WHERE key='created_at'").fetchone():
            set_meta(conn, "created_at", str(t_start))
        set_meta(conn, "last_indexed_at", str(time.time()))
        set_meta(conn, "catalog_imported_at", str(time.time()))
        conn.execute(
            """UPDATE import_runs SET
                finished_at=?, books_added=?, books_updated=?, books_skipped=?,
                books_failed=?, chunks_created=?, status=?
               WHERE id=?""",
            (
                time.time(), books_added, books_updated, skipped,
                books_failed, chunks_created, "completed", run_id,
            ),
        )

    conn.close()

    return {
        "books_added": books_added,
        "books_updated": books_updated,
        "books_skipped": skipped,
        "books_removed": removed_count,
        "books_failed": books_failed,
        "chunks_created": chunks_created,
        "elapsed_sec": elapsed,
        "n_title": n_title,
        "n_summary": n_summary,
    }
