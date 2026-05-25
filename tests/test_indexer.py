"""Unit-тесты инкрементального импорта (src/pipeline/indexer.py).

E5SmallEmbedder и load_catalog мокируются — GPU не нужен.
Все тесты используют реальный файл SQLite (sqlite-vec требует file-based DB).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalog_import.xlsx_loader import CatalogBook
from src.pipeline.indexer import _row_hash, run_import


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_book(
    catalog_id: int,
    title: str = "Test Book",
    author: str | None = "Test Author",
    summary: str | None = None,
    year: int | None = 2020,
    file_format: str = "pdf",
    file_size_mb: float = 5.0,
    filename: str = "test.pdf",
    folder: str = "books",
) -> CatalogBook:
    return CatalogBook(
        catalog_id=catalog_id,
        title=title,
        author=author,
        year=year,
        publisher=None,
        category=None,
        section=None,
        subsection=None,
        file_format=file_format,
        file_size_mb=file_size_mb,
        filename=filename,
        folder=folder,
        summary=summary,
    )


def _make_embedder_mock(n_chunks: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.model_name = "intfloat/multilingual-e5-small"
    mock.dim = 384
    mock.encode_passages.side_effect = (
        lambda texts, show_progress=False: np.zeros((len(texts), 384), dtype=np.float32)
    )
    return mock


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "test.sqlite"


@pytest.fixture
def two_books() -> list[CatalogBook]:
    return [
        _make_book(1, title="Python Book", author="A. Author"),
        _make_book(2, title="Rust Book", author="B. Author"),
    ]


def _run(db_path: Path, books: list[CatalogBook], rebuild: bool = False) -> dict:
    """Вспомогательный запуск run_import с замоканным embedder и load_catalog."""
    mock_embedder = _make_embedder_mock()
    with patch("src.pipeline.indexer.load_catalog", return_value=books), \
         patch("src.pipeline.indexer.E5SmallEmbedder", return_value=mock_embedder):
        return run_import(Path("dummy.xlsx"), db_path, rebuild=rebuild)


# ─── row_hash ─────────────────────────────────────────────────────────────────

class TestRowHash:

    def test_same_book_same_hash(self):
        b = _make_book(1, title="Book", author="Author")
        assert _row_hash(b) == _row_hash(b)

    def test_title_change_changes_hash(self):
        b1 = _make_book(1, title="Book A")
        b2 = _make_book(1, title="Book B")
        assert _row_hash(b1) != _row_hash(b2)

    def test_summary_change_changes_hash(self):
        b1 = _make_book(1, summary="Summary one about Python development")
        b2 = _make_book(1, summary="Summary two about Rust systems")
        assert _row_hash(b1) != _row_hash(b2)

    def test_format_change_does_not_change_hash(self):
        """Смена формата файла не меняет хеш (не влияет на переиндекс)."""
        b1 = _make_book(1, file_format="pdf")
        b2 = _make_book(1, file_format="epub")
        assert _row_hash(b1) == _row_hash(b2)


# ─── Первый импорт ────────────────────────────────────────────────────────────

class TestFirstImport:

    def test_all_books_added(self, db_path, two_books):
        stats = _run(db_path, two_books)
        assert stats["books_added"] == 2
        assert stats["books_updated"] == 0
        assert stats["books_skipped"] == 0

    def test_chunks_created(self, db_path, two_books):
        stats = _run(db_path, two_books)
        # Каждая книга имеет минимум 1 title-чанк
        assert stats["chunks_created"] >= 2
        assert stats["n_title"] == 2

    def test_embedder_called(self, db_path, two_books):
        mock_embedder = _make_embedder_mock()
        mock_cls = MagicMock(return_value=mock_embedder)
        with patch("src.pipeline.indexer.load_catalog", return_value=two_books), \
             patch("src.pipeline.indexer.E5SmallEmbedder", mock_cls):
            run_import(Path("dummy.xlsx"), db_path)
        assert mock_cls.call_count == 1
        assert mock_embedder.encode_passages.call_count == 1

    def test_import_runs_record_completed(self, db_path, two_books):
        _run(db_path, two_books)
        from src.index.db import open_db
        conn = open_db(db_path)
        row = conn.execute(
            "SELECT status, books_added FROM import_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["status"] == "completed"
        assert row["books_added"] == 2
        conn.close()


# ─── Инкрементальный: без изменений ──────────────────────────────────────────

class TestNoChanges:

    def test_all_books_skipped(self, db_path, two_books):
        _run(db_path, two_books)
        stats = _run(db_path, two_books)
        assert stats["books_skipped"] == 2
        assert stats["books_added"] == 0
        assert stats["books_updated"] == 0

    def test_embedder_not_loaded(self, db_path, two_books):
        """Early exit: embedder НЕ создаётся при отсутствии изменений."""
        _run(db_path, two_books)  # первый импорт

        mock_cls = MagicMock(return_value=_make_embedder_mock())
        with patch("src.pipeline.indexer.load_catalog", return_value=two_books), \
             patch("src.pipeline.indexer.E5SmallEmbedder", mock_cls):
            run_import(Path("dummy.xlsx"), db_path)

        assert mock_cls.call_count == 0

    def test_elapsed_fast(self, db_path, two_books):
        """Без изменений импорт занимает < 5 сек (нет CUDA-загрузки)."""
        import time
        _run(db_path, two_books)
        t0 = time.time()
        _run(db_path, two_books)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"Слишком долго без изменений: {elapsed:.2f}s"

    def test_db_not_modified(self, db_path, two_books):
        """Счётчики в БД не изменяются при повторном запуске."""
        _run(db_path, two_books)
        from src.index.db import open_db
        conn = open_db(db_path)
        n_books_before = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        n_chunks_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()

        _run(db_path, two_books)

        conn = open_db(db_path)
        assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == n_books_before
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == n_chunks_before
        conn.close()


# ─── Инкрементальный: обновление ─────────────────────────────────────────────

class TestUpdate:

    def test_changed_book_reindexed(self, db_path, two_books):
        _run(db_path, two_books)

        # Меняем title первой книги
        updated = [
            _make_book(1, title="Python Book Updated Edition"),
            two_books[1],
        ]
        stats = _run(db_path, updated)
        assert stats["books_updated"] == 1
        assert stats["books_skipped"] == 1
        assert stats["books_added"] == 0

    def test_updated_title_reflected_in_db(self, db_path, two_books):
        _run(db_path, two_books)

        updated = [
            _make_book(1, title="New Title For Book One"),
            two_books[1],
        ]
        _run(db_path, updated)

        from src.index.db import open_db
        conn = open_db(db_path)
        row = conn.execute("SELECT title FROM books WHERE catalog_id=1").fetchone()
        assert row["title"] == "New Title For Book One"
        conn.close()

    def test_old_chunks_replaced(self, db_path):
        """После обновления у книги ровно столько чанков, сколько в новой версии."""
        book = _make_book(1, title="Original Title", author="Author")
        _run(db_path, [book])

        from src.index.db import open_db
        conn = open_db(db_path)
        book_id = conn.execute("SELECT id FROM books WHERE catalog_id=1").fetchone()[0]
        old_chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE book_id=?", (book_id,)
        ).fetchone()[0]
        conn.close()

        updated = [_make_book(1, title="Updated Title", author="New Author")]
        _run(db_path, [updated[0]])

        conn = open_db(db_path)
        new_chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE book_id=?", (book_id,)
        ).fetchone()[0]
        # Новые чанки вставлены (могут быть равны, если структура та же)
        assert new_chunk_count > 0
        # Старые векторы не висят как orphans
        old_vec_count = conn.execute(
            "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id NOT IN (SELECT id FROM chunks)"
        ).fetchone()[0]
        assert old_vec_count == 0
        conn.close()

    def test_duplicate_of_reset_on_update(self, db_path, two_books):
        """При обновлении книги duplicate_of сбрасывается."""
        _run(db_path, two_books)

        from src.index.db import open_db
        conn = open_db(db_path)
        conn.execute("UPDATE books SET duplicate_of=2 WHERE catalog_id=1")
        conn.commit()
        conn.close()

        updated = [_make_book(1, title="Python Book Changed"), two_books[1]]
        _run(db_path, updated)

        conn = open_db(db_path)
        row = conn.execute("SELECT duplicate_of FROM books WHERE catalog_id=1").fetchone()
        assert row["duplicate_of"] is None
        conn.close()


# ─── Инкрементальный: новая книга ────────────────────────────────────────────

class TestNewBook:

    def test_new_book_added_on_second_run(self, db_path, two_books):
        _run(db_path, two_books)

        three_books = two_books + [_make_book(3, title="Go Book")]
        stats = _run(db_path, three_books)
        assert stats["books_added"] == 1
        assert stats["books_skipped"] == 2

    def test_total_count_increases(self, db_path, two_books):
        _run(db_path, two_books)
        _run(db_path, two_books + [_make_book(3, title="Go Book")])

        from src.index.db import open_db
        conn = open_db(db_path)
        n = conn.execute("SELECT COUNT(*) FROM books WHERE status='imported'").fetchone()[0]
        assert n == 3
        conn.close()


# ─── Инкрементальный: удаление ───────────────────────────────────────────────

class TestRemoved:

    def test_missing_book_marked_removed(self, db_path, two_books):
        _run(db_path, two_books)
        stats = _run(db_path, [two_books[0]])  # вторая книга убрана
        assert stats["books_removed"] == 1

    def test_removed_book_status_in_db(self, db_path, two_books):
        _run(db_path, two_books)
        _run(db_path, [two_books[0]])

        from src.index.db import open_db
        conn = open_db(db_path)
        row = conn.execute("SELECT status FROM books WHERE catalog_id=2").fetchone()
        assert row["status"] == "removed"
        conn.close()

    def test_removed_book_reappears(self, db_path, two_books):
        """Книга, помеченная 'removed', при возвращении в каталог переиндексируется."""
        _run(db_path, two_books)
        _run(db_path, [two_books[0]])      # book 2 removed
        stats = _run(db_path, two_books)   # book 2 back

        assert stats["books_updated"] == 1  # переиндексирована через to_update

        from src.index.db import open_db
        conn = open_db(db_path)
        row = conn.execute("SELECT status FROM books WHERE catalog_id=2").fetchone()
        assert row["status"] == "imported"
        conn.close()


# ─── Rebuild ──────────────────────────────────────────────────────────────────

class TestRebuild:

    def test_rebuild_resets_db(self, db_path, two_books):
        _run(db_path, two_books)
        stats = _run(db_path, two_books, rebuild=True)
        assert stats["books_added"] == 2
        assert stats["books_skipped"] == 0

    def test_rebuild_clears_chunks(self, db_path, two_books):
        _run(db_path, two_books)

        from src.index.db import open_db
        conn = open_db(db_path)
        n_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()

        _run(db_path, two_books, rebuild=True)

        conn = open_db(db_path)
        n_after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()

        assert n_after == n_before  # то же число (rebuilt), не накопилось
