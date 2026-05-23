"""Unit-тесты для src/index/dedup.py."""
import sqlite3

import pytest

from src.index.dedup import (
    HAMMING_THRESHOLD,
    compute_simhash,
    hamming_distance,
    run_dedup,
)


# ─── compute_simhash ──────────────────────────────────────────────────────────

class TestComputeSimhash:

    def test_returns_int(self):
        assert isinstance(compute_simhash("hello world"), int)

    def test_same_text_same_hash(self):
        text = "Книга о программировании на Python для начинающих разработчиков."
        assert compute_simhash(text) == compute_simhash(text)

    def test_empty_text_returns_zero(self):
        assert compute_simhash("") == 0

    def test_identical_texts_distance_zero(self):
        text = "Полное руководство по алгоритмам и структурам данных"
        h1 = compute_simhash(text)
        h2 = compute_simhash(text)
        assert hamming_distance(h1, h2) == 0

    def test_near_duplicate_small_distance(self):
        """Текст с одним изменённым словом → малое расстояние Хэмминга."""
        base = (
            "Книга посвящена разработке программного обеспечения с использованием "
            "методологии TDD. Рассматриваются ключевые концепции написания тестов "
            "и рефакторинга кода. Предназначена для опытных разработчиков."
        )
        modified = base.replace("опытных", "начинающих")
        h1 = compute_simhash(base)
        h2 = compute_simhash(modified)
        assert hamming_distance(h1, h2) <= 8

    def test_completely_different_texts_large_distance(self):
        """Совершенно разные тексты → большое расстояние."""
        h1 = compute_simhash("Python Django REST API веб-разработка")
        h2 = compute_simhash("Квантовая физика теория относительности Эйнштейн")
        assert hamming_distance(h1, h2) > 5

    def test_hash_fits_in_signed_64_bits(self):
        h = compute_simhash("произвольный текст для тестирования simhash алгоритма")
        assert -(1 << 63) <= h <= (1 << 63) - 1

    def test_respects_50k_char_limit(self):
        """Очень длинный текст — не падает, даёт стабильный результат."""
        long_text = "слово " * 20_000  # ~120K символов
        h1 = compute_simhash(long_text)
        h2 = compute_simhash(long_text)
        assert h1 == h2


# ─── hamming_distance ─────────────────────────────────────────────────────────

class TestHammingDistance:

    def test_identical(self):
        assert hamming_distance(0xABCD, 0xABCD) == 0

    def test_one_bit(self):
        assert hamming_distance(0b0000, 0b0001) == 1

    def test_all_bits_differ(self):
        assert hamming_distance(0xFFFF_FFFF_FFFF_FFFF, 0x0) == 64

    def test_symmetric(self):
        a, b = 0x1234567890ABCDEF, 0xFEDCBA0987654321
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_threshold_boundary(self):
        h = 0b111 << 10  # 3 единичных бита
        assert hamming_distance(0, h) == 3
        assert hamming_distance(0, h) <= HAMMING_THRESHOLD


# ─── run_dedup ────────────────────────────────────────────────────────────────

@pytest.fixture
def dedup_db():
    """Минимальная in-memory БД с несколькими книгами для теста дедупликации."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            catalog_id INTEGER,
            title TEXT NOT NULL,
            file_format TEXT,
            file_size_mb REAL,
            summary TEXT,
            text_simhash INTEGER,
            duplicate_of INTEGER
        );
    """)
    yield conn
    conn.close()


def _insert(conn, book_id, title, fmt, size_mb, summary=None, simhash=None):
    conn.execute(
        "INSERT INTO books (id, catalog_id, title, file_format, file_size_mb, summary, text_simhash) "
        "VALUES (?,?,?,?,?,?,?)",
        (book_id, book_id, title, fmt, size_mb, summary, simhash),
    )
    conn.commit()


class TestRunDedup:

    def test_no_duplicates(self, dedup_db):
        _insert(dedup_db, 1, "Book A", "pdf", 10.0, summary="Python разработка Django REST")
        _insert(dedup_db, 2, "Book B", "epub", 8.0, summary="Квантовая физика Эйнштейн")
        stats = run_dedup(dedup_db)
        assert stats["duplicates_marked"] == 0
        assert stats["groups_found"] == 0

    def test_exact_duplicate_marked(self, dedup_db):
        """Одинаковый текст саммари → дубликат помечается."""
        summary = "Книга о разработке программного обеспечения с TDD и чистым кодом."
        _insert(dedup_db, 1, "Book A", "epub", 10.0, summary=summary)
        _insert(dedup_db, 2, "Book B", "pdf", 8.0, summary=summary)
        stats = run_dedup(dedup_db)
        assert stats["duplicates_marked"] == 1
        assert stats["groups_found"] == 1

    def test_canonical_is_epub_over_pdf(self, dedup_db):
        """epub предпочтительнее pdf при прочих равных."""
        summary = "Идентичный текст для двух книг о Python программировании и разработке."
        _insert(dedup_db, 1, "Book PDF", "pdf", 10.0, summary=summary)
        _insert(dedup_db, 2, "Book EPUB", "epub", 8.0, summary=summary)
        run_dedup(dedup_db)
        row = dedup_db.execute("SELECT duplicate_of FROM books WHERE id = 1").fetchone()
        assert row[0] == 2  # PDF (id=1) → дубликат epub (id=2)

    def test_canonical_larger_file_wins_same_format(self, dedup_db):
        """При одинаковом формате побеждает больший файл."""
        summary = "Руководство по алгоритмам и структурам данных для программистов."
        _insert(dedup_db, 1, "Small PDF", "pdf", 5.0, summary=summary)
        _insert(dedup_db, 2, "Big PDF", "pdf", 15.0, summary=summary)
        run_dedup(dedup_db)
        row = dedup_db.execute("SELECT duplicate_of FROM books WHERE id = 1").fetchone()
        assert row[0] == 2  # меньший → дубликат большего

    def test_backfills_null_simhash(self, dedup_db):
        """Книги без simhash получают его в ходе dedup."""
        _insert(dedup_db, 1, "Book A", "pdf", 5.0, summary="Текст книги о Python")
        # simhash не задан (NULL)
        row = dedup_db.execute("SELECT text_simhash FROM books WHERE id = 1").fetchone()
        assert row[0] is None
        stats = run_dedup(dedup_db)
        assert stats["backfilled"] == 1
        row = dedup_db.execute("SELECT text_simhash FROM books WHERE id = 1").fetchone()
        assert row[0] is not None

    def test_idempotent(self, dedup_db):
        """Повторный запуск dedup даёт тот же результат."""
        summary = "Одинаковый текст для двух книг о веб-разработке и JavaScript."
        _insert(dedup_db, 1, "Book PDF", "pdf", 5.0, summary=summary)
        _insert(dedup_db, 2, "Book EPUB", "epub", 8.0, summary=summary)
        stats1 = run_dedup(dedup_db)
        stats2 = run_dedup(dedup_db)
        assert stats1["duplicates_marked"] == stats2["duplicates_marked"]
        dup = dedup_db.execute("SELECT duplicate_of FROM books WHERE id = 1").fetchone()
        assert dup[0] == 2

    def test_three_copies_one_canonical(self, dedup_db):
        """Три копии одной книги → один canonical, два дублика."""
        summary = "Полное руководство по Kubernetes оркестрации контейнеров DevOps."
        _insert(dedup_db, 1, "K8s EPUB", "epub", 20.0, summary=summary)
        _insert(dedup_db, 2, "K8s PDF", "pdf", 15.0, summary=summary)
        _insert(dedup_db, 3, "K8s DJVU", "djvu", 10.0, summary=summary)
        run_dedup(dedup_db)
        dups = dedup_db.execute(
            "SELECT id FROM books WHERE duplicate_of IS NOT NULL"
        ).fetchall()
        assert len(dups) == 2
        canonical = dedup_db.execute(
            "SELECT id FROM books WHERE duplicate_of IS NULL"
        ).fetchone()
        assert canonical[0] == 1  # epub — canonical
