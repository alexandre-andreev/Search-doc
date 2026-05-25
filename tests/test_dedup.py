"""Unit-тесты для src/index/dedup.py."""
import sqlite3

import pytest

from src.index.dedup import (
    HAMMING_THRESHOLD,
    PLACEHOLDER_CLUSTER_LIMIT,
    compute_simhash,
    hamming_distance,
    normalize_title,
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

    def test_stats_include_title_groups_found(self, dedup_db):
        """run_dedup возвращает поле title_groups_found."""
        _insert(dedup_db, 1, "Book A", "pdf", 5.0, summary="Python разработка Django REST API веб")
        stats = run_dedup(dedup_db)
        assert "title_groups_found" in stats


# ─── normalize_title ──────────────────────────────────────────────────────────

class TestNormalizeTitle:

    def test_lowercase(self):
        assert normalize_title("Clean Code") == "clean code"

    def test_strips_punctuation(self):
        assert normalize_title("Clean Code: A Handbook") == "clean code a handbook"

    def test_strips_hyphen(self):
        assert normalize_title("Программист-фанатик") == "программист фанатик"

    def test_strips_russian_edition_marker(self):
        result = normalize_title("Программист-фанатик 2-е издание")
        assert "издание" not in result
        assert "программист" in result

    def test_strips_english_edition_marker(self):
        result = normalize_title("Clean Code 3rd edition")
        assert "edition" not in result
        assert "clean" in result

    def test_empty_title(self):
        assert normalize_title("") == ""

    def test_whitespace_normalized(self):
        result = normalize_title("  Python   Programming  ")
        assert "  " not in result


# ─── Two-pass dedup ───────────────────────────────────────────────────────────

class TestTwoPassDedup:

    def test_placeholder_summary_not_clustered(self, dedup_db):
        """Книги с placeholder-саммари не должны склеиваться между собой.

        PLACEHOLDER_CLUSTER_LIMIT + 1 книг с одинаковым коротким саммари →
        все остаются non-duplicate.
        """
        placeholder = "Не удалось определить содержимое"
        for book_id in range(1, PLACEHOLDER_CLUSTER_LIMIT + 2):
            _insert(dedup_db, book_id, f"Книга {book_id}", "pdf", float(book_id),
                    summary=placeholder)
        stats = run_dedup(dedup_db)
        assert stats["duplicates_marked"] == 0

    def test_real_duplicate_with_same_summary_still_caught(self, dedup_db):
        """Реальные дубли (одинаковое длинное саммари) поймаются через Pass 1."""
        summary = (
            "Книга посвящена написанию чистого поддерживаемого кода. "
            "Рассматриваются принципы SOLID, рефакторинг и паттерны. "
            "Рекомендуется для опытных разработчиков программного обеспечения."
        )
        _insert(dedup_db, 1, "Clean Code EPUB", "epub", 10.0, summary=summary)
        _insert(dedup_db, 2, "Clean Code PDF",  "pdf",  8.0, summary=summary)
        stats = run_dedup(dedup_db)
        assert stats["duplicates_marked"] == 1

    def test_title_duplicate_different_summaries_caught(self, dedup_db):
        """Pass 2: одинаковый заголовок, разные саммари → дубль пойман."""
        _insert(dedup_db, 1, "Программист-фанатик", "epub", 10.0,
                summary="Книга Чеда Фаулера о карьере разработчика и мастерстве программирования.")
        _insert(dedup_db, 2, "Программист-фанатик", "pdf",  7.0,
                summary=None)  # нет саммари
        stats = run_dedup(dedup_db)
        assert stats["duplicates_marked"] == 1
        assert stats["title_groups_found"] >= 1

    def test_title_duplicate_hyphen_vs_space(self, dedup_db):
        """normalize_title выравнивает 'Программист-фанатик' и 'Программист фанатик'."""
        summary_a = "Книга о мастерстве и карьере программиста с практическими советами."
        _insert(dedup_db, 1, "Программист-фанатик", "epub", 10.0, summary=summary_a)
        _insert(dedup_db, 2, "Программист фанатик", "pdf",  7.0, summary=None)
        stats = run_dedup(dedup_db)
        assert stats["duplicates_marked"] == 1

    def test_different_titles_not_clustered_by_title(self, dedup_db):
        """Разные книги с разными заголовками не склеиваются через Pass 2."""
        _insert(dedup_db, 1, "Чистый код", "epub", 10.0, summary=None)
        _insert(dedup_db, 2, "Совершенный код", "pdf", 8.0, summary=None)
        stats = run_dedup(dedup_db)
        assert stats["duplicates_marked"] == 0

    def test_placeholder_books_can_still_be_caught_via_title(self, dedup_db):
        """Книга с placeholder-саммари, но с совпадающим заголовком → поймана через Pass 2."""
        placeholder = "Не удалось определить содержимое"
        _insert(dedup_db, 1, "Think Like a Programmer", "epub", 12.0, summary=placeholder)
        _insert(dedup_db, 2, "Think Like a Programmer", "pdf",   9.0, summary=placeholder)
        stats = run_dedup(dedup_db)
        # Pass 1 пропускает их (popular simhash), Pass 2 ловит по title
        assert stats["duplicates_marked"] == 1
