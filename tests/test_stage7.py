"""Тесты Stage 7: фильтры aggregate_to_books, команды categories/book/open."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.index.db import apply_schema, open_db
from src.search.ranker import aggregate_to_books


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    conn = open_db(db_path)
    apply_schema(conn)
    return db_path, conn


def _insert_book(conn, catalog_id: int, title: str, section: str | None = None,
                 year: int | None = None, file_format: str | None = None) -> int:
    """Вставляет книгу + title-чанк + вектор-заглушку. Возвращает chunk_id."""
    cur = conn.execute(
        """INSERT INTO books (catalog_id, title, section, year, file_format,
               xlsx_row_hash, status, text_simhash)
           VALUES (?,?,?,?,?,?,?,?)""",
        (catalog_id, title, section, year, file_format,
         f"hash{catalog_id}", "imported", 0),
    )
    book_id = cur.lastrowid
    cur2 = conn.execute(
        """INSERT INTO chunks (book_id, chunk_kind, chunk_index, text, text_hash, char_count)
           VALUES (?,?,?,?,?,?)""",
        (book_id, "title", 0, title, f"h{catalog_id}", len(title)),
    )
    chunk_id = cur2.lastrowid
    conn.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, bytes(384 * 4)),
    )
    conn.commit()
    return chunk_id


def _all_hits(conn) -> list[tuple[int, float]]:
    """Все chunk_ids с равным скором 1.0."""
    return [(r[0], 1.0) for r in conn.execute("SELECT id FROM chunks").fetchall()]


# ─── aggregate_to_books: без фильтров ─────────────────────────────────────────

class TestAggregateNoFilter:

    def test_returns_all_books(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Book A")
        _insert_book(conn, 2, "Book B")
        _insert_book(conn, 3, "Book C")

        results = aggregate_to_books(_all_hits(conn), conn, top_k=10)
        assert len(results) == 3
        conn.close()

    def test_empty_filter_dict_ignored(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Book A", section="Tech")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"section": None, "year_from": None, "format": None},
        )
        assert len(results) == 1
        conn.close()

    def test_none_filters_ignored(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Book A")

        results = aggregate_to_books(_all_hits(conn), conn, top_k=10, filters=None)
        assert len(results) == 1
        conn.close()


# ─── aggregate_to_books: section filter ──────────────────────────────────────

class TestSectionFilter:

    def test_section_keeps_matching(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Python Book", section="Программирование")
        _insert_book(conn, 2, "History",     section="История")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"section": "Программирование"},
        )
        assert len(results) == 1
        assert results[0]["title"] == "Python Book"
        conn.close()

    def test_section_no_match_returns_empty(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Book", section="История")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"section": "Программирование"},
        )
        assert results == []
        conn.close()

    def test_null_section_in_db_excluded(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Sectioned Book", section="Tech")
        _insert_book(conn, 2, "No Section Book", section=None)

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"section": "Tech"},
        )
        assert len(results) == 1
        conn.close()


# ─── aggregate_to_books: year_from filter ────────────────────────────────────

class TestYearFromFilter:

    def test_year_from_keeps_newer(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Old Book", year=2015)
        _insert_book(conn, 2, "New Book", year=2022)

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"year_from": 2020},
        )
        assert len(results) == 1
        assert results[0]["title"] == "New Book"
        conn.close()

    def test_year_from_exact_boundary(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Exact Year", year=2020)

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"year_from": 2020},
        )
        assert len(results) == 1
        conn.close()

    def test_null_year_excluded(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "No Year Book", year=None)
        _insert_book(conn, 2, "2021 Book",    year=2021)

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"year_from": 2020},
        )
        assert len(results) == 1
        assert results[0]["title"] == "2021 Book"
        conn.close()


# ─── aggregate_to_books: format filter ───────────────────────────────────────

class TestFormatFilter:

    def test_single_format(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "EPUB Book", file_format="epub")
        _insert_book(conn, 2, "PDF Book",  file_format="pdf")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"format": "epub"},
        )
        assert len(results) == 1
        assert results[0]["title"] == "EPUB Book"
        conn.close()

    def test_multiple_formats(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "EPUB Book", file_format="epub")
        _insert_book(conn, 2, "PDF Book",  file_format="pdf")
        _insert_book(conn, 3, "DJVU Book", file_format="djvu")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"format": "epub,pdf"},
        )
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"EPUB Book", "PDF Book"}
        conn.close()

    def test_format_case_insensitive(self, tmp_path):
        """LOWER() в SQL: формат 'PDF' в БД матчится фильтром 'pdf'."""
        _, conn = _make_db(tmp_path)
        conn.execute(
            """INSERT INTO books (catalog_id, title, file_format,
               xlsx_row_hash, status, text_simhash)
               VALUES (77, 'Uppercase PDF', 'PDF', 'h77', 'imported', 0)"""
        )
        book_id = conn.execute("SELECT id FROM books WHERE catalog_id=77").fetchone()[0]
        cur = conn.execute(
            "INSERT INTO chunks (book_id, chunk_kind, chunk_index, text, text_hash, char_count) VALUES (?,?,?,?,?,?)",
            (book_id, "title", 0, "Uppercase PDF", "hh77", 12),
        )
        conn.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (cur.lastrowid, bytes(384 * 4)),
        )
        conn.commit()

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10, filters={"format": "pdf"}
        )
        assert len(results) == 1
        conn.close()

    def test_format_with_spaces_trimmed(self, tmp_path):
        """Пробелы вокруг форматов в 'epub, pdf' обрезаются."""
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "EPUB Book", file_format="epub")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"format": " epub , pdf "},
        )
        assert len(results) == 1
        conn.close()


# ─── aggregate_to_books: combined filters ────────────────────────────────────

class TestCombinedFilters:

    def test_section_plus_year(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Match",     section="Tech", year=2022, file_format="epub")
        _insert_book(conn, 2, "Old",       section="Tech", year=2015, file_format="epub")
        _insert_book(conn, 3, "Wrong Sec", section="Art",  year=2022, file_format="epub")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"section": "Tech", "year_from": 2020},
        )
        assert len(results) == 1
        assert results[0]["title"] == "Match"
        conn.close()

    def test_all_three_filters(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Match",     section="Tech", year=2022, file_format="epub")
        _insert_book(conn, 2, "Wrong Fmt", section="Tech", year=2022, file_format="pdf")
        _insert_book(conn, 3, "Old",       section="Tech", year=2015, file_format="epub")

        results = aggregate_to_books(
            _all_hits(conn), conn, top_k=10,
            filters={"section": "Tech", "year_from": 2020, "format": "epub"},
        )
        assert len(results) == 1
        assert results[0]["title"] == "Match"
        conn.close()

    def test_filters_dont_affect_score_ordering(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Low Score",  section="Tech", year=2022, file_format="pdf")
        _insert_book(conn, 2, "High Score", section="Tech", year=2022, file_format="pdf")

        # High Score получает скор 2.0, Low — 1.0
        hits = [
            (_all_hits(conn)[1][0], 2.0),  # chunk for book 2
            (_all_hits(conn)[0][0], 1.0),  # chunk for book 1
        ]
        results = aggregate_to_books(
            hits, conn, top_k=10, filters={"section": "Tech"},
        )
        assert results[0]["title"] == "High Score"
        conn.close()


# ─── book lookup logic ────────────────────────────────────────────────────────

class TestBookLookup:

    def test_book_found_by_catalog_id(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 42, "Special Book", section="Tech", year=2021, file_format="epub")
        conn.execute(
            "UPDATE books SET author='Alice', publisher='Pub' WHERE catalog_id=42"
        )
        conn.commit()

        row = conn.execute(
            "SELECT catalog_id, title, author FROM books WHERE catalog_id=42"
        ).fetchone()
        assert row["catalog_id"] == 42
        assert row["title"] == "Special Book"
        assert row["author"] == "Alice"
        conn.close()

    def test_book_not_found_returns_none(self, tmp_path):
        _, conn = _make_db(tmp_path)
        row = conn.execute(
            "SELECT * FROM books WHERE catalog_id=99999"
        ).fetchone()
        assert row is None
        conn.close()

    def test_file_path_construction(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Test", file_format="pdf")
        conn.execute(
            "UPDATE books SET folder='/books/tech', filename='test.pdf' WHERE catalog_id=1"
        )
        conn.commit()

        row = conn.execute(
            "SELECT folder, filename FROM books WHERE catalog_id=1"
        ).fetchone()
        file_path = str(Path(row["folder"]) / row["filename"])
        assert file_path.endswith("test.pdf")
        assert "books" in file_path
        conn.close()


# ─── categories query ─────────────────────────────────────────────────────────

class TestCategoriesQuery:

    def test_sections_grouped_correctly(self, tmp_path):
        _, conn = _make_db(tmp_path)
        _insert_book(conn, 1, "Python",  section="Программирование")
        _insert_book(conn, 2, "Django",  section="Программирование")
        _insert_book(conn, 3, "History", section="История")
        _insert_book(conn, 4, "No Sec",  section=None)

        rows = conn.execute(
            """SELECT section, COUNT(*) AS n FROM books
               WHERE status='imported' AND section IS NOT NULL
               GROUP BY section ORDER BY section"""
        ).fetchall()
        conn.close()

        sections = {r[0]: r[1] for r in rows}
        assert sections["Программирование"] == 2
        assert sections["История"] == 1
        assert None not in sections

    def test_total_books_count(self, tmp_path):
        _, conn = _make_db(tmp_path)
        for i in range(5):
            _insert_book(conn, i + 1, f"Book {i}", section="Tech")

        total = conn.execute(
            "SELECT COUNT(*) FROM books WHERE status='imported'"
        ).fetchone()[0]
        conn.close()
        assert total == 5
