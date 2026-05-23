import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CatalogBook:
    catalog_id: int
    title: str
    author: str | None
    year: int | None
    publisher: str | None
    category: str | None
    section: str | None
    subsection: str | None
    file_format: str | None
    file_size_mb: float | None
    filename: str
    folder: str
    summary: str | None


@dataclass
class IndexChunk:
    text: str
    book_idx: int   # индекс в списке books
    kind: str       # 'title' | 'summary'


def safe_int(value, field_name: str, row_idx: int, warnings: list[str]) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value) if not np.isnan(value) else None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    match = re.search(r"\d{4}", s)
    if match:
        return int(match.group())
    warnings.append(f"  строка {row_idx}: поле '{field_name}' = {value!r} не распознано как число")
    return None


def safe_str(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def load_catalog(path: Path) -> list[CatalogBook]:
    if not path.exists():
        print(f"Каталог не найден: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(path, sheet_name="Каталог")

    required = ["№", "Название", "Имя файла"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"В каталоге отсутствуют обязательные колонки: {missing}", file=sys.stderr)
        sys.exit(1)

    books: list[CatalogBook] = []
    warnings: list[str] = []
    skipped = 0

    for row_idx, row in df.iterrows():
        catalog_id = safe_int(row.get("№"), "№", row_idx, warnings)
        title = safe_str(row.get("Название"))
        filename = safe_str(row.get("Имя файла"))

        if catalog_id is None or not title or not filename:
            skipped += 1
            continue

        category = safe_str(row.get("Категория"))
        section = subsection = None
        if category and "/" in category:
            section, subsection = category.split("/", 1)
        elif category:
            section = category

        books.append(CatalogBook(
            catalog_id=catalog_id,
            title=title,
            author=safe_str(row.get("Автор")),
            year=safe_int(row.get("Год"), "Год", row_idx, warnings),
            publisher=safe_str(row.get("Издательство")),
            category=category,
            section=section,
            subsection=subsection,
            file_format=safe_str(row.get("Формат")),
            file_size_mb=safe_float(row.get("Размер МБ")),
            filename=filename,
            folder=safe_str(row.get("Папка")) or "",
            summary=safe_str(row.get("Саммари")),
        ))

    if warnings:
        print(f"  Предупреждений при разборе: {len(warnings)}")
        for w in warnings[:10]:
            print(w)
        if len(warnings) > 10:
            print(f"  ... и ещё {len(warnings) - 10}")
    if skipped:
        print(f"  Пропущено строк без обязательных полей: {skipped}")

    return books


def build_chunks(books: list[CatalogBook]) -> list[IndexChunk]:
    chunks: list[IndexChunk] = []
    for idx, book in enumerate(books):
        author_part = f". {book.author}" if book.author else ""
        title_text = f"{book.title}{author_part}"
        chunks.append(IndexChunk(text=title_text, book_idx=idx, kind="title"))

        if book.summary and len(book.summary) >= 100:
            text = book.summary[:2000]
            chunks.append(IndexChunk(text=text, book_idx=idx, kind="summary"))

    return chunks
