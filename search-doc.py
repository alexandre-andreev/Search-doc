#!/usr/bin/env python
"""search-doc — локальный семантический поиск по библиотеке книг."""

import json
import os
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_DB = Path("cache/semantic_index.sqlite")
DEFAULT_CATALOG = Path("data/catalog.xlsx")
LOG_PATH = Path("cache/search-doc.log")


def _require_db(fmt: str = "text") -> None:
    """Выходит с кодом 2 если БД не существует."""
    if not DEFAULT_DB.exists():
        msg = "БД не найдена. Сначала запустите: python search-doc.py import"
        if fmt == "json":
            click.echo(json.dumps({"error": msg, "schema_version": "1.0"}))
        else:
            click.echo(f"Ошибка: {msg}", err=True)
        sys.exit(2)


def _open_db_safe():
    """Открывает БД с внятной диагностикой вместо stacktrace."""
    from src.index.db import open_db
    try:
        return open_db(DEFAULT_DB)
    except Exception as exc:
        click.echo(f"Ошибка открытия БД: {exc}", err=True)
        sys.exit(2)


# ─── cli group ────────────────────────────────────────────────────────────────

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Подробный вывод")
@click.pass_context
def cli(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    from src.util.logging import setup_logging
    setup_logging(LOG_PATH, verbose=verbose)


# ─── import ──────────────────────────────────────────────────────────────────

@cli.command("import")
@click.option("--catalog", default=str(DEFAULT_CATALOG), show_default=True,
              help="Путь к catalog.xlsx")
@click.option("--rebuild", is_flag=True,
              help="Принудительная полная переиндексация (удалить старую БД)")
@click.pass_context
def cmd_import(ctx, catalog, rebuild):
    """Импортировать каталог книг в SQLite и построить индекс."""
    from src.pipeline.indexer import run_import

    catalog_path = Path(catalog)
    if not catalog_path.exists():
        click.echo(f"Ошибка: каталог не найден: {catalog_path}", err=True)
        sys.exit(2)

    try:
        stats = run_import(catalog_path, DEFAULT_DB, rebuild=rebuild)
    except RuntimeError as exc:
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(2)
    except KeyboardInterrupt:
        click.echo("\nПрервано.", err=True)
        sys.exit(130)

    added = stats["books_added"]
    updated = stats.get("books_updated", 0)
    skipped = stats.get("books_skipped", 0)
    removed = stats.get("books_removed", 0)

    click.echo(
        f"\nДобавлено: {added}, обновлено: {updated}, "
        f"пропущено: {skipped}, удалено: {removed}"
    )
    click.echo(
        f"Создано {stats['chunks_created']} чанков "
        f"(title: {stats['n_title']}, summary: {stats['n_summary']}), "
        f"время {stats['elapsed_sec']:.1f} сек"
    )
    if stats["books_failed"]:
        click.echo(f"Ошибок: {stats['books_failed']}", err=True)
        sys.exit(3)


# ─── status ──────────────────────────────────────────────────────────────────

@cli.command("status")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]),
              show_default=True)
@click.pass_context
def cmd_status(ctx, fmt):
    """Показать состояние индекса."""
    _require_db(fmt)
    conn = _open_db_safe()

    from src.index.db import get_meta

    n_books = conn.execute("SELECT COUNT(*) FROM books WHERE status='imported'").fetchone()[0]
    n_dups = conn.execute("SELECT COUNT(*) FROM books WHERE duplicate_of IS NOT NULL").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_title = conn.execute("SELECT COUNT(*) FROM chunks WHERE chunk_kind='title'").fetchone()[0]
    n_summary = conn.execute("SELECT COUNT(*) FROM chunks WHERE chunk_kind='summary'").fetchone()[0]
    n_vectors = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]

    model = get_meta(conn, "embedding_model", "—")
    dim = get_meta(conn, "embedding_dim", "—")
    schema_ver = get_meta(conn, "schema_version", "—")
    last_indexed = get_meta(conn, "last_indexed_at", None)
    conn.close()

    if fmt == "json":
        click.echo(json.dumps({
            "books": n_books,
            "duplicates_marked": n_dups,
            "chunks": n_chunks,
            "chunks_title": n_title,
            "chunks_summary": n_summary,
            "vectors": n_vectors,
            "embedding_model": model,
            "embedding_dim": dim,
            "schema_version": schema_ver,
            "last_indexed_at": last_indexed,
        }, ensure_ascii=False, indent=2))
    else:
        click.echo(f"Книг:        {n_books} (помечено дублей: {n_dups})")
        click.echo(f"Чанков:      {n_chunks} (title: {n_title}, summary: {n_summary})")
        click.echo(f"Векторов:    {n_vectors}")
        click.echo(f"Модель:      {model}, dim: {dim}")
        click.echo(f"Схема:       v{schema_ver}")
        if last_indexed:
            import datetime
            ts = datetime.datetime.fromtimestamp(float(last_indexed))
            click.echo(f"Индексировано: {ts.strftime('%Y-%m-%d %H:%M:%S')}")


# ─── dedup ───────────────────────────────────────────────────────────────────

@cli.command("dedup")
@click.pass_context
def cmd_dedup(ctx):
    """Найти и пометить дублирующиеся книги по simhash саммари."""
    _require_db()
    conn = _open_db_safe()

    from src.index.dedup import HAMMING_THRESHOLD, run_dedup

    click.echo(f"Запускаю дедупликацию (порог Хэмминга <= {HAMMING_THRESHOLD})...")
    stats = run_dedup(conn)
    conn.close()

    if stats["backfilled"]:
        click.echo(f"  Вычислено simhash: {stats['backfilled']} книг")
    click.echo(f"  Групп дубликатов:  {stats['groups_found']}")
    click.echo(f"    из них по title: {stats['title_groups_found']}")
    click.echo(f"  Помечено дублей:   {stats['duplicates_marked']}")

    if stats["duplicates_marked"] == 0:
        click.echo("Дубликатов не найдено.")


# ─── categories ──────────────────────────────────────────────────────────────

@cli.command("categories")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]),
              show_default=True)
@click.pass_context
def cmd_categories(ctx, fmt):
    """Список разделов и подразделов каталога."""
    _require_db(fmt)
    conn = _open_db_safe()

    rows = conn.execute(
        """
        SELECT section, subsection, COUNT(*) AS n
        FROM books
        WHERE status = 'imported' AND section IS NOT NULL
        GROUP BY section, subsection
        ORDER BY section, subsection
        """
    ).fetchall()

    total = conn.execute(
        "SELECT COUNT(*) FROM books WHERE status='imported'"
    ).fetchone()[0]
    conn.close()

    # Группируем по section
    from collections import defaultdict
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append({"name": row[1], "count": row[2]})

    sections = [
        {
            "section": sec,
            "count": sum(s["count"] for s in subs),
            "subsections": subs,
        }
        for sec, subs in sorted(grouped.items())
    ]

    if fmt == "json":
        click.echo(json.dumps(
            {"schema_version": "1.0", "total_books": total, "sections": sections},
            ensure_ascii=False, indent=2,
        ))
    else:
        click.echo(f"Всего книг: {total}\n")
        for s in sections:
            click.echo(f"{s['section']}  ({s['count']})")
            for sub in s["subsections"]:
                name = sub["name"] or "(без подраздела)"
                click.echo(f"  {name} — {sub['count']}")


# ─── book ────────────────────────────────────────────────────────────────────

@cli.command("book")
@click.argument("catalog_id", type=int)
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "text"]),
              show_default=True)
@click.pass_context
def cmd_book(ctx, catalog_id, fmt):
    """Показать детали книги по catalog_id (№ из каталога)."""
    _require_db(fmt)
    conn = _open_db_safe()

    row = conn.execute(
        """
        SELECT id, catalog_id, title, author, year, publisher,
               category, section, subsection,
               file_format, file_size_mb, filename, folder, summary,
               status, duplicate_of, indexed_at
        FROM books WHERE catalog_id = ?
        """,
        (catalog_id,),
    ).fetchone()

    if row is None:
        conn.close()
        msg = f"Книга с catalog_id={catalog_id} не найдена в индексе."
        if fmt == "json":
            click.echo(json.dumps({"error": msg, "schema_version": "1.0"}))
        else:
            click.echo(f"Ошибка: {msg}", err=True)
        sys.exit(2)

    folder = row["folder"] or ""
    filename = row["filename"] or ""
    file_path = str(Path(folder) / filename) if folder and filename else filename or folder

    n_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE book_id = ?", (row["id"],)
    ).fetchone()[0]
    conn.close()

    data = {
        "schema_version": "1.0",
        "book_id": row["id"],
        "catalog_id": row["catalog_id"],
        "title": row["title"],
        "author": row["author"],
        "year": row["year"],
        "publisher": row["publisher"],
        "category": row["category"],
        "section": row["section"],
        "subsection": row["subsection"],
        "file_format": row["file_format"],
        "file_size_mb": row["file_size_mb"],
        "file_path": file_path,
        "summary": row["summary"],
        "status": row["status"],
        "duplicate_of": row["duplicate_of"],
        "chunks_indexed": n_chunks,
    }

    if fmt == "json":
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        click.echo(f"#{data['catalog_id']}  {data['title']}")
        if data.get("author"):
            click.echo(f"  Автор:    {data['author']}")
        if data.get("year"):
            click.echo(f"  Год:      {data['year']}")
        if data.get("category"):
            click.echo(f"  Раздел:   {data['category']}")
        click.echo(f"  Формат:   {data.get('file_format', '—')}")
        click.echo(f"  Размер:   {data.get('file_size_mb') or '—'} МБ")
        click.echo(f"  Путь:     {data['file_path']}")
        click.echo(f"  Статус:   {data['status']}")
        if data.get("duplicate_of"):
            click.echo(f"  Дубль:    canonical book_id={data['duplicate_of']}")
        click.echo(f"  Чанков:   {data['chunks_indexed']}")
        if data.get("summary"):
            snippet = data["summary"][:300].replace("\n", " ")
            click.echo(f"  Саммари:  {snippet}…")


# ─── open ────────────────────────────────────────────────────────────────────

@cli.command("open")
@click.argument("catalog_id", type=int)
@click.pass_context
def cmd_open(ctx, catalog_id):
    """Открыть файл книги системным просмотрщиком."""
    _require_db()
    conn = _open_db_safe()

    row = conn.execute(
        "SELECT folder, filename, title FROM books WHERE catalog_id = ?",
        (catalog_id,),
    ).fetchone()
    conn.close()

    if row is None:
        click.echo(f"Ошибка: книга с catalog_id={catalog_id} не найдена.", err=True)
        sys.exit(2)

    folder = row["folder"] or ""
    filename = row["filename"] or ""

    if not filename:
        click.echo(f"Ошибка: у книги «{row['title']}» не задано имя файла.", err=True)
        sys.exit(2)

    file_path = Path(folder) / filename if folder else Path(filename)

    if not file_path.exists():
        click.echo(f"Ошибка: файл не найден: {file_path}", err=True)
        sys.exit(2)

    try:
        os.startfile(str(file_path))
        click.echo(f"Открываю: {file_path}")
    except Exception as exc:
        click.echo(f"Ошибка открытия файла: {exc}", err=True)
        sys.exit(2)


# ─── search ──────────────────────────────────────────────────────────────────

@cli.command("search")
@click.argument("query")
@click.option("--top", default=10, show_default=True, help="Количество результатов")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "text"]),
              show_default=True)
@click.option("--no-related", is_flag=True, help="Не вычислять related_books")
@click.option("--semantic-only", is_flag=True, help="Только семантический поиск (без FTS)")
@click.option("--fts-only", is_flag=True, help="Только FTS-поиск (без эмбеддингов, без related)")
@click.option("--section", default=None, help='Фильтр по разделу, напр. "Программирование"')
@click.option("--year-from", "year_from", default=None, type=int,
              help="Фильтр: книги не старше N года")
@click.option("--file-format", "file_format", default=None,
              help='Фильтр по формату файла, напр. "pdf" или "pdf,epub"')
@click.pass_context
def cmd_search(ctx, query, top, fmt, no_related, semantic_only, fts_only,
               section, year_from, file_format):
    """Семантический поиск по библиотеке книг."""
    _require_db(fmt)

    from src.index.db import get_meta, open_db
    from src.search.fts import fts_search
    from src.search.ranker import aggregate_to_books, is_technical_query, rrf_fuse

    filters = {
        "section": section,
        "year_from": year_from,
        "format": file_format,
    } if (section or year_from or file_format) else None

    conn = _open_db_safe()
    embedding_model = get_meta(conn, "embedding_model", "intfloat/multilingual-e5-small")

    t0 = time.perf_counter()

    if fts_only:
        # Быстрый путь: эмбеддер не нужен
        fts_hits = fts_search(conn, query, k=50)
        fused_hits = fts_hits
        w_sem, w_fts = 0.0, 1.0
        strategy = "fts"
        raw_fts_scores = {cid: score for cid, score in fts_hits}
        raw_sem_scores: dict = {}
        can_compute_related = False
    else:
        from src.embedder.e5_small import E5SmallEmbedder
        from src.search.semantic import semantic_search

        try:
            embedder = E5SmallEmbedder()
        except RuntimeError as exc:
            click.echo(f"Ошибка: {exc}", err=True)
            sys.exit(2)

        query_vec = embedder.encode_query(query)
        sem_hits = semantic_search(conn, query_vec, k=50)
        raw_sem_scores = {cid: score for cid, score in sem_hits}

        if semantic_only:
            fused_hits = sem_hits
            w_sem, w_fts = 1.0, 0.0
            strategy = "semantic"
            raw_fts_scores = None
        else:
            fts_hits = fts_search(conn, query, k=50)
            technical = is_technical_query(query)
            fused_hits, (w_sem, w_fts) = rrf_fuse(sem_hits, fts_hits, is_technical=technical)
            strategy = "hybrid"
            raw_fts_scores = {cid: score for cid, score in fts_hits}

        can_compute_related = True

    results = aggregate_to_books(
        fused_hits, conn, top_k=top,
        raw_semantic=raw_sem_scores,
        raw_fts=raw_fts_scores,
        filters=filters,
    )

    for r in results:
        r.pop("_duplicate_of", None)

    related_books: list[dict] = []
    if results and not no_related and can_compute_related:
        from src.search.related import build_related_payload, find_related
        top1_book_id = results[0]["book_id"]
        main_book_ids = {r["book_id"] for r in results}
        related_hits = find_related(conn, top1_book_id, k=15, exclude_book_ids=main_book_ids)
        related_books = build_related_payload(conn, related_hits, max_results=5)

    search_time_ms = round((time.perf_counter() - t0) * 1000)
    conn.close()

    payload = {
        "schema_version": "1.0",
        "query": query,
        "search_strategy": strategy,
        "weights": {"semantic": round(w_sem, 2), "fts": round(w_fts, 2)},
        "embedding_model": embedding_model,
        "filters_applied": {
            "section": section,
            "year_from": year_from,
            "format": file_format,
        },
        "search_time_ms": search_time_ms,
        "results": [
            {**r, "rank": i + 1}
            for i, r in enumerate(results)
        ],
        "related_books": related_books,
    }

    if fmt == "json":
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_text_results(payload)

    if not results:
        sys.exit(1)


def _print_text_results(payload: dict) -> None:
    """Человекочитаемый вывод результатов поиска."""
    results = payload["results"]
    filters = payload.get("filters_applied", {})

    click.echo(f"\nЗапрос: {payload['query']!r}  [{payload['search_strategy']}]")
    active_filters = {k: v for k, v in filters.items() if v is not None}
    if active_filters:
        click.echo(f"Фильтры: {active_filters}")
    click.echo(f"Найдено: {len(results)}, время: {payload['search_time_ms']} мс")
    click.echo("=" * 70)

    for r in results:
        click.echo(f"\n#{r['rank']}  [{r['score']:.3f}]  {r['title']}")
        if r.get("author"):
            click.echo(f"     Автор:   {r['author']}")
        if r.get("year"):
            click.echo(f"     Год:     {r['year']}")
        if r.get("category"):
            click.echo(f"     Раздел:  {r['category']}")
        click.echo(f"     Формат:  {r.get('file_format', '—')}")
        click.echo(f"     Путь:    {r.get('file_path', '—')}")
        click.echo(f"     Совпало: {', '.join(r.get('matched_in', []))}")
        for mc in r.get("matched_chunks", [])[:1]:
            snippet = mc["text"][:200].replace("\n", " ")
            click.echo(f"     Сниппет: {snippet}...")

    if payload.get("related_books"):
        click.echo("\n" + "-" * 70)
        click.echo("ПОХОЖИЕ КНИГИ (related):")
        for r in payload["related_books"]:
            click.echo(f"  #{r['rank']}  [{r.get('similarity_to_top_result', 0):.3f}]  {r['title']}")


if __name__ == "__main__":
    cli()
