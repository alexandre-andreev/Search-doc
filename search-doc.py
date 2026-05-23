#!/usr/bin/env python
"""search-doc — локальный семантический поиск по библиотеке книг."""

import json
import sys
import time
from pathlib import Path

import click

# Добавляем корень проекта в sys.path, чтобы импорты src.* работали
sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_DB = Path("cache/semantic_index.sqlite")
DEFAULT_CATALOG = Path("data/catalog.xlsx")
LOG_PATH = Path("cache/search-doc.log")


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

    click.echo(
        f"\nИмпортировано {stats['books_added']} книг, "
        f"создано {stats['chunks_created']} чанков "
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
    if not DEFAULT_DB.exists():
        msg = "БД не найдена. Сначала запустите: python search-doc.py import"
        if fmt == "json":
            click.echo(json.dumps({"error": msg}))
        else:
            click.echo(f"Ошибка: {msg}", err=True)
        sys.exit(2)

    from src.index.db import get_meta, open_db

    try:
        conn = open_db(DEFAULT_DB)
    except Exception as exc:
        click.echo(f"Ошибка открытия БД: {exc}", err=True)
        sys.exit(2)

    n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
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
        click.echo(f"Книг:      {n_books}")
        click.echo(f"Чанков:    {n_chunks} (title: {n_title}, summary: {n_summary})")
        click.echo(f"Векторов:  {n_vectors}")
        click.echo(f"Модель:    {model}, dim: {dim}")
        click.echo(f"Схема:     v{schema_ver}")
        if last_indexed:
            import datetime
            ts = datetime.datetime.fromtimestamp(float(last_indexed))
            click.echo(f"Индексировано: {ts.strftime('%Y-%m-%d %H:%M:%S')}")


# ─── search ──────────────────────────────────────────────────────────────────

@cli.command("search")
@click.argument("query")
@click.option("--top", default=10, show_default=True, help="Количество результатов")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "text"]),
              show_default=True)
@click.option("--no-related", is_flag=True, help="Не вычислять related_books")
@click.option("--semantic-only", is_flag=True, help="Только семантический поиск (без FTS)")
@click.pass_context
def cmd_search(ctx, query, top, fmt, no_related, semantic_only):
    """Семантический поиск по библиотеке книг."""
    if not DEFAULT_DB.exists():
        msg = "БД не найдена. Сначала запустите: python search-doc.py import"
        if fmt == "json":
            click.echo(json.dumps({"error": msg, "schema_version": "1.0"}))
        else:
            click.echo(f"Ошибка: {msg}", err=True)
        sys.exit(2)

    from src.embedder.e5_small import E5SmallEmbedder
    from src.index.db import get_meta, open_db
    from src.search.fts import fts_search
    from src.search.ranker import aggregate_to_books, is_technical_query, rrf_fuse
    from src.search.semantic import semantic_search

    try:
        conn = open_db(DEFAULT_DB)
        embedder = E5SmallEmbedder()
    except RuntimeError as exc:
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(2)

    embedding_model = get_meta(conn, "embedding_model", "intfloat/multilingual-e5-small")

    t0 = time.perf_counter()

    query_vec = embedder.encode_query(query)
    sem_hits = semantic_search(conn, query_vec, k=50)

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

    raw_sem_scores = {cid: score for cid, score in sem_hits}
    results = aggregate_to_books(
        fused_hits, conn, top_k=top,
        raw_semantic=raw_sem_scores,
        raw_fts=raw_fts_scores,
    )

    # Убираем служебное поле _duplicate_of из вывода
    for r in results:
        r.pop("_duplicate_of", None)

    search_time_ms = round((time.perf_counter() - t0) * 1000)

    # related_books — заглушка до Этапа 4
    related_books: list[dict] = []

    conn.close()

    payload = {
        "schema_version": "1.0",
        "query": query,
        "search_strategy": strategy,
        "weights": {"semantic": round(w_sem, 2), "fts": round(w_fts, 2)},
        "embedding_model": embedding_model,
        "filters_applied": {"section": None, "year_from": None, "format": None},
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
    click.echo(f"\nЗапрос: {payload['query']!r}")
    click.echo(f"Найдено: {len(results)}, время: {payload['search_time_ms']} мс")
    click.echo("=" * 70)

    for r in results:
        click.echo(
            f"\n#{r['rank']}  [{r['score']:.3f}]  {r['title']}"
        )
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
