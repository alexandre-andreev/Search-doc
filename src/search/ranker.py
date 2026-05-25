from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

KIND_BONUS: dict[str, float] = {
    "title": 1.5,
    "summary": 1.3,
    "body": 1.0,
}

CHUNK_TEXT_LIMIT = 500

# Из project_spec.yaml
TECHNICAL_WHITELIST: frozenset[str] = frozenset({
    "pytest", "junit", "tdd", "ddd", "rest", "grpc", "sql", "css", "html",
    "http", "jwt", "api", "sdk", "cli", "solid", "llm", "gpt", "rag", "nlp",
    "cnn", "ml", "ai", "go", "rust", "java", "swift", "kotlin", "ruby", "php",
    "bash", "shell", "linux", "docker", "kubernetes", "k8s", "terraform",
    "ansible", "jenkins", "aws", "gcp", "azure", "postgres", "postgresql",
    "mysql", "sqlite", "redis", "mongodb", "kafka", "react", "vue", "angular",
    "django", "flask", "fastapi", "spring", "express", "numpy", "pandas",
    "tensorflow", "pytorch", "keras",
})

# RRF-константы (асимметричные: приоритет семантике)
RRF_K_SEMANTIC = 60
RRF_K_FTS = 200

WEIGHTS_DESCRIPTIVE = (0.85, 0.15)
WEIGHTS_TECHNICAL = (0.40, 0.60)


def is_technical_query(query: str) -> bool:
    """
    Возвращает True, если запрос содержит технические токены.

    Токен считается техническим если:
    1. Его lowercase-версия входит в TECHNICAL_WHITELIST, или
    2. Он короткий (≤8 символов), весь в верхнем регистре (как LLM, API), или
    3. Он CamelCase (как JavaScript, TypeScript).
    """
    tokens = re.findall(r'\b[A-Za-z0-9_]+\b', query)
    for token in tokens:
        if token.lower() in TECHNICAL_WHITELIST:
            return True
        if len(token) <= 8 and len(token) >= 2 and token.isupper():
            return True
        if len(token) <= 8 and re.match(r'^[A-Z][a-z]+[A-Z]', token):
            return True
    return False


def rrf_fuse(
    semantic_hits: list[tuple[int, float]],
    fts_hits: list[tuple[int, float]],
    is_technical: bool,
    k_sem: int = RRF_K_SEMANTIC,
    k_fts: int = RRF_K_FTS,
) -> tuple[list[tuple[int, float]], tuple[float, float]]:
    """
    Reciprocal Rank Fusion.

    score = w_sem * 1/(k_sem + rank_sem) + w_fts * 1/(k_fts + rank_fts)

    Возвращает:
      fused_hits: [(chunk_id, rrf_score)], отсортированные по убыванию
      weights: (w_semantic, w_fts) — использованные веса
    """
    w_sem, w_fts = WEIGHTS_TECHNICAL if is_technical else WEIGHTS_DESCRIPTIVE

    sem_rank = {cid: i + 1 for i, (cid, _) in enumerate(semantic_hits)}
    fts_rank = {cid: i + 1 for i, (cid, _) in enumerate(fts_hits)}

    all_ids = set(sem_rank) | set(fts_rank)

    scores: dict[int, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in sem_rank:
            score += w_sem / (k_sem + sem_rank[cid])
        if cid in fts_rank:
            score += w_fts / (k_fts + fts_rank[cid])
        scores[cid] = score

    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused, (w_sem, w_fts)


def aggregate_to_books(
    hits: list[tuple[int, float]],
    conn: sqlite3.Connection,
    top_k: int = 10,
    raw_semantic: dict[int, float] | None = None,
    raw_fts: dict[int, float] | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """
    Агрегирует чанки в книги:
        book_score = max(chunk_score * kind_bonus) + log(1 + N) * 0.1

    raw_semantic / raw_fts — опциональные словари сырых оценок для отображения
    в matched_chunks (не влияют на ранжирование).

    filters — опциональный словарь {section, year_from, format} для фильтрации результатов.
    """
    if not hits:
        return []

    chunk_ids = [h[0] for h in hits]
    score_map = {h[0]: h[1] for h in hits}

    placeholders = ",".join("?" * len(chunk_ids))

    where_parts = [f"c.id IN ({placeholders})"]
    params: list = list(chunk_ids)

    if filters:
        if filters.get("section"):
            where_parts.append("b.section = ?")
            params.append(filters["section"])
        if filters.get("year_from") is not None:
            where_parts.append("(b.year IS NOT NULL AND b.year >= ?)")
            params.append(int(filters["year_from"]))
        if filters.get("format"):
            fmts = [f.strip().lower() for f in str(filters["format"]).split(",") if f.strip()]
            if fmts:
                ph = ",".join("?" * len(fmts))
                where_parts.append(f"LOWER(COALESCE(b.file_format, '')) IN ({ph})")
                params.extend(fmts)

    rows = conn.execute(
        f"""
        SELECT
            c.id        AS chunk_id,
            c.book_id,
            c.chunk_kind,
            c.text,
            b.catalog_id,
            b.title,
            b.author,
            b.year,
            b.publisher,
            b.category,
            b.section,
            b.subsection,
            b.file_format,
            b.file_size_mb,
            b.filename,
            b.folder,
            b.summary,
            b.duplicate_of
        FROM chunks c
        JOIN books b ON c.book_id = b.id
        WHERE {" AND ".join(where_parts)}
        """,
        params,
    ).fetchall()

    book_data: dict[int, dict] = {}
    for row in rows:
        chunk_id = row["chunk_id"]
        book_id = row["book_id"]
        raw_score = score_map.get(chunk_id, 0.0)
        kind = row["chunk_kind"]
        bonused = raw_score * KIND_BONUS.get(kind, 1.0)

        if book_id not in book_data:
            folder = row["folder"] or ""
            filename = row["filename"] or ""
            file_path = str(Path(folder) / filename) if folder and filename else filename or folder

            book_data[book_id] = {
                "book_id": book_id,
                "catalog_id": row["catalog_id"],
                "title": row["title"],
                "author": row["author"],
                "year": row["year"],
                "publisher": row["publisher"],
                "category": row["category"],
                "section": row["section"],
                "subsection": row["subsection"],
                "file_format": row["file_format"],
                "file_path": file_path,
                "filename": filename,
                "summary": row["summary"],
                "duplicate_of": row["duplicate_of"],
                "chunks": [],
            }

        book_data[book_id]["chunks"].append({
            "chunk_id": chunk_id,
            "kind": kind,
            "text": row["text"],
            "raw_score": raw_score,
            "bonused_score": bonused,
        })

    results: list[dict] = []
    for book_id, data in book_data.items():
        chunks = data["chunks"]
        max_bonused = max(c["bonused_score"] for c in chunks)
        n_matched = len(chunks)
        final_score = max_bonused + math.log(1 + n_matched) * 0.1

        kinds_matched = sorted(set(c["kind"] for c in chunks))

        matched_chunks = []
        for c in sorted(chunks, key=lambda x: -x["raw_score"]):
            cid = c["chunk_id"]
            sem_score = raw_semantic.get(cid) if raw_semantic else c["raw_score"]
            fts_score = raw_fts.get(cid) if raw_fts else None
            matched_chunks.append({
                "kind": c["kind"],
                "text": c["text"][:CHUNK_TEXT_LIMIT],
                "semantic_score": round(sem_score, 3) if sem_score is not None else None,
                "fts_score": round(fts_score, 3) if fts_score is not None else None,
            })

        results.append({
            "book_id": book_id,
            "catalog_id": data["catalog_id"],
            "title": data["title"],
            "author": data["author"],
            "year": data["year"],
            "publisher": data["publisher"],
            "category": data["category"],
            "section": data["section"],
            "subsection": data["subsection"],
            "file_format": data["file_format"],
            "file_path": data["file_path"],
            "filename": data["filename"],
            "score": round(final_score, 3),
            "matched_in": kinds_matched,
            "summary": data["summary"],
            "matched_chunks": matched_chunks,
            "duplicates": [],
            "_duplicate_of": data["duplicate_of"],
        })

    results.sort(key=lambda r: -r["score"])

    # Фильтрация дубликатов и сбор поля duplicates
    canonical = [r for r in results if r["_duplicate_of"] is None]
    dup_in_results = [r for r in results if r["_duplicate_of"] is not None]

    # Пути дубликатов из текущей выдачи
    dup_paths: dict[int, list[str]] = defaultdict(list)
    for r in dup_in_results:
        canon_id = r["_duplicate_of"]
        if r["file_path"]:
            dup_paths[canon_id].append(r["file_path"])

    # Пути дубликатов из БД (книги, не попавшие в поиск)
    canon_ids = [r["book_id"] for r in canonical]
    if canon_ids:
        ph = ",".join("?" * len(canon_ids))
        for db_row in conn.execute(
            f"SELECT duplicate_of, filename, folder FROM books WHERE duplicate_of IN ({ph})",
            canon_ids,
        ).fetchall():
            canon_id, fname, fdir = db_row
            fdir = fdir or ""
            fname = fname or ""
            fp = str(Path(fdir) / fname) if fdir and fname else fname or fdir
            if fp:
                dup_paths[canon_id].append(fp)

    for r in canonical:
        seen = set()
        r["duplicates"] = [p for p in dup_paths.get(r["book_id"], []) if not (p in seen or seen.add(p))]

    return canonical[:top_k]
