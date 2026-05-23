"""
Eval-раннер для семантического поиска через SQLite-индекс.

Адаптация pre_mvp_eval.py: вместо NumPy-индекса читает из БД через sqlite-vec.
Логика evaluate_query, print_eval_report перенесена без изменений.

Использование:
    python eval/run_eval.py
    python eval/run_eval.py --verbose
    python eval/run_eval.py --eval eval/eval_queries_v2.yaml
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.embedder.e5_small import E5SmallEmbedder
from src.index.db import open_db
from src.search.fts import fts_search
from src.search.ranker import aggregate_to_books, is_technical_query, rrf_fuse
from src.search.semantic import semantic_search

DEFAULT_DB = Path("cache/semantic_index.sqlite")
DEFAULT_EVAL = Path("eval/eval_queries_v2.yaml")


# ─── Eval-типы и вспомогательные функции (перенос из pre_mvp_eval.py) ───────

@dataclass
class QueryEvalResult:
    id: str
    query: str
    notes: str
    expected_top_5: list[str]
    expected_top_10: list[str]
    found_top_5: list[str]
    found_top_10: list[str]
    missed_top_5: list[str]
    missed_top_10: list[str]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    results: list[dict]


def match_expectation(expected_substring: str, book_titles_and_filenames: list[str]) -> bool:
    needle = expected_substring.lower()
    return any(needle in s.lower() for s in book_titles_and_filenames)


def find_position_of_expectation(expected_substring: str, results: list[dict]) -> int | None:
    needle = expected_substring.lower()
    for i, r in enumerate(results, start=1):
        if needle in (r.get("title") or "").lower() or needle in (r.get("filename") or "").lower():
            return i
    return None


def evaluate_query(query_def: dict, results: list[dict]) -> QueryEvalResult:
    if "relevant_books" in query_def:
        return _evaluate_v2(query_def, results)
    return _evaluate_v1(query_def, results)


def _evaluate_v2(query_def: dict, results: list[dict]) -> QueryEvalResult:
    relevant = query_def.get("relevant_books", []) or []
    min_5 = int(query_def.get("min_in_top_5", 1))
    min_10 = int(query_def.get("min_in_top_10", 2))

    titles_top_5  = [f"{r['title']}|{r.get('filename','')}" for r in results[:5]]
    titles_top_10 = [f"{r['title']}|{r.get('filename','')}" for r in results[:10]]

    found_5  = [b for b in relevant if match_expectation(b, titles_top_5)]
    found_10 = [b for b in relevant if match_expectation(b, titles_top_10)]
    missed_5  = [b for b in relevant if b not in found_5]
    missed_10 = [b for b in relevant if b not in found_10]

    recall_5  = min(len(found_5),  min_5)  / min_5  if min_5  > 0 else 1.0
    recall_10 = min(len(found_10), min_10) / min_10 if min_10 > 0 else 1.0

    positions = []
    for b in relevant:
        pos = find_position_of_expectation(b, results)
        if pos is not None:
            positions.append(pos)
    mrr = 1.0 / min(positions) if positions else 0.0

    return QueryEvalResult(
        id=query_def["id"],
        query=query_def["query"],
        notes=query_def.get("notes", "") or "",
        expected_top_5=relevant,
        expected_top_10=[f"min {min_5}/{min_10}"],
        found_top_5=found_5,
        found_top_10=found_10,
        missed_top_5=missed_5,
        missed_top_10=missed_10,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        mrr=mrr,
        results=results,
    )


def _evaluate_v1(query_def: dict, results: list[dict]) -> QueryEvalResult:
    expected_5  = query_def.get("expected_in_top_5",  []) or []
    expected_10 = query_def.get("expected_in_top_10", []) or []

    titles_filenames_top_5  = [f"{r['title']}|{r.get('filename','')}" for r in results[:5]]
    titles_filenames_top_10 = [f"{r['title']}|{r.get('filename','')}" for r in results[:10]]

    found_5  = [e for e in expected_5  if match_expectation(e, titles_filenames_top_5)]
    missed_5 = [e for e in expected_5  if e not in found_5]

    all_expected_10 = list(set(expected_5) | set(expected_10))
    found_10  = [e for e in all_expected_10 if match_expectation(e, titles_filenames_top_10)]
    missed_10 = [e for e in all_expected_10 if e not in found_10]

    recall_5  = len(found_5)  / len(expected_5)       if expected_5       else 1.0
    recall_10 = len(found_10) / len(all_expected_10)  if all_expected_10  else 1.0

    positions = []
    for e in expected_5 + expected_10:
        pos = find_position_of_expectation(e, results)
        if pos is not None:
            positions.append(pos)
    mrr = 1.0 / min(positions) if positions else 0.0

    return QueryEvalResult(
        id=query_def["id"],
        query=query_def["query"],
        notes=query_def.get("notes", "") or "",
        expected_top_5=expected_5,
        expected_top_10=expected_10,
        found_top_5=found_5,
        found_top_10=found_10,
        missed_top_5=missed_5,
        missed_top_10=missed_10,
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        mrr=mrr,
        results=results,
    )


def print_eval_report(eval_results: list[QueryEvalResult], verbose: bool = False):
    print()
    print("=" * 78)
    print("  EVAL ОТЧЁТ")
    print("=" * 78)

    print(f"\n  {'ID':<30} {'R@5':>6} {'R@10':>6} {'MRR':>6}   Статус")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6}   {'-'*15}")

    for r in eval_results:
        status = "OK" if r.recall_at_10 >= 0.5 else ("~" if r.recall_at_10 > 0 else "FAIL")
        print(f"  {r.id:<30} {r.recall_at_5:>6.2f} {r.recall_at_10:>6.2f} "
              f"{r.mrr:>6.2f}   {status}")

    n = len(eval_results)
    avg_r5  = float(np.mean([r.recall_at_5  for r in eval_results]))
    avg_r10 = float(np.mean([r.recall_at_10 for r in eval_results]))
    avg_mrr = float(np.mean([r.mrr          for r in eval_results]))
    perfect = sum(1 for r in eval_results if r.recall_at_10 >= 1.0)
    bad     = sum(1 for r in eval_results if r.recall_at_10 == 0)

    print()
    print("  СВОДКА:")
    print(f"    Запросов:                      {n}")
    print(f"    Средний R@5:                   {avg_r5:.3f}")
    print(f"    Средний R@10:                  {avg_r10:.3f}")
    print(f"    Средний MRR:                   {avg_mrr:.3f}")
    print(f"    Идеальных (R@10=1.0):          {perfect}/{n}")
    print(f"    Полных провалов (R@10=0):       {bad}/{n}")

    print()
    print("  ACCEPTANCE CRITERIA (Этап 2):")
    ok_r5  = avg_r5  >= 0.78
    ok_mrr = avg_mrr >= 0.74
    print(f"    R@5  >= 0.78:  {avg_r5:.3f}  {'OK' if ok_r5  else 'FAIL'}")
    print(f"    MRR  >= 0.74:  {avg_mrr:.3f}  {'OK' if ok_mrr else 'FAIL'}")

    failed = [r for r in eval_results if r.recall_at_10 < 0.5]
    if failed:
        print()
        print("  " + "-" * 76)
        print("  ДЕТАЛИ ПРОВАЛИВШИХСЯ ЗАПРОСОВ (R@10 < 0.5):")
        print("  " + "-" * 76)
        for r in failed:
            print(f"\n  [{r.id}]  query: \"{r.query}\"")
            print(f"      R@5={r.recall_at_5:.2f}  R@10={r.recall_at_10:.2f}  MRR={r.mrr:.2f}")
            if r.missed_top_10:
                print(f"      НЕ НАЙДЕНО:")
                for m in r.missed_top_10:
                    print(f"         · {m}")
            print(f"      ТОП-5 ВЫДАЧИ:")
            for i, res in enumerate(r.results[:5], 1):
                print(f"         {i}. [{res['score']:.3f}] {res['title'][:60]}")

    if verbose:
        print()
        print("  " + "-" * 76)
        print("  ВСЕ ЗАПРОСЫ -- ТОП-5:")
        print("  " + "-" * 76)
        for r in eval_results:
            print(f"\n  [{r.id}]  \"{r.query}\"")
            for i, res in enumerate(r.results[:5], 1):
                print(f"    {i}. [{res['score']:.3f}] {res['title'][:60]}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db",   default=str(DEFAULT_DB),   type=Path)
    parser.add_argument("--eval", default=str(DEFAULT_EVAL), type=Path, dest="eval_path")
    parser.add_argument("--top",  default=10, type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--semantic-only", action="store_true",
                        help="Использовать только семантический поиск (без FTS)")
    args = parser.parse_args()

    print("=" * 78)
    print("  EVAL — семантический поиск через SQLite-индекс")
    print("=" * 78)

    if not args.db.exists():
        print(f"БД не найдена: {args.db}", file=sys.stderr)
        print("Сначала запустите: python search-doc.py import", file=sys.stderr)
        sys.exit(2)

    if not args.eval_path.exists():
        print(f"Eval-файл не найден: {args.eval_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nОткрываю БД: {args.db}")
    conn = open_db(args.db)

    print("Загружаю embedder...")
    t0 = time.time()
    embedder = E5SmallEmbedder()
    print(f"  Готово за {time.time()-t0:.1f} сек")

    with open(args.eval_path, encoding="utf-8") as f:
        eval_data = yaml.safe_load(f)

    queries_defs = eval_data.get("queries", [])
    print(f"\nЗапросов в наборе: {len(queries_defs)}")
    print("Прогоняю eval...\n")

    eval_results: list[QueryEvalResult] = []
    latencies: list[float] = []

    for qd in queries_defs:
        t_q = time.perf_counter()
        query_vec = embedder.encode_query(qd["query"])
        sem_hits = semantic_search(conn, query_vec, k=50)

        if args.semantic_only:
            fused_hits = sem_hits
            raw_fts = None
        else:
            fts_hits = fts_search(conn, qd["query"], k=50)
            technical = is_technical_query(qd["query"])
            fused_hits, _ = rrf_fuse(sem_hits, fts_hits, is_technical=technical)
            raw_fts = {cid: score for cid, score in fts_hits}

        raw_sem = {cid: score for cid, score in sem_hits}
        results = aggregate_to_books(
            fused_hits, conn, top_k=args.top,
            raw_semantic=raw_sem,
            raw_fts=raw_fts,
        )
        elapsed_ms = (time.perf_counter() - t_q) * 1000
        latencies.append(elapsed_ms)

        er = evaluate_query(qd, results)
        eval_results.append(er)

    conn.close()

    print(f"Среднее время поиска: {np.mean(latencies):.0f} мс  "
          f"(p95: {np.percentile(latencies, 95):.0f} мс)")

    print_eval_report(eval_results, verbose=args.verbose)

    print()
    print("=" * 78)
    print("  Готово.")
    print("=" * 78)


if __name__ == "__main__":
    main()
