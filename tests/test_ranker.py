"""Unit-тесты для src/search/ranker.py — is_technical_query и rrf_fuse."""
import pytest
from src.search.ranker import (
    RRF_K_FTS,
    RRF_K_SEMANTIC,
    WEIGHTS_DESCRIPTIVE,
    WEIGHTS_TECHNICAL,
    is_technical_query,
    rrf_fuse,
)


# ─── is_technical_query ──────────────────────────────────────────────────────

class TestIsTechnicalQuery:

    def test_whitelist_lowercase(self):
        assert is_technical_query("pytest фикстуры") is True

    def test_whitelist_mixed_case(self):
        assert is_technical_query("обучение с React") is True

    def test_whitelist_go(self):
        assert is_technical_query("разработка на Go Golang") is True

    def test_whitelist_llm(self):
        assert is_technical_query("LLM в продакшене") is True

    def test_whitelist_k8s(self):
        assert is_technical_query("деплой в K8s") is True

    def test_whitelist_sql(self):
        assert is_technical_query("оптимизация SQL запросов") is True

    def test_all_caps_short(self):
        """Токен ≤8 символов и весь в верхнем регистре → технический."""
        assert is_technical_query("вопросы про API") is True

    def test_all_caps_long_not_in_whitelist(self):
        """ALLCAPS > 8 символов и не в whitelist — технический через isupper."""
        # "ALGORITHM" = 9 символов, не в whitelist — НЕ технический
        assert is_technical_query("алгоритмы ALGORITHM") is False

    def test_camelcase_short(self):
        """CamelCase ≤8 символов → технический."""
        # "MacAddr" = 7 символов, CamelCase
        assert is_technical_query("MacAddr") is True

    def test_camelcase_javascript_not_technical(self):
        """JavaScript = 10 символов, не в whitelist → НЕ технический (баг-фикс)."""
        assert is_technical_query("функциональное программирование на JavaScript") is False

    def test_camelcase_typescript_not_technical(self):
        """TypeScript = 10 символов, не в whitelist → НЕ технический."""
        assert is_technical_query("разработка на TypeScript") is False

    def test_purely_russian_descriptive(self):
        assert is_technical_query("как развиваться программисту и расти в карьере") is False

    def test_empty(self):
        assert is_technical_query("") is False

    def test_no_ascii_tokens(self):
        assert is_technical_query("администрирование серверов linux") is True  # linux в whitelist

    def test_tdd(self):
        assert is_technical_query("разработка через тестирование TDD") is True

    def test_docker(self):
        assert is_technical_query("контейнеризация Docker") is True


# ─── rrf_fuse ────────────────────────────────────────────────────────────────

class TestRrfFuse:

    def test_empty_inputs(self):
        fused, (ws, wf) = rrf_fuse([], [], is_technical=False)
        assert fused == []

    def test_semantic_only(self):
        sem = [(10, 0.9), (20, 0.8), (30, 0.7)]
        fused, (ws, wf) = rrf_fuse(sem, [], is_technical=False)
        ids = [cid for cid, _ in fused]
        assert ids == [10, 20, 30]

    def test_fts_only(self):
        fts = [(10, 5.0), (20, 3.0)]
        fused, (ws, wf) = rrf_fuse([], fts, is_technical=True)
        ids = [cid for cid, _ in fused]
        assert ids == [10, 20]

    def test_descriptive_weights(self):
        sem = [(1, 0.9)]
        fts = [(1, 5.0)]
        _, (ws, wf) = rrf_fuse(sem, fts, is_technical=False)
        assert ws == WEIGHTS_DESCRIPTIVE[0]
        assert wf == WEIGHTS_DESCRIPTIVE[1]

    def test_technical_weights(self):
        sem = [(1, 0.9)]
        fts = [(1, 5.0)]
        _, (ws, wf) = rrf_fuse(sem, fts, is_technical=True)
        assert ws == WEIGHTS_TECHNICAL[0]
        assert wf == WEIGHTS_TECHNICAL[1]

    def test_rrf_formula_correctness(self):
        """Проверяем точную формулу RRF для одного документа."""
        ws, wf = WEIGHTS_DESCRIPTIVE
        sem = [(42, 0.9)]  # rank 1
        fts = [(42, 5.0)]  # rank 1
        fused, _ = rrf_fuse(sem, fts, is_technical=False)
        expected = ws / (RRF_K_SEMANTIC + 1) + wf / (RRF_K_FTS + 1)
        assert len(fused) == 1
        assert abs(fused[0][1] - expected) < 1e-9

    def test_ordering_by_rrf_score(self):
        """Документ с лучшим суммарным RRF идёт первым."""
        # doc A: sem rank 1, fts rank 10
        # doc B: sem rank 2, fts rank 1
        sem = [(100, 0.95), (200, 0.90)]
        fts = [(200, 8.0), (100, 2.0)]
        fused_desc, _ = rrf_fuse(sem, fts, is_technical=False)
        # При is_technical=False sem весит больше → doc 100 (sem rank 1) должен быть выше
        ws, wf = WEIGHTS_DESCRIPTIVE
        score_100 = ws / (RRF_K_SEMANTIC + 1) + wf / (RRF_K_FTS + 2)
        score_200 = ws / (RRF_K_SEMANTIC + 2) + wf / (RRF_K_FTS + 1)
        if score_100 >= score_200:
            assert fused_desc[0][0] == 100
        else:
            assert fused_desc[0][0] == 200

    def test_doc_only_in_fts(self):
        """Документ только в FTS — всё равно попадает в результат."""
        sem = [(1, 0.9)]
        fts = [(1, 5.0), (99, 3.0)]
        fused, _ = rrf_fuse(sem, fts, is_technical=False)
        ids = [cid for cid, _ in fused]
        assert 99 in ids

    def test_doc_only_in_semantic(self):
        """Документ только в semantic — всё равно попадает в результат."""
        sem = [(1, 0.9), (77, 0.5)]
        fts = [(1, 5.0)]
        fused, _ = rrf_fuse(sem, fts, is_technical=False)
        ids = [cid for cid, _ in fused]
        assert 77 in ids

    def test_scores_positive(self):
        sem = [(i, 1.0 - i * 0.01) for i in range(10)]
        fts = [(i, 10.0 - i) for i in range(5)]
        fused, _ = rrf_fuse(sem, fts, is_technical=True)
        for _, score in fused:
            assert score > 0

    def test_weights_sum_to_one(self):
        assert abs(WEIGHTS_DESCRIPTIVE[0] + WEIGHTS_DESCRIPTIVE[1] - 1.0) < 1e-9
        assert abs(WEIGHTS_TECHNICAL[0] + WEIGHTS_TECHNICAL[1] - 1.0) < 1e-9
