"""Unit-тесты для src/catalog_import/xlsx_loader.py — safe_int, safe_str."""
import numpy as np
import pandas as pd
import pytest

from src.catalog_import.xlsx_loader import safe_int, safe_str


def _warn():
    """Вспомогательный список предупреждений."""
    return []


class TestSafeInt:

    def test_none(self):
        assert safe_int(None, "год", 1, _warn()) is None

    def test_pandas_na(self):
        assert safe_int(pd.NA, "год", 1, _warn()) is None

    def test_numpy_nan(self):
        assert safe_int(float("nan"), "год", 1, _warn()) is None

    def test_int(self):
        assert safe_int(2016, "год", 1, _warn()) == 2016

    def test_numpy_integer(self):
        assert safe_int(np.int64(2023), "год", 1, _warn()) == 2023

    def test_float_whole(self):
        assert safe_int(2016.0, "год", 1, _warn()) == 2016

    def test_string_int(self):
        assert safe_int("2020", "год", 1, _warn()) == 2020

    def test_string_range_takes_first(self):
        """"2016-2017" → берёт первое 4-значное число."""
        assert safe_int("2016-2017", "год", 1, _warn()) == 2016

    def test_string_with_garbage(self):
        """Строка с числом внутри мусора → извлекает 4-значное число."""
        assert safe_int("year:2022!", "год", 1, _warn()) == 2022

    def test_string_no_number_returns_none_with_warning(self):
        warnings = _warn()
        result = safe_int("abc", "год", 5, warnings)
        assert result is None
        assert len(warnings) == 1
        assert "строка 5" in warnings[0]

    def test_empty_string(self):
        assert safe_int("", "год", 1, _warn()) is None

    def test_whitespace_string(self):
        assert safe_int("   ", "год", 1, _warn()) is None


class TestSafeStr:

    def test_none(self):
        assert safe_str(None) is None

    def test_pandas_na(self):
        assert safe_str(pd.NA) is None

    def test_normal_string(self):
        assert safe_str("Python") == "Python"

    def test_strips_whitespace(self):
        assert safe_str("  hello  ") == "hello"

    def test_empty_string_returns_none(self):
        assert safe_str("") is None

    def test_whitespace_only_returns_none(self):
        assert safe_str("   ") is None

    def test_converts_number(self):
        assert safe_str(42) == "42"
