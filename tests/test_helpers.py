"""Test helpers, mostly index management and book-keeping."""
import pytest
import sympy
from skverify.helpers import axis_idx, normalize_slice, _AXIS_SYMBOLS


class TestAxisSymbols:
    def test_five_distinct_integer_symbols(self):
        syms = [axis_idx(a) for a in range(5)]
        assert len(set(syms)) == 5
        assert all(s.is_integer for s in syms)

    def test_stable_identity(self):
        assert axis_idx(0) is axis_idx(0)
        assert axis_idx(0) == sympy.Symbol("i", integer=True)

    def test_beyond_five_refused(self):
        with pytest.raises(NotImplementedError):
            axis_idx(5)


class TestNormalizeSlice:
    def test_defaults(self):
        assert normalize_slice(slice(None, None), 10) == (0, 10)

    def test_negatives(self):
        assert normalize_slice(slice(1, -1), 10) == (1, 9)
        assert normalize_slice(slice(-4, None), 10) == (6, 10)

    def test_explicit(self):
        assert normalize_slice(slice(2, 7), 10) == (2, 7)

    def test_step_refused(self):
        with pytest.raises(NotImplementedError):
            normalize_slice(slice(None, None, 2), 10)
        with pytest.raises(NotImplementedError):
            normalize_slice(slice(None, None, -1), 10)
