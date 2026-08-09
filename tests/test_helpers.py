"""Test helpers, mostly index management and book-keeping."""

import numpy as np
import pytest
import sympy
from skverify.helpers import (
    axis_idx,
    normalize_slice,
    _AXIS_SYMBOLS,
    normalize_key,
)


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


class TestNormalizeKey:
    def test_single_slice_1d(self):
        assert normalize_key(slice(1, None), (5,)) == ((1, 5),)

    def test_int_and_slice_2d(self):
        assert normalize_key((1, slice(None)), (4, 7)) == (1, (0, 7))

    def test_short_key_padded(self):
        # u[1:] on 2-D means u[1:, :]
        assert normalize_key(slice(1, None), (4, 7)) == ((1, 4), (0, 7))

    def test_bare_int_padded(self):
        assert normalize_key(2, (4, 7)) == (2, (0, 7))

    def test_ellipsis_leading(self):
        assert normalize_key((Ellipsis, 0), (2, 3, 4)) == ((0, 2), (0, 3), 0)

    def test_ellipsis_middle(self):
        assert normalize_key((0, Ellipsis, 1), (2, 3, 4)) == (0, (0, 3), 1)

    def test_ellipsis_trailing(self):
        assert normalize_key((0, Ellipsis), (2, 3, 4)) == (0, (0, 3), (0, 4))

    def test_ellipsis_alone(self):
        assert normalize_key(Ellipsis, (2, 3)) == ((0, 2), (0, 3))

    def test_redundant_ellipsis(self):
        # key already covers every axis; `...` expands to nothing
        assert normalize_key((0, 1, Ellipsis), (2, 3)) == (0, 1)

    def test_negative_int(self):
        assert normalize_key(-1, (5,)) == (4,)

    def test_negative_slice_per_axis_lengths(self):
        # each axis resolves negatives against ITS OWN length
        assert normalize_key((slice(None, -1), slice(-2, None)), (4, 7)) == (
            (0, 3),
            (5, 7),
        )

    def test_all_ints_full_drop(self):
        assert normalize_key((1, 2), (4, 7)) == (1, 2)

    def test_numpy_integer_accepted(self):
        assert normalize_key(np.int64(-1), (5,)) == (4,)

    def test_double_ellipsis_refused(self):
        with pytest.raises(IndexError):
            normalize_key((Ellipsis, 0, Ellipsis), (2, 3, 4))

    def test_too_many_indices_refused(self):
        with pytest.raises(IndexError):
            normalize_key((0, 0, 0), (4, 7))

    def test_out_of_bounds_refused(self):
        with pytest.raises(IndexError):
            normalize_key(5, (5,))
        with pytest.raises(IndexError):
            normalize_key(-6, (5,))

    def test_bool_refused_before_int(self):
        # isinstance(True, int) is True, bool must be caught first
        with pytest.raises(NotImplementedError):
            normalize_key(True, (5,))
        with pytest.raises(NotImplementedError):
            normalize_key(np.bool_(False), (5,))

    def test_newaxis_refused(self):
        with pytest.raises(NotImplementedError):
            normalize_key((None, slice(None)), (5,))

    def test_fancy_indexing_refused(self):
        with pytest.raises(NotImplementedError):
            normalize_key([0, 1], (5,))
        with pytest.raises(NotImplementedError):
            normalize_key(np.array([0, 1]), (5,))

    def test_step_slice_refused(self):
        with pytest.raises(NotImplementedError):
            normalize_key(slice(None, None, 2), (5,))
