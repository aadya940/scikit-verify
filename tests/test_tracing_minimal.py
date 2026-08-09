"""The minimal 'trace all Python' layer: coercion guards, duck facts,
and the __wrapped__ fallback that runs numpy's real bodies on Pairs."""

import numpy as np
import pytest
import sympy

from skverify import Pair

U = sympy.IndexedBase("u")


def make():
    return Pair.array("u", np.array([1.0, 2.0, 4.0, 7.0]))


class TestCoercionGuards:
    def test_float_refuses(self):
        x = Pair(3.0, sympy.Symbol("x"))
        with pytest.raises(NotImplementedError):
            float(x)

    def test_int_refuses(self):
        x = Pair(3.0, sympy.Symbol("x"))
        with pytest.raises(NotImplementedError):
            int(x)

    def test_complex_refuses(self):
        x = Pair(3.0, sympy.Symbol("x"))
        with pytest.raises(NotImplementedError):
            complex(x)

    def test_value_is_the_deliberate_exit(self):
        x = Pair(3.0, sympy.Symbol("x"))
        assert x.value == 3.0


class TestDuckFacts:
    def test_ndim_shape_dtype(self):
        u = Pair.array("u", np.zeros((4, 7)))
        assert u.ndim == 2
        assert u.shape == (4, 7)
        assert u.dtype == np.float64

    def test_scalar_pair_facts(self):
        x = Pair(3.0, sympy.Symbol("x"))
        assert x.ndim == 0
        assert x.shape == ()


class TestWrappedFallback:
    def test_diff_traces_through_numpy_source(self):
        u = make()
        r = np.diff(u)
        # numpy's real body ran; elements are scalar Pairs, unrolled
        assert [e.formula for e in r] == [
            U[1] - U[0],
            U[2] - U[1],
            U[3] - U[2],
        ]
        assert [e.value for e in r] == list(np.diff(u.value))

    def test_second_difference(self):
        u = make()
        r = np.diff(u, n=2)
        assert [e.formula for e in r] == [
            U[0] - 2 * U[1] + U[2],
            U[1] - 2 * U[2] + U[3],
        ]
        assert [e.value for e in r] == list(np.diff(u.value, n=2))

    def test_table_entries_still_win(self):
        # np.sum is curated: indexed Sum formula, not unrolled elements
        u = make()
        s = np.sum(u)
        assert isinstance(s.formula, sympy.Sum)

    def test_dot_traces_as_inner_product(self):
        # np.dot's python wrapper runs; decompression + dunders do the rest
        u = make()
        r = np.dot(u, u)
        assert r.formula == U[0] ** 2 + U[1] ** 2 + U[2] ** 2 + U[3] ** 2
        assert r.value == np.dot(u.value, u.value)

    def test_unsupported_op_inside_body_is_loud(self):
        # np.median's body needs Pair < Pair (not supported yet):
        # dies mid-trace with a loud error, never silently
        u = make()
        with pytest.raises((NotImplementedError, TypeError)):
            np.median(u)
