"""Rank broadcasting: smaller operand aligns to the trailing axes."""

import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx

I, J, K = axis_idx(0), axis_idx(1), axis_idx(2)
U = sympy.IndexedBase("u")
V = sympy.IndexedBase("v")


def make_uv():
    u = Pair.array("u", np.arange(28.0).reshape(4, 7))
    v = Pair.array("v", np.arange(7.0))
    return u, v


class TestRankBroadcast:
    def test_2d_plus_1d(self):
        u, v = make_uv()
        r = u + v
        # v runs along the columns: its letter becomes j
        assert sympy.simplify(r.formula - (U[I, J] + V[J])) == 0
        assert r.domain == ((0, 4), (0, 7))
        assert np.allclose(r.value, u.value + v.value)

    def test_1d_plus_2d_order(self):
        u, v = make_uv()
        r = v + u
        assert sympy.simplify(r.formula - (V[J] + U[I, J])) == 0
        assert np.allclose(r.value, v.value + u.value)

    def test_scalar_still_fine(self):
        u, _ = make_uv()
        r = u + 2.0
        assert r.domain == ((0, 4), (0, 7))

    def test_3d_plus_1d(self):
        w = Pair.array("w", np.arange(24.0).reshape(2, 3, 4))
        v = Pair.array("v", np.arange(4.0))
        r = w - v
        W = sympy.IndexedBase("w")
        assert sympy.simplify(r.formula - (W[I, J, K] - V[K])) == 0
        assert np.allclose(r.value, w.value - v.value)

    def test_3d_plus_2d(self):
        w = Pair.array("w", np.arange(24.0).reshape(2, 3, 4))
        m = Pair.array("m", np.arange(12.0).reshape(3, 4))
        r = w + m
        W, M = sympy.IndexedBase("w"), sympy.IndexedBase("m")
        # m's axes (i, j) land on w's trailing axes (j, k)
        assert sympy.simplify(r.formula - (W[I, J, K] + M[J, K])) == 0
        assert np.allclose(r.value, w.value + m.value)

    def test_mul_broadcasts_too(self):
        u, v = make_uv()
        r = u * v
        assert sympy.simplify(r.formula - (U[I, J] * V[J])) == 0
        assert np.allclose(r.value, u.value * v.value)


class TestBroadcastRefusals:
    def test_tail_size_mismatch(self):
        u = Pair.array("u", np.zeros((4, 7)))
        w = Pair.array("w", np.zeros(4))  # 4 != 7: numpy refuses, so do we
        with pytest.raises(ValueError):
            u + w

    def test_extent_one_stretches(self):
        # (4,1) meeting (4,7): the 1 stretches, its letter pins to 0
        u = Pair.array("u", np.arange(28.0).reshape(4, 7))
        c = Pair.array("c", np.arange(4.0).reshape(4, 1))
        r = u + c
        U, C = sympy.IndexedBase("u"), sympy.IndexedBase("c")
        I, J = axis_idx(0), axis_idx(1)
        assert sympy.simplify(r.formula - (U[I, J] + C[I, 0])) == 0
        assert r.domain == ((0, 4), (0, 7))
        assert np.allclose(r.value, u.value + c.value)

    def test_extent_one_row_vector(self):
        # (1,7) meeting (4,7)
        u = Pair.array("u", np.arange(28.0).reshape(4, 7))
        c = Pair.array("c", np.arange(7.0).reshape(1, 7))
        r = u * c
        U, C = sympy.IndexedBase("u"), sympy.IndexedBase("c")
        I, J = axis_idx(0), axis_idx(1)
        assert sympy.simplify(r.formula - (U[I, J] * C[0, J])) == 0
        assert np.allclose(r.value, u.value * c.value)

    def test_newaxis_then_broadcast(self):
        # the lsq pattern: y (n,2) * w[:, None] (n,1)
        y = Pair.array("y", np.arange(8.0).reshape(4, 2))
        w = Pair.array("w", np.arange(4.0) + 1)
        r = y * w[:, None]
        assert np.allclose(r.value, y.value * (w.value[:, None]))
        assert r.domain == ((0, 4), (0, 2))
