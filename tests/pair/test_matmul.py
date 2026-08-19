"""Matmul as a contraction Sum: every rank pairing numpy supports."""

import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx

I, J, K = axis_idx(0), axis_idx(1), axis_idx(2)
KD = sympy.Symbol("k", integer=True)  # the contraction dummy
KD2 = sympy.Symbol("k2", integer=True)  # dodges the rank-3 letter k
A = sympy.IndexedBase("a")
B = sympy.IndexedBase("b")


def pairs(shape_a, shape_b, seed=0):
    rng = np.random.default_rng(seed)
    return (
        Pair.array("a", rng.standard_normal(shape_a)),
        Pair.array("b", rng.standard_normal(shape_b)),
    )


def evaluate(pair, arrs):
    """Formula lane evaluated pointwise; must match the value lane."""
    mapping = {}
    for name, arr in arrs.items():
        base = sympy.IndexedBase(name)
        for idx, v in np.ndenumerate(arr):
            mapping[base[idx if len(idx) > 1 else idx[0]]] = sympy.Float(float(v))
    value = np.asarray(pair.value)
    letters = tuple(axis_idx(ax) for ax in range(value.ndim))
    out = np.empty(value.shape)
    for pos in np.ndindex(value.shape):
        point = pair.formula.subs(dict(zip(letters, pos)), simultaneous=True)
        out[pos] = float(sympy.N(point.doit().xreplace(mapping)))
    return out


class TestFormulas:
    def test_2d_2d(self):
        a, b = pairs((3, 4), (4, 5))
        r = a @ b
        assert r.formula == sympy.Sum(A[I, KD] * B[KD, J], (KD, 0, 3))
        assert r.domain == ((0, 3), (0, 5))

    def test_1d_1d_scalar(self):
        a, b = pairs(4, 4)
        r = a @ b
        assert r.formula == sympy.Sum(A[KD] * B[KD], (KD, 0, 3))
        assert r.domain is None
        assert np.isclose(r.value, a.value @ b.value)

    def test_2d_1d(self):
        a, b = pairs((3, 4), 4)
        r = a @ b
        assert r.formula == sympy.Sum(A[I, KD] * B[KD], (KD, 0, 3))
        assert r.domain == (0, 3)

    def test_1d_2d(self):
        a, b = pairs(4, (4, 5))
        r = a @ b
        assert r.formula == sympy.Sum(A[KD] * B[KD, I], (KD, 0, 3))
        assert r.domain == (0, 5)

    def test_batched_3d(self):
        a, b = pairs((2, 3, 4), (2, 4, 5))
        r = a @ b
        assert r.formula == sympy.Sum(A[I, J, KD2] * B[I, KD2, K], (KD2, 0, 3))
        assert r.domain == ((0, 2), (0, 3), (0, 5))

    def test_broadcast_batch_pins_zero(self):
        # a's extent-1 batch axis broadcasts: it indexes at 0
        a, b = pairs((1, 3, 4), (5, 4, 2))
        r = a @ b
        assert r.formula == sympy.Sum(A[0, J, KD2] * B[I, KD2, K], (KD2, 0, 3))
        assert r.domain == ((0, 5), (0, 3), (0, 2))

    def test_3d_meets_2d(self):
        a, b = pairs((2, 3, 4), (4, 5))
        r = a @ b
        assert r.formula == sympy.Sum(A[I, J, KD2] * B[KD2, K], (KD2, 0, 3))
        assert r.domain == ((0, 2), (0, 3), (0, 5))

    def test_fresh_dummy_in_chain(self):
        # (a @ b) @ c: the outer contraction must not capture the inner k
        a, b = pairs((3, 4), (4, 5))
        c = Pair.array("c", np.random.default_rng(1).standard_normal((5, 2)))
        r = (a @ b) @ c
        dummies = [s.variables[0] for s in r.formula.atoms(sympy.Sum)]
        assert len(set(dummies)) == 2


class TestAllRankPairings:
    SHAPES = [
        ((4,), (4,)),
        ((4,), (4, 5)),
        ((3, 4), (4,)),
        ((3, 4), (4, 5)),
        ((2, 3, 4), (4,)),
        ((4,), (2, 4, 5)),
        ((2, 3, 4), (4, 5)),
        ((3, 4), (2, 4, 5)),
        ((2, 3, 4), (2, 4, 5)),
        ((1, 3, 4), (5, 4, 2)),
        ((5, 3, 4), (1, 4, 2)),
        ((2, 1, 3, 4), (6, 4, 5)),
        ((2, 6, 3, 4), (1, 1, 4, 5)),
    ]

    @pytest.mark.parametrize("shape_a,shape_b", SHAPES)
    def test_differential(self, shape_a, shape_b):
        a, b = pairs(shape_a, shape_b)
        r = a @ b
        expected = np.matmul(a.value, b.value)
        assert np.allclose(np.asarray(r.value), expected)
        assert np.allclose(
            evaluate(r, {"a": a.value, "b": b.value}), expected
        )

    @pytest.mark.parametrize("shape_a,shape_b", SHAPES)
    def test_np_matmul_matches_operator(self, shape_a, shape_b):
        a, b = pairs(shape_a, shape_b)
        assert np.matmul(a, b).formula == (a @ b).formula


class TestDispatchEdges:
    def test_rmatmul_concrete_uniform(self):
        # ones @ pair: the concrete side lifts as a constant field
        b = Pair.array("b", np.arange(4.0))
        r = np.ones((3, 4)) @ b
        assert r.formula == sympy.Sum(1.0 * B[KD], (KD, 0, 3))
        assert np.allclose(r.value, np.ones((3, 4)) @ b.value)

    def test_concrete_operand_becomes_named_table(self):
        b = Pair.array("b", np.arange(4.0))
        r = np.arange(8.0).reshape(2, 4) @ b
        assert "const_" in str(r.formula)  # values disclosed, not hidden
        assert np.allclose(r.value, np.arange(8.0).reshape(2, 4) @ b.value)

    def test_scalar_operand_raises(self):
        a = Pair.array("a", np.arange(4.0))
        with pytest.raises((ValueError, TypeError)):
            a @ 2.0

    def test_shape_mismatch_raises(self):
        a, b = pairs((3, 4), (5, 6))
        with pytest.raises(ValueError):
            a @ b

    def test_mask_operand_bridges(self):
        # (a > 0) @ b counts through the 0/1 bridge, not raw Booleans
        a, b = pairs(4, 4)
        r = (a > 0) @ b
        assert r.formula.has(sympy.Piecewise)
        assert np.isclose(
            float(evaluate(r, {"a": a.value, "b": b.value})),
            (a.value > 0) @ b.value,
        )


class TestDot:
    def test_dot_matches_matmul_2d(self):
        a, b = pairs((3, 4), (4, 5))
        assert np.dot(a, b).formula == (a @ b).formula

    def test_dot_matches_matmul_1d(self):
        a, b = pairs(4, 4)
        assert np.dot(a, b).formula == (a @ b).formula

    def test_dot_scalar_multiplies(self):
        a = Pair.array("a", np.arange(4.0))
        r = np.dot(a, 3.0)
        assert r.formula == 3.0 * A[I]
        assert np.allclose(r.value, a.value * 3.0)

    def test_dot_nd_refuses(self):
        a, b = pairs((2, 3, 4), (2, 4, 5))
        with pytest.raises(NotImplementedError, match="matmul"):
            np.dot(a, b)
