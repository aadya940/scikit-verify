"""Per-axis reduction: one letter bound in a Sum, survivors renumbered."""

import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx

I, J, K = axis_idx(0), axis_idx(1), axis_idx(2)
JD = sympy.Symbol("j", integer=True)  # the Sum dummy
P = sympy.IndexedBase("p")


@pytest.fixture
def p():
    # non-square: catches axis mixups a square shape forgives
    return Pair.array("p", np.arange(12.0).reshape(3, 4))


@pytest.fixture
def w():
    return Pair.array("w", np.arange(24.0).reshape(2, 3, 4))


def evaluate(pair, arrs):
    expr = pair.formula.doit()
    mapping = {}
    for name, arr in arrs.items():
        base = sympy.IndexedBase(name)
        for idx, v in np.ndenumerate(arr):
            mapping[base[idx if len(idx) > 1 else idx[0]]] = float(v)
    return expr.xreplace(mapping)


class TestSumAxis:
    def test_axis0(self, p):
        s = np.sum(p, axis=0)
        assert s.formula == sympy.Sum(P[JD, I], (JD, 0, 2))
        assert s.domain == (0, 4)
        assert np.allclose(s.value, p.value.sum(axis=0))

    def test_axis1(self, p):
        s = np.sum(p, axis=1)
        assert s.formula == sympy.Sum(P[I, JD], (JD, 0, 3))
        assert s.domain == (0, 3)
        assert np.allclose(s.value, p.value.sum(axis=1))

    def test_negative_axis(self, p):
        assert np.sum(p, axis=-1).formula == np.sum(p, axis=1).formula

    def test_3d_middle_axis(self, w):
        # axis=1: survivor BELOW (i stays) and ABOVE (k renumbers to j)
        W = sympy.IndexedBase("w")
        s = np.sum(w, axis=1)
        assert s.formula == sympy.Sum(W[I, JD, J], (JD, 0, 2))
        assert s.domain == ((0, 2), (0, 4))
        assert np.allclose(s.value, w.value.sum(axis=1))

    def test_values_evaluate(self, p):
        # the formula, numerically evaluated per surviving index, matches
        s = np.sum(p, axis=0)
        arr = p.value
        for col in range(4):
            got = float(
                s.formula.doit()
                .subs(I, col)
                .xreplace(
                    {P[r, c]: float(arr[r, c]) for r in range(3) for c in range(4)}
                )
            )
            assert np.isclose(got, arr.sum(axis=0)[col])


class TestLetterInvariantAfterReduce:
    def test_reduced_plus_vector(self, p):
        # the u[2]+v sibling: Sum over rows, then add a fresh 4-vector.
        # Letters must line up (both survivors speak `i`).
        v = Pair.array("v", np.arange(4.0))
        V = sympy.IndexedBase("v")
        r = np.sum(p, axis=0) + v
        assert sympy.simplify(r.formula - (sympy.Sum(P[JD, I], (JD, 0, 2)) + V[I])) == 0
        assert np.allclose(r.value, p.value.sum(axis=0) + v.value)

    def test_softmax_with_axis(self, p):
        # rows normalized: exp(p)/sum(exp(p), axis=1) needs broadcasting
        # of the reduced result back against the 2-D array... which is
        # rank-broadcast: (3,) meeting (3,4) trailing-aligns WRONG here,
        # so numpy itself requires keepdims/reshape; we just check the
        # reduced piece lifts and evaluates.
        e = np.exp(p)
        s = np.sum(e, axis=1)
        assert s.domain == (0, 3)
        assert np.allclose(s.value, np.exp(p.value).sum(axis=1))


class TestRefusals:
    def test_axis_tuple_refused(self, p):
        with pytest.raises(NotImplementedError):
            np.sum(p, axis=(0, 1))

    def test_keepdims_keeps_the_axis(self, p):
        r = np.sum(p, axis=0, keepdims=True)
        assert np.shape(r.value) == (1,) + np.shape(p.value)[1:]
        full = np.sum(p, axis=0)
        assert r.formula == full.formula


class TestUnchangedPaths:
    def test_full_reduction_2d(self, p):
        s = np.sum(p)
        assert s.domain is None
        assert np.isclose(s.value, p.value.sum())

    def test_1d_axis0_still_scalar(self):
        v = Pair.array("v", np.arange(5.0))
        s = np.sum(v, axis=0)
        assert s.domain is None
        assert np.isclose(s.value, v.value.sum())
