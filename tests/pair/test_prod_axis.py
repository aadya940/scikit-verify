"""Per-axis product reductions preserve symbolic Product formulas."""

import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx

AXIS0, AXIS1 = axis_idx(0), axis_idx(1)
JD = sympy.Symbol("j", integer=True)
P = sympy.IndexedBase("p")


@pytest.fixture
def p():
    return Pair.array("p", np.arange(1.0, 13.0).reshape(3, 4))


@pytest.fixture
def w():
    return Pair.array("w", np.arange(1.0, 25.0).reshape(2, 3, 4))


class TestProdAxis:
    def test_axis0(self, p):
        product = np.prod(p, axis=0)
        assert product.formula == sympy.Product(P[JD, AXIS0], (JD, 0, 2))
        assert product.domain == (0, 4)
        assert np.allclose(product.value, np.prod(p.value, axis=0))

    def test_axis1(self, p):
        product = np.prod(p, axis=1)
        assert product.formula == sympy.Product(P[AXIS0, JD], (JD, 0, 3))
        assert product.domain == (0, 3)
        assert np.allclose(product.value, np.prod(p.value, axis=1))

    def test_negative_axis(self, p):
        assert np.prod(p, axis=-1).formula == np.prod(p, axis=1).formula

    def test_3d_middle_axis_renumbers_survivors(self, w):
        W = sympy.IndexedBase("w")
        product = np.prod(w, axis=1)
        assert product.formula == sympy.Product(W[AXIS0, JD, AXIS1], (JD, 0, 2))
        assert product.domain == ((0, 2), (0, 4))
        assert np.allclose(product.value, np.prod(w.value, axis=1))

    def test_formula_evaluates(self, p):
        product = np.prod(p, axis=0)
        for col in range(4):
            got = product.formula.doit().subs(AXIS0, col).xreplace(
                {P[row, col]: float(p.value[row, col]) for row in range(3)}
            )
            assert np.isclose(float(got), np.prod(p.value, axis=0)[col])

    def test_keepdims(self, p):
        product = np.prod(p, axis=0, keepdims=True)
        assert np.shape(product.value) == (1,) + np.shape(p.value)[1:]
        assert product.formula == np.prod(p, axis=0).formula

    def test_keepdims_full(self, p):
        r = np.prod(p, keepdims=True)
        assert np.shape(r.value) == (1, 1)
        assert r.formula == np.prod(p).formula

    def test_axis_tuple_refused(self, p):
        with pytest.raises(NotImplementedError):
            np.prod(p, axis=(0, 1))

    def test_axis_tuple_len1_normalized(self, p):
        # scipy normalizes (0,) -> 0
        assert np.prod(p, axis=(0,)).formula == np.prod(p, axis=0).formula


class TestProdUnchangedPaths:
    def test_full_reduction_2d(self, p):
        s = np.prod(p)
        assert s.domain is None
        assert np.isclose(s.value, np.prod(p.value))

    def test_full_reduction_formula(self):
        v = Pair.array("v", np.arange(1.0, 6.0))
        s = np.prod(v)
        V = sympy.IndexedBase("v")
        assert s.formula == sympy.Product(V[JD], (JD, 0, 4))
        assert np.isclose(s.value, np.prod(v.value))

    def test_1d_axis0_still_scalar(self):
        v = Pair.array("v", np.arange(1.0, 6.0))
        s = np.prod(v, axis=0)
        assert s.domain is None
        assert np.isclose(s.value, np.prod(v.value))

    def test_scalar_pair(self):
        s = Pair(3.0, sympy.Symbol("s"))
        r = np.prod(s)
        assert r is s
        # scalar with axis should also be identity
        assert np.prod(s, axis=0) is s

    def test_object_ndarray_bag_full(self):
        v = Pair.array("v", np.arange(1.0, 6.0))
        bag = np.array([v[i] for i in range(5)], dtype=object)
        r = np.prod(bag)
        # bag product via element dunders keeps trace
        assert np.isclose(float(r.value), np.prod(v.value))
        # formula should be product of element formulas
        assert r.formula == (
            v[0].formula * v[1].formula * v[2].formula
            * v[3].formula * v[4].formula
        )

    def test_object_ndarray_bag_axis(self):
        p = Pair.array("p", np.arange(1.0, 7.0).reshape(2, 3))
        bag = np.empty((2, 3), dtype=object)
        for idx in np.ndindex(2, 3):
            bag[idx] = p[idx]
        r = np.prod(bag, axis=0)
        assert r.shape == (3,)
        assert np.allclose([float(x.value) for x in r], np.prod(p.value, axis=0))
        r1 = np.prod(bag, axis=1)
        assert r1.shape == (2,)
        assert np.allclose([float(x.value) for x in r1], np.prod(p.value, axis=1))


class TestProdDtype:
    def test_dtype_float_allowed(self, p):
        r = np.prod(p, dtype=float)
        assert np.isclose(r.value, np.prod(p.value))
        # float dtype does not change formula
        assert r.formula == np.prod(p).formula

    def test_dtype_int_refused(self, p):
        with pytest.raises(NotImplementedError, match="non-float dtype"):
            np.prod(p, dtype=int)

    def test_dtype_int_refused_1d(self):
        v = Pair.array("v", np.arange(1.0, 6.0))
        with pytest.raises(NotImplementedError):
            np.prod(v, dtype=np.int32)


class TestProdWhereOut:
    def test_where_masks_with_one(self, p):
        mask = np.array([[True, False, True, False],
                         [True, True, False, True],
                         [False, True, True, True]])
        r = np.prod(p, where=mask)
        # excluded elements become 1
        expected = np.prod(np.where(mask, p.value, 1.0))
        assert np.isclose(r.value, expected)
        # formula is a Product over a Piecewise with identity 1
        assert r.formula.has(sympy.Product)
        assert r.formula.has(sympy.Piecewise)
        assert r.formula.has(P)
        # numeric evaluation via doit + const-table mapping
        const_base = next(
            b for b in r.formula.atoms(sympy.IndexedBase)
            if str(b) != "p"
        )
        mapping = {
            P[row, col]: float(p.value[row, col])
            for row in range(3) for col in range(4)
        }
        mapping.update(
            {
                const_base[row, col]: int(mask[row, col])
                for row in range(3) for col in range(4)
            }
        )
        assert np.isclose(
            float(sympy.N(r.formula.doit().xreplace(mapping))), expected
        )

        # also check per-axis where
        r0 = np.prod(p, axis=0, where=mask)
        expected0 = np.prod(np.where(mask, p.value, 1.0), axis=0)
        assert np.allclose(r0.value, expected0)
        # per-axis formula is Product over Piecewise with identity 1
        assert isinstance(r0.formula, sympy.Product)
        assert r0.formula.has(P)
        assert r0.formula.has(sympy.Piecewise)
        const_base0 = next(
            b for b in r0.formula.atoms(sympy.IndexedBase)
            if str(b) != "p"
        )
        expected_pw = sympy.Piecewise(
            (P[JD, AXIS0], sympy.Ne(const_base0[JD, AXIS0], 0)),
            (1.0, True),
        )
        assert r0.formula.function == expected_pw
        assert r0.formula.limits == ((JD, 0, 2),)
        # per-column doit evaluation
        for col in range(4):
            expr = r0.formula.doit().subs(AXIS0, col)
            m = {P[row, col]: float(p.value[row, col]) for row in range(3)}
            m.update(
                {const_base0[row, col]: int(mask[row, col]) for row in range(3)}
            )
            got = float(sympy.N(expr.xreplace(m)))
            assert np.isclose(got, expected0[col])

    def test_where_pair_mask(self):
        v = Pair.array("v", np.array([1.0, 2.0, 3.0, 4.0]))
        cond = v > 2.0
        r = np.prod(v, where=cond)
        expected = np.prod(np.where(v.value > 2.0, v.value, 1.0))
        assert np.isclose(r.value, expected)
        # formula is Product over Piecewise with pair condition
        V = sympy.IndexedBase("v")
        JD2 = sympy.Symbol("j", integer=True)
        assert r.formula == sympy.Product(
            sympy.Piecewise((V[JD2], V[JD2] > 2.0), (1.0, True)),
            (JD2, 0, 3),
        )
        assert r.formula.has(sympy.Piecewise)
        assert r.formula.has(V)
        assert np.isclose(
            float(r.formula.doit().xreplace(
                {V[i]: float(v.value[i]) for i in range(4)}
            )),
            expected,
        )

    def test_out_pair(self, p):
        out = Pair.array("o", np.zeros(4))
        r = np.prod(p, axis=0, out=out)
        assert r is out
        assert np.allclose(out.value, np.prod(p.value, axis=0))
        assert out.formula == np.prod(p, axis=0).formula

    def test_out_raw_refused(self, p):
        buf = np.zeros(4)
        with pytest.raises(NotImplementedError, match="untraced buffer"):
            np.prod(p, axis=0, out=buf)

    def test_out_keepdims(self, p):
        out = Pair.array("o", np.zeros((1, 4)))
        r = np.prod(p, axis=0, keepdims=True, out=out)
        assert r is out
        assert np.shape(out.value) == (1, 4)

    def test_unsupported_kwargs_refused(self, p):
        with pytest.raises(NotImplementedError):
            np.prod(p, initial=1)
