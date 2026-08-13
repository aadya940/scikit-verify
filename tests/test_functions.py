"""__array_function__ entries."""

import numpy as np
import pytest
import sympy

from skverify import Pair, IDX

U = sympy.IndexedBase("u")


def make(n=10):
    return Pair.array("u", np.linspace(0.1, 0.9, n))


class TestWhere:
    def test_raw_mask_gets_ne(self):
        u = make()
        w = np.where(u, u, -u)
        assert w.formula == sympy.Piecewise(
            (U[IDX], sympy.Ne(U[IDX], 0)), (-U[IDX], True)
        )
        assert np.allclose(w.value, np.where(u.value, u.value, -u.value))

    def test_scalar_branch(self):
        w = np.where(make(), 1.0, 0.0)
        assert w.formula == sympy.Piecewise((1.0, sympy.Ne(U[IDX], 0)), (0.0, True))


class TestSum:
    def test_telescoping(self):
        u = make()
        s = np.sum(u[1:] - u[:-1])
        assert s.domain is None
        assert s.value == pytest.approx(
            u.value[-1] - u.value[0]
        )  # the math checks itself

    def test_inclusive_bounds(self):
        s = np.sum(make(4))
        j = next(iter(s.formula.variables))
        assert s.formula.limits == ((j, 0, 3),)  # half-open (0,4) => Sum to 3


class TestConstantFields:
    def test_zeros_plus_array(self):
        u = make(10)
        z = np.zeros(10) + u
        assert z.formula == U[IDX]
        assert z.domain == (0, 10)

    def test_zeros_like(self):
        z = np.zeros_like(make())
        assert z.formula == sympy.Integer(0)
        assert z.domain == (0, 10)

    def test_nonuniform_ndarray_refused(self):
        with pytest.raises(NotImplementedError):
            np.array([1.0, 2.0, 3.0]) + make(3)

    def test_unmapped_function_traces_with_guards(self):
        # unmapped + pure-Python: numpy's body runs on Pairs; its internal
        # comparisons are recorded as branch conditions and the trace lives
        r = np.median(make())
        assert float(r.value) == np.median(make().value)
