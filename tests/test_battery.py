"""Pins from the adversarial testing hour: edges the batteries caught."""

import numpy as np
import pytest
import sympy
from scipy.interpolate import make_interp_spline, make_smoothing_spline

from skverify import Pair, to_sympy


class TestEdges:
    def test_0d_array_is_scalar_symbol(self):
        p = Pair.array("u", np.array(3.0))
        assert p.formula == sympy.Symbol("u", real=True)
        assert p.domain is None

    def test_empty_array_sum(self):
        s = np.sum(Pair.array("u", np.array([])))
        assert s.formula == sympy.Sum(
            sympy.IndexedBase("u")[sympy.Symbol("j", integer=True)],
            (sympy.Symbol("j", integer=True), 0, -1),
        )

    def test_double_newaxis(self):
        u = Pair.array("u", np.arange(3.0))
        assert u[None, :, None].value.shape == (1, 3, 1)

    def test_matmul_chain_nests_fresh_dummies(self):
        a = Pair.array("a", np.eye(2))
        r = a @ a @ a @ a
        dummies = {s.variables[0] for s in r.formula.atoms(sympy.Sum)}
        assert len(dummies) == 3  # every contraction its own dummy

    def test_nan_and_inf_flow_concrete(self):
        p = Pair.array("u", np.array([np.nan, np.inf, 1.0])) + 1
        assert np.isnan(p.value[0]) and np.isinf(p.value[1])


class TestGateVariants:
    def test_interp_natural_bc(self):
        x = np.linspace(0, 30)
        y = np.sin(x)
        spl = to_sympy(_interp_natural, x, y)
        ref = make_interp_spline(x, y, bc_type="natural")
        assert np.allclose([c.value for c in spl.c], ref.c)

    def test_interp_linear_order(self):
        x = np.linspace(0, 30)
        y = np.sin(x)
        spl = to_sympy(_interp_k1, x, y)
        ref = make_interp_spline(x, y, k=1)
        assert np.allclose([c.value for c in spl.c], ref.c)

    def test_scale_n100_smoothing(self):
        # deep scatter nesting must not hit the recursion limit, and the
        # derivation stays flat in n
        x = np.linspace(0, 4, 100)
        y = x**2
        spl = to_sympy(_smooth, x, y)
        lines = spl.c[0].derivation().splitlines()
        assert len(lines) < 1200


def _interp_natural(x, y):
    return make_interp_spline(x, y, bc_type="natural")


def _interp_k1(x, y):
    return make_interp_spline(x, y, k=1)


def _smooth(x, y):
    return make_smoothing_spline(x, y, lam=0.1)
