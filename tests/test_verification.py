"""Formula-lane verification: for lifted functions the FORMULA must
evaluate to the value, not merely the value match scipy."""

import numpy as np
import pytest
import sympy

from scipy.stats import skew, variation

from skverify import Pair, to_sympy
from skverify.helpers import axis_idx

Y = np.linspace(1.0, 2.0, 10) ** 2


def _eval_scalar(formula, name, arr):
    mapping = {
        sympy.IndexedBase(name)[k]: sympy.Float(float(v))
        for k, v in enumerate(arr)
    }
    return float(sympy.N(formula.doit().xreplace(mapping).doit()))


def test_skew_formula_evaluates():
    r = to_sympy(_skew, Y)
    assert np.isclose(_eval_scalar(r.formula, "y", Y), skew(Y), rtol=1e-6)


def test_variation_formula_evaluates():
    r = to_sympy(_variation, Y)
    assert np.isclose(_eval_scalar(r.formula, "y", Y), variation(Y), rtol=1e-6)


def test_percentile_formula_evaluates():
    p = Pair.array("u", np.array([3.0, 1.0, 4.0, 1.5, 9.0]))
    r = np.percentile(p, 37.5)
    got = _eval_scalar(r.formula, "u", p.value)
    assert np.isclose(got, np.percentile(p.value, 37.5))


def test_self_aliasing_shift_write():
    # p[1:] = p[:-1]: value-semantic RHS makes this exact
    base = np.array([1.0, 2.0, 3.0, 4.0])
    p = Pair.array("u", base.copy())
    m = base.copy()
    p[1:] = p[:-1]
    m[1:] = m[:-1].copy()
    assert np.allclose(np.asarray(Pair._value_of(p.value), dtype=float), m)
    mapping = {
        sympy.IndexedBase("u")[k]: sympy.Float(base[k]) for k in range(4)
    }
    for k in range(4):
        f = p.formula.subs(axis_idx(0), k).doit().xreplace(mapping)
        assert np.isclose(float(sympy.N(f.doit())), m[k])


def _skew(y):
    return skew(y)


def _variation(y):
    return variation(y)
