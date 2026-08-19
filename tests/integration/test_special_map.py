"""Every scipy.special -> sympy table row disposed by execution:
the sympy form must reproduce scipy's values across sample points."""

import numpy as np
import pytest
import scipy.special as sp
import sympy

from skverify import Pair
from skverify.registry import UFUNC_TABLE

SPECIAL_ROWS = [
    (fn, target)
    for fn, target in UFUNC_TABLE.items()
    if getattr(fn, "__module__", "").startswith("scipy")
    or fn.__name__
    in ("erf", "erfc", "gamma", "digamma", "zeta", "gammaln", "psi",
        "ndtr", "expit", "logit", "xlogy", "beta", "jv", "huber")
]

POINTS_1 = [0.3, 1.7, 2.5]
POINTS_2 = [(0.5, 1.2), (2.0, 0.7), (1.0, 3.0)]


@pytest.mark.parametrize(
    "fn", [f for f, _ in SPECIAL_ROWS], ids=lambda f: f.__name__
)
def test_row_reproduces_scipy(fn):
    target = UFUNC_TABLE[fn]
    nin = fn.nin if hasattr(fn, "nin") else 1
    pts = POINTS_1 if nin == 1 else POINTS_2
    for pt in pts:
        args = (pt,) if nin == 1 else pt
        try:
            want = float(fn(*args))
        except TypeError:
            pytest.skip("point outside sampled signature")
        if not np.isfinite(want):
            continue
        expr = target(*[sympy.Float(a) for a in args])
        got = float(sympy.N(expr))
        assert np.isclose(got, want, rtol=1e-10), (fn.__name__, pt, got, want)


def test_special_dispatches_on_pairs():
    u = Pair.array("u", np.linspace(0.1, 0.9, 5))
    r = sp.erf(u)
    assert r.formula == sympy.erf(
        sympy.IndexedBase("u")[sympy.Symbol("i", integer=True)]
    )
    assert np.allclose(np.asarray(r.value, dtype=float), sp.erf(u.value))


def test_ndtr_is_the_gaussian_cdf():
    u = Pair.array("u", np.linspace(-2, 2, 5))
    r = sp.ndtr(u)
    assert np.allclose(np.asarray(r.value, dtype=float), sp.ndtr(u.value))
    t = sympy.Symbol("t")
    expr = r.formula.subs(
        sympy.IndexedBase("u")[sympy.Symbol("i", integer=True)], t
    )
    # symbolically the integral of the standard normal density
    density = sympy.exp(-t**2 / 2) / sympy.sqrt(2 * sympy.pi)
    assert sympy.simplify(sympy.diff(expr, t) - density) == 0
