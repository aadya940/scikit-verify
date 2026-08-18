"""The gate: make_smoothing_spline, unmodified scipy, traced end to end.

x (knots) and y (data) are both symbolic; the returned BSpline carries
coefficient formulas in terms of x[j] and opaque solver atoms, scipy's
own input validation appears as recorded guards, and the values match
the untraced function exactly.
"""

import numpy as np
import sympy
from scipy.interpolate import make_smoothing_spline

from skverify import to_sympy
from skverify.pair import _GUARDS, _OPAQUE, Pair


def drive(x, y):
    # module-level import above: instrumentation reaches callees via
    # the function's globals
    return make_smoothing_spline(x, y, lam=0.1)


def test_make_smoothing_spline_lifts():
    x = np.linspace(0, 4, 10)
    y = x**2 + 0.1 * np.sin(5 * x)
    spl = to_sympy(drive, x, y)

    assert spl.c.dtype == object
    assert all(isinstance(c, Pair) for c in spl.c)

    X = sympy.IndexedBase("x")
    assert spl.c[0].formula.has(X[0])  # symbolic in the knots

    records = {entry[0]: dict(entry[1]) for entry in _OPAQUE}
    assert records["design_matrix"]["residual"] == "ok"  # matches sympy's basis
    assert records["solve_banded"]["residual"] == "ok"  # A @ x == b, this run

    assert any(g.has(X[0]) for g in _GUARDS)  # ascending-x validation recorded

    ref = make_smoothing_spline(x, y, lam=0.1)
    ours = np.array([c.value for c in spl.c], dtype=float)
    assert np.allclose(ours, ref.c)
