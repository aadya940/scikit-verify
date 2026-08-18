"""Class-based APIs lift: the twin mechanism (instrumented method
hierarchies, explicit super(), runtime class twinning in the doorman)."""

import numpy as np
from scipy.interpolate import CubicSpline, KroghInterpolator

from skverify import Pair, to_sympy


def _cubic(x, y):
    return CubicSpline(x, y)


def _krogh(x, y):
    return KroghInterpolator(x[:6], y[:6])


def test_cubic_spline_lifts():
    x = np.linspace(0, 30)
    y = np.sin(x) + x**2 / 100
    spl = to_sympy(_cubic, x, y)
    ref = CubicSpline(x, y)
    got = Pair._value_of(np.asarray(spl.c))
    assert np.allclose(np.asarray(got, dtype=float), ref.c)


def test_krogh_lifts():
    x = np.linspace(0, 30)
    y = np.sin(x)
    spl = to_sympy(_krogh, x, y)
    ref = KroghInterpolator(x[:6], y[:6])
    got = Pair._value_of(np.asarray(getattr(spl, "c")))
    assert np.allclose(np.asarray(got, dtype=float), np.asarray(ref.c, dtype=float))
