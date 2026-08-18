"""Gate 3: make_lsq_spline lifts end to end.

New walls crossed, all general: .view(float) as a neutral method,
newaxis indexing routed through extent-1 reshape, tuple-returning
Cython kernels (data_matrix, fpback) as multi-output atoms, and
multi-out-parameter in-place kernels (qr_reduce mutates A and y_w)
rewritten to tuple assignments of opaque atoms."""

import numpy as np
from scipy.interpolate import make_lsq_spline

from skverify import to_sympy
from skverify.pair import _OPAQUE, Pair

T = np.r_[(0.0,) * 4, np.linspace(5, 25, 5), (30.0,) * 4]


def drive(x, y):
    return make_lsq_spline(x, y, t=T)


def test_make_lsq_spline_lifts():
    x = np.linspace(0, 30)
    y = 8 * x + 50
    spl = to_sympy(drive, x, y)

    assert all(isinstance(c, Pair) for c in spl.c)
    ref = make_lsq_spline(x, y, t=T)
    assert np.allclose(np.array([c.value for c in spl.c]), ref.c)

    names = [entry[-1][0] for entry in _OPAQUE]
    assert any(n.startswith("data_matrix") for n in names)
    assert any(n.startswith("qr_reduce") for n in names)
    assert any(n.startswith("fpback") for n in names)
    assert "fpback" in str(spl.c[0].formula)
