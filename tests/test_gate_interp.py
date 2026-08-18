"""Gate 2: make_interp_spline lifts end to end.

Walls crossed on the way (all general mechanisms): Pair.size/.flags
duck facts, astype/copy passthrough for object arrays, concrete-lane
isfinite checks, out-parameter Cython calls (_coloc) as assigned
opaque atoms, neutral functions passed by reference, multi-output
LAPACK atoms (gbsv), opaque calls on copied buffers so overwrite_*
routines cannot mutate traced inputs, and extent-1 reshape."""

import numpy as np
from scipy.interpolate import make_interp_spline

from skverify import to_sympy
from skverify.pair import _OPAQUE, Pair


def drive(x, y):
    return make_interp_spline(x, y)


def test_make_interp_spline_lifts():
    x = np.linspace(0, 30)
    y = 8 * x + 50
    spl = to_sympy(drive, x, y)

    assert all(isinstance(c, Pair) for c in spl.c)
    ref = make_interp_spline(x, y)
    assert np.allclose(np.array([c.value for c in spl.c]), ref.c)

    names = [entry[-1][0] for entry in _OPAQUE]
    assert any(n.startswith("coloc") for n in names)  # collocation atom
    assert any("dgbsv" in n for n in names)  # banded-solve atom

    # the solution atom flows into the coefficients
    assert "dgbsv" in str(spl.c[0].formula)
