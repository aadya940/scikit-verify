"""Variable capture at broadcast: a reduced operand meeting a higher
rank must never have a free letter renamed onto its own bound dummy."""

import numpy as np
import sympy

from skverify import Pair
from skverify.helpers import axis_idx


def test_reduced_operand_broadcast_is_capture_free():
    X0 = np.arange(12.0).reshape(4, 3)
    u = Pair.array("X", X0)
    # column demean-and-square: the historical silent-capture shape
    s2 = np.sum((u - np.sum(u, axis=0)) ** 2, axis=0)
    vals = {
        sympy.IndexedBase("X")[i, j]: sympy.Float(X0[i, j])
        for i in range(4)
        for j in range(3)
    }
    want = np.sum((X0 - X0.sum(0)) ** 2, axis=0)
    for col in range(3):
        got = float(
            sympy.N(
                s2.formula.subs(axis_idx(0), col).doit().xreplace(vals).doit()
            )
        )
        assert np.isclose(got, want[col])


def test_no_diagonal_artifacts():
    # the bug's signature was X[i, i] terms appearing from nowhere
    u = Pair.array("X", np.arange(12.0).reshape(4, 3))
    d = u - np.sum(u, axis=0)
    i = sympy.Symbol("i", integer=True)
    X = sympy.IndexedBase("X")
    assert not d.formula.has(X[i, i])
