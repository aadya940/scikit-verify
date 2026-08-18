"""table entries + the parametrized differential loop."""

import numpy as np
import pytest
import sympy

from skverify import Pair, IDX
from skverify.registry import UFUNC_TABLE

U = sympy.IndexedBase("u")


def make():
    return Pair.array("u", np.random.default_rng(0).uniform(0.15, 0.85, 8))


class TestNamedEntries:
    def test_sin(self):
        u = make()
        assert np.sin(u).formula == sympy.sin(U[IDX])
        assert np.allclose(np.sin(u).value, np.sin(u.value))

    def test_renamed_arcsin(self):
        assert np.arcsin(make()).formula == sympy.asin(U[IDX])

    def test_binary_maximum(self):
        u = make()
        m = np.maximum(u[1:], u[:-1])
        assert m.formula == sympy.Max(U[IDX + 1], U[IDX])
        assert m.domain == (0, 7)

    def test_priority_interop(self):
        u = make()
        assert (2.0 * np.exp(u)).formula == 2.0 * sympy.exp(U[IDX])


ELEMENTWISE = [
    (np_fn, sp_fn)
    for np_fn, sp_fn in UFUNC_TABLE.items()
    if np_fn.nin == 1  # unary only for the loop
]

SAFE = {"arccosh": (1.1, 3.0), "arctanh": (-0.9, 0.9)}


@pytest.mark.parametrize(
    "np_fn,sp_fn", ELEMENTWISE, ids=[f.__name__ for f, _ in ELEMENTWISE]
)
def test_differential_whole_table(np_fn, sp_fn):
    """Every table entry, forever: formula evaluated == value computed."""
    # NumPy and SymPy may disagree outside the domain.
    lo, hi = SAFE.get(np_fn.__name__, (0.15, 0.85))
    u = Pair.array("u", np.random.default_rng(0).uniform(lo, hi, 8))
    out = np_fn(u)
    for k in range(3):
        evaluated = float(out.formula.subs(U[IDX], u.value[k]))
        assert evaluated == pytest.approx(out.value[k], rel=1e-9)


class TestRefusals:
    def test_unmapped_ufunc(self):
        with pytest.raises(NotImplementedError):
            np.frexp(make())

    def test_add_reduce_is_sum(self):
        r = np.add.reduce(make())
        assert isinstance(r.formula, sympy.Sum)

    def test_max_reduce_is_lazy_max(self):
        r = np.maximum.reduce(make())
        assert isinstance(r.formula, sympy.Max)
        assert float(r.value) == make().value.max()

    def test_out_kwarg_refused(self):
        u = make()
        with pytest.raises(NotImplementedError):
            np.sin(u, out=np.empty(8))
