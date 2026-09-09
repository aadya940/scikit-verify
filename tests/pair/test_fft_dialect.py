"""The FFT family as dialect entries: the DFT is a closed form at any
n, so fft/ifft/rfft emit Sums instead of sealing as atoms, and
transform mathematics (round trips, linearity, Parseval) becomes
checkable instead of opaque."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy
from skverify.helpers import axis_idx
from skverify.testing import check_formula

V = sympy.IndexedBase("v")
i = sympy.Symbol("i", integer=True)
VALS = np.array([0.7, -1.2, 2.5, 0.3])
N = 4


class TestFormulaEqualsValue:
    @pytest.mark.parametrize("fn,np_fn", [
        (lambda v: np.fft.fft(v), np.fft.fft),
        (lambda v: np.fft.rfft(v), np.fft.rfft),
        (lambda v: np.fft.ifft(v), np.fft.ifft),
    ], ids=["fft", "rfft", "ifft"])
    def test_every_entry(self, fn, np_fn):
        out = to_sympy(fn, VALS.copy())
        want = np_fn(VALS)
        subs = {V[k]: VALS[k] for k in range(N)}
        k0 = axis_idx(0)
        for kk in range(len(np.atleast_1d(want))):
            got = complex(sympy.N(out.formula.subs(k0, kk).doit().xreplace(subs)))
            assert abs(got - complex(want[kk])) < 1e-10, kk

    def test_roundtrip_composes(self):
        out = to_sympy(lambda v: np.fft.ifft(np.fft.fft(v)), VALS.copy())
        subs = {V[k]: VALS[k] for k in range(N)}
        k0 = axis_idx(0)
        for kk in range(N):
            got = complex(sympy.N(out.formula.subs(k0, kk).doit().xreplace(subs)))
            assert abs(got - VALS[kk]) < 1e-10


class TestTransformMathematics:
    def test_linearity_is_exact(self):
        # fft(a + b) - fft(a) - fft(b) == 0, symbolically
        U, W = sympy.IndexedBase("u"), sympy.IndexedBase("w")

        def lin(u, w):
            return (np.fft.fft(u + w) - np.fft.fft(u) - np.fft.fft(w)).real

        v = check_formula(
            lin, (VALS.copy(), VALS[::-1].copy()),
            sympy.Integer(0) * U[i] * W[i], indices=(i,), explore=False,
        )
        assert v.matches, v.message()

    def test_parseval(self):
        # sum |X[k]|^2 == n * sum v[j]^2 for real input
        def power(v):
            return (np.abs(np.fft.fft(v)) ** 2).sum()

        j = sympy.Dummy("j", integer=True)
        spec = N * sympy.Sum(V[j] ** 2, (j, 0, N - 1))
        v = check_formula(power, (VALS.copy(),), spec, explore=False)
        assert v.matches, v.message()

    def test_dc_component_is_the_sum(self):
        # X[0] is just the sum of the samples
        def dc(v):
            return np.fft.fft(v).real[0]

        j = sympy.Dummy("j", integer=True)
        v = check_formula(dc, (VALS.copy(),),
                          sympy.Sum(V[j], (j, 0, N - 1)), explore=False)
        assert v.matches, v.message()


class TestHonestDegradation:
    """Unsupported variants fall back to the sealed-atom path: closed
    form where the dialect speaks, disclosed atom elsewhere, never a
    hard failure for working numpy code."""

    def test_norm_variant_degrades_to_atom(self):
        out = to_sympy(lambda v: np.fft.fft(v, norm="ortho"), VALS.copy())
        assert "fft_0" in str(out.formula)
        assert any("fft" in str(r[0]) for r in out.unchecked
                   if isinstance(r, tuple))

    def test_2d_degrades_to_atom(self):
        out = to_sympy(lambda a: np.fft.fft(a), np.arange(4.0).reshape(2, 2))
        assert "fft_0" in str(out.formula)
        v = np.asarray(out.value)
        assert np.allclose(v, np.fft.fft(np.arange(4.0).reshape(2, 2)))
