"""Broad exercise of the numpy dialect and Pair operations: mapped
functions traced end to end, and refusal branches confirmed to fire.
Value-lane equals numpy throughout."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy, Pair

V = np.array([0.7, -1.2, 2.5, 0.3, 0.4])
M = np.arange(6.0).reshape(2, 3) + 1


def _val(fn, *a):
    out = to_sympy(fn, *a)
    return np.asarray(out.value, dtype=float)


@pytest.mark.parametrize("fn,arg", [
    (lambda v: np.cumsum(v), V), (lambda v: np.cumprod(v), V),
    (lambda v: np.diff(v), V), (lambda v: np.ediff1d(v), V),
    (lambda v: np.gradient(v), V), (lambda v: np.trapezoid(v), V),
    (lambda v: np.ptp(v), V), (lambda v: np.clip(v, 0.0, 1.0), V),
    (lambda v: np.sort(v), V), (lambda v: np.argmax(v), V),
    (lambda v: np.cumsum(v).sum(), V), (lambda v: np.flip(v), V),
    (lambda v: np.roll(v, 2), V), (lambda v: np.tile(v, 2), V),
    (lambda v: np.repeat(v, 2), V), (lambda v: np.abs(v).sum(), V),
    (lambda v: np.maximum(v, 0.0), V), (lambda v: np.sign(v), V),
    (lambda v: np.cumsum(v) - v, V), (lambda v: np.where(v > 0, v, -v), V),
])
def test_vector_functions_match_numpy(fn, arg):
    assert np.allclose(_val(fn, arg.copy()),
                       np.asarray(fn(arg.copy()), dtype=float), equal_nan=True)


@pytest.mark.parametrize("fn,arg", [
    (lambda a: a.T @ a, M), (lambda a: a @ a.T, M),
    (lambda a: np.trace(a @ a.T), M), (lambda a: a.reshape(-1).sum(), M),
    (lambda a: np.transpose(a).sum(), M), (lambda a: a.ravel().sum(), M),
    (lambda a: np.outer(a[0], a[1]).sum(), M),
    (lambda a: np.kron(a[0], a[1]).sum(), M),
])
def test_matrix_functions_match_numpy(fn, arg):
    assert np.allclose(_val(fn, arg.copy()),
                       np.asarray(fn(arg.copy()), dtype=float))


class TestRefusals:
    def test_astype_int_refuses(self):
        with pytest.raises(NotImplementedError):
            to_sympy(lambda a: a.astype(int).mean(), np.array([1.4, 2.6]))

    def test_fft_norm_variant_seals(self):
        out = to_sympy(lambda v: np.fft.fft(v, norm="ortho"), V.copy())
        assert any("fft" in str(r[0]) for r in out.unchecked
                   if isinstance(r, tuple))

    def test_gradient_repeated_axis(self):
        with pytest.raises(ValueError, match="repeated axis"):
            to_sympy(lambda a: np.gradient(a, axis=(0, 0)),
                     np.ones((3, 4)))


class TestPairOperations:
    def test_getitem_slice_and_int(self):
        p = Pair.array("v", V.copy())
        assert np.isclose(float(p[0].value), V[0])
        assert np.allclose(np.asarray(p[1:3].value, float), V[1:3])

    def test_inplace_add(self):
        def f(v):
            v = v.copy()
            v += 1.0
            return v.sum()
        assert np.isclose(_val(f, V.copy()), (V + 1.0).sum())

    def test_dtype_reports_numeric(self):
        p = Pair.array("v", V.copy())
        assert p.dtype == np.dtype(float)

    def test_reshape_and_transpose_methods(self):
        def f(a):
            return a.reshape(3, 2).T.sum()
        assert np.isclose(_val(f, M.copy()), M.reshape(3, 2).T.sum())

    def test_comparison_returns_mask_semantics(self):
        def f(v):
            return v[v > 0].sum()
        assert np.isclose(_val(f, V.copy()), V[V > 0].sum())
