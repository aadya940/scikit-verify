"""N-D __getitem__: formulas, bounds, and the value lane, differentially."""

import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx

I, J, K = axis_idx(0), axis_idx(1), axis_idx(2)
U = sympy.IndexedBase("u")
W = sympy.IndexedBase("w")


@pytest.fixture
def u():
    # non-square on purpose: catches axis-order mixups a square shape forgives
    return Pair.array("u", np.arange(28.0).reshape(4, 7))


@pytest.fixture
def w():
    return Pair.array("w", np.arange(24.0).reshape(2, 3, 4))


class TestSlices:
    def test_both_axes(self, u):
        r = u[1:, :-1]
        assert r.formula == U[I + 1, J]
        assert r.domain == ((0, 3), (0, 6))
        assert np.allclose(r.value, u.value[1:, :-1])

    def test_short_key_pads(self, u):
        assert u[1:].formula == u[1:, :].formula
        assert u[1:].domain == u[1:, :].domain

    def test_ellipsis(self, u):
        r = u[..., 1:]
        assert r.formula == U[I, J + 1]
        assert r.domain == ((0, 4), (0, 6))

    def test_negatives(self, u):
        r = u[1:-1, -2:]
        assert r.formula == U[I + 1, J + 5]
        assert r.domain == ((0, 2), (0, 2))
        assert np.allclose(r.value, u.value[1:-1, -2:])

    def test_composition(self, u):
        r = u[1:, :][1:, :]
        assert r.formula == U[I + 2, J]
        assert r.domain == ((0, 2), (0, 7))
        assert np.allclose(r.value, u.value[2:, :])


class TestIntDrops:
    def test_int_keeps_original_symbols(self, u):
        r = u[2]
        assert r.formula == U[2, J]  # survivor stays `j`, NOT renumbered to `i`
        assert r.domain == (0, 7)
        assert np.allclose(r.value, u.value[2])

    def test_negative_int(self, u):
        assert u[-1].formula == U[3, J]

    def test_all_ints_scalar(self, u):
        r = u[2, 3]
        assert r.formula == U[2, 3]
        assert r.domain is None
        assert r.value == u.value[2, 3]

    def test_mixed_3d(self, w):
        r = w[1, 1:, 2]
        assert r.formula == W[1, J + 1, 2]
        assert r.domain == (0, 2)
        assert np.allclose(r.value, w.value[1, 1:, 2])


class TestArithmetic:
    def test_2d_stencil_merges(self, u):
        d = u[1:, :] - u[:-1, :]
        assert sympy.simplify(d.formula - (U[I + 1, J] - U[I, J])) == 0
        assert d.domain == ((0, 3), (0, 7))
        assert np.allclose(d.value, np.diff(u.value, axis=0))

    def test_per_axis_mismatch_refused(self, u):
        with pytest.raises(ValueError):
            u[1:, :] - u[:, 1:]


class TestRefusals:
    def test_step_slice_lifts(self, u):
        # gained with stride support: rows 0 and 2 of the 4x7 array
        r = u[::2, :]
        assert r.formula == U[2 * I, J]
        assert r.domain == ((0, 2), (0, 7))
        assert np.allclose(r.value, u.value[::2, :])

    def test_newaxis(self, u):
        with pytest.raises(NotImplementedError):
            u[:, None]

    def test_fancy(self, u):
        with pytest.raises(NotImplementedError):
            u[[0, 1]]

    def test_bool_scalar(self, u):
        with pytest.raises(NotImplementedError):
            u[True]

    def test_too_many_indices(self, u):
        with pytest.raises(IndexError):
            u[0, 0, 0]

    def test_double_ellipsis(self, u):
        with pytest.raises(IndexError):
            u[..., ...]
