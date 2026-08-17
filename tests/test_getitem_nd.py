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
    def test_int_renames_survivor(self, u):
        # the letter invariant: letter k always means result-axis k
        r = u[2]
        assert r.formula == U[2, I]  # survivor takes axis-0's letter
        assert r.domain == (0, 7)
        assert np.allclose(r.value, u.value[2])

    def test_negative_int(self, u):
        assert u[-1].formula == U[3, I]

    def test_all_ints_scalar(self, u):
        r = u[2, 3]
        assert r.formula == U[2, 3]
        assert r.domain is None
        assert r.value == u.value[2, 3]

    def test_mixed_3d(self, w):
        r = w[1, 1:, 2]
        assert r.formula == W[1, I + 1, 2]  # sole survivor renamed to i
        assert r.domain == (0, 2)
        assert np.allclose(r.value, w.value[1, 1:, 2])

    def test_dropped_row_adds_to_vector(self, u):
        # the bug the invariant fixes: u[2] + v must share ONE letter
        v = Pair.array("v", np.arange(7.0))
        V = sympy.IndexedBase("v")
        r = u[2] + v
        assert sympy.simplify(r.formula - (U[2, I] + V[I])) == 0
        assert np.allclose(r.value, u.value[2] + v.value)


class TestTranspose:
    def test_2d(self, u):
        r = u.T
        assert r.formula == U[J, I]  # new position (a, b) reads old (b, a)
        assert r.domain == ((0, 7), (0, 4))  # non-square: extents swap
        assert np.allclose(r.value, u.value.T)

    def test_round_trip(self, u):
        r = u.T.T
        assert r.formula == U[I, J]
        assert r.domain == ((0, 4), (0, 7))
        assert np.allclose(r.value, u.value)

    def test_np_transpose_dispatches(self, u):
        assert np.transpose(u).formula == U[J, I]

    def test_3d_permutation(self, w):
        r = w.transpose((2, 0, 1))  # result pos 0 reads old axis 2, ...
        assert r.formula == W[J, K, I]
        assert r.domain == ((0, 4), (0, 2), (0, 3))
        assert np.allclose(r.value, w.value.transpose((2, 0, 1)))

    def test_1d_noop(self):
        v = Pair.array("v", np.arange(5.0))
        assert v.T is v  # like numpy: 1-D transpose is itself

    def test_transpose_then_slice(self, u):
        r = u.T[1:, :]  # 7x4, drop first row
        assert r.formula == U[J, I + 1]
        assert r.domain == ((0, 6), (0, 4))
        assert np.allclose(r.value, u.value.T[1:, :])

    def test_transposed_stencil_arithmetic(self, u):
        d = u.T[1:, :] - u.T[:-1, :]
        assert np.allclose(d.value, np.diff(u.value.T, axis=0))
        assert sympy.simplify(d.formula - (U[J, I + 1] - U[J, I])) == 0


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

    def test_fancy_gathers_concrete_positions(self, u):
        out = u[[0, 1]]
        assert out.dtype == object
        assert out[0].formula == U[0, I]
        assert np.allclose(out[1].value, u.value[1])

    def test_bool_scalar(self, u):
        with pytest.raises(NotImplementedError):
            u[True]

    def test_too_many_indices(self, u):
        with pytest.raises(IndexError):
            u[0, 0, 0]

    def test_double_ellipsis(self, u):
        with pytest.raises(IndexError):
            u[..., ...]
