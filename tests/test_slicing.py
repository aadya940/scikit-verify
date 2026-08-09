"""slice lifting, domains, alignment."""

import numpy as np
import pytest
import sympy

from skverify import Pair, IDX

U = sympy.IndexedBase("u")


def make(n=12):
    return Pair.array("u", np.linspace(0.0, 1.0, n))


class TestSliceLifting:
    def test_forward_shift(self):
        u = make()
        assert make()[1:].formula == U[IDX + 1]
        assert u[1:].domain == (0, 11)

    def test_first_difference(self):
        u = make()
        d = u[1:] - u[:-1]
        assert sympy.simplify(d.formula - (U[IDX + 1] - U[IDX])) == 0
        assert d.domain == (0, 11)
        assert np.allclose(d.value, np.diff(u.value))

    def test_second_difference(self):
        u = make()
        d = u[2:] - 2 * u[1:-1] + u[:-2]
        assert sympy.simplify(d.formula - (U[IDX + 2] - 2 * U[IDX + 1] + U[IDX])) == 0
        assert d.domain == (0, 10)
        assert np.allclose(d.value, np.diff(u.value, 2))

    def test_interior_slice(self):
        u = make()
        v = u[2:-3]
        assert v.formula == U[IDX + 2]
        assert v.domain == (0, 7)
        assert np.allclose(v.value, u.value[2:-3])

    def test_slice_of_slice(self):
        u = make()
        v = u[1:][2:]  # shifts compose: +1 then +2
        assert v.formula == U[IDX + 3]
        assert v.domain == (0, 9)

    def test_negative_start(self):
        u = make()
        v = u[-4:]
        assert v.formula == U[IDX + 8]
        assert v.domain == (0, 4)

    def test_full_slice_identity(self):
        u = make()
        v = u[:]
        assert v.formula == U[IDX]
        assert v.domain == (0, 12)


class TestAlignment:
    def test_matching_lengths_merge(self):
        u = make()
        assert (u[1:] - u[:-1]).domain == (0, 11)

    def test_mismatch_raises(self):
        u = make()
        with pytest.raises(ValueError):
            u[1:] - u[:-2]

    def test_scalar_broadcasts(self):
        u = make()
        assert (2.0 * u[1:]).domain == (0, 11)


class TestSliceRefusals:
    def test_step_slice_lifts(self):
        # gained with stride support: n=12, every other point
        u = make()
        assert u[::2].formula == U[2 * IDX]
        assert u[::2].domain == (0, 6)
        assert np.allclose(u[::2].value, u.value[::2])

    def test_reverse_lifts(self):
        u = make()
        assert u[::-1].formula == U[11 - IDX]
        assert u[::-1].domain == (0, 12)
        assert np.allclose(u[::-1].value, u.value[::-1])

    def test_integer_index_gives_scalar(self):
        # gained when 1-D merged into the N-D path
        u = make()
        assert u[3].formula == U[3]
        assert u[3].domain is None
        assert u[3].value == u.value[3]


class TestEdgeShapes:
    def test_len_two(self):
        u = Pair.array("u", np.array([1.0, 2.0]))
        d = u[1:] - u[:-1]
        assert d.domain == (0, 1)
        assert d.value[0] == 1.0

    def test_empty_slice(self):
        u = make()
        v = u[5:5]
        assert v.domain == (0, 0)
