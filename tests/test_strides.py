"""Strided slices: steps and flips as affine index maps.

u = [10, 20, 30, 40, 50]
u[::-1] -> u[4 - i]     bounds (0, 5)
u[::2]  -> u[2*i]       bounds (0, 3)
u[1::2] -> u[2*i + 1]   bounds (0, 2)
Every case also checked differentially against real numpy.
"""

import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx

I, J = axis_idx(0), axis_idx(1)
U = sympy.IndexedBase("u")


def make(n=5):
    return Pair.array("u", np.arange(1.0, n + 1))


def same(a, b):
    return sympy.simplify(a - b) == 0


class TestFlip:
    def test_flip_formula(self):
        u = make(5)
        r = u[::-1]
        assert same(r.formula, U[4 - I])
        assert r.domain == (0, 5)
        assert np.allclose(r.value, u.value[::-1])

    def test_flip_roundtrip(self):
        u = make(5)
        r = u[::-1][::-1]
        assert same(r.formula, U[I])
        assert r.domain == (0, 5)
        assert np.allclose(r.value, u.value)

    def test_flip_of_slice(self):
        u = make(6)
        r = u[1:5][::-1]  # [20,30,40,50] flipped -> [50,40,30,20]
        assert np.allclose(r.value, u.value[1:5][::-1])
        assert same(r.formula, U[4 - I])  # positions 4,3,2,1
        assert r.domain == (0, 4)


class TestSteps:
    def test_even_step(self):
        u = make(5)
        r = u[::2]  # [10, 30, 50]
        assert same(r.formula, U[2 * I])
        assert r.domain == (0, 3)
        assert np.allclose(r.value, u.value[::2])

    def test_offset_step(self):
        u = make(5)
        r = u[1::2]  # [20, 40]
        assert same(r.formula, U[2 * I + 1])
        assert r.domain == (0, 2)
        assert np.allclose(r.value, u.value[1::2])

    def test_step_with_stop(self):
        u = make(10)
        r = u[1:8:3]  # indices 1, 4, 7
        assert same(r.formula, U[3 * I + 1])
        assert r.domain == (0, 3)
        assert np.allclose(r.value, u.value[1:8:3])

    def test_negative_step_with_bounds(self):
        u = make(6)
        r = u[4:0:-2]  # indices 4, 2
        assert np.allclose(r.value, u.value[4:0:-2])
        assert same(r.formula, U[4 - 2 * I])
        assert r.domain == (0, 2)


class TestComposition:
    def test_step_then_flip(self):
        u = make(7)
        r = u[::2][::-1]  # indices 0,2,4,6 -> 6,4,2,0
        assert np.allclose(r.value, u.value[::2][::-1])
        assert same(r.formula, U[6 - 2 * I])
        assert r.domain == (0, 4)

    def test_flip_then_step(self):
        u = make(7)
        r = u[::-1][::2]  # indices 6,5,4,3,2,1,0 -> 6,4,2,0
        assert np.allclose(r.value, u.value[::-1][::2])
        assert same(r.formula, U[6 - 2 * I])
        assert r.domain == (0, 4)

    def test_step_of_step(self):
        u = make(9)
        r = u[::2][::2]  # indices 0,2,4,6,8 -> 0,4,8
        assert np.allclose(r.value, u.value[::2][::2])
        assert same(r.formula, U[4 * I])
        assert r.domain == (0, 3)


class TestNd:
    def test_mixed_2d(self):
        u = Pair.array("u", np.arange(28.0).reshape(4, 7))
        r = u[::2, ::-1]
        assert np.allclose(r.value, u.value[::2, ::-1])
        assert same(r.formula, U[2 * I, 6 - J])
        assert r.domain == ((0, 2), (0, 7))

    def test_stride_then_stencil(self):
        # coarsened stencil: the every-other-point difference
        u = make(9)
        d = u[::2][1:] - u[::2][:-1]
        assert np.allclose(d.value, np.diff(u.value[::2]))
        assert same(d.formula, U[2 * I + 2] - U[2 * I])
        assert d.domain == (0, 4)
