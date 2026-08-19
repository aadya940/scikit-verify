"""the milestone kernels, end to end."""

import numpy as np
import pytest
import sympy

from skverify import Pair, IDX, to_sympy

U = sympy.IndexedBase("u")
c_, dt_, dx_, a_ = (sympy.Symbol(s, real=True) for s in ("c", "dt", "dx", "alpha"))


def upwind(u, c, dt, dx):
    return u[1:] - (c * dt / dx) * (u[1:] - u[:-1])


def ftcs_heat(u, alpha, dt, dx):
    return u[1:-1] + (alpha * dt / dx**2) * (u[2:] - 2 * u[1:-1] + u[:-2])


class TestUpwind:
    def test_formula(self):
        out = to_sympy(upwind, np.linspace(0, 1, 16), 0.9, 0.01, 0.1)
        expect = U[IDX + 1] - c_ * dt_ / dx_ * (U[IDX + 1] - U[IDX])
        assert sympy.simplify(out.formula - expect) == 0
        assert out.domain == (0, 15)

    def test_value_lane(self):
        u, c, dt, dx = np.linspace(0, 1, 16), 0.9, 0.01, 0.1
        out = to_sympy(upwind, u, c, dt, dx)
        assert np.allclose(out.value, upwind(u, c, dt, dx))  # traced == plain run


class TestFTCSHeat:
    def test_formula(self):
        out = to_sympy(ftcs_heat, np.linspace(0, 1, 16), 0.05, 0.01, 0.1)
        expect = U[IDX + 1] + a_ * dt_ / dx_**2 * (U[IDX + 2] - 2 * U[IDX + 1] + U[IDX])
        assert sympy.simplify(out.formula - expect) == 0
        assert out.domain == (0, 14)

    def test_value_lane(self):
        args = (np.linspace(0, 1, 16), 0.05, 0.01, 0.1)
        assert np.allclose(to_sympy(ftcs_heat, *args).value, ftcs_heat(*args))
