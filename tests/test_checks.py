"""The first validators: three-valued verdicts, symbolic decisions."""

import numpy as np
import sympy

from skverify import Pair, to_sympy
from skverify import checks
from skverify.helpers import axis_idx

I = axis_idx(0)
U = sympy.IndexedBase("u")
P = sympy.IndexedBase("p")


def birth_death(p, b, d):
    stay = 1.0 - b - d
    return b * p[:-2] + stay * p[1:-1] + d * p[2:]


def leaky(p, b, d):
    return b * p[:-2] + (1.0 - b - d) * p[1:-1]


class TestAgainst:
    def test_matching_formula_proven(self):
        u = Pair.array("u", np.arange(5.0))
        d = u[1:] - u[:-1]
        e = checks.against(d, U[I + 1] - U[I])
        assert e.verdict == checks.PROVEN

    def test_differing_formula_refuted_with_residual(self):
        u = Pair.array("u", np.arange(5.0))
        d = u[1:] - u[:-1]
        e = checks.against(d, U[I + 2] - U[I])
        assert e.verdict == checks.REFUTED
        assert e.detail != 0

    def test_reference_as_string_like_expr(self):
        u = Pair.array("u", np.arange(5.0))
        r = u * 2.0
        assert checks.against(r, 2.0 * U[I]).verdict == checks.PROVEN


class TestConservesMass:
    def test_full_update_proven(self):
        b, d = sympy.symbols("b d", real=True)
        out = to_sympy(birth_death, np.full(16, 1 / 16), 0.3, 0.2)
        e = checks.conserves_mass(out)
        assert e.verdict == checks.PROVEN

    def test_leak_refuted(self):
        out = to_sympy(leaky, np.full(16, 1 / 16), 0.3, 0.2)
        e = checks.conserves_mass(out)
        assert e.verdict == checks.REFUTED

    def test_scaling_refuted(self):
        u = Pair.array("u", np.arange(5.0))
        e = checks.conserves_mass(u * 2.0)
        assert e.verdict == checks.REFUTED
        assert sympy.simplify(e.detail - 1) == 0  # excess mass, 2 - 1


class TestCentered:
    def test_laplacian_proven(self):
        u = Pair.array("u", np.arange(6.0))
        lap = u[2:] - 2 * u[1:-1] + u[:-2]
        e = checks.centered(lap, at=1)
        assert e.verdict == checks.PROVEN

    def test_forward_difference_refuted(self):
        u = Pair.array("u", np.arange(6.0))
        fwd = u[1:] - u[:-1]
        e = checks.centered(fwd, at=0)
        assert e.verdict == checks.REFUTED

    def test_kdv_bad_derivative_off_center(self):
        u = Pair.array("u", np.arange(8.0))
        good = u[2:] - u[:-2]     # offsets 0, 2 about 1
        bad_shifted = u[3:] - u[1:-2]  # offsets 1, 3: not about 1
        assert checks.centered(good, at=1).verdict == checks.PROVEN
        assert checks.centered(bad_shifted, at=1).verdict == checks.REFUTED

    def test_no_letters_unknown(self):
        x = Pair(3.0, sympy.Symbol("x"))
        assert checks.centered(x).verdict == checks.UNKNOWN


class TestChecksOnRealKernels:
    def test_kdv_bad_derivative_refuted_end_to_end(self):
        u = Pair.array("u", np.cosh(np.linspace(-3, 3, 12)) ** -2)
        good_ux = (u[2:] - u[:-2])
        bad_ux = (u[3:] - u[1:-2])
        assert checks.centered(good_ux, at=1).verdict == checks.PROVEN
        assert checks.centered(bad_ux, at=1).verdict == checks.REFUTED

    def test_markov_2d_walk_conserves_interior(self):
        def walk_step(p, left, right, up, down):
            stay = 1.0 - left - right - up - down
            return (
                stay * p[1:-1, 1:-1]
                + right * p[1:-1, :-2]
                + left * p[1:-1, 2:]
                + down * p[:-2, 1:-1]
                + up * p[2:, 1:-1]
            )

        out = to_sympy(walk_step, np.full((6, 6), 1 / 36), 0.1, 0.1, 0.1, 0.1)
        assert checks.conserves_mass(out).verdict == checks.PROVEN
