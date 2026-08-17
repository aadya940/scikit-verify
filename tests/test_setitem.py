"""Item assignment: the formula becomes a scatter, checked on both lanes."""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy
from skverify.helpers import axis_idx

I, J = axis_idx(0), axis_idx(1)
U = sympy.IndexedBase("u")
D = sympy.IndexedBase("d")


def evaluate_1d(pair, arrays):
    mapping = {}
    for name, arr in arrays.items():
        base = sympy.IndexedBase(name)
        for k, v in enumerate(arr):
            mapping[base[k]] = sympy.Float(v)
    lo, hi = pair._axis_bounds[0]
    out = np.empty(hi - lo)
    for k in range(hi - lo):
        out[k] = float(sympy.N(pair.formula.subs(I, k).doit().xreplace(mapping)))
    return out


class TestIntWrite:
    def test_boundary_write(self):
        u = Pair.array("u", np.zeros(6))
        u[0] = 5.0
        assert u.formula == sympy.Piecewise((5.0, sympy.Eq(I, 0)), (U[I], True))
        assert u.value[0] == 5.0

    def test_negative_index(self):
        u = Pair.array("u", np.zeros(6))
        u[-1] = 3.0
        assert u.formula.args[0][1] == sympy.Eq(I, 5)
        assert u.value[5] == 3.0

    def test_sequential_writes_stack(self):
        u = Pair.array("u", np.arange(4.0))
        u[0] = 0.0
        u[3] = 0.0
        arrs = {"u": np.arange(4.0)}
        assert np.allclose(evaluate_1d(u, arrs), u.value)


class TestSliceWrite:
    def test_stencil_fill(self):
        d = Pair.array("d", np.arange(8.0))
        out = np.zeros_like(d)
        out[1:-1] = d[2:] - d[:-2]
        expected = sympy.Piecewise(
            (D[I + 1] - D[I - 1], sympy.And(sympy.Ge(I, 1), sympy.Lt(I, 7))),
            (0, True),
        )
        assert out.formula == expected
        want = np.zeros(8)
        want[1:-1] = np.arange(8.0)[2:] - np.arange(8.0)[:-2]
        assert np.allclose(out.value, want)

    def test_full_slice_overwrites(self):
        u = Pair.array("u", np.zeros(4))
        v = Pair.array("v", np.arange(4.0))
        u[:] = v * 2.0
        V = sympy.IndexedBase("v")
        assert u.formula == 2.0 * V[I]

    def test_value_reindexed_into_region(self):
        u = Pair.array("u", np.zeros(6))
        v = Pair.array("v", np.arange(3.0))
        u[2:5] = v
        V = sympy.IndexedBase("v")
        assert u.formula.args[0][0] == V[I - 2]
        assert np.allclose(u.value[2:5], np.arange(3.0))

    def test_strided_write_refused(self):
        u = Pair.array("u", np.zeros(6))
        with pytest.raises(NotImplementedError):
            u[::2] = 1.0


class TestMaskedWrite:
    def test_scalar_under_mask(self):
        m = Pair.array("m", np.arange(5.0))
        m[m > 2.5] = 0.0
        M = sympy.IndexedBase("m")
        assert m.formula == sympy.Piecewise(
            (0.0, sympy.Gt(M[I], 2.5)), (M[I], True)
        )
        assert np.allclose(m.value, [0, 1, 2, 0, 0])

    def test_array_under_mask_refused(self):
        m = Pair.array("m", np.arange(5.0))
        with pytest.raises(NotImplementedError):
            m[m > 2.5] = np.arange(5.0)


class TestAliasing:
    def test_slices_are_value_semantic(self):
        u = Pair.array("u", np.arange(6.0))
        w = u[1:]
        u[1] = 99.0
        assert w.value[0] == 1.0  # a copy, not a numpy view

    def test_scalar_pair_refuses(self):
        x = Pair(3.0, sympy.Symbol("x"))
        with pytest.raises(TypeError):
            x[0] = 1.0


class TestKernels:
    def test_build_then_fill_traces(self):
        def heat_step(u, r):
            dudt = np.zeros_like(u)
            dudt[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]
            return u + r * dudt

        out = to_sympy(heat_step, np.linspace(0, 1, 8) ** 2, 0.1)
        R = sympy.Symbol("r", real=True)
        arrs = np.linspace(0, 1, 8) ** 2
        want = arrs + 0.1 * np.concatenate(
            [[0], arrs[2:] - 2 * arrs[1:-1] + arrs[:-2], [0]]
        )
        assert np.allclose(out.value, want)
        assert out.formula.has(R)

    def test_boundary_conditions_kernel(self):
        def dirichlet(u, bc):
            u = u * 1.0
            u[0] = bc
            u[-1] = bc
            return u

        out = to_sympy(dirichlet, np.arange(5.0), 7.0)
        assert out.value[0] == 7.0 and out.value[4] == 7.0
        assert np.allclose(out.value[1:4], [1, 2, 3])
