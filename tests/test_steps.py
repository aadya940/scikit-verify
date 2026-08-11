""".steps: the derivation, every intermediate formula in execution order."""

import numpy as np
import sympy

from skverify import Pair, to_sympy
from skverify.helpers import axis_idx

I = axis_idx(0)
U = sympy.IndexedBase("u")


class TestSteps:
    def test_last_step_is_the_formula(self):
        u = Pair.array("u", np.arange(4.0))
        r = 2.0 * u[1:] - u[:-1]
        assert r.steps[-1] == r.formula

    def test_chain_in_execution_order(self):
        u = Pair.array("u", np.arange(4.0))
        r = np.exp(u) * 2.0
        assert r.steps == [U[I], sympy.exp(U[I]), 2.0 * sympy.exp(U[I])]

    def test_operands_contribute_their_history(self):
        a = Pair(2.0, sympy.Symbol("a"))
        b = Pair(3.0, sympy.Symbol("b"))
        r = (a + b) * a
        A, B = sympy.Symbol("a"), sympy.Symbol("b")
        assert r.steps == [A, B, A + B, A, (A + B) * A]

    def test_different_code_paths_different_steps(self):
        # the point of the design: steps record what RAN, not what exists
        def kernel(u, flag):
            if flag:
                return np.exp(u)
            return np.sin(u)

        e = to_sympy(kernel, np.arange(3.0), True)
        s = to_sympy(kernel, np.arange(3.0), False)
        assert e.steps == [U[I], sympy.exp(U[I])]
        assert s.steps == [U[I], sympy.sin(U[I])]

    def test_reductions_carry_history(self):
        u = Pair.array("u", np.arange(4.0))
        r = np.sum(np.exp(u))
        assert r.steps[0] == U[I]
        assert r.steps[1] == sympy.exp(U[I])
        assert isinstance(r.steps[-1], sympy.Sum)

    def test_where_merges_all_three_histories(self):
        u = Pair.array("u", np.arange(4.0))
        w = np.where(u, np.exp(u), 0.0 * u)
        assert sympy.exp(U[I]) in w.steps
        assert w.steps[-1] == w.formula
