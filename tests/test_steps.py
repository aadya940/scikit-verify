""".steps: the derivation as a parents DAG, each distinct fact once."""

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
        # a is shared by both branches; the DAG lists it ONCE
        a = Pair(2.0, sympy.Symbol("a"))
        b = Pair(3.0, sympy.Symbol("b"))
        r = (a + b) * a
        A, B = sympy.Symbol("a"), sympy.Symbol("b")
        assert r.steps == [A, B, A + B, (A + B) * A]

    def test_diamond_dedup(self):
        # (a+b) reused twice: eager concat doubled it, the DAG cannot
        a = Pair(2.0, sympy.Symbol("a"))
        b = Pair(3.0, sympy.Symbol("b"))
        s = a + b
        r = s * s + s
        assert r.steps.count(sympy.Symbol("a") + sympy.Symbol("b")) == 1

    def test_no_quadratic_blowup(self):
        # eager concat gave O(n^2) steps for a fold chain; DAG gives O(n)
        u = Pair.array("u", np.arange(4.0))
        acc = u
        for _ in range(50):
            acc = acc + u
        assert len(acc.steps) == 51

    def test_setitem_keeps_prewrite_state(self):
        u = Pair.array("u", np.arange(4.0))
        v = np.exp(Pair.array("v", np.arange(4.0)))
        u[1:3] = v[1:3]
        steps = u.steps
        assert steps[-1] == u.formula  # the scatter Piecewise
        assert U[I] in steps  # the pre-write state survives as a parent
        V = sympy.IndexedBase("v")
        assert sympy.exp(V[I]) in steps  # the value's own history rides in

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


class TestCseSteps:
    def _ols_slope(self):
        x = Pair.array("x", np.arange(4.0))
        y = Pair.array("y", np.arange(4.0) * 2.0)
        sx, sy = np.sum(x), np.sum(y)
        sxx, sxy = np.sum(x * x), np.sum(x * y)
        return (4 * sxy - sx * sy) / (4 * sxx - sx * sx)

    def test_substituting_back_is_exact(self):
        # the workstream rule: the pretty view can never drift from truth
        r = self._ols_slope()
        assignments, steps = r.cse_steps()
        for reduced, original in zip(steps, r.steps):
            for sym, expr in reversed(assignments):
                reduced = reduced.subs(sym, expr)
            assert reduced == original

    def test_shared_sum_named_once(self):
        r = self._ols_slope()
        assignments, _ = r.cse_steps()
        text = " ; ".join(str(e) for _, e in assignments)
        # Sum(x[j], ...) is used twice in the slope but assigned once
        assert text.count("Sum(x[j], (j, 0, 3))") == 1

    def test_bound_index_stays_inside_its_sum(self):
        # cse must not hoist u[j]*v[j] (bound j) to a top-level name
        u = Pair.array("u", np.arange(4.0))
        v = Pair.array("v", np.arange(4.0))
        r = np.sum(u * v) + np.sum(u * v) * 2.0
        assignments, steps = r.cse_steps()
        for _, expr in assignments:
            if not expr.has(sympy.Sum):
                assert not any(
                    str(s) == "j" for s in expr.free_symbols
                ), f"bound index leaked into assignment {expr}"
        for reduced, original in zip(steps, r.steps):
            for sym, expr in reversed(assignments):
                reduced = reduced.subs(sym, expr)
            assert reduced == original

    def test_derivation_reads_top_to_bottom(self):
        r = self._ols_slope()
        text = r.derivation()
        assert "t0 = " in text
        assert text.splitlines()[-1].startswith("result: ")

    def test_derivation_on_write_history(self):
        u = Pair.array("u", np.zeros(4))
        v = np.exp(Pair.array("v", np.arange(4.0)))
        u[1:3] = v[1:3]
        text = u.derivation()
        assert text.splitlines()[-1].startswith("result: ")
        assert "Piecewise" in text
