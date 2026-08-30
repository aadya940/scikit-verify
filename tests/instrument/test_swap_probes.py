"""The memoized rewriter must match sympy's scope-aware substitution
exactly, and the API exit must hand users evaluated normal forms."""

import numpy as np
import pytest
import sympy

from skverify.helpers import (
    axis_idx,
    has_probe,
    ops_capped,
    reevaluated,
    swap_probes,
    tree_nodes_capped,
)

I, J = axis_idx(0), axis_idx(1)
U = sympy.IndexedBase("u")
D = sympy.Dummy("probe")


class TestSwapSemantics:
    def test_plain_symbol(self):
        e = D**2 + sympy.sin(D)
        assert swap_probes(e, {D: I + 1}, I) == e.subs(D, I + 1)

    def test_simultaneous_swap(self):
        e = U[I, J] + I * J
        want = e.subs({I: J, J: I}, simultaneous=True)
        assert swap_probes(e, {I: J, J: I}, I) == want

    def test_bound_sum_dummy_shadows(self):
        e = sympy.Sum(U[J] * I, (J, 0, 4)) + J
        want = e.subs({I: I + 1, J: I}, simultaneous=True)
        assert swap_probes(e, {I: I + 1, J: I}, I) == want

    def test_nested_binders(self):
        k = sympy.Symbol("k", integer=True)
        e = sympy.Sum(sympy.Product(U[J] + k, (k, 0, J)), (J, 0, 3)) + k
        want = e.subs({k: 7, J: 9}, simultaneous=True)
        assert swap_probes(e, {k: sympy.Integer(7), J: sympy.Integer(9)}, I) == want

    def test_indexed_slot(self):
        lbl = sympy.Symbol("slot")
        sl = sympy.IndexedBase(lbl)
        meaning = U[I] * 2 + 1
        got = swap_probes(sl[3] + sl[J], {lbl: meaning}, I)
        assert got == (U[3] * 2 + 1) + (U[J] * 2 + 1)

    def test_indexed_slot_meaning_with_bound_axis(self):
        # a meaning that BINDS the axis symbol must keep its binder
        lbl = sympy.Symbol("slot")
        sl = sympy.IndexedBase(lbl)
        meaning = sympy.Sum(U[I], (I, 0, 4))  # closed over i
        got = swap_probes(sl[2], {lbl: meaning}, I)
        assert got == meaning  # nothing free to substitute

    def test_base_label_never_malformed(self):
        lbl = sympy.Symbol("slot")
        sl = sympy.IndexedBase(lbl)
        got = swap_probes(sl[J] + lbl, {lbl: U[I] + 1}, I)
        # the Indexed slot substitutes elementwise, the bare label wholesale
        assert got == (U[J] + 1) + (U[I] + 1)

    def test_dag_shape_terminates_fast(self):
        big = sympy.Integer(0)
        for k in range(200):
            big = sympy.Piecewise((U[k] + big + D, sympy.Eq(I, k)), (big, True))
        out = swap_probes(big, {D: sympy.Integer(1)}, I)
        assert not has_probe(out, {D})

    def test_unevaluated_context_respected(self):
        with sympy.evaluate(False):
            got = swap_probes(D + D, {D: sympy.Integer(2)}, I)
        assert got.func is sympy.Add  # not collapsed to 4 while held
        assert sympy.sympify(str(got)) == 4


class TestSizeGuards:
    def test_ops_capped_exact_when_small(self):
        e = U[I] * 2 + sympy.sin(U[J])
        assert ops_capped(e, 1000) == int(sympy.count_ops(e))

    def test_ops_capped_none_when_over(self):
        e = sympy.Add(*[U[k] * k for k in range(50)])
        assert ops_capped(e, 3) is None

    def test_dag_never_hangs(self):
        big = sympy.Integer(0)
        for k in range(400):
            big = sympy.Piecewise((U[k] + big, sympy.Eq(I, k)), (big, True))
        assert ops_capped(big, 10_000) is None  # returns, quickly
        assert tree_nodes_capped(big, 50_000) is None


class TestReevaluated:
    def test_normalizes_held_arithmetic(self):
        with sympy.evaluate(False):
            e = sympy.Add(sympy.Integer(2), sympy.Integer(3)) * sympy.Abs(
                sympy.Integer(-4)
            )
        assert reevaluated(e) == 20

    def test_sum_passes_verbatim(self):
        s = sympy.Sum(U[J], (J, 0, 4))
        assert reevaluated(s) is s

    def test_ancestor_of_held_untouched(self):
        from skverify.recurrence import Iterate

        it = Iterate(
            sympy.Lambda((J,), J + 1), sympy.Integer(0), sympy.Integer(3)
        )
        with sympy.evaluate(False):
            mixed = sympy.Abs(it) + sympy.Add(sympy.Integer(1), sympy.Integer(1))
        out = reevaluated(mixed)
        assert out.has(Iterate)  # held survives, no eval hang path

    def test_clean_sibling_still_normalizes(self):
        s = sympy.Sum(U[J], (J, 0, 4))
        with sympy.evaluate(False):
            clean = sympy.Add(sympy.Integer(2), sympy.Integer(3))
            e = sympy.Add(s, clean)
        out = reevaluated(e)
        assert out == s + 5


class TestExitNormalForm:
    def test_traced_formula_is_evaluated(self):
        from skverify import to_sympy

        def f(x):
            return np.abs(x * 2.0) + x.sum()

        out = to_sympy(f, np.array([1.0, -2.0, 3.0]))
        f_ = out.formula
        elements = list(f_) if hasattr(f_, "__len__") else [f_]
        for e in elements:
            if isinstance(e, sympy.Basic) and not e.has(sympy.Sum, sympy.Product):
                rebuilt = e.func(*e.args) if e.args else e
                assert rebuilt == e  # already in evaluated normal form
