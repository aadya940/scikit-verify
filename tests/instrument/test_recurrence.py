"""Loops as domains: the recurrence folder.

A loop past FOLD_START iterations folds its carried formula into a
held Iterate (constant size, unrolls on demand). Short loops keep
their fully unrolled formulas. A body that stops matching the
template breaks the fold and the eager exact formulas resume.
"""

import numpy as np
import sympy

from skverify import to_sympy
from skverify.recurrence import FOLD_START, Iterate


def nonlinear_growth(x):
    c = x[0]
    for _ in range(25):
        c = c + 0.1 * c * c
    return c


def short_loop(x):
    c = x[0]
    for _ in range(4):
        c = c + 0.1 * c * c
    return c


class TestFold:
    def test_snowball_folds_to_held_iterate(self):
        out = to_sympy(nonlinear_growth, np.array([0.1]))
        f = out.formula
        assert f.func is Iterate
        step = f.args[0]
        s, n = step.variables
        assert step.expr == 0.1 * s**2 + s
        X = sympy.IndexedBase("x")
        got = float(sympy.N(f.subs(X[0], 0.1).doit()))
        expect = float(nonlinear_growth(np.array([0.1])))
        assert np.isclose(got, expect)

    def test_short_loops_stay_unrolled(self):
        out = to_sympy(short_loop, np.array([0.1]))
        assert not out.formula.atoms(Iterate)
        X = sympy.IndexedBase("x")
        got = float(sympy.N(out.formula.subs(X[0], 0.1)))
        assert np.isclose(got, float(short_loop(np.array([0.1]))))

    def test_iteration_indexed_template(self):
        def indexed(x):
            c = x[0]
            for k in range(2, 20):
                c = c + 0.1 * c * c + float(k)
            return c

        out = to_sympy(indexed, np.array([0.1]))
        expect = float(indexed(np.array([0.1])))
        X = sympy.IndexedBase("x")
        got = float(sympy.N(out.formula.subs(X[0], 0.1).doit()))
        assert np.isclose(got, expect)

    def test_broken_template_falls_back_exact(self):
        def shape_shift(x):
            c = x[0]
            for k in range(FOLD_START + 5):
                if k == FOLD_START + 3:
                    c = c * c  # body changes shape mid-fold
                else:
                    c = c + 0.1 * c * c
            return c

        out = to_sympy(shape_shift, np.array([0.05]))
        expect = float(shape_shift(np.array([0.05])))
        X = sympy.IndexedBase("x")
        f = out.formula
        if f.atoms(Iterate):
            f = f.replace(
                lambda e: e.func is Iterate, lambda e: e.doit(deep=False)
            )
        got = float(sympy.N(f.subs(X[0], 0.05).doit()))
        assert np.isclose(got, expect)
        # repair contract: no probe Dummy may survive into the result
        assert not any(
            isinstance(sym, sympy.Dummy) for sym in out.formula.free_symbols
        )
