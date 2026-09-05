"""assume= facts simplify formulas before comparison (issue #41).

Each rule gets three kinds of test: the unit behavior of the helper,
the end-to-end effect through check_formula, and a too-weak fact that
must fire nothing.
"""

import numpy as np
import pytest
import sympy

from skverify.testing import _apply_assumptions, check_formula

a, b, x = sympy.symbols("a b x")
T = sympy.IndexedBase("t")


class TestRules:
    def test_min_collapses_under_ordering(self):
        e = sympy.Min(a, b) * 2 + 1
        assert _apply_assumptions(e, [a < b]) == 2 * a + 1

    def test_max_collapses_under_ordering(self):
        assert _apply_assumptions(sympy.Max(a, b), [a < b]) == b

    def test_nonstrict_ordering_also_collapses(self):
        assert _apply_assumptions(sympy.Min(a, b), [a <= b]) == a

    def test_reversed_statement_of_the_same_fact(self):
        assert _apply_assumptions(sympy.Min(a, b), [b > a]) == a

    def test_abs_positive(self):
        assert _apply_assumptions(sympy.Abs(x) + 1, [x > 0]) == x + 1

    def test_abs_nonnegative(self):
        assert _apply_assumptions(sympy.Abs(x), [x >= 0]) == x

    def test_abs_negative(self):
        assert _apply_assumptions(sympy.Abs(x), [x < 0]) == -x

    def test_sign_positive(self):
        assert _apply_assumptions(sympy.sign(x) * 5, [x > 0]) == 5

    def test_sign_negative(self):
        assert _apply_assumptions(sympy.sign(x), [x < 0]) == -1

    def test_eq_substitutes(self):
        e = T[0] + T[1] * 2
        assert _apply_assumptions(e, [sympy.Eq(T[0], 0)]) == 2 * T[1]

    def test_matches_up_to_expand(self):
        # the fact states 2*(b - a) > 0; Abs carries 2*b - 2*a
        e = sympy.Abs(2 * b - 2 * a)
        assert _apply_assumptions(e, [2 * (b - a) > 0]) == 2 * b - 2 * a

    def test_nested_min_resolves_over_rounds(self):
        e = sympy.Min(a, sympy.Min(a, b))
        got = _apply_assumptions(e, [a < b])
        assert got == a


class TestTooWeakFactsFireNothing:
    def test_ne_leaves_min_alone(self):
        e = sympy.Min(a, b)
        assert _apply_assumptions(e, [sympy.Ne(a, b)]) == e

    def test_wrong_direction_leaves_abs_alone(self):
        e = sympy.Abs(x)
        assert _apply_assumptions(e, [x < 5]) == e  # x < 5 says nothing about sign

    def test_unrelated_symbols_leave_min_alone(self):
        c = sympy.Symbol("c")
        e = sympy.Min(a, c)
        assert _apply_assumptions(e, [a < b]) == e

    def test_no_facts_is_identity(self):
        e = sympy.Min(a, b) + sympy.Abs(x)
        assert _apply_assumptions(e, []) == e


def f_min(u, w):
    return np.minimum(u, w)


def f_absx(v):
    return np.abs(v)


def f_shifted(t):
    return t[0] + t[1] * 3.0


class TestEndToEnd:
    def test_min_trace_matches_under_ordering(self):
        # traced formula is Min(u, w); the spec is just u, provable
        # only because assume says u < w
        u = sympy.Symbol("u")
        v = check_formula(f_min, (1.0, 2.0), u, assume=[u < sympy.Symbol("w")])
        assert v.tier == "exact"

    def test_min_without_the_fact_does_not_prove(self):
        u = sympy.Symbol("u")
        v = check_formula(f_min, (1.0, 2.0), u)
        assert v.tier != "exact"

    def test_abs_trace_matches_on_positive_domain(self):
        V = sympy.IndexedBase("v")
        i = sympy.Symbol("i", integer=True)
        v = check_formula(
            f_absx, (np.array([1.0, 2.0, 3.0]),), V[i], indices=(i,),
            assume=[V[k] > 0 for k in range(3)],
        )
        assert v.tier == "exact"

    def test_eq_substitution_closes_the_proof(self):
        # code computes t[0] + 3*t[1]; the spec drops t[0], valid
        # only on the domain where assume pins t[0] to zero
        v = check_formula(
            f_shifted, (np.array([0.0, 2.0]),), 3 * T[1],
            assume=[sympy.Eq(T[0], 0)],
        )
        assert v.tier == "exact"

    def test_both_sides_normalized_spec_with_min(self):
        # the SPEC carries the Min; assume collapses it to meet the code
        u, w = sympy.symbols("u w")
        v = check_formula(f_min, (1.0, 2.0), sympy.Min(u, w), assume=[u < w])
        assert v.tier == "exact"
