"""Rigor for recompression: every folded formula must EVALUATE back to the
concrete result, across sizes, strides, and boundary shapes. The fold is a
claim; these tests are the differential check of that claim."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy
from skverify.api import _fold_add, _recompress
from skverify.helpers import axis_idx

I = axis_idx(0)


def evaluate(formula, subs_arrays):
    """Numerically evaluate a folded formula against named arrays."""
    expr = formula.doit() if formula.has(sympy.Sum) else formula
    mapping = {}
    for name, arr in subs_arrays.items():
        base = sympy.IndexedBase(name)
        for k, v in enumerate(arr):
            mapping[base[k]] = float(v)
    return float(expr.xreplace(mapping))


class TestFoldedFormulasEvaluate:
    """The strongest property: substitute real numbers into the folded
    formula and it must equal the numerical result exactly."""

    @pytest.mark.parametrize("n", [4, 5, 9, 16, 33])
    def test_trapezoid_all_sizes(self, n):
        from scipy.integrate import trapezoid

        y = np.random.default_rng(n).uniform(-1, 1, n)
        t = to_sympy(lambda y: trapezoid(y, dx=0.1), y)
        assert np.isclose(evaluate(t.formula, {"y": y}), t.value)
        assert np.isclose(t.value, trapezoid(y, dx=0.1))

    @pytest.mark.parametrize("n", [5, 9, 17])
    def test_simpson_all_sizes(self, n):
        from scipy.integrate import simpson

        y = np.random.default_rng(n).uniform(-1, 1, n)
        s = to_sympy(lambda y: simpson(y, dx=0.5), y)
        assert np.isclose(evaluate(s.formula, {"y": y}), s.value)

    @pytest.mark.parametrize("n", [4, 8, 20])
    def test_dot_all_sizes(self, n):
        rng = np.random.default_rng(n)
        a, b = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        d = to_sympy(np.dot, a, b)
        assert np.isclose(evaluate(d.formula, {"a": a, "b": b}), d.value)

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("n", [6, 11])
    def test_diff_rule_evaluates_elementwise(self, n, order):
        x = np.random.default_rng(n * order).uniform(-1, 1, n)
        s = to_sympy(np.diff, x, order)
        expected = np.diff(x, order)
        base = sympy.IndexedBase("a")
        mapping = {base[k]: float(v) for k, v in enumerate(x)}
        for k in range(len(expected)):
            got = float(s.formula.subs(I, k).xreplace(mapping))
            assert np.isclose(got, expected[k])


class TestCumulativeFold:
    """Growing elements (running sums) fold as prefix sums of a proven
    shiftable difference: elem(i) = elem[0] + Sum(diff(j), (j, 0, i-1))."""

    @pytest.mark.parametrize("n", [4, 6, 11, 20])
    def test_cumtrapz_all_sizes(self, n):
        from scipy.integrate import cumulative_trapezoid

        y = np.random.default_rng(n).uniform(-1, 1, n)
        b = to_sympy(cumulative_trapezoid, y)
        assert b.formula.has(sympy.Sum)  # folded, not an Array
        Y = sympy.IndexedBase("y")
        m = {Y[k]: float(v) for k, v in enumerate(y)}
        for k in range(n - 1):
            got = float(b.formula.subs(I, k).doit().xreplace(m))
            assert np.isclose(got, b.value[k])
        assert np.allclose(b.value, cumulative_trapezoid(y))

    def test_empty_prefix_is_first_element(self):
        # at i=0 the Sum is empty: rule must equal element 0 exactly
        from scipy.integrate import cumulative_trapezoid

        y = np.linspace(0, 1, 5) ** 3
        b = to_sympy(cumulative_trapezoid, y)
        Y = sympy.IndexedBase("y")
        m = {Y[k]: float(v) for k, v in enumerate(y)}
        assert np.isclose(float(b.formula.subs(I, 0).doit().xreplace(m)), b.value[0])

    def test_shape_generic_cumulative_rule(self):
        from scipy.integrate import cumulative_trapezoid

        a = to_sympy(cumulative_trapezoid, np.linspace(0, 1, 5))
        b = to_sympy(cumulative_trapezoid, np.linspace(0, 1, 9))
        assert a.formula == b.formula  # same rule, only domain moves
        assert a.domain == (0, 4) and b.domain == (0, 8)


class TestFoldRefusals:
    def test_truly_irregular_elements_refuse(self):
        # neither shifted copies nor a shiftable difference
        U = sympy.IndexedBase("u")
        assert _recompress([U[0], 3 * U[5] ** 2, U[1] + 7, sympy.log(U[2])]) is None

    def test_too_few_terms_refuse(self):
        U = sympy.IndexedBase("u")
        assert _fold_add(U[0] + 2 * U[1]) is None

    def test_symbolic_indices_refuse(self):
        # already-indexed formulas are not re-folded
        U = sympy.IndexedBase("u")
        assert _fold_add(U[I] + U[I + 1] + U[I + 2] + U[I + 3]) is None

    def test_irregular_coefficients_refuse(self):
        U = sympy.IndexedBase("u")
        expr = (
            1.0 * U[0]
            + 2.0 * U[1]
            + 5.0 * U[2]
            + 11.0 * U[3]
            + 3.0 * U[4]
            + 7.0 * U[5]
            + 13.0 * U[6]
        )
        assert _fold_add(expr) is None

    def test_recompress_short_list_refuses(self):
        U = sympy.IndexedBase("u")
        assert _recompress([U[0]]) is None


class TestDeterminism:
    def test_same_input_same_formula(self):
        from scipy.integrate import trapezoid

        y = np.linspace(0, 1, 10)
        a = to_sympy(lambda y: trapezoid(y, dx=0.2), y)
        b = to_sympy(lambda y: trapezoid(y, dx=0.2), y)
        assert a.formula == b.formula

    def test_shape_generic_rule(self):
        # same rule at different sizes; only the domain moves
        s5 = to_sympy(np.diff, np.linspace(0, 1, 5))
        s9 = to_sympy(np.diff, np.linspace(0, 1, 9))
        assert s5.formula == s9.formula
        assert s5.domain == (0, 4) and s9.domain == (0, 8)
