"""The @specifies scaffold, exercised through the public API only.
Ported from the spec-diff burn-in: same mathematics, now behind
check_formula and the decorator."""

import numpy as np
import pytest
import sympy

from skverify.testing import Verdict, _describe_difference, check_formula, specifies

N = 5
V = sympy.IndexedBase("v")
i = sympy.Symbol("i", integer=True)
VALS = np.array([0.7, -1.2, 2.5, 0.3, -0.4])


def _mean():
    j = sympy.Dummy("j", integer=True)
    return sympy.Sum(V[j], (j, 0, N - 1)) / N


def f_mean(v):
    return v.mean()


def f_affine(v):
    return 3.0 * v + 1.0


def f_zscore(v):
    return (v - v.mean()) / v.std()


def f_affine_wrong(v):
    return 3.0 * v + 1.5


class TestCheckFormula:
    def test_scalar_exact(self):
        v = check_formula(f_mean, (VALS.copy(),), _mean())
        assert v.tier == "exact" and v.matches
        assert v.shape == ()

    def test_entrywise_exact(self):
        v = check_formula(
            f_affine, (VALS.copy(),), 3 * V[i] + 1, indices=(i,)
        )
        assert v.tier == "exact"
        assert v.shape == (5,)

    def test_float_constant_tier(self):
        j = sympy.Dummy("j", integer=True)
        mean = _mean()
        std = sympy.sqrt(sympy.Sum((V[j] - mean) ** 2, (j, 0, N - 1)) / N)
        v = check_formula(
            f_zscore, (VALS.copy(),), (V[i] - mean) / std, indices=(i,)
        )
        assert v.tier == "float-constant" and v.matches
        assert "rational points" in v.detail

    def test_differs_with_counterexample(self):
        v = check_formula(
            f_affine_wrong, (VALS.copy(),), 3 * V[i] + 1, indices=(i,)
        )
        assert v.tier == "differs" and not v.matches
        assert "spec value" in v.counterexample
        assert "your spec" in v.message()

    def test_incomplete_on_refusal(self):
        def f(v):
            return np.interp(0.5, v, v)

        v = check_formula(f, (np.sort(VALS.copy()),), _mean())
        assert v.tier == "incomplete"
        assert "not a code bug" in v.detail


class TestDecorator:
    def test_passing_spec(self):
        @specifies(3 * V[i] + 1, indices=(i,))
        def check():
            return f_affine, (VALS.copy(),)

        check()

    def test_failing_spec_raises_with_both_formulas(self):
        @specifies(3 * V[i] + 1, indices=(i,))
        def check():
            return f_affine_wrong, (VALS.copy(),)

        with pytest.raises(AssertionError, match="your spec"):
            check()

    def test_property_rung(self):
        @specifies.property(
            lambda F: sympy.Eq(
                sum(F.subs(i, k) for k in range(N)), 0
            )
        )
        def check():
            def center(v):
                return v - v.mean()

            return center, (VALS.copy(),)

        check()


class TestDiffersMessage:
    """Regression tests for the formula-difference diagnostic."""

    def test_large_diagnostic_bounded(self):
        v = check_formula(
            f_affine_wrong, (VALS.copy(),), 3 * V[i] + 1, indices=(i,)
        )
        assert v.tier == "differs"
        msg = v.message()
        assert "your spec" in msg
        assert "the code" in msg
        assert len(msg.splitlines()) <= 25

    def test_denominator_difference_named(self):
        n = sympy.Symbol("n")
        j = sympy.Dummy("j", integer=True)
        mean = sympy.Sum(V[j], (j, 0, n - 1)) / n
        spec = sympy.Sum((V[j] - mean) ** 2, (j, 0, n - 1)) / n
        code = sympy.Sum((V[j] - mean) ** 2, (j, 0, n - 1)) / (n - 1)
        verdict = Verdict(
            tier="differs", shape=(), spec=spec, traced=code,
            counterexample={
                "v": [1, 4, 2, 8, 5],
                "spec value": 6.5599,
                "code value": 8.1999,
            },
        )
        msg = verdict.message()
        assert "difference: denominator" in msg
        assert "n vs n - 1" in msg

    def test_numerator_difference_named(self):
        j = sympy.Dummy("j", integer=True)
        mean = sympy.Sum(V[j], (j, 0, N - 1)) / N
        spec = sympy.Sum((V[j] - mean) ** 2, (j, 0, N - 1)) / N
        code = sympy.Sum(V[j] ** 2, (j, 0, N - 1)) / N
        verdict = Verdict(
            tier="differs", shape=(), spec=spec, traced=code,
            counterexample={
                "v": [1, 4, 2, 8, 5],
                "spec value": 6.5599,
                "code value": 30.0,
            },
        )
        msg = verdict.message()
        assert "difference: numerator" in msg

    def test_sum_term_difference_identified(self):
        a, b, c = sympy.symbols("a b c")
        verdict = Verdict(
            tier="differs", shape=(), spec=a + b + c, traced=a + b,
        )
        msg = verdict.message()
        assert "difference:" in msg
        assert "c" in msg

    def test_reordered_sums_no_false_hint(self):
        a, b, c = sympy.symbols("a b c")
        verdict = Verdict(
            tier="differs", shape=(), spec=a + b + c, traced=c + a + b,
        )
        msg = verdict.message()
        assert "difference:" not in msg

    def test_fallback_no_misleading_hint(self):
        a, b, c, d = sympy.symbols("a b c d")
        verdict = Verdict(
            tier="differs", shape=(), spec=a * b, traced=c * d,
        )
        msg = verdict.message()
        assert "difference:" not in msg

    def test_both_num_and_den_differ_no_hint(self):
        a, b, c, d = sympy.symbols("a b c d")
        verdict = Verdict(
            tier="differs", shape=(), spec=a / b, traced=c / d,
        )
        msg = verdict.message()
        assert "difference:" not in msg
