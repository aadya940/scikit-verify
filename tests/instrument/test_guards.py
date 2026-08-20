"""Branch capture: scalar conditions decide via the concrete lane and are
recorded; to_sympy exposes them as .preconditions."""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy

X = sympy.Symbol("x", real=True)
DT = sympy.Symbol("dt", real=True)
U = sympy.IndexedBase("u")


def kernel(x, dt):
    if dt > 0.5:
        return x * 0.9
    return x + dt


class TestBranchCapture:
    def test_taken_branch_and_its_condition(self):
        out = to_sympy(kernel, np.arange(4.0), 0.01)
        XB = sympy.IndexedBase("x")  # named from kernel's own signature
        assert out.formula == XB[sympy.Symbol("i", integer=True)] + DT
        assert out.preconditions == sympy.Le(DT, 0.5)  # Not(dt > 0.5)

    def test_other_branch_other_condition(self):
        out = to_sympy(kernel, np.arange(4.0), 0.9)
        assert out.preconditions == sympy.Gt(DT, 0.5)

    def test_no_branches_means_true(self):
        out = to_sympy(lambda x: x * 2.0, np.arange(3.0))
        assert out.preconditions is sympy.true

    def test_multiple_guards_conjoin(self):
        def f(x):
            if x > 0:
                if x < 10:
                    return x * 2.0
            return x

        out = to_sympy(f, 3.0)
        assert out.preconditions == sympy.And(sympy.Gt(X, 0), sympy.Lt(X, 10))

    def test_while_records_per_iteration(self):
        def f(x):
            s = x
            while s < 10:
                s = s * 2.0
            return s

        out = to_sympy(f, 3.0)
        # 3 -> 6 -> 12: two Trues then the exit False, all over evolving s
        assert out.preconditions.count(sympy.Lt) >= 1
        assert float(out.value) == 12.0

    def test_assert_becomes_hypothesis(self):
        def f(x):
            assert x > 0, "validation gate"
            return np.sqrt(x)

        out = to_sympy(f, 4.0)
        assert out.preconditions == sympy.Gt(X, 0)

    def test_array_condition_still_refuses(self):
        u = Pair.array("u", np.arange(4.0))
        with pytest.raises(NotImplementedError):
            if u > 0:  # ambiguous, like numpy's own error
                pass

    def test_non_condition_truthiness_still_refuses(self):
        u = Pair.array("u", np.arange(4.0))
        with pytest.raises(NotImplementedError):
            bool(u)

    def test_guards_reset_between_traces(self):
        to_sympy(kernel, np.arange(4.0), 0.9)
        out = to_sympy(lambda x: x * 2.0, np.arange(3.0))
        assert out.preconditions is sympy.true  # no leakage from prior trace


class TestGuardedLibraryCode:
    def test_polyval_lifts_horner_elementwise(self):
        out = to_sympy(np.polyval, np.array([2.0, 3.0, 5.0]), np.arange(4.0))
        assert np.allclose(out.value, np.polyval([2.0, 3.0, 5.0], np.arange(4.0)))
        P = sympy.IndexedBase("p")
        assert out.formula.has(P[0]) and out.formula.has(P[2])

    def test_searchsorted_counting_form(self):
        # the insertion index IS a count: Sum_k [a[k] < v], exact for
        # sorted bins -- the ordering rides in the preconditions
        x = np.array([10.0, 20.0, 30.0, 40.0])
        s = to_sympy(lambda a: np.searchsorted(a, 25), x)
        assert int(s.value) == 2
        A = sympy.IndexedBase("a")
        subs = {A[k]: x[k] for k in range(4)}
        assert s.formula.doit().subs(subs) == 2
        assert s.preconditions.has(sympy.Le(A[1], A[2]))

    def test_median_lifts_path_scoped(self):
        out = to_sympy(np.median, np.array([3.0, 1.0, 2.0]))
        assert float(out.value) == 2.0
        assert out.preconditions is not sympy.true  # the sort's comparisons
