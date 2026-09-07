"""explore() visits every branch; covers() is the proof it did.

The case that matters most is the last one: a spec that is correct on
the path the input happens to take and wrong on the other branch.
Single-path checking passes it; explore=True catches it.
"""

import numpy as np
import pytest
import sympy

from skverify.explore import covers, explore
from skverify.testing import check_formula

V = sympy.IndexedBase("v")
X = sympy.IndexedBase("x")
i = sympy.Symbol("i", integer=True)


def two_branch(v):
    if v.sum() > 0:
        return v * 2.0
    return v * 3.0


def three_branch(x):
    if x[0] > 1.0:
        return x * 2.0
    if x[0] > 0.0:
        return x + 1.0
    return x


def nested_infeasible(v):
    if v[0] > 2.0:
        if v[0] < 1.0:  # unreachable under the outer guard
            return v * 9.0
        return v * 2.0
    return v


def transcendental_guard(v):
    if np.exp(v[0]) > 50.0:  # outside the linear fragment
        return v * 2.0
    return v


class TestExploration:
    def test_both_branches_found_and_proven(self):
        r = explore(two_branch, (np.array([1.0, 2.0]),))
        assert len(r.paths) == 2
        assert r.complete
        assert covers(two_branch, (np.array([1.0, 2.0]),))

    def test_three_branches_with_infeasible_combination(self):
        r = explore(three_branch, (np.array([1.0, 2.0]),))
        assert len(r.paths) == 3
        assert r.complete
        # x <= 0 AND x > 1 was refuted, not silently dropped
        assert len(r.infeasible) >= 1

    def test_unreachable_branch_proven_infeasible(self):
        r = explore(nested_infeasible, (np.array([0.0, 0.0]),))
        assert r.complete
        conds = [str(p.condition) for p in r.paths]
        assert not any("9.0" in c for c in conds)  # dead branch never traced

    def test_transcendental_guard_solver_free_witnessing(self):
        # exp(v0) > 50 is outside LRA, but the SAMPLER can still land
        # on both sides; completeness holds iff every leftover region
        # was refuted or witnessed. Whatever the outcome, it must be
        # stated, never guessed.
        r = explore(transcendental_guard, (np.array([1.0, 2.0]),))
        assert len(r.paths) >= 1
        assert r.complete or r.undecided  # named, either way

    def test_path_conditions_are_disjoint_evidence(self):
        r = explore(three_branch, (np.array([1.0, 2.0]),))
        # every explored path carries its own guard set
        sigs = {frozenset(p.condition) for p in r.paths}
        assert len(sigs) == len(r.paths)


class TestCheckEverywhere:
    def test_spec_wrong_only_on_the_untraced_branch(self):
        # THE case this feature exists for. Spec matches the traced
        # (positive-sum) branch and is wrong on the other one.
        spec = 2 * V[i]

        v1 = check_formula(
            two_branch, (np.array([1.0, 2.0]),), spec, indices=(i,)
        )
        assert v1.matches  # single-path: passes, honestly qualified

        v2 = check_formula(
            two_branch, (np.array([1.0, 2.0]),), spec, indices=(i,),
            explore=True,
        )
        assert v2.tier == "differs"
        assert "on the path where" in v2.detail

    def test_piecewise_spec_matches_everywhere(self):
        def relu2(v):
            if v.sum() > 0:
                return v * 2.0
            return v * 3.0

        # per-branch specs, checked with each branch's guard assumed:
        # here one spec that IS the function, via explore on both
        j = sympy.Dummy("j", integer=True)
        total = sympy.Sum(V[j], (j, 0, 1))
        spec = sympy.Piecewise((2 * V[i], total > 0), (3 * V[i], True))
        v = check_formula(
            relu2, (np.array([1.0, 2.0]),), spec, indices=(i,),
            explore=True,
        )
        assert v.matches
        assert "all 2 path(s)" in v.detail

    def test_exact_on_all_paths_reports_coverage(self):
        def affine_split(x):
            if x[0] > 0.0:
                return 3.0 * x + 1.0
            return 3.0 * x - 1.0

        spec = sympy.Piecewise(
            (3 * X[i] + 1, X[0] > 0), (3 * X[i] - 1, True)
        )
        v = check_formula(
            affine_split, (np.array([1.0, 2.0]),), spec, indices=(i,),
            explore=True,
        )
        assert v.matches
        assert "coverage proven" in v.detail


class TestSharpEdges:
    def test_path_cap_never_claims_completeness(self):
        # 2^5 = 32 sign branches, cap at 4: must say capped, not proven
        def many(v):
            out = 0.0
            for k in range(5):
                if v[k] > 0:
                    out = out + v[k]
                else:
                    out = out - v[k]
            return out

        r = explore(many, (np.array([1.0, -1.0, 1.0, -1.0, 1.0]),),
                    max_paths=4)
        assert r.capped
        assert not r.complete
        assert "path cap" in r.summary()

    def test_far_domain_guard_is_reachable(self):
        # v[0] > 100 lives outside the default sampling window; the
        # escalating range must still find a witness
        def far(v):
            if v[0] > 100.0:
                return v * 2.0
            return v

        r = explore(far, (np.array([1.0, 2.0]),))
        assert len(r.paths) == 2
        assert r.complete

    def test_two_array_args_rebuilt_by_name(self):
        # the branch guard is on the SECOND argument; a positional
        # rebuild would write the witness into the first
        def g(u, w):
            if w[0] > 0:
                return u + w
            return u - w

        r = explore(g, (np.array([1.0, 2.0]), np.array([1.0, 2.0])))
        assert len(r.paths) == 2
        assert r.complete

    def test_scalar_argument_witnessed(self):
        def h(v, alpha):
            if alpha > 0.5:
                return v * alpha
            return v

        r = explore(h, (np.array([1.0, 2.0]), 0.1))
        assert len(r.paths) == 2
        assert r.complete
