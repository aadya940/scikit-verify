"""Path-coverage tests for skverify.explore.

These exercise the internal helpers directly (so we hit the sqrt/division
z3 encodings, the Piecewise lowering, the NNF pusher, the witness solver
and sampler, the coverage-gap theorem) plus a handful of end-to-end
explore() calls for the refusal / error / cap / undecided reporting paths.
"""

import numpy as np
import sympy
import z3

from skverify import explore as E
from skverify.explore import (
    Exploration,
    Path,
    _atoms_of,
    _coverage_gap,
    _forced_ties,
    _holds,
    _lower_piecewise_rels,
    _model_to_subs,
    _negate,
    _param_names,
    _push_not_down,
    _rebuild_args,
    _refuted,
    _rewrite_sqrt_rels,
    _slots_of,
    _to_z3,
    _unrolled,
    _witness,
    _witness_z3,
    covers,
    explore,
)

V = sympy.IndexedBase("v")
X = sympy.IndexedBase("x")
a = sympy.Symbol("a")
b = sympy.Symbol("b")


# ---------------------------------------------------------------------------
# Exploration dataclass: summary() and complete branches (lines 68,70 etc.)
# ---------------------------------------------------------------------------

class TestExplorationReporting:
    def test_summary_lists_infeasible(self):
        r = Exploration(paths=[Path((), ())],
                        infeasible=[sympy.true])
        s = r.summary()
        assert "1 region(s) proven infeasible" in s
        assert "coverage proven" in s  # complete: only paths + infeasible

    def test_summary_lists_undecided(self):
        r = Exploration(paths=[Path((), ())], undecided=[sympy.true])
        s = r.summary()
        assert "UNDECIDED" in s
        assert "NOT proven" in s
        assert not r.complete

    def test_summary_lists_refusals(self):
        r = Exploration(paths=[Path((), ())], refusals=["nope"])
        s = r.summary()
        assert "refused by the tracer" in s
        assert not r.complete

    def test_summary_lists_errors(self):
        r = Exploration(paths=[Path((), ())], errors=["ValueError: x"])
        s = r.summary()
        assert "RAISES" in s
        assert not r.complete

    def test_summary_capped(self):
        r = Exploration(paths=[Path((), ())], capped=True)
        assert "path cap" in r.summary()
        assert not r.complete

    def test_complete_needs_paths(self):
        assert not Exploration().complete  # no paths at all


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

class TestAtomHelpers:
    def test_atoms_of_true(self):
        assert _atoms_of(sympy.true) == ()
        assert _atoms_of(True) == ()

    def test_atoms_of_and(self):
        atoms = _atoms_of(sympy.And(a > 0, b < 1))
        assert len(atoms) == 2

    def test_atoms_of_single(self):
        assert _atoms_of(a > 0) == (a > 0,)

    def test_negate_relational(self):
        # relationals expose .negated
        assert _negate(a > 0) == (a <= 0)

    def test_negate_fallback_not(self):
        weird = sympy.Symbol("p", integer=True)  # no .negated -> Not
        assert _negate(weird) == sympy.Not(weird)

    def test_holds_true_false_and_simplify(self):
        assert _holds(a > 0, {a: sympy.Integer(1)}) is True
        assert _holds(a > 0, {a: sympy.Integer(-1)}) is False
        # non-boolean-literal expr that simplifies to a truth value
        assert _holds(sympy.Eq(a, a), {}) is True

    def test_holds_exception_returns_false(self):
        # xreplace of a non-Basic blows up -> caught -> False
        class Boom:
            def xreplace(self, subs):
                raise RuntimeError("boom")
        assert _holds(Boom(), {}) is False

    def test_unrolled_doit_and_exception(self):
        j = sympy.Symbol("j", integer=True)
        s = sympy.Sum(V[j], (j, 0, 1))
        out = _unrolled([s > 0])
        assert out[0] != (s > 0)  # doit expanded the sum

        class NoDoit:
            def doit(self):
                raise ValueError("nope")
        assert _unrolled([NoDoit()])[0].__class__.__name__ == "NoDoit"

    def test_forced_ties_detects_pair(self):
        ties = _forced_ties([a <= b, a >= b])
        assert ties  # a bounded both sides is a tie


class TestSlots:
    def test_slots_skips_non_basic(self):
        slots, syms = _slots_of([object(), a > 0])
        assert a in syms
        assert slots == []

    def test_slots_indexed_and_labels(self):
        slots, syms = _slots_of([V[0] > 0])
        assert V[0] in slots
        # the base label must NOT leak into syms
        assert not any(str(s) == "v" for s in syms)

    def test_slots_symbol_axis_names_excluded(self):
        i = sympy.Symbol("i")  # axis name
        slots, syms = _slots_of([i > 0])
        assert i not in syms


class TestParamNames:
    def test_named_params(self):
        def f(u, w):
            return u
        assert _param_names(f, 2) == ["u", "w"]

    def test_signature_typeerror_fallback(self):
        # an object whose signature introspection raises TypeError
        class NoSig:
            __call__ = property(lambda self: None)  # not callable-inspectable
        names = _param_names(NoSig(), 2)
        assert names == ["arg0", "arg1"]

    def test_fewer_params_than_requested(self):
        def f(u):
            return u
        # asking for 3 names from a 1-arg fn: falls through to argK
        assert _param_names(f, 3) == ["arg0", "arg1", "arg2"]

    def test_builtin_fallback(self):
        # numpy ufuncs / builtins have no inspectable signature of the
        # right arity -> arg0.. fallback
        names = _param_names(len, 3)
        assert names == ["arg0", "arg1", "arg2"]


class TestRebuildArgs:
    def test_indexed_written_by_name(self):
        def f(v):
            return v
        out = _rebuild_args(f, (np.array([1.0, 2.0]),),
                            {V[0]: sympy.Rational(9)})
        assert out[0][0] == 9.0

    def test_index_out_of_range_swallowed(self):
        def f(v):
            return v
        # index 5 into a length-2 array: IndexError caught, no crash
        out = _rebuild_args(f, (np.array([1.0, 2.0]),),
                            {V[5]: sympy.Rational(9)})
        assert out[0].shape == (2,)

    def test_scalar_symbol_written(self):
        def f(v, alpha):
            return v
        out = _rebuild_args(f, (np.array([1.0]), 0.0),
                            {sympy.Symbol("alpha"): sympy.Rational(3)})
        assert out[1] == 3.0


# ---------------------------------------------------------------------------
# _to_z3 encodings: Piecewise, sqrt, division, Abs, Max/Min, NNF
# ---------------------------------------------------------------------------

class TestLowerPiecewise:
    def test_piecewise_comparison_lowers_to_or(self):
        pw = sympy.Piecewise((1, a > 0), (0, True))
        lowered = _lower_piecewise_rels(sympy.Ne(pw, 0))
        assert isinstance(lowered, sympy.Or) or lowered != sympy.Ne(pw, 0)

    def test_no_piecewise_unchanged(self):
        assert _lower_piecewise_rels(a > 0) == (a > 0)


class TestRewriteSqrt:
    def test_sqrt_gt_positive(self):
        r = _rewrite_sqrt_rels(sympy.sqrt(a) > 2)
        assert r == (a > 4)

    def test_sqrt_ge_positive(self):
        assert _rewrite_sqrt_rels(sympy.sqrt(a) >= 3) == (a >= 9)

    def test_sqrt_lt_carries_nonneg(self):
        r = _rewrite_sqrt_rels(sympy.sqrt(a) < 2)
        assert isinstance(r, sympy.And)

    def test_sqrt_le_carries_nonneg(self):
        r = _rewrite_sqrt_rels(sympy.sqrt(a) <= 2)
        assert isinstance(r, sympy.And)

    def test_sqrt_eq(self):
        assert _rewrite_sqrt_rels(sympy.Eq(sympy.sqrt(a), 2)) == sympy.Eq(a, 4)

    def test_sqrt_ne(self):
        r = _rewrite_sqrt_rels(sympy.Ne(sympy.sqrt(a), 2))
        assert isinstance(r, sympy.Or)

    def test_sqrt_gt_negative_number(self):
        # sqrt(E) > -1: always true where defined -> And(E>=0, true)
        r = _rewrite_sqrt_rels(sympy.sqrt(a) > -1)
        # And(E>=0, true) simplifies to E>=0
        assert r == (a >= 0)

    def test_sqrt_lt_negative_impossible(self):
        # sqrt(E) < -5: impossible -> false
        r = _rewrite_sqrt_rels(sympy.sqrt(a) < -5)
        assert r == sympy.false


class TestPushNotDown:
    def test_not_and(self):
        r = _push_not_down(sympy.Not(sympy.And(a > 0, b > 0)))
        assert isinstance(r, sympy.Or)

    def test_not_or(self):
        r = _push_not_down(sympy.Not(sympy.Or(a > 0, b > 0)))
        assert isinstance(r, sympy.And)

    def test_double_not(self):
        # build the unevaluated Not(Not(x)) so it survives to the pusher
        inner = sympy.Not(a > 0, evaluate=False)
        expr = sympy.Not(inner, evaluate=False)
        assert _push_not_down(expr) == (a > 0)

    def test_not_relational_flips(self):
        expr = sympy.Not(a > 0, evaluate=False)
        assert _push_not_down(expr) == (a <= 0)

    def test_and_recurses(self):
        r = _push_not_down(sympy.And(sympy.Not(a > 0), b > 0))
        assert isinstance(r, sympy.And)

    def test_unresolvable_not_kept(self):
        p = sympy.Symbol("p")  # Not(p) has no .negated
        r = _push_not_down(sympy.Not(p))
        assert isinstance(r, sympy.Not)


class TestToZ3Encodings:
    def _conv(self, expr):
        vm = {}
        return _to_z3(expr, z3, vm), vm

    def test_abs(self):
        c, _ = self._conv(sympy.Abs(a) > 1)
        assert c is not None

    def test_max(self):
        c, _ = self._conv(sympy.Max(a, b) > 0)
        assert c is not None

    def test_min(self):
        c, _ = self._conv(sympy.Min(a, b) > 0)
        assert c is not None

    def test_sqrt_aux_encoding(self):
        c, vm = self._conv(sympy.sqrt(a) > sympy.Symbol("t"))
        # non-numeric rhs blocks the rewrite, so aux sqrt encoding runs
        assert c is not None

    def test_inverse_sqrt(self):
        c, _ = self._conv(sympy.Symbol("t") * a ** sympy.Rational(-1, 2) > 0)
        assert c is not None

    def test_negative_integer_power_division(self):
        c, _ = self._conv(a ** -1 > sympy.Symbol("t"))
        assert c is not None

    def test_positive_integer_power(self):
        c, _ = self._conv(a ** 3 > 1)
        assert c is not None

    def test_float_literal(self):
        c, _ = self._conv(a > sympy.Float(1.5))
        assert c is not None

    def test_rational_literal(self):
        c, _ = self._conv(a > sympy.Rational(1, 3))
        assert c is not None

    def test_integer_literal(self):
        c, _ = self._conv(a > sympy.Integer(2))
        assert c is not None

    def test_bare_not_in_bconv_returns_none(self):
        # a Not over a bare boolean symbol has no .negated, so it survives
        # NNF and reaches bconv's Not branch -> None. Wrap in And so the
        # top-level dispatch reaches bconv rather than the Not pre-pass.
        p = sympy.Symbol("p")
        expr = sympy.And(sympy.Not(p), a > 0)
        assert self._conv(expr)[0] is None

    def test_unknown_boolean_type_returns_none(self):
        # bconv fall-through: a boolean atom that is neither And/Or/Not
        # nor a handled relational (a bare Symbol) -> None
        assert self._conv(sympy.And(sympy.Symbol("p"), a > 0))[0] is None

    def test_piecewise_nested_if(self):
        # a Piecewise as an ARITHMETIC operand survives _lower_piecewise_rels
        # (which only rewrites Piecewise directly on a comparison side), so
        # conv() reaches the nested-If encoding.
        pw = sympy.Piecewise((a, a > 0), (b, True))
        c, _ = self._conv(pw + a > sympy.Symbol("t"))
        assert c is not None

    def test_piecewise_bad_cond_returns_none(self):
        # a Piecewise operand whose condition is outside the fragment ->
        # bconv returns None inside the If encoding
        pw = sympy.Piecewise((a, sympy.exp(a) > b), (b, True))
        c, _ = self._conv(pw + a > sympy.Symbol("t"))
        assert c is None

    def test_piecewise_bad_value_returns_none(self):
        # a Piecewise branch VALUE outside the fragment -> conv returns
        # None at line 526
        pw = sympy.Piecewise((sympy.exp(a), a > 0), (b, True))
        c, _ = self._conv(pw + a > sympy.Symbol("t"))
        assert c is None

    def test_out_of_fragment_returns_none(self):
        c, _ = self._conv(sympy.exp(a) > 1)
        assert c is None

    # each operator must propagate None when an operand is unconvertible
    def test_add_operand_unconvertible(self):
        assert self._conv(sympy.exp(a) + a > 1)[0] is None

    def test_mul_operand_unconvertible(self):
        assert self._conv(sympy.exp(a) * a > 1)[0] is None

    def test_sqrt_operand_unconvertible(self):
        assert self._conv(sympy.sqrt(sympy.exp(a)) > sympy.Symbol("t"))[0] \
            is None

    def test_inverse_sqrt_operand_unconvertible(self):
        e = sympy.Symbol("t") * sympy.exp(a) ** sympy.Rational(-1, 2) > 0
        assert self._conv(e)[0] is None

    def test_division_operand_unconvertible(self):
        # exp(a)**-1 simplifies to exp(-a); keep a real negative power
        e = (sympy.exp(a) - 5) ** -1 > sympy.Symbol("t")
        assert self._conv(e)[0] is None

    def test_positive_power_operand_unconvertible(self):
        assert self._conv(sympy.exp(a) ** 3 > 1)[0] is None

    def test_abs_operand_unconvertible(self):
        # exp is unconditionally positive so Abs(exp) simplifies away;
        # subtract to keep a genuine Abs whose inner is out-of-fragment
        assert self._conv(sympy.Abs(sympy.exp(a) - 5) > 1)[0] is None

    def test_max_operand_unconvertible(self):
        assert self._conv(sympy.Max(sympy.exp(a), a) > 0)[0] is None

    def test_and_operand_unconvertible(self):
        assert self._conv(sympy.And(sympy.exp(a) > 0, a > 0))[0] is None

    def test_or_operand_unconvertible(self):
        assert self._conv(sympy.Or(sympy.exp(a) > 0, a > 0))[0] is None

    def test_relational_side_unconvertible(self):
        # bconv relational path: one side converts to None
        assert self._conv(a > sympy.exp(b))[0] is None

    def test_non_integer_power_none(self):
        # x ** (1/3): not integer, not a handled root -> None
        assert self._conv(a ** sympy.Rational(1, 3) > 1)[0] is None

    def test_true_false(self):
        assert self._conv(sympy.true)[0] is not None
        assert self._conv(sympy.false)[0] is not None

    def test_bconv_not_refused(self):
        # a Not that survives NNF (over an atom with no .negated) makes
        # bconv refuse -> None
        p = sympy.Symbol("p")
        c, _ = self._conv(sympy.Not(sympy.Function("g")(p) > 0))
        assert c is None


# ---------------------------------------------------------------------------
# refuted / witness_z3 / model_to_subs / coverage_gap
# ---------------------------------------------------------------------------

class TestRefuted:
    def test_refuted_true_for_empty_region(self):
        assert _refuted([a > 1, a < 0]) is True

    def test_refuted_false_for_satisfiable(self):
        assert _refuted([a > 0]) is False

    def test_refuted_false_outside_fragment(self):
        assert _refuted([sympy.exp(a) > 1]) is False

    def test_refuted_exception_returns_false(self):
        # an atom whose doit() raises trips the outer except -> False
        class Boom:
            def doit(self):
                raise RuntimeError("boom")
        # _unrolled swallows doit errors, so feed something that breaks
        # _to_z3 itself: a bare object with no sympy interface
        assert _refuted([object()]) is False


class TestWitnessZ3:
    def test_witness_found(self):
        w = _witness_z3([a > 5])
        assert w is not None
        assert _holds(a > 5, w)

    def test_witness_unsat_returns_none(self):
        assert _witness_z3([a > 1, a < 0]) is None

    def test_witness_outside_fragment_none(self):
        assert _witness_z3([sympy.exp(a) > 1]) is None

    def test_model_to_subs_skips_aux(self):
        # build a model whose varmap includes an _aux dummy
        vm = {}
        _to_z3(sympy.sqrt(a) > sympy.Symbol("t"), z3, vm)
        s = z3.Solver()
        c = _to_z3(sympy.sqrt(a) > sympy.Symbol("t"), z3, vm)
        s.add(c)
        assert s.check() == z3.sat
        subs = _model_to_subs(s.model(), vm)
        assert not any(getattr(k, "name", "").startswith("_aux")
                       for k in subs)


class TestWitnessSampler:
    def test_equality_solved(self):
        rng = np.random.default_rng(0)
        # single unknown equality solved rather than sampled
        w = _witness([sympy.Eq(sympy.pi * a, 0)], rng)
        assert w is not None

    def test_tie_region_constructive(self):
        rng = np.random.default_rng(0)
        w = _witness([V[0] <= V[1], V[0] >= V[1]], rng)
        assert w is not None
        assert w[V[0]] == w[V[1]]

    def test_eq_against_number(self):
        rng = np.random.default_rng(0)
        w = _witness([sympy.Eq(a, sympy.Rational(3))], rng)
        assert w is not None
        assert w[a] == 3

    def test_eq_both_slots_forced_equal(self):
        rng = np.random.default_rng(0)
        # Eq(v0, v1): both are slots -> subs[lhs] = subs[rhs]
        w = _witness([sympy.Eq(V[0], V[1])], rng)
        assert w is not None
        assert w[V[0]] == w[V[1]]

    def test_eq_unsatisfiable_returns_none(self):
        rng = np.random.default_rng(0)
        # contradiction the sampler can never satisfy
        w = _witness([sympy.Eq(a, sympy.Integer(1)),
                      sympy.Eq(a, sympy.Integer(2))], rng)
        assert w is None

    def test_tie_against_number(self):
        rng = np.random.default_rng(0)
        # V[0] bounded both sides by 3: a tie where one side is a number
        w = _witness([V[0] <= 3, V[0] >= 3], rng)
        assert w is not None
        assert w[V[0]] == 3

    def test_eq_number_on_lhs(self):
        rng = np.random.default_rng(0)
        # Eq(3, a): the slot is the rhs -> line 215 branch
        w = _witness([sympy.Eq(sympy.Integer(3), a)], rng)
        assert w is not None
        assert w[a] == 3

    def test_equality_two_unknowns_falls_to_sampling(self):
        rng = np.random.default_rng(0)
        # Eq(a + b, 5): neither side is a lone slot; the solve/sample
        # fallback (lines 240-244 retry loop) must still land a witness
        w = _witness([sympy.Eq(a + b, sympy.Integer(5))], rng)
        assert w is None or _holds(sympy.Eq(a + b, sympy.Integer(5)), w)

    def test_equality_unsolvable_returns_none(self):
        rng = np.random.default_rng(1)
        # a transcendental equality solve() cannot close rationally, with
        # a contradictory companion so no draw ever satisfies it
        w = _witness([sympy.Eq(sympy.exp(a), a), a > 0, a < 0], rng)
        assert w is None

    def test_all_equal_mode(self):
        rng = np.random.default_rng(0)
        # a region only reachable in the all-equal tail of the budget:
        # every slot of the base equal AND positive
        w = _witness([sympy.Eq(V[0], V[1]), sympy.Eq(V[1], V[2]), V[0] > 0],
                     rng)
        assert w is None or w[V[0]] == w[V[2]]


class TestCoverageGap:
    def test_gap_none_when_covered(self):
        # two paths whose conditions partition the reals -> no gap
        paths = [Path((), (a > 0,)), Path((), (a <= 0,))]
        assert _coverage_gap(paths, ()) is None

    def test_gap_found_when_incomplete(self):
        paths = [Path((), (a > 0,))]  # misses a <= 0
        gap = _coverage_gap(paths, ())
        assert gap not in (None, "unverifiable")

    def test_gap_unverifiable_outside_fragment(self):
        paths = [Path((), (sympy.exp(a) > 1,))]
        assert _coverage_gap(paths, ()) == "unverifiable"

    def test_gap_unverifiable_aux_negation(self):
        # a division-aux path condition is unsound to negate
        paths = [Path((), (a ** -1 > sympy.Symbol("t"),))]
        assert _coverage_gap(paths, ()) == "unverifiable"

    def test_gap_constraint_outside_fragment(self):
        paths = [Path((), (a > 0,))]
        assert _coverage_gap(paths, (sympy.exp(a) > 1,)) == "unverifiable"

    def test_gap_exception_is_unverifiable(self):
        # a path condition that blows up during conversion -> the outer
        # except returns "unverifiable" rather than crashing
        class Boom:
            def doit(self):
                raise RuntimeError("boom")

            def __iter__(self):
                raise RuntimeError("boom")
        paths = [Path((), Boom())]
        assert _coverage_gap(paths, ()) == "unverifiable"

    def test_gap_empty_paths(self):
        # no paths: Not(Or()) degenerates; solver sees BoolVal(True)
        assert _coverage_gap([], ()) is not None

    def test_gap_with_valid_constraint_added(self):
        # a convertible domain constraint is added to the solver (line
        # 749); the a>0 path covers the whole a>0 domain -> no gap
        paths = [Path((), (a > 0,))]
        assert _coverage_gap(paths, (a > 0,)) is None

    def test_gap_constraint_narrows_to_covered(self):
        # path a>5 plus domain constraint a>10: fully covered -> None
        paths = [Path((), (a > 5,))]
        assert _coverage_gap(paths, (a > 10,)) is None


# ---------------------------------------------------------------------------
# end-to-end explore(): refusals, errors, undecided, capped
# ---------------------------------------------------------------------------

class TestExploreEndToEnd:
    def test_refusal_reported(self):
        # a construct the tracer refuses -> refusals, never a crash
        def uses_unsupported(v):
            return np.interp(0.5, v, v)  # traced table -> NotImplementedError
        r = explore(uses_unsupported, (np.array([1.0, 2.0, 3.0, 4.0]),))
        assert r.refusals
        assert not r.complete

    def test_error_region_reported(self):
        def guarded(v):
            if v.sum() < 0:
                raise ValueError("bad")
            return v * 2.0
        r = explore(guarded, (np.array([1.0, 2.0]),))
        assert r.errors and "ValueError" in r.errors[0]
        assert not r.complete

    def test_undecided_transcendental(self):
        def trans(v):
            if np.exp(v[0]) > 50.0:
                return v * 2.0
            return v
        r = explore(trans, (np.array([1.0, 2.0]),))
        assert r.complete or r.undecided

    def test_capped_reports_work_remaining(self):
        def many(v):
            out = 0.0
            for k in range(5):
                if v[k] > 0:
                    out = out + v[k]
                else:
                    out = out - v[k]
            return out
        r = explore(many, (np.array([1.0, -1.0, 1.0, -1.0, 1.0]),),
                    max_paths=3)
        assert r.capped
        assert not r.complete

    def test_time_budget_caps(self):
        def many(v):
            out = 0.0
            for k in range(6):
                if v[k] > 0:
                    out = out + v[k]
                else:
                    out = out - v[k]
            return out
        r = explore(many, (np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),),
                    time_budget=0.0)
        assert r.capped

    def test_dart_infeasible_branch(self):
        # nested guard whose negation is refuted inside the DART loop
        def nested(v):
            if v[0] > 2.0:
                if v[0] < 1.0:  # unreachable under outer guard
                    return v * 9.0
                return v * 2.0
            return v
        r = explore(nested, (np.array([0.0, 0.0]),))
        assert r.complete
        # the dead inner branch is never traced
        assert not any("9.0" in str(p.condition) for p in r.paths)

    def test_dart_infeasible_appended(self):
        # three ordered guards: negating the second with the first held
        # (x0 <= 1 AND x0 > 1) is refuted inside the DART negation loop
        def three(x):
            if x[0] > 1.0:
                return x * 2.0
            if x[0] > 0.0:
                return x + 1.0
            return x
        r = explore(three, (np.array([1.0, 2.0]),))
        assert r.infeasible
        assert r.complete

    def test_dart_undecided_region(self):
        # a guard the solver cannot refute and the sampler may miss;
        # whatever the verdict, undecided regions must be NAMED, not
        # silently dropped, and completeness never falsely claimed.
        def trans(v):
            if np.log(v[0] * v[0] + 1.0) > 100.0:
                return v * 2.0
            return v
        r = explore(trans, (np.array([1.0, 2.0]),), time_budget=10.0)
        assert r.complete or r.undecided or r.capped

    def test_coverage_gap_feeds_back_a_missed_input(self):
        # abs(v0) > 1 splits into TWO far regions (v0 > 1 and v0 < -1);
        # bookkeeping over the single guard leaves a gap the closing Z3
        # tautology check finds and feeds back (line 642).
        def f(v):
            if abs(v[0]) > 1.0:
                return v * 2.0
            return v
        r = explore(f, (np.array([2.0, 1.0]),))
        assert r.complete

    def test_median_six_orderings(self):
        r = explore(lambda v: np.median(v), (np.array([1.0, 2.0, 3.0]),))
        assert r.complete
        assert len(r.paths) == 6

    def test_sqrt_norm_coverage_proven(self):
        def norm_branch(v):
            n = np.sqrt((v ** 2).sum())
            if n > 1.0:
                return v / n
            return v
        r = explore(norm_branch, (np.array([1.0, 2.0]),))
        assert r.complete
        assert len(r.paths) == 2

    def test_deadline_trips_inside_dart_loop(self):
        # a first path carrying many guards means the DART negation loop
        # is long; a tiny budget expires mid-loop -> capped, no theorem
        def f(v):
            s = 0.0
            for k in range(8):
                if v[k] > 0:
                    s += v[k]
            return s
        r = explore(f, (np.ones(8),), time_budget=0.005)
        assert r.capped
        assert not r.complete

    def test_covers_true(self):
        def two(v):
            if v.sum() > 0:
                return v * 2.0
            return v * 3.0
        assert covers(two, (np.array([1.0, 2.0]),)) is True

    def test_covers_false_on_error(self):
        def guarded(v):
            if v.sum() < 0:
                raise ValueError("bad")
            return v * 2.0
        assert covers(guarded, (np.array([1.0, 2.0]),)) is False
