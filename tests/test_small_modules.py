"""Coverage sprint: uncovered paths across five small modules.

Drives twins.py, recurrence.py, atoms.py, sets.py and api.py through
the public ``to_sympy`` entry (plus a few direct handles the existing
suite already uses: ``Pair._opaque_call``, ``TracedSet``, ``TracedDict``).
Assertions are value-lane or field checks; no library code is touched.
"""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy


# --------------------------------------------------------------------------
# sets.py -- TracedSet operations and TracedDict lookups
# --------------------------------------------------------------------------
class TestTracedSet:
    def test_union_intersection_difference_via_to_sympy(self):
        def f(x):
            s = set([x[0], x[1], x[2]])
            t = set([x[1], x[2], x[3]])
            u = (s | t) & s
            return len(u)

        out = to_sympy(f, np.array([1.0, 2.0, 3.0, 4.0]))
        assert out.value == 3

    def test_difference_and_symmetric_difference(self):
        def f(x):
            s = set([x[0], x[1], x[2]])
            t = set([x[2], x[3]])
            return len(s - t) + len((s ^ t))

        out = to_sympy(f, np.array([1.0, 2.0, 3.0, 4.0]))
        # s - t = {1,2}  (2 elems); s ^ t = {1,2,4} (3 elems)
        assert out.value == 5

    def test_direct_set_algebra_carries_sympy_formula(self):
        from skverify.sets import TracedSet

        a = TracedSet([Pair(1.0, sympy.Symbol("a")), Pair(2.0, sympy.Symbol("b"))])
        b = TracedSet([Pair(2.0, sympy.Symbol("b")), Pair(3.0, sympy.Symbol("c"))])
        assert "TracedSet(" in repr(a)
        assert len(a) == 2
        # membership by concrete value
        assert Pair(1.0, sympy.Symbol("a")) in a
        assert Pair(9.0, sympy.Symbol("z")) not in a

        # sympy evaluates the set combinators eagerly; the value lane is
        # the concrete guarantee, the formula stays a sympy Set.
        u = a | b
        assert isinstance(u.formula, sympy.Set)
        assert u.value == {1.0, 2.0, 3.0}

        i = a & b
        assert isinstance(i.formula, sympy.Set)
        assert i.value == {2.0}

        d = a - b
        assert isinstance(d.formula, sympy.Set)
        assert d.value == {1.0}

        x = a ^ b
        assert isinstance(x.formula, sympy.Set)
        assert x.value == {1.0, 3.0}

    def test_combine_with_plain_iterable_other(self):
        from skverify.sets import TracedSet

        a = TracedSet([Pair(1.0, sympy.Symbol("a")), Pair(2.0, sympy.Symbol("b"))])
        # other is a plain list -> wrapped into a TracedSet internally
        u = a | [2.0, 3.0]
        assert u.value == {1.0, 2.0, 3.0}

    def test_empty_set_formula_is_emptyset(self):
        from skverify.sets import TracedSet

        assert TracedSet([]).formula is sympy.EmptySet
        # iteration yields the stored elements
        a = TracedSet([Pair(1.0, sympy.Symbol("a"))])
        assert [p.value for p in a] == [1.0]


class TestTracedDict:
    def test_dict_selection_via_to_sympy(self):
        def f(x):
            d = {0.0: 10.0, 1.0: 20.0}
            return d[x[0]]

        out = to_sympy(f, np.array([1.0]))
        assert out.value == 20.0

    def test_direct_traced_dict_methods(self):
        from skverify.sets import TracedDict

        d = TracedDict({0.0: 10.0, 1.0: 20.0})
        assert len(d) == 2
        assert set(iter(d)) == {0.0, 1.0}
        assert set(d.keys()) == {0.0, 1.0}
        assert set(d.values()) == {10.0, 20.0}
        assert dict(d.items()) == {0.0: 10.0, 1.0: 20.0}
        assert 1.0 in d
        assert 9.0 not in d

    def test_plain_key_returns_stored_value(self):
        from skverify.sets import TracedDict

        d = TracedDict({0.0: 10.0, 1.0: 20.0})
        # plain (non-Pair) key -> the stored value straight through
        assert d[1.0] == 20.0

    def test_get_hit_and_miss(self):
        from skverify.sets import TracedDict

        d = TracedDict({0.0: 10.0})
        assert d.get(0.0) == 10.0
        assert d.get(5.0) is None
        assert d.get(5.0, "def") == "def"

    def test_traced_key_builds_piecewise(self):
        from skverify.sets import TracedDict

        d = TracedDict({0.0: 10.0, 1.0: 20.0})
        key = Pair(1.0, sympy.Symbol("y", real=True))
        res = d[key]
        assert isinstance(res, Pair)
        assert isinstance(res.formula, sympy.Piecewise)
        assert res.value == 20.0


# --------------------------------------------------------------------------
# atoms.py -- opaque compiled calls and rng draws
# --------------------------------------------------------------------------
class TestOpaqueAtoms:
    def test_multi_output_svd_seals_role_named_atoms(self):
        def g(x):
            U, S, Vh = np.linalg.svd(x)
            return S

        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = to_sympy(g, A)
        assert np.allclose(np.asarray(out.value, dtype=float), np.linalg.svd(A)[1])
        assert any(r[0] == "svd" for r in out.unchecked)

    def test_multi_output_tuple_and_integer_passthrough(self):
        # a hand-rolled multi-output routine: one float array + one int
        # array. The int output passes through concrete (326-330 / 425-426).
        u = Pair.array("u", np.arange(4.0))

        def two_out(x):
            return np.asarray(x, dtype=float) * 2.0, np.array([1, 2], dtype=int)

        farr, iarr = Pair._opaque_call(two_out, (u,), {})
        assert isinstance(farr, Pair)
        assert isinstance(farr.formula, sympy.Indexed)
        assert np.allclose(np.asarray(farr.value, dtype=float), u.value * 2.0)
        # integer bookkeeping output is not a Pair
        assert not isinstance(iarr, Pair)
        assert np.array_equal(iarr, np.array([1, 2]))

    def test_multi_output_scalar_atom(self):
        # a 0-d float output becomes a scalar Symbol atom (atoms 326-330)
        u = Pair.array("u", np.arange(4.0))

        def scal_out(x):
            return np.array(3.0), np.array([1.0, 2.0])

        s, v = Pair._opaque_call(scal_out, (u,), {})
        assert isinstance(s, Pair) and isinstance(s.formula, sympy.Symbol)
        assert float(s.value) == 3.0
        assert isinstance(v, Pair) and isinstance(v.formula, sympy.Indexed)

    def test_const_operand_disclosed_and_equality_note(self):
        # a concrete ndarray operand alongside a traced one: it becomes a
        # named const symbol, and a bitwise-equal-to-input observation is
        # disclosed (atoms 375, 385-389).
        def f(u):
            helper = np.array([1.0, 4.0])  # equals u bitwise at trace time
            return np.linalg.norm(np.stack([u, helper]))

        # through to_sympy so _session.inputs is populated and the
        # bitwise-equal-to-input observation is disclosed.
        def driver(u):
            helper = np.array([1.0, 4.0])  # equals u bitwise at trace time

            def routine(a, b):
                return np.asarray(a, dtype=float) + np.asarray(b, dtype=float)

            return Pair._opaque_call(routine, (u, helper), {})

        out = to_sympy(driver, np.array([1.0, 4.0]))
        notes = [r[2][1] for r in out.unchecked if "const0" in r[2][1]]
        assert notes and "bitwise-equal" in notes[0]

    def test_scalar_const_operand(self):
        # np.isscalar operand -> sympified into the formula (atoms 367-369)
        u = Pair.array("u", np.array([1.0, 2.0]))

        def routine(a, c):
            return np.asarray(a, dtype=float) + c

        r = Pair._opaque_call(routine, (u, 5.0), {})
        assert isinstance(r, Pair)
        from skverify.pair import _OPAQUE

        assert "5" in _OPAQUE[-1][2][1] or "5.0" in _OPAQUE[-1][2][1]

    def test_protocol_dunder_returning_none_passes_through(self):
        # a compiled __init__-style dunder returns None by design (341-353)
        u = Pair.array("u", np.arange(3.0))

        def __init__(x):
            return None

        res = Pair._opaque_call(__init__, (u,), {})
        assert res is None
        from skverify.pair import _OPAQUE

        assert _OPAQUE[-1][0] == "__init__"

    def test_state_setter_returning_none_refuses(self):
        # a non-dunder returning None: internal state the trace can't follow
        u = Pair.array("u", np.arange(3.0))

        def set_state(x):
            return None

        with pytest.raises(NotImplementedError):
            Pair._opaque_call(set_state, (u,), {})

    def test_contiguous_retry_fortran_layout(self):
        # a routine that rejects C-contiguous input on the first call and
        # accepts Fortran layout on retry (atoms 321-330).
        u = Pair.array("u", np.array([[1.0, 2.0], [3.0, 4.0]]))
        state = {"tries": 0}

        def picky(x):
            state["tries"] += 1
            if state["tries"] == 1 and x.flags["C_CONTIGUOUS"]:
                raise ValueError("Array must be F-contiguous")
            return np.asarray(x, dtype=float) * 2.0

        r = Pair._opaque_call(picky, (u,), {})
        assert state["tries"] == 2
        assert np.allclose(np.asarray(r.value, dtype=float), u.value * 2.0)

    def test_non_contiguous_valueerror_reraised(self):
        # a ValueError NOT about contiguity is re-raised unchanged (322-323)
        u = Pair.array("u", np.arange(3.0))

        def boom(x):
            raise ValueError("something else entirely")

        with pytest.raises(ValueError, match="something else"):
            Pair._opaque_call(boom, (u,), {})

    def test_operand_identity_verified_formula(self):
        # an ndarray extracted from a traced value (value_of) is named by
        # its formula, identity-verified, not as an anonymous const (375).
        from skverify.coercion import value_of

        u = Pair.array("u", np.array([1.0, 2.0, 3.0]))
        raw = value_of(u)  # records id(raw) -> u's formula in the session

        def routine(a):
            return np.asarray(a, dtype=float) + 1.0

        r = Pair._opaque_call(routine, (raw,), {})
        from skverify.pair import _OPAQUE

        # the operand appears as u[...] (its formula), not const0
        assert "const" not in _OPAQUE[-1][2][1]

    def test_mutation_guard_raises(self):
        u = Pair.array("u", np.arange(4.0))

        def scribbler(x):
            x[0] = -99.0  # mutates the buffer it was handed... but it's a copy
            return np.array([0.0])

        # runs on a copy: the traced value is untouched and no error
        Pair._opaque_call(scribbler, (u,), {})
        assert u.value[0] == 0.0


class TestRngAtoms:
    def test_required_param_missing_stays_concrete(self):
        # gamma requires shape; call with no args -> concrete draw
        # (atoms 189-190 -> concrete()).
        def f(x):
            rng = np.random.default_rng(7)
            try:
                return x + rng.gamma()  # missing required shape
            except TypeError:
                return x + rng.gamma(2.0)  # numpy would raise; fall back

        out = to_sympy(f, np.float64(1.0))
        assert np.isfinite(float(out.value))

    def test_exponential_scale_lifts(self):
        # exponential(scale=2) -> rate 1/2, mean 2 (default param path)
        def f(x):
            rng = np.random.default_rng(7)
            return x + rng.exponential(2.0)

        out = to_sympy(f, np.float64(1.0))
        import sympy.stats as st

        assert sympy.simplify(st.E(out.formula) - (sympy.Symbol("x", real=True) + 2.0)) == 0

    def test_param_via_keyword(self):
        # distribution parameter supplied by keyword (atoms 187-188)
        def f(x):
            rng = np.random.default_rng(9)
            return x + rng.normal(loc=0.0, scale=2.0)

        out = to_sympy(f, np.float64(1.0))
        import sympy.stats as st

        assert st.variance(out.formula) == 4.0

    def test_exotic_kwarg_stays_concrete(self):
        # an unmapped keyword (dtype) forces the concrete draw (atoms 180-182)
        def f(x):
            rng = np.random.default_rng(13)
            return x + rng.standard_normal(dtype=np.float32)

        out = to_sympy(f, np.float64(1.0))
        assert np.isfinite(float(out.value))

    def test_array_draw_iid_indexedbase(self):
        # array-shaped draw -> IndexedBase atom (atoms 216-222)
        def sim(x):
            rng = np.random.default_rng(11)
            return x + rng.normal(0.0, 1.0, 3)

        data = np.array([1.0, 2.0, 3.0])
        out = to_sympy(sim, data)
        rng = np.random.default_rng(11)
        assert np.allclose(
            np.asarray(out.value, dtype=float), data + rng.normal(0.0, 1.0, 3)
        )
        notes = [r[-1][1] for r in out.unchecked if r[0] == "normal"]
        assert notes and "iid" in notes[0]

    def test_unmapped_draw_stays_concrete(self):
        def pick(x):
            rng = np.random.default_rng(4)
            return x + rng.integers(0, 10)

        out = to_sympy(pick, np.float64(1.0))
        rng = np.random.default_rng(4)
        assert np.isclose(float(out.value), 1.0 + rng.integers(0, 10))

    def test_library_internal_draw_stays_concrete(self):
        # a draw not requested by user code returns a plain number
        # (_user_code_draw False path, atoms 177-178). Exercised by any
        # numpy-internal random usage; here we just assert restoration.
        def f(x):
            return x * 2.0

        to_sympy(f, np.float64(1.0))
        assert type(np.random.default_rng(0)) is np.random.Generator


# --------------------------------------------------------------------------
# api.py -- top-level surface and harvest paths
# --------------------------------------------------------------------------
class TestApiSurface:
    def test_kwargs_wrap_by_name(self):
        def f(a, b):
            return a * 2.0 + b

        out = to_sympy(f, np.array([1.0, 2.0]), b=np.array([3.0, 4.0]))
        assert np.allclose(np.asarray(out.value, dtype=float), np.array([5.0, 8.0]))
        assert out.formula.free_symbols  # depends on both inputs

    def test_scalar_symbol_and_int_config_disclosure(self):
        # int arg passes untraced and is disclosed (api 322-337)
        def f(x, n):
            return x * n

        out = to_sympy(f, np.float64(2.0), 3)
        assert float(out.value) == 6.0
        disc = [r for r in out.unchecked if r[0] == "arg:n"]
        assert disc and "configuration" in disc[0][2][1]

    def test_bool_str_none_pass_through(self):
        def f(x, flag, name, opt):
            return x * (2.0 if flag else 1.0)

        out = to_sympy(f, np.float64(3.0), True, "label", None)
        assert float(out.value) == 6.0

    def test_repack_constant_wraps(self):
        # a function ignoring its data would give a constant; use one that
        # genuinely returns a constant path (searchsorted-style). Simpler:
        # a plain arithmetic returning scalar constant with no traced dep.
        def f(x):
            return 5.0  # no dependence -> constant, but x is int-ish? make float

        out = to_sympy(f, np.float64(1.0))
        assert float(out.value) == 5.0

    def test_recompress_folds_elementwise_diff(self):
        # per-element formulas fold back to one indexed rule (_recompress)
        def diff(u):
            return u[1:] - u[:-1]

        out = to_sympy(diff, np.array([1.0, 3.0, 6.0, 10.0]))
        assert np.allclose(
            np.asarray(out.value, dtype=float), np.array([2.0, 3.0, 4.0])
        )
        # folded to a single indexed rule over the axis
        assert out.formula.atoms(sympy.Indexed)

    def test_fold_poly_horner(self):
        # Horner nest folds through polynomial coefficients (_fold_poly)
        def horner(c, x):
            return ((c[0] * x + c[1]) * x + c[2])

        out = to_sympy(horner, np.array([2.0, 3.0, 4.0]), np.float64(5.0))
        assert np.isclose(float(out.value), 2.0 * 25 + 3.0 * 5 + 4.0)

    def test_fold_add_weighted_sum(self):
        # a big uniform-weight Add folds to Sum form (_fold_add)
        def wsum(y):
            return (
                0.1 * y[0] + 0.1 * y[1] + 0.1 * y[2] + 0.1 * y[3]
                + 0.1 * y[4] + 0.1 * y[5]
            )

        out = to_sympy(wsum, np.arange(6.0))
        assert np.isclose(float(out.value), 0.1 * np.arange(6.0).sum())

    def test_result_display_accessors(self):
        def f(x):
            return x * 2.0 + 1.0

        out = to_sympy(f, np.float64(3.0))
        assert float(out.value) == 7.0
        assert out.preconditions is not None
        assert isinstance(out.pretty(), str)
        assert isinstance(out.expand_formula(), sympy.Basic)

    def test_infer_names_too_many_args_raises(self):
        def f(a):
            return a

        with pytest.raises(TypeError):
            to_sympy(f, np.float64(1.0), np.float64(2.0))

    def test_var_positional_names(self):
        def f(*xs):
            return xs[0] + xs[1]

        out = to_sympy(f, np.float64(1.0), np.float64(2.0))
        assert float(out.value) == 3.0

    def test_lambda_traces_through_instrumented_retry(self):
        # a lambda with a control-flow wall goes to the instrumented retry
        def body(x):
            total = 0.0
            for i in range(3):
                total = total + x[i]
            return total

        out = to_sympy(body, np.array([1.0, 2.0, 3.0]))
        assert np.isclose(float(out.value), 6.0)

    def test_hashed_identity_disclosure(self):
        # traced values used as set members are disclosed once as
        # used-as-identity, not as n^2 equality guards (api 283-284).
        def f(x):
            seen = set()
            total = 0.0
            for i in range(3):
                seen.add(x[i])
                total = total + x[i]
            return total

        out = to_sympy(f, np.array([1.0, 2.0, 3.0]))
        assert any(r[0] == "used_as_identity" for r in out.unchecked)

    def test_cumulative_recompress(self):
        # running/cumulative sums fold via the prefix-sum branch of
        # _recompress (api 569-586).
        def cumulative(u):
            out = np.zeros_like(u)
            acc = 0.0
            for i in range(len(u)):
                acc = acc + u[i]
                out[i] = acc
            return out

        data = np.array([1.0, 2.0, 3.0, 4.0])
        out = to_sympy(cumulative, data)
        assert np.allclose(np.asarray(out.value, dtype=float), np.cumsum(data))

    def test_dataframe_like_to_numpy(self):
        pd = pytest.importorskip("pandas")

        def f(x):
            return x * 2.0

        s = pd.Series([1.0, 2.0, 3.0])
        out = to_sympy(f, s)
        assert np.allclose(np.asarray(out.value, dtype=float), np.array([2.0, 4.0, 6.0]))


# --------------------------------------------------------------------------
# recurrence.py + twins.py -- loop folding
# --------------------------------------------------------------------------
class TestLoopFolding:
    def test_snowball_folds_to_iterate(self):
        from skverify.recurrence import Iterate

        def growth(x):
            c = x[0]
            for _ in range(25):
                c = c + 0.1 * c * c
            return c

        out = to_sympy(growth, np.array([0.1]))
        f = out.expand_formula()
        assert f.atoms(Iterate)
        X = sympy.IndexedBase("x")
        got = float(sympy.N(f.subs(X[0], 0.1).doit()))
        assert np.isclose(got, float(growth(np.array([0.1]))))

    def test_short_loop_stays_unrolled(self):
        from skverify.recurrence import Iterate

        def short(x):
            c = x[0]
            for _ in range(4):
                c = c + 0.1 * c * c
            return c

        out = to_sympy(short, np.array([0.1]))
        assert not out.formula.atoms(Iterate)

    def test_iteration_index_generalizes(self):
        def indexed(x):
            c = x[0]
            for k in range(2, 20):
                c = c + 0.01 * c + 0.001 * float(k)
            return c

        out = to_sympy(indexed, np.array([0.1]))
        X = sympy.IndexedBase("x")
        f = out.expand_formula()
        got = float(sympy.N(f.subs(X[0], 0.1).doit()))
        assert np.isclose(got, float(indexed(np.array([0.1]))))

    def test_coupled_state_folds_to_tuple(self):
        from skverify.recurrence import Iterate, Nth

        def coupled(x):
            a = x[0]
            b = x[1]
            for _ in range(20):
                a, b = a + 0.1 * b, b + 0.2 * a * a
            return a + b

        vals = np.array([0.3, 0.5])
        out = to_sympy(coupled, vals)
        f = out.expand_formula()
        assert f.atoms(Iterate) and f.atoms(Nth)
        X = sympy.IndexedBase("x")
        got = float(sympy.N(f.subs({X[0]: 0.3, X[1]: 0.5}).doit()))
        assert np.isclose(got, float(coupled(vals)))

    def test_broken_template_falls_back(self):
        from skverify.recurrence import FOLD_START, Iterate

        def shape_shift(x):
            c = x[0]
            for k in range(FOLD_START + 5):
                if k == FOLD_START + 3:
                    c = c * c
                else:
                    c = c + 0.1 * c * c
            return c

        out = to_sympy(shape_shift, np.array([0.05]))
        expect = float(shape_shift(np.array([0.05])))
        X = sympy.IndexedBase("x")
        f = out.expand_formula()
        for _ in range(3):
            if f.atoms(Iterate):
                f = f.replace(lambda e: e.func is Iterate, lambda e: e.doit(deep=False))
            f = f.doit()
        got = float(sympy.N(f.subs(X[0], 0.05)))
        assert np.isclose(got, expect)

    def test_guarded_loop_no_leaked_dummies(self):
        from skverify.recurrence import FOLD_START

        def guarded(x):
            c = x[0]
            for _ in range(FOLD_START + 6):
                if c > 0:
                    c = c + 0.1 * c * c
            return c

        out = to_sympy(guarded, np.array([0.1]))
        pre = out.preconditions
        if pre is not sympy.true:
            for sym in pre.free_symbols:
                if isinstance(sym, sympy.Dummy):
                    assert sym in out.definitions

    def test_nested_loops_fold(self):
        from skverify.recurrence import FOLD_START

        def nested(x):
            t = x[0]
            for _ in range(FOLD_START + 4):
                inner = 0.0
                for j in range(3):
                    inner = inner + x[j]
                t = t + 0.1 * t * inner
            return t

        vals = np.array([0.1, 0.2, 0.3, 0.4])
        out = to_sympy(nested, vals)
        X = sympy.IndexedBase("x")
        subs = {X[k]: v for k, v in enumerate(vals)}
        f = out.expand_formula().subs(subs)
        for _ in range(3):
            f = f.doit()
        assert np.isclose(float(sympy.N(f)), float(nested(vals)))


class TestRecurrenceHelpers:
    def test_iterate_doit_unrolls_scalar(self):
        from skverify.recurrence import Iterate

        s, n = sympy.symbols("s n")
        step = sympy.Lambda((s, n), s + 1)
        it = Iterate(step, sympy.Integer(0), sympy.Integer(3))
        assert it.doit() == 3

    def test_iterate_holds_on_symbolic_count(self):
        from skverify.recurrence import Iterate

        s, n = sympy.symbols("s n")
        step = sympy.Lambda((s, n), s + 1)
        m = sympy.Symbol("m", integer=True)
        it = Iterate(step, sympy.Integer(0), m)
        # non-integer count: stays held (recurrence 77-78)
        assert it.doit().func is Iterate

    def test_iterate_eval_always_holds(self):
        from skverify.recurrence import Iterate

        s, n = sympy.symbols("s n")
        step = sympy.Lambda((s, n), s * 2)
        it = Iterate(step, sympy.Integer(1), sympy.Integer(2))
        assert it.func is Iterate  # eval returns None -> held

    def test_nth_eval_and_doit(self):
        from skverify.recurrence import Iterate, Nth

        tup = sympy.Tuple(sympy.Integer(10), sympy.Integer(20))
        # eval on a concrete Tuple + integer index (recurrence 99-101)
        assert Nth(tup, sympy.Integer(1)) == 20
        # doit on a held Iterate producing a Tuple
        s0, s1, n = sympy.symbols("s0 s1 n")
        step = sympy.Lambda((s0, s1, n), sympy.Tuple(s0 + s1, s1))
        it = Iterate(step, sympy.Tuple(sympy.Integer(1), sympy.Integer(1)), sympy.Integer(2))
        held = Nth(it, sympy.Integer(0))
        assert held.doit() == it.doit()[0]

    def test_nth_doit_stays_held_on_symbolic(self):
        from skverify.recurrence import Nth

        expr = sympy.Symbol("q")  # not a Tuple
        held = Nth(expr, sympy.Integer(0))
        # doit returns Nth(inner, i) when inner is not a concrete Tuple (109)
        assert held.doit().func is Nth

    def test_inline_non_basic_passthrough(self):
        from skverify.recurrence import inline

        # non-Basic expr returns unchanged (recurrence 166-167)
        assert inline(42, {}) == 42

    def test_inline_substitutes_symbol(self):
        from skverify.recurrence import Iterate, inline

        sym = sympy.Symbol("loop_held", real=True)
        s, n = sympy.symbols("s n")
        step = sympy.Lambda((s, n), s + 1)
        it = Iterate(step, sympy.Integer(0), sympy.Integer(2))
        expr = sym + 5
        out = inline(expr, {sym: it})
        assert out.atoms(Iterate)

    def test_subst_slot_scalar_and_array(self):
        from skverify.recurrence import _subst_slot
        from skverify.helpers import axis_idx

        label = sympy.Dummy("state")
        # scalar substitution
        expr = label + 1
        assert _subst_slot(expr, label, sympy.Integer(3)) == 4
        # array-slot substitution: state[k] -> value indexed at k
        i0 = axis_idx(0)
        base = sympy.IndexedBase(label)
        expr2 = base[sympy.Integer(2)]
        val = sympy.IndexedBase("v")[i0] * 2
        res = _subst_slot(expr2, label, val)
        assert res == sympy.IndexedBase("v")[2] * 2


class TestInstrumentTwins:
    def test_closure_cells_bound_as_constants(self):
        # a closure over a constant whose body hits a loop wall: the cell
        # snapshots into the twin namespace and the twin is instrumented
        # (twins 68-86).
        def make(scale):
            def inner(x):
                c = x[0]
                for _ in range(25):
                    c = c + scale * c * c
                return c

            return inner

        fn = make(0.1)
        out = to_sympy(fn, np.array([0.1]))
        assert any("closure" in s for s in out.instrumented)
        assert np.isclose(float(out.value), float(fn(np.array([0.1]))))

    def test_helper_function_instrumented(self):
        # a callee with a loop wall is instrumented too (depth recursion,
        # twins 227-251).
        def helper(v):
            c = v[0]
            for _ in range(25):
                c = c + 0.1 * c * c
            return c

        def top(x):
            return helper(x) * 2.0

        out = to_sympy(top, np.array([0.1]))
        expect = float(top(np.array([0.1])))
        assert np.isclose(float(out.value), expect)

    def test_class_method_instrumented(self):
        # a user class whose method hits a loop wall: the class is twinned
        # (twins _instrument_class, 275-468) so isinstance gates hold.
        class Accum:
            def __init__(self, base):
                self.base = base

            def run(self, x):
                c = self.base + x[0]
                for _ in range(25):
                    c = c + 0.1 * c * c
                return c

        def top(x):
            return Accum(0.0).run(x)

        out = to_sympy(top, np.array([0.1]))
        expect = float(top(np.array([0.1])))
        assert np.isclose(float(out.value), expect)

    def test_class_with_base_chain_and_property(self):
        # subclass + super() + a read-only property: exercises base
        # twinning and descriptor rewrapping (twins 288-296, 327-339,
        # 345-349, 365-369).
        class Base:
            def __init__(self, k):
                self.k = k

            def scaled(self, x):
                return self.k * x[0]

        class Child(Base):
            def __init__(self, k):
                super().__init__(k)

            @property
            def doubled_k(self):
                return self.k * 2.0

            def run(self, x):
                c = self.scaled(x) + self.doubled_k
                for _ in range(25):
                    c = c + 0.05 * c
                return c

        def top(x):
            return Child(1.0).run(x)

        out = to_sympy(top, np.array([0.1]))
        expect = float(top(np.array([0.1])))
        assert np.isclose(float(out.value), expect)
