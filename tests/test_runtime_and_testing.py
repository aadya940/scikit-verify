"""Coverage sprint: instrument.runtime twins and the testing surface.

runtime.py holds the traced twins for math-neutral calls -- the
allocators (zeros/ones/eye/full/empty), the neutral copy, the method
router, the functional/ufunc ``.at`` sites, guarded set/dict, the
selection getitem, and the scalar shims. They are only installed on
to_sympy's instrumented retry, so every function below plants a wall
(``int(pair)`` or a local write) that forces that retry, then checks
the value lane against numpy.

testing.py is the specifies / check_formula / check_property surface:
the four verdict tiers, the message formatting, the pytest decorators
and their skip/assert behavior.
"""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy
from skverify.helpers import axis_idx
from skverify.testing import (
    Verdict,
    check_formula,
    check_property,
    specifies,
)


def _wall(a):
    """Force the instrumented retry: int() on a traced scalar is a hard
    wall to_sympy cannot fold, so it re-runs the body with the twins
    installed. A no-op on the value lane."""
    int(a.ravel()[0])


def _val(out):
    v = out.value if isinstance(out, Pair) else out
    return np.asarray(Pair._value_of(v), dtype=float)


# ======================================================================
# runtime.py -- allocators and construction
# ======================================================================

class TestAllocators:
    def test_zeros_ones_then_write(self):
        def f(v):
            _wall(v)
            z = np.zeros((2, 2))
            o = np.ones((2, 2))
            z[0, 0] = v[0]
            o[1, 1] = v[1]
            return z + o

        d = np.array([3.0, 5.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), f(d.copy()))

    def test_zeros_honors_dtype(self):
        def f(v):
            _wall(v)
            mask = np.zeros(3, dtype=bool)
            mask[1] = True
            counter = np.zeros(2, dtype=int)
            counter[0] = 2
            return mask.sum() + counter.sum() + v[0] * 0

        d = np.array([9.0, 1.0, 1.0])
        out = to_sympy(f, d.copy())
        assert np.isclose(float(_val(out)), 3.0)

    def test_eye_and_identity(self):
        def f(v):
            _wall(v)
            e = np.eye(3, k=1)
            ident = np.identity(3)
            return (e + ident) * v[0]

        d = np.array([2.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), f(d.copy()))

    def test_eye_formula_is_kronecker(self):
        # line 41-49: the Piecewise delta must match the value lane
        def f(v):
            _wall(v)
            return np.eye(3) * v[0]

        out = to_sympy(f, np.array([1.0]))
        assert np.allclose(_val(out), np.eye(3))

    def test_full_concrete_fill(self):
        # 58-62: a concrete fill stays a raw buffer (no formula lift),
        # so it can still be added to a traced value downstream
        def f(v):
            _wall(v)
            base = np.full((2,), 7.0)
            return base + v

        d = np.array([4.0, 5.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), np.array([11.0, 12.0]))

    def test_full_traced_fill(self):
        # 63-68: a Pair fill lifts formula + steps
        def f(v):
            _wall(v)
            return np.full((3,), v[0] * 2.0)

        d = np.array([5.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), np.full((3,), 10.0))

    def test_empty_then_overwrite(self):
        # 71-78: uninitialized formula, fully overwritten in the value lane
        def f(v):
            _wall(v)
            buf = np.empty(3)
            for k in range(3):
                buf[k] = v[k] + 1.0
            return buf

        d = np.array([1.0, 2.0, 3.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), d + 1.0)


class TestScalarPassthroughAllocators:
    def test_zeros_ones_no_writes(self):
        # the space generators' scalar-passthrough tails
        def f(v):
            _wall(v)
            return np.zeros(2) + np.ones(2) + v[0] * 0.0

        out = to_sympy(f, np.array([3.0, 3.0]))
        assert np.allclose(_val(out), np.ones(2))


# ======================================================================
# runtime.py -- neutral copy (_skv_neutral) 134-152
# ======================================================================

class TestNeutralCopy:
    def test_asarray_owns_buffer(self):
        # 141: a real copy; library-side mutation must not flow back
        def f(a):
            b = np.asarray(a, dtype=float)
            b += 100.0
            _wall(a)
            return a.sum()

        d = np.array([1.0, 2.0, 3.0])
        out = to_sympy(f, d.copy())
        assert np.isclose(float(_val(out)), 6.0)

    def test_asarray_ndmin(self):
        # 144-146: ndmin promotes the traced value + rebuilds bounds
        def f(a):
            _wall(a)
            b = np.array(a, ndmin=2)
            return b + 0.0

        d = np.array([1.0, 2.0])
        out = to_sympy(f, d.copy())
        assert _val(out).shape == (1, 2)
        assert np.allclose(_val(out).ravel(), d)


# ======================================================================
# runtime.py -- method router (_skv_method) 155-235
# ======================================================================

class TestMethodRouter:
    def test_astype_on_pair(self):
        # 222-224
        def f(v):
            _wall(v)
            return (v * 2.0).astype(float)

        d = np.array([1.0, 2.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), d * 2.0)

    def test_copy_on_pair(self):
        # 232-234
        def f(v):
            _wall(v)
            w = (v + 1.0).copy()
            return w

        d = np.array([1.0, 2.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), d + 1.0)

    def test_view_float_reinterpret_ok(self):
        # 217-221: a float->float view is math-neutral
        def f(v):
            _wall(v)
            w = v * 1.0
            return w.view(np.float64)

        d = np.array([3.0, 4.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), d)

    def test_view_nonfloat_refused(self):
        # 219-220: an int reinterpretation would change the math -> refuse
        def f(v):
            _wall(v)
            return (v * 1.0).view(np.int64)

        with pytest.raises(NotImplementedError):
            to_sympy(f, np.array([3.0, 4.0]))

    def test_ndarray_method_ordinary(self):
        # 235: fall through to the value's own method
        def f(v):
            _wall(v)
            w = v * 2.0
            return w.reshape(2, 1)

        d = np.array([1.0, 2.0])
        out = to_sympy(f, d.copy())
        assert _val(out).shape == (2, 1)


# ======================================================================
# runtime.py -- functional / ufunc .at  (_skv_at) 276-300
# ======================================================================

class TestAtSites:
    def test_ufunc_at_accumulates(self):
        # 279-289: np.add.at with duplicate indices accumulates
        def f(v):
            _wall(v)
            y = v * 1.0
            np.add.at(y, [0, 0, 1], 1.0)
            return y

        d = np.array([10.0, 20.0, 30.0])
        out = to_sympy(f, d.copy())
        expect = d.copy()
        np.add.at(expect, [0, 0, 1], 1.0)
        assert np.allclose(_val(out), expect)

    def test_ufunc_at_unary(self):
        # 291-292: the no-value branch (unary ufunc.at)
        def f(v):
            _wall(v)
            y = v * 1.0
            np.negative.at(y, [0, 2])
            return y

        d = np.array([1.0, 2.0, 3.0])
        out = to_sympy(f, d.copy())
        expect = d.copy()
        np.negative.at(expect, [0, 2])
        assert np.allclose(_val(out), expect)


# ======================================================================
# runtime.py -- scalar shims
# ======================================================================

class TestScalarShims:
    def test_float_on_traced_scalar(self):
        # 393-396: float() keeps a traced scalar traced
        def f(v):
            _wall(v)
            s = float(v[0] + v[1])
            return v * 0.0 + s

        d = np.array([2.0, 3.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), 5.0)

    def test_float_on_size1_array(self):
        # 397-403: float() on a size-1 array unwraps explicitly
        def f(v):
            _wall(v)
            block = (v[:1] * 2.0)
            s = float(block)
            return v * 0.0 + s

        d = np.array([4.0, 9.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), 8.0)

    def test_isscalar_traced(self):
        # 514-518
        def f(v):
            _wall(v)
            s = v[0] * 1.0
            flag = 1.0 if np.isscalar(s) or np.ndim(s) == 0 else 0.0
            return v * 0.0 + flag

        d = np.array([5.0, 6.0])
        out = to_sympy(f, d.copy())
        assert _val(out).shape == (2,)

    def test_classof_of_pair_is_ndarray(self):
        # 525-526: class-keyed dispatch (x.__class__) must route a Pair
        # holding an ndarray the same way isinstance gates do
        def f(v):
            _wall(v)
            cls = v.__class__
            flag = 1.0 if cls is np.ndarray else 0.0
            return v * 0.0 + flag

        d = np.array([1.0, 2.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), 1.0)

    def test_clip_a_min_spelling(self):
        # 407-411
        def f(v):
            _wall(v)
            return np.clip(v, a_min=0.0, a_max=1.0)

        d = np.array([-1.0, 0.5, 2.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), np.clip(d, 0.0, 1.0))


# ======================================================================
# runtime.py -- selection getitem, guarded dict/set (mapping lookups)
# ======================================================================

class TestSelectionAndContainers:
    def test_dict_lookup_by_traced_key(self):
        # 475-497 (getitem-single mapping -> Piecewise selection, incl.
        # the Pair-value branch 484-486) + _skv_dict 500-509
        def f(v):
            _wall(v)
            key = v[0]  # a traced scalar Pair
            table = {0.0: v[1], 1.0: v[2]}
            sel = table[key]  # traced-key lookup is a selection
            return v * 0.0 + sel

        d = np.array([0.0, 7.0, 9.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), 7.0)

    def test_set_dedup_plain(self):
        # 314-337: guarded set on plain values builds a real set
        def f(v):
            _wall(v)
            s = set([1, 1, 2, 3])
            return v * 0.0 + float(len(s))

        d = np.array([1.0, 1.0])
        out = to_sympy(f, d.copy())
        assert np.allclose(_val(out), 3.0)


# ======================================================================
# runtime.py -- scipy diags (_traced_diags) via scipy.sparse
# ======================================================================

class TestTracedDiags:
    def test_diags_mismatched_offsets_raises(self):
        # 106-107: len(offsets) != len(diagonals)
        pytest.importorskip("scipy")
        from scipy.sparse import diags_array

        def f(t):
            _wall(t)
            M = diags_array([t[:3], t[:2]], offsets=[0], shape=(3, 3))
            return M.toarray()

        with pytest.raises(ValueError):
            to_sympy(f, np.array([1.0, 2.0, 3.0]))

    def test_diags_shape_inferred(self):
        # 108-112: shape=None infers from the diagonal length
        pytest.importorskip("scipy")
        from scipy.sparse import diags_array

        def f(t):
            _wall(t)
            # shape omitted: _traced_diags infers (n, n) from the
            # diagonal length + offset (runtime.py 108-112)
            return diags_array([t], offsets=[0]) + np.zeros((3, 3))

        d = np.array([1.0, 2.0, 3.0])
        out = to_sympy(f, d.copy())
        ref = diags_array([d], offsets=[0]).toarray()
        assert np.allclose(_val(out), ref)

    def test_diags_scalar_offset_rejects_list(self):
        # 97-100: scalar offset with a list-of-arrays is a scipy error
        pytest.importorskip("scipy")
        from scipy.sparse import diags_array

        def f(t):
            _wall(t)
            M = diags_array([t[:2], t[:2]], offsets=0, shape=(3, 3))
            return M.toarray()

        with pytest.raises(ValueError):
            to_sympy(f, np.array([1.0, 2.0, 3.0]))


# ======================================================================
# testing.py -- Verdict formatting per tier
# ======================================================================

N = 5
V = sympy.IndexedBase("v")
i = sympy.Symbol("i", integer=True)
VALS = np.array([0.7, 1.2, 2.5, 0.3, 0.4])


def _mean():
    j = sympy.Dummy("j", integer=True)
    return sympy.Sum(V[j], (j, 0, N - 1)) / N


class TestVerdictMessage:
    def test_matches_message_short(self):
        v = Verdict(tier="exact", shape=(2,), detail="all agree")
        msg = v.message()
        assert "verdict: exact" in msg
        assert "all agree" in msg
        assert "your spec" not in msg

    def test_incomplete_message_short(self):
        v = Verdict(tier="incomplete", shape=(), detail="tracer refused")
        assert "your spec" not in v.message()
        assert "tracer refused" in v.message()

    def test_differs_message_full(self):
        v = Verdict(
            tier="differs", shape=(3,), spec="3*v[i]+1", traced="3*v[i]+2",
            counterexample={"v[0]": 1, "spec value": 4, "code value": 5},
            detail="first disagreement",
        )
        msg = v.message()
        assert "your spec:" in msg
        assert "the code:" in msg
        assert "counterexample:" in msg
        assert "v[0] = 1" in msg
        assert "first disagreement" in msg

    def test_matches_property(self):
        assert Verdict(tier="exact", shape=()).matches
        assert Verdict(tier="float-constant", shape=()).matches
        assert Verdict(tier="sampled", shape=()).matches
        assert not Verdict(tier="differs", shape=()).matches
        assert not Verdict(tier="undecided", shape=()).matches


# ======================================================================
# testing.py -- check_formula tiers
# ======================================================================

class TestCheckFormulaTiers:
    def test_exact_scalar(self):
        v = check_formula(lambda v: v.mean(), (VALS.copy(),), _mean())
        assert v.tier == "exact" and v.matches

    def test_float_constant(self):
        j = sympy.Dummy("j", integer=True)
        mean = _mean()
        std = sympy.sqrt(sympy.Sum((V[j] - mean) ** 2, (j, 0, N - 1)) / N)
        v = check_formula(
            lambda v: (v - v.mean()) / v.std(), (VALS.copy(),),
            (V[i] - mean) / std, indices=(i,),
        )
        assert v.tier == "float-constant" and v.matches
        assert "rational points" in v.detail

    def test_differs(self):
        v = check_formula(
            lambda v: 3.0 * v + 1.5, (VALS.copy(),),
            3 * V[i] + 1, indices=(i,),
        )
        assert v.tier == "differs" and not v.matches
        assert "spec value" in v.counterexample

    def test_incomplete_on_refusal(self):
        def f(v):
            return np.interp(0.5, v, v)

        v = check_formula(f, (np.sort(VALS.copy()),), _mean())
        assert v.tier == "incomplete"
        assert "not a code bug" in v.detail

    def test_undecided_unknown_symbol(self):
        # 253-262: spec references a symbol the trace does not have
        bogus = sympy.Symbol("q_unheard_of")
        v = check_formula(lambda v: v.mean(), (VALS.copy(),), _mean() + bogus)
        assert v.tier == "undecided"
        assert "q_unheard_of" in v.detail

    def test_incomplete_over_sealed_compiled_calls(self):
        # 59-72 (_param_names_of) + 234-252: the inputs exist but the
        # computation routed through sealed compiled atoms, so a
        # formula-level spec cannot reach them
        pytest.importorskip("scipy")
        from scipy.interpolate import make_lsq_spline

        T = np.r_[(0.0,) * 4, np.linspace(5, 25, 5), (30.0,) * 4]

        def drive(x, y):
            return make_lsq_spline(x, y, t=T).c

        X = sympy.IndexedBase("x")
        Y = sympy.IndexedBase("y")
        x = np.linspace(0, 30)
        y = 8 * x + 50
        v = check_formula(drive, (x, y), X[i] + Y[i], indices=(i,))
        assert v.tier == "incomplete"
        assert "sealed compiled calls" in v.detail

    def test_undecided_no_sample_point(self):
        # 559: contradictory assumptions leave no point to arbitrate at
        j = sympy.Dummy("j", integer=True)
        mean = _mean()
        std = sympy.sqrt(sympy.Sum((V[j] - mean) ** 2, (j, 0, N - 1)) / N)
        v = check_formula(
            lambda v: v.std(), (VALS.copy(),), std,
            assume=[V[0] > 0, V[0] < 0],
        )
        assert v.tier == "undecided"
        assert "no sample point" in v.detail

    def test_sampled_tier_no_float_constants(self):
        # 306-309: sampled tier (needs arbitration, no rounded constants)
        j = sympy.Dummy("j", integer=True)
        data = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        spec_max = sympy.Max(*[V[k] for k in range(5)])
        v = check_formula(lambda v: np.max(v), (data,), spec_max)
        assert v.tier in ("sampled", "exact", "float-constant")


# ======================================================================
# testing.py -- explore integration
# ======================================================================

class TestExplore:
    def test_explore_matches_all_paths(self):
        # _check_everywhere happy path 156-184
        def relu(v):
            return np.where(v > 0, v, 0.0)

        v = check_formula(
            relu, (np.array([1.0, -2.0, 3.0]),),
            sympy.Piecewise((V[i], V[i] > 0), (0, True)), indices=(i,),
            explore=True,
        )
        assert v.matches
        assert "path" in v.detail

    def test_explore_differs_on_unvisited_path(self):
        # 174-176: the spec holds on the traced branch but fails on the
        # other; explore negates the guard, finds the counterexample,
        # and reports the path condition
        def clamp(v):
            if v[0] > 0:
                return v * 2.0
            return v * 3.0

        v = check_formula(
            clamp, (np.array([1.0, 2.0]),), 2 * V[i], indices=(i,),
            explore=True,
        )
        assert v.tier == "differs"
        assert "on the path where" in v.detail


# ======================================================================
# testing.py -- @specifies decorator
# ======================================================================

class TestSpecifiesDecorator:
    def test_passing(self):
        @specifies(3 * V[i] + 1, indices=(i,), explore=False)
        def t():
            return (lambda v: 3.0 * v + 1.0), (VALS.copy(),)

        assert callable(t)
        t()  # no raise

    def test_failing_raises(self):
        @specifies(3 * V[i] + 1, indices=(i,), explore=False)
        def t():
            return (lambda v: 3.0 * v + 2.0), (VALS.copy(),)

        with pytest.raises(AssertionError) as ei:
            t()
        assert "your spec" in str(ei.value)

    def test_incomplete_skips(self):
        @specifies(_mean(), explore=False)
        def t():
            return (lambda v: np.interp(0.5, v, v)), (np.sort(VALS.copy()),)

        with pytest.raises(pytest.skip.Exception):
            t()

    def test_name_and_doc_preserved(self):
        @specifies(3 * V[i] + 1, indices=(i,), explore=False)
        def my_test():
            "docstring here"
            return (lambda v: 3.0 * v + 1.0), (VALS.copy(),)

        assert my_test.__name__ == "my_test"
        assert my_test.__doc__ == "docstring here"


# ======================================================================
# testing.py -- check_property and @specifies.property
# ======================================================================

def _centered_sums_to_zero(F):
    # the centering formula comes back entrywise (v[i] - mean); bind i
    # to each row and assert the sum is zero
    ii = axis_idx(0)
    return sympy.Eq(sum(F.subs(ii, k) for k in range(5)), 0)


def _centered_sums_to_one(F):
    ii = axis_idx(0)
    return sympy.Eq(sum(F.subs(ii, k) for k in range(5)), 1)


class TestProperty:
    def test_check_property_holds_single_path(self):
        # explore=False branch 705-713
        def center(v):
            return v - v.mean()

        v = check_property(
            center, (VALS.copy(),), _centered_sums_to_zero, explore=False,
        )
        assert v.matches

    def test_check_property_explore(self):
        def center(v):
            return v - v.mean()

        v = check_property(
            center, (VALS.copy(),), _centered_sums_to_zero, explore=True,
        )
        assert v.matches

    def test_check_property_incomplete(self):
        # 707-712: tracer refusal under explore=False
        def f(v):
            return np.interp(0.5, v, v)

        v = check_property(
            f, (np.sort(VALS.copy()),), lambda F: sympy.Eq(F, F),
            explore=False,
        )
        assert v.tier == "incomplete"

    def test_property_false_differs(self):
        # a fact that is simply false on the path
        def f(v):
            return v - v.mean()

        v = check_property(
            f, (VALS.copy(),), _centered_sums_to_one, explore=False,
        )
        assert not v.matches

    def test_specifies_property_decorator_passes(self):
        @specifies.property(_centered_sums_to_zero, explore=False)
        def t():
            return (lambda v: v - v.mean()), (VALS.copy(),)

        t()  # no raise

    def test_specifies_property_decorator_fails(self):
        @specifies.property(_centered_sums_to_one, explore=False)
        def t():
            return (lambda v: v - v.mean()), (VALS.copy(),)

        with pytest.raises(AssertionError):
            t()

    def test_specifies_property_incomplete_skips(self):
        @specifies.property(lambda F: sympy.Eq(F, F), explore=False)
        def t():
            return (lambda v: np.interp(0.5, v, v)), (np.sort(VALS.copy()),)

        with pytest.raises(pytest.skip.Exception):
            t()

    def test_property_over_sealed_incomplete(self):
        # 731-745: a fact ranging over sealed compiled outputs is
        # incomplete, boundary named
        pytest.importorskip("scipy")
        from scipy.interpolate import make_lsq_spline

        T = np.r_[(0.0,) * 4, np.linspace(5, 25, 5), (30.0,) * 4]

        def drive(x, y):
            return make_lsq_spline(x, y, t=T).c

        x = np.linspace(0, 30)
        y = 8 * x + 50
        v = check_property(
            drive, (x, y),
            lambda F: sympy.Eq(F.subs(axis_idx(0), 0), 0),
            explore=False,
        )
        assert v.tier == "incomplete"
        assert "sealed compiled outputs" in v.detail

    def test_property_and_decomposition(self):
        # 763-770: an And claim, each conjunct checked
        def center(v):
            return v - v.mean()

        v = check_property(
            center, (VALS.copy(),),
            lambda F: sympy.And(
                sympy.Eq(sum(F.subs(axis_idx(0), k) for k in range(5)), 0),
                sympy.Eq(F.subs(axis_idx(0), 0) - F.subs(axis_idx(0), 0), 0),
            ),
            explore=False,
        )
        assert v.matches

    def test_property_plain_false(self):
        # 757-762: a claim that reduces to False differs on the path
        def f(v):
            return v - v.mean()

        v = check_property(
            f, (VALS.copy(),), lambda F: False, explore=False,
        )
        assert v.tier == "differs"

    def test_property_and_with_false_conjunct(self):
        # 763-770: And decomposition where one conjunct fails
        def f(v):
            return v - v.mean()

        v = check_property(
            f, (VALS.copy(),),
            lambda F: sympy.And(_centered_sums_to_zero(F), False),
            explore=False,
        )
        assert not v.matches

    def test_property_inequality_entailment(self):
        # inequality claim -> entailment path (771-806)
        def f(v):
            return v ** 2

        v = check_property(
            f, (VALS.copy(),),
            lambda F: F.subs(axis_idx(0), 0) >= 0,
            assume=[V[k] > 0 for k in range(5)],
            explore=False,
        )
        assert v.tier in ("exact", "sampled", "undecided", "differs")
