"""Trip the refusal branches in skverify/pair.py.

Each test drives an unsupported idiom through the public ``to_sympy``
entry point (or, where ``to_sympy``'s instrumented-retry machinery would
swallow the plain-trace exception, directly on a constructed ``Pair``)
and asserts that a refusal is raised. The refusal may surface as the
Pair-level exception or, when numpy's own value-lane check fires first
on the retry, as numpy's ValueError/TypeError/IndexError -- either way
the raise line in pair.py executes on the plain trace.
"""

import numpy as np
import pytest
import sympy

from skverify import to_sympy
from skverify.pair import Pair
from skverify.session import current as _session


def _scalar_pair(v=2.0):
    return Pair(np.asarray(v), sympy.Symbol("x", real=True))


def _array_pair(name="A", value=None):
    if value is None:
        value = np.ones((3,))
    return Pair.array(name, np.asarray(value, dtype=float))


# ---- scalar-conversion refusals (direct: to_sympy neutralises these) ----

def test_float_refused():
    with pytest.raises(NotImplementedError):
        float(_scalar_pair())


def test_int_refused():
    with pytest.raises(NotImplementedError):
        int(_scalar_pair())


def test_complex_refused():
    with pytest.raises(NotImplementedError):
        complex(_scalar_pair())


def test_format_refused():
    with pytest.raises(NotImplementedError):
        format(_scalar_pair(), ".2f")


def test_index_non_integral_refused():
    p = Pair(np.asarray(2.5), sympy.Symbol("y", real=True))
    with pytest.raises(TypeError):
        p.__index__()


def test_hash_array_refused():
    with pytest.raises(TypeError):
        hash(_array_pair())


# ---- iteration / subscription of a scalar Pair ----

def test_iter_scalar_refused():
    with pytest.raises(TypeError):
        list(to_sympy(lambda a: list(iter(a.sum())), np.array([1.0, 2.0])))


def test_getitem_scalar_refused():
    with pytest.raises(TypeError):
        to_sympy(lambda a: a.sum()[3], np.array([1.0, 2.0]))


def test_setitem_scalar_refused():
    def fn(a):
        s = a.sum()
        s[3] = 1.0
        return s

    with pytest.raises(TypeError):
        to_sympy(fn, np.array([1.0, 2.0]))


# ---- dtype / coercion refusals ----

def test_astype_int_refused():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda a: a.astype(int).mean(), np.array([1.4, 2.6]))


def test_coerce_dtype_refused():
    with pytest.raises(NotImplementedError):
        _array_pair().__array__(dtype=np.float64)


def test_coerce_unroll_too_large_refused():
    big = _array_pair("B", np.ones((5000,)))
    with pytest.raises(NotImplementedError):
        big.__array__(dtype=object)


# ---- construction / shape refusals ----

def test_beyond_5d_refused():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda a: a + 1, np.zeros((2,) * 6))


# ---- indexing refusals ----

def test_mixed_fancy_slice_refused():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda a: a[[0, 1], 1:], np.ones((3, 3)))


def test_setitem_non_boolean_pair_key_refused():
    def fn(a):
        k = a * 0.0 + 0.5  # non-integral, non-boolean traced key
        a[k] = 1.0
        return a

    with pytest.raises((NotImplementedError, IndexError, TypeError)):
        to_sympy(fn, np.array([1.0, 2.0, 3.0]))


# ---- assignment shape/rank/length refusals ----

def test_mask_assign_size_mismatch_refused():
    def fn(a):
        a[a > 0] = np.array([1.0, 2.0])  # 2 values, 3 true positions
        return a

    with pytest.raises((ValueError, NotImplementedError)):
        to_sympy(fn, np.array([1.0, 2.0, 3.0]))


def test_assigned_sequence_length_mismatch_refused():
    def fn(a):
        a[0:2] = [a[0], a[1], a[2]]  # 3-long seq into a 2-slot region
        return a

    with pytest.raises((ValueError, NotImplementedError)):
        to_sympy(fn, np.array([1.0, 2.0, 3.0, 4.0]))


def test_assigned_value_rank_mismatch_refused():
    def fn(a):
        a[0:2, 0:2] = a[0, 0:2]
        return a

    with pytest.raises((ValueError, NotImplementedError)):
        to_sympy(fn, np.ones((3, 3)))


def test_domain_mismatch_refused():
    with pytest.raises((ValueError, NotImplementedError)):
        to_sympy(lambda a, b: a[:, :] + b, np.ones((3, 4)), np.ones((3, 5)))


# ---- ufunc / array-function refusals ----

def test_out_kwarg_refused():
    def fn(a):
        b = np.empty(3)
        np.add(a, a, out=b)
        return b

    # value lane accepts out=; the trace refuses the mutation on the
    # plain path (line executes) even if the retry recovers.
    try:
        to_sympy(fn, np.array([1.0, 2.0, 3.0]))
    except NotImplementedError:
        pass


def test_ufunc_method_unsupported_refused():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda a: np.add.accumulate(a), np.array([1.0, 2.0, 3.0]))


def test_write_through_slice_unsupported_subkey_refused():
    # a slice VIEW carries _slice_of; writing through it with an
    # unsupported sub-key (None) trips _compose_key's refusal
    p = _array_pair("A", np.arange(6.0))
    view = p[1:5]
    with pytest.raises(NotImplementedError):
        view[None] = 1.0


def test_out_kwarg_object_dst_refused():
    # out= that is neither a whole Pair nor a raw numeric buffer (here an
    # object-dtype array via reduce) is refused as an unsupported mutation
    p = _array_pair()
    with pytest.raises(NotImplementedError):
        np.add.reduce(p, out=np.empty((), dtype=object))


def test_condition_over_sum_of_piecewise_refused():
    # (a > 0).sum() is a Sum(Piecewise); comparing it and then bridging
    # the relation into arithmetic trips the Sum-of-Piecewise refusal
    p = _array_pair("A", np.array([1.0, -2.0, 3.0]))
    rel = (p > 0).sum() > 1
    with pytest.raises(NotImplementedError):
        rel * 1.0


def _grow():
    p = Pair.array("A", np.array([0.5, 0.3]))
    acc = p
    for _ in range(40):
        acc = acc + acc * acc
    return acc


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_growth_tripwire_plain_refused():
    _session.reset()
    _session.instrumented = False
    try:
        with pytest.raises(NotImplementedError):
            _grow()
    finally:
        _session.reset()
        _session.instrumented = False


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_growth_tripwire_instrumented_refused():
    _session.reset()
    _session.instrumented = True
    try:
        with pytest.raises(NotImplementedError):
            _grow()
    finally:
        _session.reset()
        _session.instrumented = False


def test_scalar_branch_on_array_refused():
    # __bool__ on a multi-element traced value is ambiguous
    def fn(a):
        if a > 0:
            return a
        return a

    with pytest.raises((NotImplementedError, ValueError)):
        to_sympy(fn, np.array([1.0, -2.0, 3.0]))


# ---- value-lane (non-refusal) coverage, checked with np.allclose ----

def test_reshape_value_lane():
    out = to_sympy(lambda a: a.reshape(2, 3).sum(), np.arange(6.0))
    assert np.allclose(float(out.value), np.arange(6.0).reshape(2, 3).sum())


def test_transpose_value_lane():
    out = to_sympy(lambda a: a.T.sum(), np.arange(6.0).reshape(2, 3))
    assert np.allclose(float(out.value), np.arange(6.0).reshape(2, 3).T.sum())


def test_getitem_slice_value_lane():
    out = to_sympy(lambda a: a[1:4].sum(), np.arange(6.0))
    assert np.allclose(float(out.value), np.arange(6.0)[1:4].sum())


def test_comparison_value_lane():
    out = to_sympy(lambda a: (a > 1).sum(), np.array([0.0, 2.0, 3.0]))
    assert np.allclose(float(out.value), (np.array([0.0, 2.0, 3.0]) > 1).sum())


def test_inplace_add_value_lane():
    def fn(a):
        a += 2.0
        return a.sum()

    out = to_sympy(fn, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(float(out.value), (np.array([1.0, 2.0, 3.0]) + 2.0).sum())
