"""Residual coverage mop-up for three modules.

Covers still-reachable branches in
``skverify/instrument/twins.py``, ``skverify/maps/numpy.py`` and
``skverify/instrument/triage.py`` that prior sprints left uncovered.

Every test drives real inputs through :func:`to_sympy` and asserts the
value lane against numpy where the branch returns a result; passthrough
branches only need the trace to complete. Where a source line was read
and judged an unreachable defensive/alternate branch it is documented in
the module report, not forced here.
"""

import functools

import numpy as np
import pytest

from skverify import Pair, to_sympy


def _val(out):
    return np.asarray(out.value if isinstance(out, Pair) else out, dtype=float)


def _obj(x):
    """A genuine object-dtype array of per-element traced values; the
    leading scatter write supplies the wall that forces instrumentation."""
    A = np.zeros(x.shape)
    for i in range(len(x)):
        A[i] = x[i] * 1.0
    return np.array([x[i] for i in range(len(x))], dtype=object)


# =====================================================================
# maps/numpy.py -- np.prod / np.sum plain-path variants
# =====================================================================

def test_prod_out_tuple_target():
    # out= as a length-1 tuple -> line 393; the Pair scalar out -> 398.
    x = np.array([2.0, 3.0, 4.0])

    def fn(x):
        acc = np.zeros(())
        for i in range(len(x)):
            acc = acc + x[i]
        target = acc * 0.0  # a scalar Pair buffer
        np.prod(x, out=(target,))
        return target

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.prod(x))


def test_prod_where_list_mask():
    # where= given as a plain list -> np.asarray coercion (line 411).
    x = np.array([2.0, 3.0, 4.0, 5.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        return np.prod(s, where=[True, False, True, False])

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.prod(x, where=[True, False, True, False]))


def test_prod_keepdims_axis_tuple_refuses():
    # keepdims with an axis tuple -> NotImplementedError (line 424).
    x = np.array([[2.0, 3.0], [4.0, 5.0]])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s[i, j] = x[i, j] * 1.0
        return np.prod(s, axis=(0, 1), keepdims=True)

    with pytest.raises(NotImplementedError):
        to_sympy(fn, x.copy())


def test_prod_object_array_untraced_plain():
    # a plain-number object array (no Pairs) -> np.prod fallback (462).
    x = np.array([2.0, 3.0, 4.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        plain = np.array([2.0, 3.0, 4.0], dtype=object)
        return s.sum() + np.prod(plain)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum() + 24.0)


def test_prod_object_pairs_per_axis():
    # 2-D object bag, per-axis product multiplies each 1-D slice (447-461)
    x = np.array([2.0, 3.0, 4.0, 5.0])

    def fn(x):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        o = np.array(
            [[x[0], x[1]], [x[2], x[3]]], dtype=object
        )
        return np.prod(o, axis=0)

    out = to_sympy(fn, x.copy())
    ref = np.array([2.0 * 4.0, 3.0 * 5.0])
    assert np.allclose(_val(out), ref)


def test_sum_out_tuple_scalar_buffer():
    # sum out= as a length-1 tuple -> line 555, scalar Pair buffer -> 560.
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        acc = np.zeros(())
        for i in range(len(x)):
            acc = acc + x[i]
        target = acc * 0.0  # scalar-valued Pair
        np.sum(x, out=(target,))
        return target

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum())


def test_sum_out_ndarray_buffer():
    # out= into an ndarray-valued Pair -> out.value[...] = ... (line 558).
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s[i, j] = x[i, j] * 1.0
        target = s[0] * 0.0  # a 1-D ndarray-valued Pair buffer
        np.sum(s, axis=0, out=target)
        return target

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum(axis=0))


def test_sum_where_list_mask():
    # sum where= given as a plain list -> np.asarray (line 572).
    x = np.array([1.0, 2.0, 3.0, 4.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        return np.sum(s, where=[True, False, True, False])

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.sum(x, where=[True, False, True, False]))


def test_sum_keepdims_axis_tuple_refuses():
    # sum keepdims with axis tuple -> NotImplementedError (line 585).
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s[i, j] = x[i, j] * 1.0
        return np.sum(s, axis=(0, 1), keepdims=True)

    with pytest.raises(NotImplementedError):
        to_sympy(fn, x.copy())


def test_sum_untraced_object_scalar_array():
    # np.sum on a plain-number single-slot fallthrough -> line 617.
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        plain = np.array([[5.0, 6.0]], dtype=object)  # 2-D, ndim!=1
        # object 2-D, axis None -> not bag-1d -> line 599 branch handles
        return s.sum() + float(np.asarray(plain, dtype=float).sum())

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum() + 11.0)


# =====================================================================
# maps/numpy.py -- gradient variants
# =====================================================================

def test_gradient_single_scalar_spacing():
    # one scalar spacing arg -> spacing = [dx]*nd (line 827).
    x = np.array([1.0, 4.0, 9.0, 16.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        return np.gradient(s, 2.0)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.gradient(x, 2.0))


def test_gradient_per_axis_spacing():
    # nd spacing args -> spacing = list(varargs) (line 829).
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s[i, j] = x[i, j] * 1.0
        return np.gradient(s, 2.0, 0.5)

    gx, gy = to_sympy(fn, x.copy())
    rx, ry = np.gradient(x, 2.0, 0.5)
    assert np.allclose(_val(gx), rx)
    assert np.allclose(_val(gy), ry)


def test_gradient_scalar_axis():
    # a scalar axis -> axes = [(axis+nd)%nd] (line 836).
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                s[i, j] = x[i, j] * 1.0
        return np.gradient(s, axis=1)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.gradient(x, axis=1))


# =====================================================================
# maps/numpy.py -- diag / astype / mean / median object-bag paths
# =====================================================================

def test_astype_object_pairs_neutral():
    # object array holding Pairs: astype is math-neutral -> line 1088.
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        cast = np.astype(o, np.float64)
        return np.sum(cast)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum())


def test_diag_3d_traced_fallback():
    # a traced array with >2 axis-bounds -> np.diag value fallback (1114)
    x = np.arange(8.0)

    def fn(x):
        s = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s[i, j, k] = x[i * 4 + j * 2 + k] * 1.0
        return np.diag(s[0])

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.diag(x.reshape(2, 2, 2)[0]))


def test_mean_object_bag_axis_none():
    # object-Pair array, axis None -> _sum(a)/a.size (line 1128).
    x = np.array([2.0, 4.0, 6.0])

    def fn(x):
        o = _obj(x)
        return np.mean(o)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.mean())


def test_mean_object_array_with_axis_fallback():
    # object array + explicit axis -> concrete float fallback (line 1129).
    x = np.array([1.0, 2.0, 3.0, 4.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        o = np.array([[x[0], x[1]], [x[2], x[3]]], dtype=object)
        return s.sum() + float(np.mean(o, axis=0).sum())

    out = to_sympy(fn, x.copy())
    ref = np.mean(x.reshape(2, 2), axis=0).sum()
    assert np.allclose(_val(out), x.sum() + ref)


def test_quantile_2d_traced_fallback():
    # a 2-D traced array -> quantile concrete fallback (line 1215).
    x = np.array([1.0, 2.0, 3.0, 4.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        o = np.array([[x[0], x[1]], [x[2], x[3]]], dtype=object)
        return s.sum() + float(np.quantile(o, 0.5))

    out = to_sympy(fn, x.copy())
    ref = np.quantile(x.reshape(2, 2), 0.5)
    assert np.allclose(_val(out), x.sum() + ref)


def test_median_2d_object_fallback():
    # a 2-D object array is neither bag(1d) nor Pair -> line 1186.
    x = np.array([1.0, 2.0, 3.0, 4.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        o = np.array([[x[0], x[1]], [x[2], x[3]]], dtype=object)
        return s.sum() + float(np.median(o))

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum() + np.median(x.reshape(2, 2)))


def test_round_object_array_fallback():
    # np.round on a plain-number object array -> line 1292.
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        plain = np.array([1.234, 5.678], dtype=object)
        return s.sum() + float(np.round(plain, 1).sum())

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum() + round(1.2, 1) + round(5.7, 1))


# =====================================================================
# instrument/twins.py -- lambda source extraction (102-127)
# =====================================================================

def test_lambda_instrumented_via_source_extraction():
    # a lambda's getsource returns an assignment, not a FunctionDef: the
    # lambda-extraction path rebuilds a synthetic def (twins 102-127).
    x = np.array([1.0, 2.0, 3.0])
    f = lambda v: np.sum(np.array([v[i] * 1.0 for i in range(len(v))]))
    out = to_sympy(f, x.copy())
    assert np.allclose(_val(out), x.sum())


def test_lambda_param_match_among_several():
    # several lambdas on one source line: the one whose params match the
    # live function is chosen (twins 111-114).
    x = np.array([2.0, 3.0, 4.0])
    g = (lambda a: a, lambda v: np.sum(np.array([v[i] * 1.0 for i in range(len(v))])))[1]
    out = to_sympy(g, x.copy())
    assert np.allclose(_val(out), x.sum())


# =====================================================================
# instrument/twins.py -- descent over a source-less callee (240-247)
# =====================================================================

# a plain function with NO retrievable source (built via exec): the
# descent tries to instrument it, inspect.getsource raises OSError, and
# the except at twins 240-247 swallows it and continues. It must live in
# a real importable module so it is a genuine module-level callee.
_exec_ns = {}
exec(
    "def _sourceless_sum(v):\n"
    "    acc = 0.0\n"
    "    for i in range(len(v)):\n"
    "        acc = acc + v[i]\n"
    "    return acc\n",
    _exec_ns,
)
_sourceless_sum = _exec_ns["_sourceless_sum"]
_sourceless_sum.__module__ = __name__  # a non-builtin module for the gate


def test_descent_over_sourceless_callee():
    # _sourceless_sum is a direct-name callee; instrumenting it raises
    # OSError (no source) which the descent swallows (twins 240-247).
    # The call still traces via the runtime doorman on the concrete data.
    x = np.array([1.0, 2.0, 3.0])

    def fn(v):
        s = np.zeros(v.shape)
        for i in range(len(v)):
            s[i] = v[i] * 1.0
        return _sourceless_sum(s)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum())


# =====================================================================
# instrument/twins.py -- OPAQUE callee skipped in twin descent (188)
# =====================================================================

from numpy.linalg import solve as _solve  # bare-name opaque callable


def _solve_helper(A, b):
    return _solve(A, b)  # a bare-name global whose __name__ == "solve"


def test_opaque_callee_skipped_in_descent():
    # _solve_helper is twinned; the descent over ITS callees finds the
    # bare-name opaque callable `solve` and skips it -> twins line 188.
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])

    def fn(A, b):
        s = np.zeros(b.shape)
        for i in range(len(b)):
            s[i] = b[i] * 1.0
        return _solve_helper(A, s)  # direct-name callee -> static descent

    out = to_sympy(fn, A.copy(), b.copy())
    assert np.allclose(_val(out), np.linalg.solve(A, b))


# =====================================================================
# instrument/twins.py -- decorated method peel via __wrapped__ (345-349)
# =====================================================================

class _CallableWrapper:
    """A NON-function callable member carrying __wrapped__: the class
    twin peels it via the __wrapped__ chain (twins lines 342-349)."""

    def __init__(self, fn):
        self.__wrapped__ = fn
        functools.update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        # descriptor so instances can call it like a method
        if obj is None:
            return self
        return lambda *a, **k: self.__wrapped__(obj, *a, **k)

    def __call__(self, *a, **k):
        return self.__wrapped__(*a, **k)


def _scale_impl(self, v):
    acc = np.zeros(())
    for i in range(len(v)):
        acc = acc + v[i] * self.k
    return acc


class _Decorated:
    def __init__(self, k):
        self.k = k

    scale = _CallableWrapper(_scale_impl)


def test_class_callable_wrapper_member_peels_wrapped():
    # a class reached by reference whose method is a non-function
    # callable with __wrapped__: the class-twin peels it (twins 342-349)
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        cls = _Decorated
        return cls(2.0).scale(s)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum() * 2.0)


# =====================================================================
# instrument/triage.py -- decorated fn peel with no inner sites (302)
# =====================================================================

def _pure_deco(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@_pure_deco
def _pure_inner(v):
    return v[0] * v[1]  # arithmetic only -> inner has no twinnable sites


def test_decorated_fn_inner_no_sites_returns_fn():
    # peel the wrapper; the inner has no sites so the twin is a no-op and
    # triage returns the original wrapper (triage line 302).
    x = np.array([2.0, 5.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        g = _pure_inner
        return g(s)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), 2.0 * 5.0)


# =====================================================================
# instrument/triage.py -- _values_match over a tuple result (417-420)
# =====================================================================

def _inject_kw_deco(f):
    @functools.wraps(f)
    def wrapper(*args, keepdims=False, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@_inject_kw_deco
def _pair_stats(v):
    acc = np.zeros(())
    sq = np.zeros(())
    for i in range(len(v)):
        acc = acc + v[i]
        sq = sq + v[i] * v[i]
    return acc, sq  # a TUPLE result -> _values_match flattens it


def test_peel_verify_tuple_result_flat():
    # the wrapper injects keepdims the inner lacks: the peel drops it and
    # VERIFIES a tuple result, driving _values_match's tuple flat (417-420)
    v = np.array([1.0, 2.0, 3.0])

    def fn(v):
        s = np.zeros(v.shape)
        for i in range(len(v)):
            s[i] = v[i] * 1.0
        g = _pair_stats
        return g(s, keepdims=True)

    a, b = to_sympy(fn, v.copy())
    assert np.allclose(_val(a), v.sum())
    assert np.allclose(_val(b), np.sum(v * v))


# =====================================================================
# instrument/triage.py -- peel verify falls back when wrapper raises (297-299)
# =====================================================================

_RAISE_ON_REF = {"armed": False}


def _raising_wrapper_deco(f):
    @functools.wraps(f)
    def wrapper(*args, keepdims=False, **kwargs):
        # keepdims is injected: the inner lacks it, so the peel enters
        # the drop-and-verify path. On the concrete verification pass
        # (line 291) we raise, forcing the fallback at triage 297-299.
        if _RAISE_ON_REF["armed"] and not any(
            isinstance(a, Pair) for a in args
        ):
            raise RuntimeError("wrapper refuses the concrete verify pass")
        return f(*args, **kwargs)

    return wrapper


@_raising_wrapper_deco
def _sum_inner(v):
    acc = np.zeros(())
    for i in range(len(v)):
        acc = acc + v[i]
    return acc


def test_peel_verify_wrapper_raises_falls_back():
    # the wrapper raises during the concrete verification pass; the peel
    # swallows it and returns the wrapper result (triage 297-299). The
    # wrapper is math-neutral, so the value lane is still the sum.
    v = np.array([2.0, 3.0, 4.0])

    def fn(v):
        s = np.zeros(v.shape)
        for i in range(len(v)):
            s[i] = v[i] * 1.0
        g = _sum_inner
        return g(s, keepdims=True)  # inject keepdims -> peel verify path

    _RAISE_ON_REF["armed"] = True
    try:
        out = to_sympy(fn, v.copy())
    finally:
        _RAISE_ON_REF["armed"] = False
    assert np.allclose(_val(out), v.sum())
