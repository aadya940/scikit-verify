"""Drive the runtime dispatch policy in skverify/instrument/triage.py.

Triage's doorman (``_skv_maybe``) fires only for calls the rewriter
cannot resolve statically -- a callable reached through a LOCAL
VARIABLE or a computed expression. Attribute calls (``np.linalg.solve``)
are rewritten at compile time and never reach the doorman. Two more
facts shape these tests:

* The instrumented retry runs only after the plain trace WALLS, so
  each function opens with a scatter write into a local ``np.zeros``
  buffer -- writing a traced value into an array is that wall.
* ``np.zeros`` allocations mask as ``float64`` while holding Pairs, so
  the ufunc object-loop branches (which key on ``dtype == object``)
  need a genuine object array, built here with ``np.array([...],
  dtype=object)`` over per-element traced values.

Where a branch returns a result the value lane is checked against
numpy/scipy; where the branch is a passthrough the trace need only
complete.
"""

import functools

import numpy as np
import pytest

from skverify import Pair, to_sympy


def _val(out):
    return np.asarray(out.value if isinstance(out, Pair) else out, dtype=float)


def _asbool(out):
    return np.asarray(out.value if isinstance(out, Pair) else out, dtype=bool)


# ---------------------------------------------------------------------
# Curated opaque boundary reached by reference: line 73 -> _skv_opaque
# ---------------------------------------------------------------------

def test_opaque_solve_reached_by_reference():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])

    def fn(A, b):
        scratch = np.zeros(b.shape)
        for i in range(len(b)):
            scratch[i] = b[i] * 1.0
        solver = np.linalg.solve  # reference -> the runtime doorman
        return solver(A, scratch)

    out = to_sympy(fn, A.copy(), b.copy())
    assert np.allclose(_val(out), np.linalg.solve(A, b))


def test_opaque_inv_traced_and_untraced_passthrough():
    # the opaque wrapper on UNTRACED data runs the real call (line 445).
    A = np.array([[4.0, 2.0], [1.0, 3.0]])

    def fn(A):
        scratch = np.zeros(A.shape)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                scratch[i, j] = A[i, j] * 1.0
        inv = np.linalg.inv
        plain = inv(np.eye(2))  # untraced -> real inv
        traced = inv(scratch)  # traced -> opaque atom
        return traced, plain

    out_t, out_p = to_sympy(fn, A.copy())
    assert np.allclose(_val(out_t), np.linalg.inv(A))
    assert np.allclose(_val(out_p), np.eye(2))


# ---------------------------------------------------------------------
# concrete_inventory: set routines and sparse constructors (100-146)
# ---------------------------------------------------------------------

def test_unique_by_reference_runs_concrete():
    x = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0])

    def fn(x):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        uniq = np.unique
        return uniq(A)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.unique(x))


def test_isin_by_reference_concrete():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0])

    def fn(x, y):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        member = np.isin
        return member(A, y)

    out = to_sympy(fn, x.copy(), y.copy())
    got = np.asarray(out.value if isinstance(out, Pair) else out)
    assert np.array_equal(np.asarray(got, dtype=bool), np.isin(x, y))


def test_unique_on_object_dtype_array():
    # a true object-dtype buffer exercises the object-dtype limb of
    # concrete_inventory's deep() (lines 106-115).
    x = np.array([2.0, 1.0, 2.0, 3.0, 1.0])

    def fn(x):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        obj = np.array([x[i] for i in range(len(x))], dtype=object)
        uniq = np.unique
        return uniq(obj)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.unique(x))


def test_coo_matrix_intifies_float_indices():
    pytest.importorskip("scipy")
    from scipy.sparse import coo_matrix

    data = np.array([10.0, 20.0, 30.0])

    def fn(data):
        scratch = np.zeros(data.shape)
        for i in range(len(data)):
            scratch[i] = data[i] * 1.0
        row = np.array([0.0, 1.0, 2.0])
        col = np.array([0.0, 1.0, 2.0])
        ctor = coo_matrix
        M = ctor((scratch, (row, col)), shape=(3, 3))
        return M.toarray()

    ref = coo_matrix(
        (data, (np.array([0, 1, 2]), np.array([0, 1, 2]))), shape=(3, 3)
    ).toarray()
    out = to_sympy(fn, data.copy())
    assert np.allclose(_val(out), ref)


def test_coo_matrix_object_dtype_kwarg_dropped():
    # a dtype=object kwarg derived from traced weights is dropped so
    # scipy does not fold the matrix to zeros (lines 134-143).
    pytest.importorskip("scipy")
    from scipy.sparse import coo_matrix

    data = np.array([1.0, 2.0, 3.0])

    def fn(data):
        scratch = np.zeros(data.shape)
        for i in range(len(data)):
            scratch[i] = data[i] * 1.0
        row = np.array([0.0, 1.0, 2.0])
        col = np.array([0.0, 1.0, 2.0])
        ctor = coo_matrix
        M = ctor((scratch, (row, col)), shape=(3, 3), dtype=object)
        return M.toarray()

    ref = coo_matrix(
        (data, (np.array([0, 1, 2]), np.array([0, 1, 2]))), shape=(3, 3)
    ).toarray()
    out = to_sympy(fn, data.copy())
    assert np.allclose(_val(out), ref)


def test_csr_matrix_from_dense_traced():
    pytest.importorskip("scipy")
    from scipy.sparse import csr_matrix

    dense = np.array([[1.0, 0.0], [0.0, 2.0]])

    def fn(d):
        scratch = np.zeros(d.shape)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                scratch[i, j] = d[i, j] * 1.0
        ctor = csr_matrix
        return ctor(scratch).toarray()

    out = to_sympy(fn, dense.copy())
    assert np.allclose(_val(out), dense)


# ---------------------------------------------------------------------
# RNG generator method passthrough: line 160
# ---------------------------------------------------------------------

def test_rng_non_dist_method_passes_through():
    # a bound Generator method NOT in RNG_DISTS returns as-is (line 160).
    # (Under trace_rng the generator is a TracedGenerator, so the
    #  distribution-shim limb 154-159 is not reachable here -- see the
    #  module report.)
    def fn(x):
        scratch = np.zeros(x.shape)
        for i in range(len(x)):
            scratch[i] = x[i] * 1.0
        rng = np.random.default_rng(1)
        shuffle = rng.permutation
        perm = shuffle(len(scratch))
        return scratch[perm]

    x = np.array([5.0, 6.0, 7.0, 8.0])
    out = to_sympy(fn, x.copy())
    ref = x[np.random.default_rng(1).permutation(len(x))]
    assert np.allclose(np.sort(_val(out)), np.sort(ref))


# ---------------------------------------------------------------------
# ufunc shim over genuine object-Pair arrays: lines 347-391
# ---------------------------------------------------------------------

def _obj(x):
    """A real object-dtype array of per-element traced values (a scatter
    write first supplies the wall that forces instrumentation)."""
    A = np.zeros(x.shape)
    for i in range(len(x)):
        A[i] = x[i] * 1.0
    return np.array([x[i] for i in range(len(x))], dtype=object)


def test_isnan_family_concrete_on_object_pairs():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        chk = np.isnan
        ci = np.isinf
        cf = np.isfinite
        return chk(o), ci(o), cf(o)

    n, i, f = to_sympy(fn, x.copy())
    assert not _asbool(n).any()
    assert not _asbool(i).any()
    assert _asbool(f).all()


def test_isnan_scalar_traced_value():
    # a scalar traced value (not an array) exercises the scalar limb of
    # ufunc_shim's conc() (line 361).
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        s = np.zeros(())
        for i in range(len(x)):
            s = s + x[i]
        chk = np.isnan
        return chk(s)

    out = to_sympy(fn, x.copy())
    assert not bool(out.value if isinstance(out, Pair) else out)


def test_ufunc_elementwise_object_loop():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        adder = np.add
        return adder(o, o)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x + x)


def test_ufunc_out_object_array():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        target = np.empty(o.shape, dtype=object)
        adder = np.add
        adder(o, o, out=target)
        return target

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x + x)


def test_ufunc_out_tuple_target():
    # out= passed as a length-1 tuple exercises line 369-370.
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        target = np.empty(o.shape, dtype=object)
        adder = np.add
        adder(o, o, out=(target,))
        return target

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x + x)


def test_ufunc_plain_number_object_array():
    # an object array carrying plain numbers (no Pairs): the ufunc shim
    # coerces to numeric and runs (lines 383-391). The object array is
    # built from an untraced constant so it holds floats, not Pairs,
    # while the scatter still forces the instrumented retry.
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        plain = np.array([4.0, 5.0, 6.0], dtype=object)
        sq = np.square
        return sq(plain)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.array([16.0, 25.0, 36.0]))


# ---------------------------------------------------------------------
# numpy dispatcher shim over object-Pair arrays: lines 315-345
# ---------------------------------------------------------------------

def test_dispatcher_shim_on_object_pairs():
    # np.concatenate is a dispatcher (has __wrapped__, not a function);
    # an object array of Pairs bypasses the __array_function__ protocol,
    # so triage routes it through the function table (line 336-342).
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        cat = np.concatenate
        return cat([o, o])

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.concatenate([x, x]))


def test_dispatcher_shim_table_entry():
    # np.clip has a function-table entry: an object-Pair array routes
    # through it (the dispatcher_shim limb, lines 336-342).
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        clip = np.clip
        return clip(o, 1.5, 2.5)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.clip(x, 1.5, 2.5))


def test_dispatcher_plain_shim_plain_number_array():
    # np.flip has no table entry -> plain_shim; a plain-number object
    # array (no Pairs) is coerced to numeric (lines 322-332).
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        plain = np.array([4.0, 5.0, 6.0], dtype=object)
        flip = np.flip
        return flip(plain)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), np.array([6.0, 5.0, 4.0]))


# ---------------------------------------------------------------------
# runtime class twin and bound-method twin: lines 203-221
# ---------------------------------------------------------------------

class _Accum:
    def __init__(self, v):
        acc = np.zeros(())
        for i in range(len(v)):
            acc = acc + v[i]
        self.total = acc


def test_class_reached_by_reference_twins():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        cls = _Accum  # class through a variable -> runtime twin (203-208)
        return cls(s).total

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum())


class _Model:
    def __init__(self, k):
        self.k = k

    def apply(self, v):
        acc = np.zeros(())
        for i in range(len(v)):
            acc = acc + v[i] * self.k
        return acc


_MODEL = _Model(2.0)


def test_bound_method_reached_by_reference_twins():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        s = np.zeros(x.shape)
        for i in range(len(x)):
            s[i] = x[i] * 1.0
        meth = _MODEL.apply  # bound method -> twin + rebind (211-220)
        return meth(s)

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum() * 2.0)


# ---------------------------------------------------------------------
# neutral asarray family reached by reference: lines 81-83
# ---------------------------------------------------------------------

def test_neutral_asarray_by_reference():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        aa = np.asarray  # neutral by resolved identity (line 81-83)
        return aa(A) * 2.0

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x * 2.0)


# ---------------------------------------------------------------------
# bound ndarray reduction on a decompressed object-Pair array: 161-177
# ---------------------------------------------------------------------

def test_bag_reduction_bound_method():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        o = _obj(x)
        m = o.mean  # bound reduction on an object-Pair array
        return m()

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.mean())


# ---------------------------------------------------------------------
# opaque wrapper on genuinely untraced data: line 445
# ---------------------------------------------------------------------

def test_opaque_wrapper_untraced_constant():
    x = np.array([1.0, 2.0, 3.0])

    def fn(x):
        A = np.zeros(x.shape)
        for i in range(len(x)):
            A[i] = x[i] * 1.0
        inv = np.linalg.inv
        C = np.array([[2.0, 0.0], [0.0, 4.0]])  # untraced constant
        return A.sum() + inv(C)[0, 0]

    out = to_sympy(fn, x.copy())
    assert np.allclose(_val(out), x.sum() + 0.5)


# ---------------------------------------------------------------------
# decorated-function peel path: lines 222-302 + _values_match 413-430
# ---------------------------------------------------------------------

def _wraps_deco(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@_wraps_deco
def _weighted(v, w):
    acc = np.zeros(())
    for i in range(len(v)):
        acc = acc + v[i] * w[i]
    return acc


def test_decorated_function_peels_via_wrapped():
    # functools.wraps sets __wrapped__: the peel unwraps and twins the
    # inner, then the simple bind succeeds (line 271).
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, 0.25, 0.25])

    def fn(v, w):
        s = np.zeros(v.shape)
        for i in range(len(v)):
            s[i] = v[i] * 1.0
        peel = _weighted
        return peel(s, w)

    out = to_sympy(fn, v.copy(), w.copy())
    assert np.allclose(_val(out), np.dot(v, w))


def _make_nonwraps():
    # a decorator that does NOT use functools.wraps: the inner function
    # hides in a closure cell under the wrapper's own name (lines
    # 235-242).
    def total(v):
        acc = np.zeros(())
        for i in range(len(v)):
            acc = acc + v[i]
        return acc

    def wrapper(*args, **kwargs):
        return total(*args, **kwargs)

    wrapper.__name__ = "total"
    return wrapper


_nonwraps_total = _make_nonwraps()


def test_decorated_nonwraps_closure_cell_peel():
    v = np.array([2.0, 3.0, 4.0])

    def fn(v):
        s = np.zeros(v.shape)
        for i in range(len(v)):
            s[i] = v[i] * 1.0
        peel = _nonwraps_total
        return peel(s)

    out = to_sympy(fn, v.copy())
    assert np.allclose(_val(out), v.sum())


def _inject_deco(f):
    @functools.wraps(f)
    def wrapper(*args, keepdims=False, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@_inject_deco
def _total_inject(v):
    acc = np.zeros(())
    for i in range(len(v)):
        acc = acc + v[i]
    return acc


def test_decorated_wrapper_injecting_kwargs_verifies():
    # the wrapper accepts keepdims but the inner does not: the peel
    # drops it, runs the twin, and VERIFIES against the wrapper on
    # concrete values (lines 281-296, driving _values_match 413-428).
    v = np.array([2.0, 3.0, 4.0])

    def fn(v):
        s = np.zeros(v.shape)
        for i in range(len(v)):
            s[i] = v[i] * 1.0
        peel = _total_inject
        return peel(s, keepdims=True)

    out = to_sympy(fn, v.copy())
    assert np.allclose(_val(out), v.sum())


# ---------------------------------------------------------------------
# plain twinned helper reached by reference: lines 303-314
# ---------------------------------------------------------------------

def test_plain_helper_function_twins_by_reference():
    def helper(v):
        acc = np.zeros(())
        for i in range(len(v)):
            acc = acc + v[i] ** 2
        return acc

    def fn(v):
        s = np.zeros(v.shape)
        for i in range(len(v)):
            s[i] = v[i] * 1.0
        h = helper
        return h(s)

    v = np.array([1.0, 2.0, 3.0])
    out = to_sympy(fn, v.copy())
    assert np.allclose(_val(out), np.sum(v**2))
