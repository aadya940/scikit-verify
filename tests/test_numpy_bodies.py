"""Exercise the happy-path and variant bodies of skverify.maps.numpy that
the refusal-heavy suites never reach: object-bag reductions, weighted and
per-axis average, percentile/quantile vector queries, searchsorted/interp
variants, place/putmask/copyto provenance, and the scalar-passthrough tails
of the space generators. Value lane is checked against numpy throughout."""

import numpy as np
import pytest

from skverify import to_sympy, Pair


def _val(out):
    return np.asarray(Pair._value_of(out), dtype=float)


def _bag_val(out):
    # object array of scalar Pairs -> float array
    arr = np.asarray(out, dtype=object)
    flat = np.array([float(Pair._value_of(e)) for e in arr.ravel()])
    return flat.reshape(arr.shape)


def _wall(a):
    # int() on a Pair is a hard wall: it forces to_sympy's instrumented
    # retry, which routes object arrays of Pairs through the map entries
    # (the bag branches). A no-op on the value.
    int(a.ravel()[0])


# ---- object-bag reductions (np.asarray on a >1-D Pair makes a bag) ----

class TestBagReductions:
    def test_prod_bag_per_axis(self):
        M = np.arange(1, 7.0).reshape(2, 3)

        def f(m):
            _wall(m)
            bag = np.array([[m[0, 0], m[0, 1], m[0, 2]],
                            [m[1, 0], m[1, 1], m[1, 2]]], dtype=object)
            return np.prod(bag, axis=0)

        out = to_sympy(f, M.copy())
        assert np.allclose(_bag_val(out), np.prod(M, axis=0))

    def test_prod_bag_flatten(self):
        V = np.array([2.0, 3.0, 4.0])

        def f(v):
            _wall(v)
            bag = np.array([v[0], v[1], v[2]], dtype=object)
            return np.prod(bag)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.prod(V))

    def test_sum_bag_per_axis(self):
        M = np.arange(1, 7.0).reshape(2, 3)

        def f(m):
            _wall(m)
            bag = np.array([[m[0, 0], m[0, 1], m[0, 2]],
                            [m[1, 0], m[1, 1], m[1, 2]]], dtype=object)
            return np.sum(bag, axis=1)

        out = to_sympy(f, M.copy())
        assert np.allclose(_bag_val(out), np.sum(M, axis=1))

    def test_sum_bag_flatten(self):
        V = np.array([1.0, 2.0, 5.0])

        def f(v):
            _wall(v)
            bag = np.array([v[0], v[1], v[2]], dtype=object)
            return np.sum(bag)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.sum(V))


# ---- prod/sum with out=, where=, keepdims (556-573 provenance) ----

class TestReduceKwargs:
    def test_sum_out_pair_scalar(self):
        V = np.array([1.0, 2.0, 3.0])

        def f(v):
            buf = np.sum(v)  # a scalar Pair
            np.sum(v * 2.0, out=buf)
            return buf

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.sum(V * 2.0))

    def test_sum_where_mask(self):
        V = np.array([1.0, -2.0, 3.0, -4.0])

        def f(v):
            return np.sum(v, where=v > 0)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.sum(V, where=V > 0))

    def test_prod_where_mask(self):
        V = np.array([1.5, 2.0, 0.5, 4.0])

        def f(v):
            return np.prod(v, where=v > 1.0)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.prod(V, where=V > 1.0))


# ---- ascontiguousarray on a Pair ----

def test_ascontiguousarray_pair():
    M = np.arange(6.0).reshape(2, 3)

    def f(m):
        return np.ascontiguousarray(m).sum()

    out = to_sympy(f, M.copy())
    assert np.allclose(_val(out), np.ascontiguousarray(M).sum())


# ---- diag: vector->matrix and matrix->diagonal ----

class TestDiag:
    def test_vector_to_diag_matrix(self):
        V = np.array([1.0, 2.0, 3.0])

        def f(v):
            return np.diag(v).sum()

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.diag(V).sum())

    def test_matrix_to_diagonal(self):
        M = np.arange(9.0).reshape(3, 3)

        def f(m):
            return np.diag(m).sum()

        out = to_sympy(f, M.copy())
        assert np.allclose(_val(out), np.diag(M).sum())


# ---- var per-axis and with ddof ----

class TestVar:
    def test_var_per_axis(self):
        M = np.arange(1, 7.0).reshape(2, 3)

        def f(m):
            return np.var(m, axis=0)

        out = to_sympy(f, M.copy())
        assert np.allclose(_val(out), np.var(M, axis=0))

    def test_var_per_axis_ddof(self):
        M = np.arange(1, 7.0).reshape(2, 3)

        def f(m):
            return np.var(m, axis=1, ddof=1)

        out = to_sympy(f, M.copy())
        assert np.allclose(_val(out), np.var(M, axis=1, ddof=1))

    def test_var_1d_pair(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            return np.var(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.var(V))

    def test_var_1d_pair_ddof(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            return np.var(v, ddof=1)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.var(V, ddof=1))

    def test_std_1d_pair(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            return np.std(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.std(V))


# ---- mean over a bag (element dunders keep the trace) ----

def test_mean_bag_flatten():
    V = np.array([2.0, 4.0, 6.0])

    def f(v):
        _wall(v)
        bag = np.array([v[0], v[1], v[2]], dtype=object)
        return np.mean(bag)

    out = to_sympy(f, V.copy())
    assert np.allclose(_val(out), np.mean(V))


# ---- median: bag path and 1-D Pair path ----

class TestMedian:
    def test_median_bag_odd(self):
        V = np.array([3.0, 1.0, 2.0])

        def f(v):
            _wall(v)
            bag = np.array([v[0], v[1], v[2]], dtype=object)
            return np.median(bag)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.median(V))

    def test_median_bag_even(self):
        V = np.array([4.0, 1.0, 3.0, 2.0])

        def f(v):
            _wall(v)
            bag = np.array([v[0], v[1], v[2], v[3]], dtype=object)
            return np.median(bag)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.median(V))

    def test_median_1d_pair_odd(self):
        V = np.array([3.0, 1.0, 2.0, 5.0, 4.0])

        def f(v):
            return np.median(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.median(V))

    def test_median_1d_pair_even(self):
        V = np.array([3.0, 1.0, 2.0, 5.0])

        def f(v):
            return np.median(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.median(V))

    def test_median_1d_pair_axis0(self):
        # axis=0 on a 1-D traced array normalizes to axis=None
        V = np.array([3.0, 1.0, 2.0, 5.0, 4.0])

        def f(v):
            return np.median(v, axis=0)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.median(V, axis=0))


# ---- percentile / quantile with vector q ----

class TestQuantile:
    def test_percentile_vector_q(self):
        V = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def f(v):
            return np.percentile(v, [25.0, 50.0, 75.0])

        out = to_sympy(f, V.copy())
        assert np.allclose(_bag_val(out), np.percentile(V, [25.0, 50.0, 75.0]))

    def test_quantile_scalar_q(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            return np.quantile(v, 0.5)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.quantile(V, 0.5))


# ---- average: returned, weights, per-axis ----

class TestAverage:
    def test_average_returned(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            avg, wsum = np.average(v, returned=True)
            return avg

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.average(V))

    def test_average_returned_weights(self):
        V = np.array([1.0, 2.0, 3.0])
        W = np.array([0.5, 0.25, 0.25])

        def f(v, w):
            avg, wsum = np.average(v, weights=w, returned=True)
            return avg

        out = to_sympy(f, V.copy(), W.copy())
        assert np.allclose(_val(out), np.average(V, weights=W))

    def test_average_1d_weights(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])
        W = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v, w):
            return np.average(v, weights=w)

        out = to_sympy(f, V.copy(), W.copy())
        assert np.allclose(_val(out), np.average(V, weights=W))

    def test_average_per_axis_1d_weights(self):
        M = np.arange(1, 7.0).reshape(2, 3)
        W = np.array([1.0, 2.0])

        def f(m, w):
            return np.average(m, axis=0, weights=w)

        out = to_sympy(f, M.copy(), W.copy())
        assert np.allclose(_val(out), np.average(M, axis=0, weights=W))

    def test_average_bag_no_weights(self):
        V = np.array([1.0, 2.0, 3.0])

        def f(v):
            _wall(v)
            bag = np.array([v[0], v[1], v[2]], dtype=object)
            return np.average(bag)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.average(V))

    def test_average_bag_weights(self):
        V = np.array([1.0, 2.0, 3.0])
        W = np.array([0.2, 0.3, 0.5])

        def f(v, w):
            _wall(v)
            bag = np.array([v[0], v[1], v[2]], dtype=object)
            return np.average(bag, weights=w)

        out = to_sympy(f, V.copy(), W.copy())
        assert np.allclose(_val(out), np.average(V, weights=W))

    def test_average_bag_per_axis(self):
        M = np.arange(1, 7.0).reshape(2, 3)

        def f(m):
            _wall(m)
            bag = np.array([[m[0, 0], m[0, 1], m[0, 2]],
                            [m[1, 0], m[1, 1], m[1, 2]]], dtype=object)
            return np.average(bag, axis=1)

        out = to_sympy(f, M.copy())
        assert np.allclose(_bag_val(out), np.average(M, axis=1))

    def test_average_numeric_data_traced_weights(self):
        # numeric data (built inside, untraced), TRACED weights: routes
        # through the wrap-data-as-disclosed-constant path
        W = np.array([0.2, 0.3, 0.5])

        def f(w):
            data = np.array([10.0, 20.0, 30.0])
            return np.average(data, weights=w)

        out = to_sympy(f, W.copy())
        exp = np.average([10.0, 20.0, 30.0], weights=W)
        assert np.allclose(_val(out), exp)

    def test_average_pair_untraced_weights(self):
        # traced data, weights given as a plain list (untraced): wraps w
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            return np.average(v, weights=[1.0, 1.0, 2.0, 2.0])

        out = to_sympy(f, V.copy())
        assert np.allclose(
            _val(out), np.average(V, weights=[1.0, 1.0, 2.0, 2.0]))

    def test_average_bag_per_axis_weights(self):
        M = np.arange(1, 7.0).reshape(2, 3)
        W = np.array([1.0, 2.0, 3.0])

        def f(m, w):
            _wall(m)
            bag = np.array([[m[0, 0], m[0, 1], m[0, 2]],
                            [m[1, 0], m[1, 1], m[1, 2]]], dtype=object)
            return np.average(bag, axis=1, weights=w)

        out = to_sympy(f, M.copy(), W.copy())
        assert np.allclose(_bag_val(out), np.average(M, axis=1, weights=W))


# ---- searchsorted variants ----

class TestSearchsorted:
    def test_searchsorted_traced_bins_scalar_v(self):
        BINS = np.array([1.0, 3.0, 5.0, 7.0])

        def f(b):
            return np.searchsorted(b, 4.0)

        out = to_sympy(f, BINS.copy())
        assert int(Pair._value_of(out)) == int(np.searchsorted(BINS, 4.0))

    def test_searchsorted_traced_bins_vector_v(self):
        BINS = np.array([1.0, 3.0, 5.0, 7.0])

        out = to_sympy(lambda b: np.searchsorted(b, np.array([2.0, 6.0])),
                       BINS.copy())
        got = np.array([int(Pair._value_of(e)) for e in out])
        assert np.array_equal(got, np.searchsorted(BINS, np.array([2.0, 6.0])))

    def test_searchsorted_concrete_bins_traced_v_scalar(self):
        def f(v):
            return np.searchsorted(np.array([0.0, 1.0, 2.0, 3.0]), v)

        out = to_sympy(f, np.array(1.5))
        assert int(Pair._value_of(out)) == int(
            np.searchsorted([0.0, 1.0, 2.0, 3.0], 1.5))

    def test_searchsorted_concrete_bins_traced_v_vector_right(self):
        V = np.array([0.5, 2.0, 3.5])

        def f(v):
            return np.searchsorted(np.array([0.0, 1.0, 2.0, 3.0]), v,
                                   side="right")

        out = to_sympy(f, V.copy())
        got = np.array([int(Pair._value_of(e)) for e in out])
        assert np.array_equal(
            got, np.searchsorted([0.0, 1.0, 2.0, 3.0], V, side="right"))


# ---- interp variants ----

class TestInterp:
    def test_interp_scalar_query(self):
        def f(x):
            return np.interp(x, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0])

        out = to_sympy(f, np.array(0.5))
        assert np.allclose(
            _val(out), np.interp(0.5, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]))

    def test_interp_vector_query(self):
        X = np.array([-1.0, 0.5, 1.5, 3.0])

        def f(x):
            return np.interp(x, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0])

        out = to_sympy(f, X.copy())
        got = _bag_val(out)
        assert np.allclose(
            got, np.interp(X, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0]))

    def test_interp_left_right(self):
        X = np.array([-5.0, 5.0])

        def f(x):
            return np.interp(x, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0],
                             left=-1.0, right=99.0)

        out = to_sympy(f, X.copy())
        got = _bag_val(out)
        assert np.allclose(got, np.interp(
            X, [0.0, 1.0, 2.0], [0.0, 10.0, 20.0], left=-1.0, right=99.0))


# ---- nan_to_num: all-finite and with non-finite entries ----

class TestNanToNum:
    def test_nan_to_num_all_finite(self):
        V = np.array([1.0, 2.0, 3.0])

        def f(v):
            return np.nan_to_num(v).sum()

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nan_to_num(V).sum())

    def test_nan_to_num_vector_nonfinite(self):
        V = np.array([1.0, np.inf, 3.0, np.nan])

        def f(v):
            return np.nan_to_num(v)

        out = to_sympy(f, V.copy())
        got = _bag_val(out)
        assert np.allclose(got, np.nan_to_num(V))

    def test_nan_to_num_scalar_nonfinite(self):
        def f(x):
            return np.nan_to_num(x)

        out = to_sympy(f, np.array(np.inf))
        assert np.allclose(_val(out), np.nan_to_num(np.inf))


# ---- np.select (chained where) ----

def test_select():
    V = np.array([-1.0, 0.5, 2.0])

    def f(v):
        return np.select([v < 0, v > 1], [-v, v * 2], default=0.0)

    out = to_sympy(f, V.copy())
    got = _bag_val(out) if isinstance(np.asarray(out, dtype=object).ravel()[0],
                                      Pair) else _val(out)
    exp = np.select([V < 0, V > 1], [-V, V * 2], default=0.0)
    assert np.allclose(got, exp)


# ---- nan-aware reductions (WHICH entries are nan is a trace fact) ----

class TestNanReduce:
    def test_nansum(self):
        V = np.array([1.0, np.nan, 3.0, 4.0])

        def f(v):
            return np.nansum(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nansum(V))

    def test_nanmean(self):
        V = np.array([1.0, np.nan, 3.0, 4.0])

        def f(v):
            return np.nanmean(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nanmean(V))

    def test_nanprod(self):
        V = np.array([2.0, np.nan, 3.0])

        def f(v):
            return np.nanprod(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nanprod(V))

    def test_nanvar(self):
        V = np.array([1.0, np.nan, 3.0, 5.0])

        def f(v):
            return np.nanvar(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nanvar(V))

    def test_nanstd(self):
        V = np.array([1.0, np.nan, 3.0, 5.0])

        def f(v):
            return np.nanstd(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nanstd(V))

    def test_nanmedian_odd(self):
        V = np.array([3.0, np.nan, 1.0, 2.0, 5.0])

        def f(v):
            return np.nanmedian(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nanmedian(V))

    def test_nanmedian_even(self):
        V = np.array([3.0, np.nan, 1.0, 2.0])

        def f(v):
            return np.nanmedian(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nanmedian(V))

    def test_nanmedian_per_axis(self):
        M = np.array([[3.0, np.nan, 1.0], [2.0, 5.0, 4.0]])

        def f(m):
            return np.nanmedian(m, axis=0)

        out = to_sympy(f, M.copy())
        got = _bag_val(out)
        assert np.allclose(got, np.nanmedian(M, axis=0))

    def test_nansum_all_nan(self):
        V = np.array([np.nan, np.nan])

        def f(v):
            return np.nansum(v)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nansum(V))

    def test_nanmedian_bag(self):
        V = np.array([3.0, np.nan, 1.0, 2.0, 5.0])

        def f(v):
            _wall(v)
            bag = np.array([v[0], v[1], v[2], v[3], v[4]], dtype=object)
            return np.nanmedian(bag)

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.nanmedian(V))


# ---- place / putmask / copyto provenance ----

class TestScatter:
    def test_copyto_masked(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            np.copyto(v, 0.0, where=v > 2.0)
            return v

        out = to_sympy(f, V.copy())
        exp = V.copy()
        np.copyto(exp, 0.0, where=exp > 2.0)
        assert np.allclose(_val(out), exp)

    def test_copyto_full(self):
        V = np.array([1.0, 2.0, 3.0])

        def f(v):
            np.copyto(v, 5.0)
            return v

        out = to_sympy(f, V.copy())
        assert np.allclose(_val(out), np.full(3, 5.0))

    def test_putmask(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            np.putmask(v, v > 2.0, 0.0)
            return v

        out = to_sympy(f, V.copy())
        exp = V.copy()
        np.putmask(exp, exp > 2.0, 0.0)
        assert np.allclose(_val(out), exp)

    def test_putmask_integer_mask(self):
        # a non-Pair, non-bool (integer) mask: coerced via != 0
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            int(v[0])  # wall -> instrumented retry keeps the mask untraced
            np.putmask(v, np.array([1, 0, 1, 0]), 0.0)
            return v

        out = to_sympy(f, V.copy())
        exp = V.copy()
        np.putmask(exp, np.array([1, 0, 1, 0]), 0.0)
        assert np.allclose(_val(out), exp)

    def test_place(self):
        V = np.array([1.0, 2.0, 3.0, 4.0])

        def f(v):
            np.place(v, v > 2.0, 0.0)
            return v

        out = to_sympy(f, V.copy())
        exp = V.copy()
        np.place(exp, exp > 2.0, 0.0)
        assert np.allclose(_val(out), exp)


# ---- trace with offset ----

def test_trace_offset():
    M = np.arange(16.0).reshape(4, 4)

    def f(m):
        return np.trace(m, offset=1)

    out = to_sympy(f, M.copy())
    assert np.allclose(_val(out), np.trace(M, offset=1))


# ---- fill_diagonal ----

class TestFillDiagonal:
    def test_fill_diagonal_scalar(self):
        M = np.zeros((3, 3))

        def f(m):
            np.fill_diagonal(m, 7.0)
            return m

        out = to_sympy(f, M.copy())
        exp = M.copy()
        np.fill_diagonal(exp, 7.0)
        assert np.allclose(_val(out), exp)

    def test_fill_diagonal_vector(self):
        M = np.zeros((3, 3))
        D = np.array([1.0, 2.0, 3.0])

        def f(m, d):
            np.fill_diagonal(m, d)
            return m

        out = to_sympy(f, M.copy(), D.copy())
        exp = M.copy()
        np.fill_diagonal(exp, D)
        assert np.allclose(_val(out), exp)


# ---- space generators, scalar (untraced) passthrough tails ----

class TestSpaceScalarPassthrough:
    def test_linspace_untraced(self):
        # start/stop are ints -> pass through, parts is None
        def f(v):
            return v.sum() + np.linspace(0, 1, 5).sum()

        out = to_sympy(f, np.array([1.0, 2.0]))
        assert np.allclose(
            _val(out), np.array([1.0, 2.0]).sum() + np.linspace(0, 1, 5).sum())

    def test_logspace_untraced(self):
        def f(v):
            return v.sum() + np.logspace(0, 2, 3).sum()

        out = to_sympy(f, np.array([1.0]))
        assert np.allclose(_val(out), 1.0 + np.logspace(0, 2, 3).sum())

    def test_geomspace_untraced(self):
        def f(v):
            return v.sum() + np.geomspace(1, 8, 4).sum()

        out = to_sympy(f, np.array([1.0]))
        assert np.allclose(_val(out), 1.0 + np.geomspace(1, 8, 4).sum())

    def test_logspace_traced(self):
        def f(a, b):
            return np.logspace(a, b, 4).sum()

        out = to_sympy(f, np.array(0.0), np.array(2.0))
        assert np.allclose(_val(out), np.logspace(0.0, 2.0, 4).sum())

    def test_geomspace_traced(self):
        def f(a, b):
            return np.geomspace(a, b, 4).sum()

        out = to_sympy(f, np.array(1.0), np.array(8.0))
        assert np.allclose(_val(out), np.geomspace(1.0, 8.0, 4).sum())


# ---- concrete checks (isclose/isfinite family) ----

class TestConcreteChecks:
    def test_isclose_on_pair(self):
        V = np.array([1.0, 2.0, 3.0])

        def f(v):
            return np.isclose(v, v)

        out = to_sympy(f, V.copy())
        assert np.all(np.asarray(out))

    def test_allclose_on_pair(self):
        V = np.array([1.0, 2.0, 3.0])

        def f(v):
            return np.allclose(v, v)

        out = to_sympy(f, V.copy())
        assert bool(out)


# ---- bincount elem_formulas (1447-1449) ----

def test_bincount_traced_weights():
    # concrete integer positions, TRACED weights -> weights go through
    # elem_formulas' Pair branch (xreplace over the index letter)
    W = np.array([0.5, 1.5, 2.0, 3.0])

    def f(w):
        return np.bincount(np.array([0, 1, 1, 2]), weights=w)

    out = to_sympy(f, W.copy())
    got = np.array([float(Pair._value_of(e)) for e in out])
    assert np.allclose(got, np.bincount([0, 1, 1, 2], weights=W))


def test_bincount_traced_x():
    # traced integer-valued positions x -> x_fs via elem_formulas Pair
    # branch; counts are Piecewise-Eq sums
    X = np.array([0.0, 1.0, 1.0, 2.0])

    def f(x):
        return np.bincount(x)

    out = to_sympy(f, X.copy())
    got = np.array([float(Pair._value_of(e)) for e in out])
    assert np.allclose(got, np.bincount([0, 1, 1, 2]))


def test_bincount_bag_x():
    # object-bag positions -> elem_formulas bag branch (1446)
    X = np.array([0.0, 1.0, 2.0, 2.0])

    def f(x):
        _wall(x)
        bag = np.array([x[0], x[1], x[2], x[3]], dtype=object)
        return np.bincount(bag)

    out = to_sympy(f, X.copy())
    got = np.array([float(Pair._value_of(e)) for e in out])
    assert np.allclose(got, np.bincount([0, 1, 2, 2]))
