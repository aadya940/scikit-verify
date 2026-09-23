"""Exercise every registered ufunc and a broad set of registered array
functions through the tracer. Elementwise dispatch and the function
table are the bulk of maps/numpy.py; this drives them directly and
checks the value lane against numpy/scipy."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy
from skverify.registry import UFUNC_TABLE


def _unary_ufuncs():
    out = []
    for np_uf in UFUNC_TABLE:
        if getattr(np_uf, "nin", None) == 1 and getattr(np_uf, "nout", 1) == 1:
            out.append(np_uf)
    return out


# positive, modest domain keeps log/sqrt/gamma/arcsin all valid
SAFE = np.array([0.3, 0.5, 0.7])


@pytest.mark.parametrize("uf", _unary_ufuncs(),
                         ids=[u.__name__ for u in _unary_ufuncs()])
def test_every_unary_ufunc_traces_and_matches(uf):
    try:
        ref = uf(SAFE)
    except (TypeError, ValueError):
        pytest.skip("ufunc needs non-float or extra args")
    if not np.all(np.isfinite(ref)):
        pytest.skip("ufunc not finite on the safe domain")
    out = to_sympy(lambda v: uf(v), SAFE.copy())
    got = np.asarray(out.value, dtype=float)
    assert np.allclose(got, np.asarray(ref, dtype=float), equal_nan=True)


V = np.array([0.7, 1.2, 2.5, 0.3, 0.9])
W = np.array([2.0, 1.0, 5.0, 3.0, 4.0])


def _v(fn, *a):
    return np.asarray(to_sympy(fn, *[x.copy() if isinstance(x, np.ndarray)
                                     else x for x in a]).value, dtype=float)


@pytest.mark.parametrize("fn,args", [
    (lambda v: np.average(v), (V,)),
    (lambda v, w: np.average(v, weights=w), (V, W)),
    (lambda v: np.mean(v), (V,)), (lambda v: np.std(v), (V,)),
    (lambda v: np.var(v), (V,)), (lambda v: np.median(v), (V,)),
    (lambda v: np.nansum(v), (V,)), (lambda v: np.nanmean(v), (V,)),
    (lambda v: np.nanstd(v), (V,)), (lambda v: np.nanvar(v), (V,)),
    (lambda v: np.nanprod(v), (V,)), (lambda v: np.nanmedian(v), (V,)),
    (lambda v: np.percentile(v, 50), (V,)),
    (lambda v: np.quantile(v, 0.25), (V,)),
    (lambda v: np.prod(v), (V,)),
    (lambda v: np.around(v, 2), (V,)), (lambda v: np.round(v, 1), (V,)),
    (lambda v: np.nan_to_num(v), (V,)),
    (lambda v, w: np.dot(v, w), (V, W)),
    (lambda v: np.trace(np.outer(v, v)), (V,)),
    (lambda v: np.transpose(np.outer(v, v)).sum(), (V,)),
    (lambda v: np.clip(v, 0.5, 2.0), (V,)),
    (lambda v: np.where(v > 1.0, v, 0.0), (V,)),
    (lambda v: np.broadcast_to(v, (2, 5)).sum(), (V,)),
    (lambda v: np.diag(np.outer(v, v)), (V,)),
])
def test_registered_functions_match_numpy(fn, args):
    ref = np.asarray(fn(*[a.copy() for a in args]), dtype=float)
    assert np.allclose(_v(fn, *args), ref, equal_nan=True)


class TestTwoOutputUfuncs:
    def test_modf(self):
        # two-output ufuncs return a tuple of traced results
        frac, whole = to_sympy(lambda v: np.modf(v), V.copy())
        rfrac, rwhole = np.modf(V)
        assert np.allclose(np.asarray(frac.value, float), rfrac)
        assert np.allclose(np.asarray(whole.value, float), rwhole)

    def test_frexp(self):
        mant, expo = to_sympy(lambda v: np.frexp(v), V.copy())
        rmant, rexpo = np.frexp(V)
        assert np.allclose(np.asarray(mant.value, float), rmant)
