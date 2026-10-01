"""The method kwarg of percentile and quantile changes the mathematics.
Each supported method is refereed against numpy, including a fractional
position that lands exactly halfway, and unsupported methods refuse."""

import numpy as np
import pytest

from skverify import to_sympy

V = np.array([3.0, 1.0, 4.0, 1.5, 2.2])


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "nearest", "midpoint"])
@pytest.mark.parametrize("q", [0.0, 0.3, 0.375, 0.5, 0.9, 1.0])
def test_quantile_methods_match_numpy(method, q):
    def f(x):
        return np.quantile(x, q, method=method)

    ref = f(V.copy())
    out = to_sympy(f, V.copy())
    assert np.isclose(float(out.value), ref), (method, q)


@pytest.mark.parametrize("method", ["lower", "nearest"])
def test_percentile_methods_match_numpy(method):
    def f(x):
        return np.percentile(x, 30, method=method)

    ref = f(V.copy())
    out = to_sympy(f, V.copy())
    assert np.isclose(float(out.value), ref)


def test_vector_q_with_method():
    def f(x):
        return np.quantile(x, [0.25, 0.75], method="midpoint").sum()

    ref = f(V.copy())
    out = to_sympy(f, V.copy())
    assert np.isclose(float(out.value), ref)


def test_unsupported_method_refuses():
    def f(x):
        return np.quantile(x, 0.5, method="inverted_cdf")

    with pytest.raises(NotImplementedError):
        to_sympy(f, V.copy())


def test_stray_kwarg_refuses():
    def f(x):
        return np.quantile(x, 0.5, keepdims=True)

    with pytest.raises(NotImplementedError):
        to_sympy(f, V.copy())
