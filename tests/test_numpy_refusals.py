"""Trip every refusal branch in skverify/maps/numpy.py.

Each test drives a numpy op through ``to_sympy`` with an input crafted to
hit one specific ``raise`` site, asserting the exception fires.

Unreachable from the public API (documented, not tested):
  * 263, 322 -- _held_sum/_held_prod binder-escape guards; upstream-bug
    detectors that the surrounding selector-resolution logic defuses first.
  * 513 -- _clip_entry kwargs refusal; np.clip is re-registered later
    (numpy.py:1944) to a lambda that drops kwargs, so _clip_entry is dead.
  * 1693, 1860 -- mutating-write / fill_diagonal into a NON-traced
    destination; the instrumented retry wraps local array constructors
    (np.zeros/np.eye) into Pairs, so a plain ndarray can never hold traced
    values through to_sympy.
  * 1700 -- mutating-write "with options"; np.copyto rejects unknown kwargs
    in its C dispatcher and np.place/np.putmask take no kwargs, so no
    leftover option can reach the check.
  * 1965, 1984, 1990, 1996 -- FFT-family refusals; np.fft.* is a curated
    OPAQUE boundary (instrument/registries.py) and has no
    __array_function__, so the _fft/_ifft/_rfft entries never run.
"""

import numpy as np
import pytest

from skverify import to_sympy

ANY = (NotImplementedError, ValueError, TypeError)


# line 150: np.where over a Sum-of-Piecewise condition
def test_where_sum_of_piecewise():
    def f(u):
        cond = np.sum(np.where(u > 0, 1.0, 0.0)) > 1  # Sum of Piecewise
        return np.where(cond, u, -u)

    with pytest.raises(NotImplementedError):
        to_sympy(f, np.array([1.0, -2.0, 3.0]))


# line 469: np.prod with axis tuple
def test_prod_axis_tuple():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.prod(v, axis=(0, 1)), np.ones((2, 3)))


# line 417/418: np.prod with a non-float dtype
def test_prod_int_dtype():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.prod(v, dtype=int), np.array([1.0, 2.0, 3.0]))


# line 420: np.prod with an unsupported kwarg (initial=)
def test_prod_unsupported_kwarg():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.prod(v, initial=2.0), np.array([1.0, 2.0, 3.0]))


# line 424: np.prod keepdims with axis tuple
def test_prod_keepdims_axis_tuple():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda v: np.prod(v, axis=(0, 1), keepdims=True), np.ones((2, 3))
        )


# line 402: reduction with out= into an untraced buffer (prod)
def test_prod_out_untraced_buffer():
    buf = np.empty(())

    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.prod(v, out=buf), np.array([1.0, 2.0, 3.0]))


# line 564: reduction with out= into an untraced buffer (sum)
def test_sum_out_untraced_buffer():
    buf = np.empty(())

    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.sum(v, out=buf), np.array([1.0, 2.0, 3.0]))


# line 578/579: np.sum with a non-float dtype
def test_sum_int_dtype():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.sum(v, dtype=int), np.array([1.0, 2.0, 3.0]))


# line 581: np.sum with an unsupported kwarg (initial=)
def test_sum_unsupported_kwarg():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.sum(v, initial=2.0), np.array([1.0, 2.0, 3.0]))


# line 585: np.sum keepdims with axis tuple
def test_sum_keepdims_axis_tuple():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda v: np.sum(v, axis=(0, 1), keepdims=True), np.ones((2, 3))
        )


# line 624: np.sum with axis tuple
def test_sum_axis_tuple():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.sum(v, axis=(0, 1)), np.ones((2, 3)))


# line 713: matmul on scalars (both operands must be at least 1-D)
def test_matmul_scalar_operands():
    with pytest.raises(ValueError):
        to_sympy(lambda a, b: np.matmul(a, b), 2.0, 3.0)


# line 768: np.dot with out=
def test_dot_out():
    buf = np.empty((2,))

    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda a, b: np.dot(a, b, out=buf),
            np.ones((2, 2)),
            np.ones((2,)),
        )


# line 772: np.dot N-D
def test_dot_nd():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda a, b: np.dot(a, b),
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)),
        )


# line 784: zeros_like with unsupported kwarg
def test_zeros_like_kwarg():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.zeros_like(v, subok=False), np.array([1.0, 2.0]))


# line 790: ones_like with unsupported kwarg
def test_ones_like_kwarg():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.ones_like(v, subok=False), np.array([1.0, 2.0]))


# line 796: full_like with unsupported kwarg
def test_full_like_kwarg():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda v: np.full_like(v, 3.0, subok=False), np.array([1.0, 2.0])
        )


# line 816: gradient on non-traced input
def test_gradient_non_traced():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda n: np.gradient(np.array([1.0, 2.0, 3.0]), n), 2.0)


# line 818: gradient edge_order must be 1 or 2
def test_gradient_bad_edge_order():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.gradient(v, edge_order=3), np.array([1.0, 2.0, 3.0]))


# line 823: gradient with non-scalar spacing
def test_gradient_nonscalar_spacing():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda v: np.gradient(v, np.array([0.0, 1.0, 2.0])),
            np.array([1.0, 2.0, 3.0]),
        )


# line 831: gradient spacing arity mismatch
def test_gradient_spacing_arity():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda v: np.gradient(v, 1.0, 2.0),
            np.array([1.0, 2.0, 3.0]),
        )


# line 840: gradient repeated axis
def test_gradient_repeated_axis():
    with pytest.raises(ValueError):
        to_sympy(
            lambda v: np.gradient(v, axis=(0, 0)),
            np.ones((3, 3)),
        )


# line 845: gradient edge_order 2 on too-small axis
def test_gradient_edge_order2_small():
    with pytest.raises(ValueError):
        to_sympy(
            lambda v: np.gradient(v, edge_order=2), np.array([1.0, 2.0])
        )


# line 913: logspace/linspace retstep not supported
def test_linspace_retstep():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda a, b: np.linspace(a, b, 5, retstep=True), 0.0, 1.0)


# line 923: space generator beyond 5D
def test_linspace_beyond_5d():
    start = np.ones((2, 2, 2, 2, 2))

    with pytest.raises(NotImplementedError):
        to_sympy(lambda a: np.linspace(a, a + 1.0, 4, axis=-1), start)


# line 962: logspace non-scalar/complex base
def test_logspace_bad_base():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda a, b: np.logspace(a, b, 5, base=np.array([2.0, 3.0])),
            0.0,
            1.0,
        )


# line 965: logspace non-positive base
def test_logspace_nonpositive_base():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda a, b: np.logspace(a, b, 5, base=-2.0), 0.0, 1.0)


# line 999: geomspace complex endpoints
def test_geomspace_complex():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda a, b: np.geomspace(a, b, 5),
            np.array(1.0 + 1.0j),
            np.array(2.0),
        )


# line 1002: geomspace cannot include zero
def test_geomspace_zero():
    with pytest.raises(ValueError):
        to_sympy(lambda a, b: np.geomspace(a, b, 5), 0.0, 4.0)


# line 1004: geomspace endpoints must share sign
def test_geomspace_sign_mismatch():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda a, b: np.geomspace(a, b, 5), -1.0, 4.0)


# line 1052: all() with axis
def test_all_axis():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.all(v > 0, axis=0), np.ones((2, 2)))


# line 1060: any() with axis
def test_any_axis():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.any(v > 0, axis=0), np.ones((2, 2)))


# line 1268: np.round at an exact half-way tie
def test_round_halfway_tie():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.round(v, 0), np.array([0.5, 1.5, 2.5]))


# line 1287: np.round on >1-D traced arrays
def test_round_multidim():
    with pytest.raises(NotImplementedError):
        to_sympy(lambda v: np.round(v, 1), np.array([[0.11, 0.22], [0.33, 0.44]]))


# line 1717: np.place with a cycled values list
def test_place_cycled_values():
    def f(v):
        np.place(v, v > 0, [7.0, 8.0])
        return v

    with pytest.raises(NotImplementedError):
        to_sympy(f, np.array([1.0, 2.0, 3.0]))


# line 1789: np.interp with period
def test_interp_period():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda x: np.interp(x, [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], period=2.0),
            np.array([0.5, 1.5]),
        )


# line 1798: np.interp with a traced table
def test_interp_traced_table():
    with pytest.raises(NotImplementedError):
        to_sympy(
            lambda xp: np.interp(np.array([0.5]), xp, np.array([0.0, 1.0, 4.0])),
            np.array([0.0, 1.0, 2.0]),
        )


# line 1865: fill_diagonal 2-D unwrapped only (wrap=True)
def test_fill_diagonal_wrap():
    def f(m):
        np.fill_diagonal(m, 9.0, wrap=True)
        return m

    with pytest.raises(NotImplementedError):
        to_sympy(f, np.ones((4, 2)))


# NOTE: the FFT-family refusals (numpy.py lines 1965/1984/1990/1996) and the
# _held_sum/_held_prod binder-escape refusals (263/322) are not exercised
# here -- see the module docstring / final report for why they are
# unreachable from the public to_sympy API.
