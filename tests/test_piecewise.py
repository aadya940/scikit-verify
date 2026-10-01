"""np.piecewise as a dialect entry. The referee throughout is numpy
itself: the traced value lane must equal what the untraced call
returns, including where conditions overlap and numpy's
last-condition-wins precedence applies."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy

V = np.array([-2.0, -0.5, 0.0, 0.7, 1.8])


def _check(fn, *args):
    ref = fn(*[np.copy(a) for a in args])
    out = to_sympy(fn, *[np.copy(a) for a in args])
    got = np.asarray(out.value, dtype=float)
    assert np.allclose(got, ref), f"value lane {got} != numpy {ref}"
    return out


def test_two_pieces_with_default():
    def f(x):
        return np.piecewise(
            x, [x < 0, x > 1], [lambda t: -t, lambda t: t**2]
        ).sum()

    out = _check(f, V)
    assert out.formula.has(sympy.Piecewise)


def test_scalar_pieces():
    def f(x):
        return np.piecewise(x, [x < 0, x >= 0], [-1.0, 1.0]).sum()

    _check(f, V)


def test_explicit_otherwise_clause():
    def f(x):
        return np.piecewise(
            x, [x < -1, x > 1], [lambda t: t + 10, lambda t: t - 10, 99.0]
        ).sum()

    _check(f, V)


def test_overlap_last_condition_wins():
    # both conditions true on x > 1: numpy applies pieces in order,
    # so the second piece must win in the formula too
    def f(x):
        return np.piecewise(
            x, [x > 0, x > 1], [lambda t: t * 2, lambda t: t * 100]
        ).sum()

    _check(f, V)


def test_callable_default():
    def f(x):
        return np.piecewise(x, [x < 0], [lambda t: 0.0, lambda t: t**3]).sum()

    _check(f, V)


def test_extra_args_passed_to_pieces():
    def f(x):
        return np.piecewise(
            x, [x < 0, x >= 0], [lambda t, k: -t * k, lambda t, k: t * k], 3.0
        ).sum()

    _check(f, V)


def test_single_condition_not_in_a_list():
    def f(x):
        return np.piecewise(x < 0, [x < 0], [1.0, 0.0]).sum() if False else \
            np.piecewise(x, x < 0, [lambda t: -t, lambda t: t]).sum()

    _check(f, V)


def test_formula_evaluates_at_other_inputs():
    # the certificate must hold beyond the traced sample
    def f(x):
        return np.piecewise(
            x, [x < 0, x > 1], [lambda t: -t, lambda t: t**2]
        ).sum()

    out = to_sympy(f, np.copy(V))
    other = np.array([2.5, -3.0, 0.5, 1.5, -0.1])
    subs = {sympy.Symbol("x"): None}
    x = sympy.IndexedBase("x")
    expr = out.formula.doit()
    got = float(expr.subs({x[i]: other[i] for i in range(5)}))
    assert np.isclose(got, f(other))


def test_concrete_mask_refused():
    def f(x):
        mask = np.array([True, False, True, False, True])
        return np.piecewise(x, [mask], [lambda t: t * 2]).sum()

    with pytest.raises(NotImplementedError):
        to_sympy(f, np.copy(V))


def test_wrong_funclist_length_raises():
    def f(x):
        return np.piecewise(x, [x < 0], [1.0, 2.0, 3.0]).sum()

    with pytest.raises(ValueError):
        to_sympy(f, np.copy(V))
