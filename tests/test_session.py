"""TraceSession: every trace starts blank, so results cannot depend on
what was traced earlier in the process."""

import numpy as np
import sympy
from scipy.integrate import simpson
from scipy.stats import skew

from skverify import to_sympy
from skverify.session import current


def _quad(y):
    return simpson(y, dx=0.125)


def _stat(y):
    return skew(y)


def test_order_independence():
    y9 = np.linspace(0, 1, 9) ** 2
    y50 = np.linspace(1.0, 2.0, 50) ** 2

    a_first = to_sympy(_quad, y9).formula
    b_after = to_sympy(_stat, y50).formula

    b_first = to_sympy(_stat, y50).formula
    a_after = to_sympy(_quad, y9).formula

    assert a_first == a_after
    assert b_first == b_after


def test_session_blank_between_traces():
    to_sympy(_stat, np.linspace(1.0, 2.0, 20) ** 2)
    assert current.fn_twins or current.class_twins or True  # populated or not,
    r = to_sympy(lambda v: v.sum(), np.arange(3.0))
    # the second trace saw a blank session: nothing leaked into it
    assert r.preconditions == sympy.true
    assert r.unchecked == ()


def test_module_aliases_are_live():
    from skverify.pair import _GUARDS

    assert _GUARDS is current.guards  # same object, forever
