"""Published spline penalty formulas checked against the NumPy construction."""

import numpy as np
import pytest
import sympy

from skverify.testing import specifies

T = sympy.IndexedBase("t")
KNOTS = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 3.0, 3.0, 3.0])
KNOT_ASSUMPTIONS = [
    sympy.Eq(T[0], 0),
    sympy.Eq(T[1], 0),
    sympy.Eq(T[2], 0),
    sympy.Eq(T[3], 0),
    sympy.Eq(T[6], T[5]),
    sympy.Eq(T[7], T[5]),
    sympy.Eq(T[8], T[5]),
    T[4] > 0,
    T[5] > T[4],
]


def penalty(t):
    """Construct the cubic B-spline penalty matrix C.T @ R @ C."""
    m = len(t) - 4
    d1 = np.zeros((m + 1, m))
    for j in range(m + 1):
        run = t[j + 3] - t[j]
        if run != 0:
            if j - 1 >= 0:
                d1[j, j - 1] = -3 / run
            if j < m:
                d1[j, j] = 3 / run

    d2 = np.zeros((m + 2, m + 1))
    for j in range(m + 2):
        run = t[j + 2] - t[j]
        if run != 0:
            if j - 1 >= 0:
                d2[j, j - 1] = -2 / run
            if j < m + 1:
                d2[j, j] = 2 / run

    c = d2 @ d1
    r = np.zeros((m + 2, m + 2))
    for p in range(m + 2):
        r[p, p] = (t[p + 2] - t[p]) / 3
        if p + 1 < m + 2:
            r[p, p + 1] = r[p + 1, p] = (t[p + 2] - t[p + 1]) / 6
    return c.T @ r @ c


def penalty_entry_00(t):
    return penalty(t)[0, 0]


def penalty_entry_12(t):
    return penalty(t)[1, 2]


@specifies(12 / T[4] ** 3, assume=KNOT_ASSUMPTIONS)
def test_penalty_entry_00_matches_the_paper():
    return penalty_entry_00, (KNOTS.copy(),)


@specifies(-12 / (T[4] ** 2 * T[5]), assume=KNOT_ASSUMPTIONS)
def test_penalty_entry_12_matches_the_paper():
    return penalty_entry_12, (KNOTS.copy(),)


def test_penalty_entry_wrong_sign_reports_both_formulas():
    @specifies(12 / (T[4] ** 2 * T[5]), assume=KNOT_ASSUMPTIONS)
    def check_wrong_sign():
        return penalty_entry_12, (KNOTS.copy(),)

    with pytest.raises(AssertionError) as exc_info:
        check_wrong_sign()

    message = str(exc_info.value)
    assert "your spec" in message
    assert "the code" in message
