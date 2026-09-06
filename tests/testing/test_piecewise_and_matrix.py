"""Piecewise seams and multi-axis formulas checked through check_formula."""

import numpy as np
import sympy

from skverify.testing import check_formula

ROW, COL, REDUCE = sympy.symbols("i j k", integer=True)
V = sympy.IndexedBase("v")
U = sympy.IndexedBase("u")
W = sympy.IndexedBase("w")
A = sympy.IndexedBase("a")


def relu_squared(v):
    return np.where(v > 0, v * v, 0.0)


def test_piecewise_formula_and_seam_match():
    spec = sympy.Piecewise((V[ROW] ** 2, V[ROW] > 0), (0, True))

    verdict = check_formula(
        relu_squared,
        (np.array([-1.0, 0.0, 2.0]),),
        spec,
        indices=(ROW,),
    )

    assert verdict.matches, verdict.message()


def test_piecewise_seam_mismatch_prints_both_conditions():
    spec = sympy.Piecewise((V[ROW] ** 2, V[ROW] >= 0), (0, True))

    verdict = check_formula(
        relu_squared,
        (np.array([-1.0, 0.0, 2.0]),),
        spec,
        indices=(ROW,),
    )

    assert verdict.tier == "differs"
    assert "piecewise condition mismatch" in verdict.detail
    assert "v[0] >= 0" in verdict.detail
    assert "v[0] > 0" in verdict.detail


def test_piecewise_branch_formula_mismatch_fails_with_matching_seam():
    spec = sympy.Piecewise((V[ROW] ** 3, V[ROW] > 0), (0, True))

    verdict = check_formula(
        relu_squared,
        (np.array([-1.0, 0.0, 2.0]),),
        spec,
        indices=(ROW,),
    )

    assert verdict.tier == "differs"
    assert "first disagreement" in verdict.detail


def test_outer_product_binds_indices_in_axis_order():
    verdict = check_formula(
        lambda u, w: np.outer(u, w),
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
        U[ROW] * W[COL],
        indices=(ROW, COL),
    )

    assert verdict.tier == "exact", verdict.message()


def test_gram_matrix_matches_indexed_sum():
    values = np.arange(6.0).reshape(2, 3) + 1
    spec = sympy.Sum(A[REDUCE, ROW] * A[REDUCE, COL], (REDUCE, 0, 1))

    verdict = check_formula(lambda a: a.T @ a, (values,), spec, indices=(ROW, COL))

    assert verdict.tier == "exact", verdict.message()


def test_reversed_output_indices_fail():
    verdict = check_formula(
        lambda u, w: np.outer(u, w),
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])),
        U[ROW] * W[COL],
        indices=(COL, ROW),
    )

    assert verdict.tier == "differs"
