"""Targeted coverage for coercion.py, checks.py, contracts.py.

Drives the type/value coercion helpers, the three-valued check
validators, and the sealed-atom defining-equation contracts through
real inputs, asserting on returned fields and value lanes.
"""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy
from skverify import checks
from skverify.coercion import value_of, formula_of, repack, numeric
from skverify.contracts import check_call, OK, FAILED, UNKNOWN
from skverify.helpers import axis_idx


# ---------------------------------------------------------------------------
# coercion.py
# ---------------------------------------------------------------------------
class TestValueOf:
    def test_scalar_passthrough(self):
        assert value_of(3.5) == 3.5

    def test_pair_scalar_value(self):
        p = Pair(2.0, sympy.Symbol("x"))
        assert value_of(p) == 2.0

    def test_pair_array_records_origin(self):
        # array-valued Pair with a real formula: id(value) -> formula is
        # recorded in the session (lines 60-67, the try/except body).
        from skverify.session import current as session

        u = Pair.array("u", np.array([1.0, 2.0, 3.0]))
        raw = value_of(u)
        assert isinstance(raw, np.ndarray)
        assert id(raw) in session.value_origins

    def test_object_array_of_pairs_to_float(self):
        # object array holding Pairs -> plain float array (lines 69-73).
        u = Pair.array("u", np.array([1.0, 2.0]))
        arr = np.array([u[0], u[1]], dtype=object)
        out = value_of(arr)
        assert out.dtype == float
        assert np.allclose(out, [1.0, 2.0])

    def test_list_of_pairs_rebuilt(self):
        # list containing a Pair is rebuilt with values (line 74-75).
        p = Pair(5.0, sympy.Symbol("y"))
        out = value_of([p, 9.0])
        assert isinstance(out, list)
        assert out == [5.0, 9.0]

    def test_object_array_without_pairs_passthrough(self):
        arr = np.array([1.0, 2.0], dtype=object)
        out = value_of(arr)
        # no Pairs inside: returned as-is (falls through)
        assert out is arr


class TestFormulaOf:
    def test_pair_formula(self):
        p = Pair(1.0, sympy.Symbol("z"))
        assert formula_of(p) == sympy.Symbol("z")

    def test_keepdims_box_single_formula(self):
        # object array where all element formulas are identical: one
        # scalar in a box, returned directly (lines 108-112).
        p = Pair(1.0, sympy.Symbol("s"))
        arr = np.array([p, p], dtype=object)
        assert formula_of(arr) == sympy.Symbol("s")

    def test_decompressed_no_pattern_refuses(self):
        # element formulas with no recompressible pattern -> refusal
        # (lines 113-120).
        a = Pair(1.0, sympy.Symbol("a"))
        b = Pair(2.0, sympy.cos(sympy.Symbol("q")))
        arr = np.array([a, b], dtype=object)
        with pytest.raises(NotImplementedError, match="provable pattern"):
            formula_of(arr)

    def test_uniform_field_constant(self):
        # a uniform field folds to one clean constant (lines 121-124/129).
        out = formula_of(np.full((3, 3), 5.0))
        assert out == sympy.Float(5.0)

    def test_uniform_nan_field(self):
        # a uniform NaN array -> the inert NaN symbol (lines 125-126).
        out = formula_of(np.full(4, np.nan))
        assert out == sympy.Symbol("NaN", real=True)

    def test_uniform_bool_field(self):
        # a uniform boolean array -> sympy Integer (lines 127-128).
        out = formula_of(np.ones(3, dtype=bool))
        assert out == sympy.Integer(1)

    def test_concrete_table_disclosed(self):
        # a small concrete non-uniform kernel -> named const table
        # (lines 130-143).
        out = formula_of(np.array([1.0, 2.0, 3.0]))
        assert out.atoms(sympy.IndexedBase)

    def test_raw_large_nonuniform_refuses(self):
        # non-uniform array too large to disclose -> refusal (lines 144-146).
        big = np.arange(5000.0)
        with pytest.raises(NotImplementedError, match="wrap it"):
            formula_of(big)

    def test_object_array_recompresses(self):
        # a decompressed object array whose per-element formulas share a
        # provable pattern folds to one indexed rule (lines 113-117).
        u = Pair.array("u", np.array([1.0, 3.0, 6.0, 10.0]))
        d = u[1:] - u[:-1]
        elems = np.array([d[k] for k in range(3)], dtype=object)
        out = formula_of(elems)
        assert out.atoms(sympy.Indexed)

    def test_scalar_nan_sentinel(self):
        assert formula_of(float("nan")) == sympy.Symbol("NaN", real=True)

    def test_scalar_number(self):
        assert formula_of(4) == sympy.Integer(4)


class TestRepack:
    def test_non_object_array_returns_none(self):
        assert repack(np.arange(3.0)) is None

    def test_empty_object_array_returns_none(self):
        assert repack(np.array([], dtype=object)) is None

    def test_unpatterned_returns_none(self):
        # object array of Pairs whose formulas do not recompress ->
        # formula_of raises, repack swallows it and returns None (190-191).
        a = Pair(1.0, sympy.Symbol("a"))
        b = Pair(2.0, sympy.cos(sympy.Symbol("q")))
        arr = np.array([a, b], dtype=object)
        assert repack(arr) is None

    def test_uniform_pairs_repacks(self):
        p = Pair(1.0, sympy.Symbol("s"))
        arr = np.array([p, p], dtype=object)
        out = repack(arr)
        assert isinstance(out, Pair)


class TestNumeric:
    def test_object_array_coerces(self):
        v = np.array([1.0, 2.0], dtype=object)
        out = numeric(v)
        assert out.dtype == float

    def test_object_array_uncoercible_passthrough(self):
        v = np.array([object(), object()], dtype=object)
        out = numeric(v)
        assert out.dtype == object

    def test_plain_array_copied(self):
        v = np.arange(3.0)
        out = numeric(v, copy=True)
        assert out is not v and np.allclose(out, v)

    def test_plain_array_nocopy(self):
        v = np.arange(3.0)
        assert numeric(v, copy=False) is v

    def test_scalar_passthrough(self):
        assert numeric(7) == 7


# ---------------------------------------------------------------------------
# checks.py
# ---------------------------------------------------------------------------
I = axis_idx(0)
U = sympy.IndexedBase("u")


class TestAgainst:
    def test_matching_after_simplify(self):
        # residual survives expand (canonical) but simplify cancels it:
        # a trig identity exercises _rung_simplify's PROVEN return (37).
        t = sympy.Symbol("t")
        x = Pair(0.5, sympy.sin(t) ** 2)
        e = checks.against(x, 1 - sympy.cos(t) ** 2)
        assert e.verdict == checks.PROVEN
        assert e.method == "simplify"

    def test_canonical_match_proven(self):
        # residual expands straight to zero: the cheapest rung decides
        # (line 29, _rung_canonical PROVEN).
        u = Pair.array("u", np.arange(5.0))
        d = u[1:] - u[:-1]
        e = checks.against(d, U[I + 1] - U[I])
        assert e.verdict == checks.PROVEN
        assert e.method == "canonical"

    def test_nonzero_constant_refuted(self):
        # constant residual -> REFUTED (line 49).
        x = Pair(1.0, sympy.Integer(1))
        e = checks.against(x, sympy.Integer(2))
        assert e.verdict == checks.REFUTED

    def test_symbolic_residual_refuted(self):
        u = Pair.array("u", np.arange(5.0))
        d = u[1:] - u[:-1]
        e = checks.against(d, U[I + 2] - U[I])
        assert e.verdict == checks.REFUTED

    def test_free_symbol_only_refuted(self):
        # residual is a bare non-indexed symbol -> REFUTED (line 51 or 52).
        x = Pair(1.0, sympy.Symbol("w"))
        e = checks.against(x, sympy.Integer(0))
        assert e.verdict == checks.REFUTED


class TestConservesMass:
    def test_full_update_proven(self):
        def birth_death(p, b, d):
            stay = 1.0 - b - d
            return b * p[:-2] + stay * p[1:-1] + d * p[2:]

        out = to_sympy(birth_death, np.full(16, 1 / 16), 0.3, 0.2)
        assert checks.conserves_mass(out).verdict == checks.PROVEN

    def test_leak_refuted(self):
        # coefficient sum != 1 with a numeric residual -> REFUTED (line 103).
        u = Pair.array("u", np.arange(5.0))
        e = checks.conserves_mass(u * 2.0)
        assert e.verdict == checks.REFUTED


class TestCentered:
    def test_laplacian_proven(self):
        u = Pair.array("u", np.arange(6.0))
        lap = u[2:] - 2 * u[1:-1] + u[:-2]
        assert checks.centered(lap, at=1).verdict == checks.PROVEN

    def test_forward_difference_refuted(self):
        u = Pair.array("u", np.arange(6.0))
        fwd = u[1:] - u[:-1]
        assert checks.centered(fwd, at=0).verdict == checks.REFUTED

    def test_no_indexed_letters_unknown(self):
        x = Pair(3.0, sympy.Symbol("x"))
        assert checks.centered(x).verdict == checks.UNKNOWN

    def test_pure_integer_index_skipped(self):
        # an index that is a bare integer constant (no axis letter) is
        # skipped, letter offsets still decide (lines 121-122).
        i = axis_idx(0)
        A = sympy.IndexedBase("A")
        f = A[i - 1, 0] + A[i + 1, 0]  # second index constant 0
        p = Pair(np.zeros(3), f, ((0, 3),))
        assert checks.centered(p, at=0).verdict == checks.PROVEN

    def test_multi_letter_index_unknown(self):
        # an index carrying two axis letters cannot be reduced (123-124).
        i, j = axis_idx(0), axis_idx(1)
        A = sympy.IndexedBase("A")
        p = Pair(np.zeros(3), A[i + j], ((0, 3),))
        assert checks.centered(p).verdict == checks.UNKNOWN

    def test_noninteger_offset_unknown(self):
        # an offset that is not an integer (a free stencil parameter)
        # -> UNKNOWN (line 128).
        i = axis_idx(0)
        A = sympy.IndexedBase("A")
        h = sympy.Symbol("h")
        p = Pair(np.zeros(3), A[i] + A[i + h], ((0, 3),))
        assert checks.centered(p).verdict == checks.UNKNOWN


class TestBanded:
    def _tridiagonal(self):
        a = Pair(np.zeros((5, 5)), sympy.Integer(0), ((0, 5), (0, 5)))
        v = Pair.array("v", np.arange(25.0).reshape(5, 5))
        for i in range(5):
            for j in range(max(0, i - 1), min(5, i + 2)):
                a[i, j] = v[i, j]
        return a

    def test_tridiagonal_proven(self):
        ev = checks.banded(self._tridiagonal())
        assert ev.verdict == checks.PROVEN
        assert ev.detail == "banded (1, 1)"

    def test_zero_matrix_proven(self):
        # a formula that is identically zero: the empty-scatter PROVEN
        # branch (line 184).
        a = Pair(np.zeros((3, 3)), sympy.Integer(0), ((0, 3), (0, 3)))
        ev = checks.banded(a)
        assert ev.verdict == checks.PROVEN
        assert "band 0, 0" in ev.detail

    def test_non_scatter_unknown(self):
        u = Pair.array("u", np.arange(6.0).reshape(2, 3))
        assert checks.banded(u).verdict == checks.UNKNOWN

    def test_relational_condition_not_concrete(self):
        # a piecewise write condition Eq(i, j) is not a concrete
        # (row, col) write -> _scatter_positions returns None (line 163).
        i, j = axis_idx(0), axis_idx(1)
        pw = sympy.Piecewise((sympy.Float(1.0), sympy.Eq(i, j)),
                             (sympy.Integer(0), sympy.true))
        p = Pair(np.zeros((3, 3)), pw, ((0, 3), (0, 3)))
        assert checks.banded(p).verdict == checks.UNKNOWN

    def test_single_axis_condition_not_concrete(self):
        # a condition fixing only one axis (Eq(i, 0)) leaves the column
        # free: not a concrete scatter (line 167).
        i = axis_idx(0)
        pw = sympy.Piecewise((sympy.Float(1.0), sympy.Eq(i, 0)),
                             (sympy.Integer(0), sympy.true))
        p = Pair(np.zeros((3, 3)), pw, ((0, 3), (0, 3)))
        assert checks.banded(p).verdict == checks.UNKNOWN


# ---------------------------------------------------------------------------
# contracts.py -- driven both end-to-end (to_sympy) and directly (check_call)
# ---------------------------------------------------------------------------
class TestSolveContract:
    def test_solve_ok_end_to_end(self):
        A0 = np.array([[4.0, 1.0], [1.0, 3.0]])
        b0 = np.array([1.0, 2.0])
        r = to_sympy(lambda A, b: np.linalg.solve(A, b), A0, b0)
        verdicts = dict(r.unchecked[0][1])
        assert verdicts["square"] == OK
        assert verdicts["residual"] == OK

    def test_solve_residual_zero_scale_ok(self):
        # A and b both zero -> scale == 0 branch (line 114 in _solve_residual).
        A0 = np.zeros((2, 2))
        b0 = np.zeros(2)
        x0 = np.zeros(2)
        _, verdicts = check_call("solve", (A0, b0), x0)
        assert dict(verdicts)["residual"] == OK

    def test_solve_residual_bad(self):
        A0 = np.array([[4.0, 1.0], [1.0, 3.0]])
        b0 = np.array([1.0, 2.0])
        _, verdicts = check_call("solve", (A0, b0), np.array([9.0, 9.0]))
        assert dict(verdicts)["residual"] == FAILED

    def test_square_requires_failed(self):
        _, verdicts = check_call("solve", (np.zeros((2, 3)), np.zeros(2)), np.zeros(3))
        assert dict(verdicts)["square"] == FAILED


class TestBandedAndDesignContracts:
    def test_solve_banded_zero_scale_ok(self):
        # all-zero banded system -> _relative_residual scale==0 branch
        # (line 56).
        args = ((0, 0), np.zeros((1, 3)), np.zeros(3))
        _, verdicts = check_call("solve_banded", args, np.zeros(3))
        assert dict(verdicts)["residual"] == OK

    def test_design_matrix_wrong_basis_failed(self):
        # a compiled design matrix that does not match sympy's own
        # B-spline basis is refuted (line 103).
        x = np.array([0.5, 1.5, 2.5])
        t = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        _, verdicts = check_call("design_matrix", (x, t, 3), np.zeros((3, 6)))
        assert dict(verdicts)["residual"] == FAILED

    def test_cholesky_non_2d_unknown(self):
        # a 1-D "factor" cannot be a Cholesky factor -> UNKNOWN (line 152).
        _, verdicts = check_call("cholesky", (np.eye(2),), np.array([1.0, 1.0]))
        assert dict(verdicts)["residual"] == UNKNOWN


class TestSvdContract:
    def test_svd_ok_end_to_end(self):
        A0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = to_sympy(lambda A: np.linalg.svd(A, full_matrices=False), A0)
        first = out[0] if isinstance(out, tuple) else out
        rec = [x for x in first.unchecked if x[0] == "svd"]
        assert rec and dict(rec[0][1])["residual"] == OK

    def test_svd_not_tuple_unknown(self):
        # result is not a 3-tuple -> UNKNOWN (lines 126-127).
        A0 = np.eye(3)
        assert dict(check_call("svd", (A0,), np.eye(3))[1])["residual"] == UNKNOWN

    def test_svd_bad_reconstruction_failed(self):
        A0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        u, s, vh = np.linalg.svd(A0)
        bad = (u, s * 2.0, vh)  # wrong singular values
        assert dict(check_call("svd", (A0,), bad)[1])["residual"] == FAILED

    def test_svd_nonorthonormal_u_failed(self):
        # reconstruction ok but U not orthonormal -> FAILED (lines 133-134).
        A0 = np.eye(2)
        u = np.array([[1.0, 1.0], [0.0, 1.0]])
        s = np.array([1.0, 1.0])
        vh = np.eye(2)
        recon = (u[:, :2] * s) @ vh
        # make A match this (non-orthonormal) reconstruction so recon passes
        assert dict(check_call("svd", (recon,), (u, s, vh))[1])["residual"] == FAILED


class TestQrContract:
    def test_qr_ok(self):
        A0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        q, r_ = np.linalg.qr(A0)
        assert dict(check_call("qr", (A0,), (q, r_))[1])["residual"] == OK

    def test_qr_not_tuple_unknown(self):
        # result not a >=2 tuple -> UNKNOWN (lines 140-141).
        A0 = np.eye(2)
        assert dict(check_call("qr", (A0,), np.eye(2))[1])["residual"] == UNKNOWN

    def test_qr_bad_failed(self):
        A0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        q, r_ = np.linalg.qr(A0)
        assert dict(check_call("qr", (A0,), (q, r_ * 2.0))[1])["residual"] == FAILED


class TestLstsqContract:
    def test_lstsq_ok(self):
        A0 = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b0 = np.array([1.0, 2.0, 2.0])
        x, *_ = np.linalg.lstsq(A0, b0, rcond=None)
        # tuple result path (line 165)
        assert dict(check_call("lstsq", (A0, b0), (x,))[1])["residual"] == OK

    def test_lstsq_array_result_ok(self):
        A0 = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b0 = np.array([1.0, 2.0, 2.0])
        x = np.linalg.lstsq(A0, b0, rcond=None)[0]
        assert dict(check_call("lstsq", (A0, b0), x)[1])["residual"] == OK

    def test_lstsq_bad_failed(self):
        A0 = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b0 = np.array([1.0, 2.0, 2.0])
        assert dict(check_call("lstsq", (A0, b0), np.array([9.0, 9.0]))[1])["residual"] == FAILED


class TestEighContract:
    def test_symmetric_ok(self):
        A0 = np.array([[2.0, 1.0], [1.0, 2.0]])
        assert dict(check_call("eigh", (A0, None), None)[1])["symmetric"] == OK

    def test_asymmetric_failed(self):
        # line 24: non-square/asymmetric FAILED inside _symmetric.
        A0 = np.array([[2.0, 1.0], [0.0, 2.0]])
        assert dict(check_call("eigh", (A0, None), None)[1])["symmetric"] == FAILED

    def test_nonsquare_symmetric_failed(self):
        A0 = np.zeros((2, 3))
        assert dict(check_call("eigh", (A0, None), None)[1])["symmetric"] == FAILED


class TestOtherContracts:
    def test_irfft_wrong_spec_shape_unknown(self):
        # spec shape mismatch -> UNKNOWN (lines 211-212).
        out = np.array([1.0, 2.0, 3.0, 4.0])
        spec = np.array([1.0 + 0j])  # wrong shape
        assert check_call("irfft", (spec,), out)[1][0][1] == UNKNOWN

    def test_irfft_too_big_unknown(self):
        out = np.zeros(4096)
        assert check_call("irfft", (np.zeros(2049, dtype=complex),), out)[1][0][1] == UNKNOWN

    def test_fft_ok(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        got = np.fft.fft(x)
        assert check_call("fft", (x,), got)[1][0][1] == OK

    def test_unknown_contract_all_unknown(self):
        name, verdicts = check_call("no_such_routine", (np.eye(2),), None)
        assert dict(verdicts)["contract"] == UNKNOWN

    def test_residual_exception_swallowed_unknown(self):
        # residual fn raises (mismatched shapes) -> UNKNOWN (lines 344-345).
        _, verdicts = check_call("solve", (np.eye(2), np.zeros(2)), "not-an-array")
        assert dict(verdicts)["residual"] == UNKNOWN
