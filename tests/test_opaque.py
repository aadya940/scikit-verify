"""Opaque compiled calls: named in the formula, contract-checked, guarded
against hidden mutation."""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy


class TestOpaqueCalls:
    def test_solve_is_named_and_checked(self):
        A0 = np.array([[4.0, 1.0], [1.0, 3.0]])
        b0 = np.array([1.0, 2.0])
        r = to_sympy(lambda A, b: np.linalg.solve(A, b), A0, b0)
        A = sympy.IndexedBase("A")
        B = sympy.IndexedBase("b")
        i, j = sympy.symbols("i j", integer=True)
        assert r.formula == sympy.Function("solve")(A[i, j], B[i])
        assert np.allclose(r.value, np.linalg.solve(A0, b0))
        name, verdicts = r.unchecked[0]
        assert name == "solve"
        assert dict(verdicts)["square"] == "ok"
        assert dict(verdicts)["residual"] == "ok"

    def test_requires_violation_is_recorded(self):
        A0 = np.zeros((2, 3))
        with pytest.raises(Exception):
            to_sympy(lambda A, b: np.linalg.solve(A, b), A0, np.zeros(2))

    def test_opaque_composes_with_arithmetic(self):
        A0 = np.array([[2.0, 0.0], [0.0, 2.0]])
        b0 = np.array([1.0, 1.0])
        r = to_sympy(lambda A, b: np.linalg.solve(A, b) * 2.0, A0, b0)
        assert r.formula.has(sympy.Function("solve"))
        assert np.allclose(r.value, np.linalg.solve(A0, b0) * 2.0)

    def test_norm_lifts_through_its_python_body(self):
        r = to_sympy(lambda u: np.linalg.norm(u), np.array([3.0, 4.0]))
        assert np.isclose(r.value, 5.0)
        assert r.unchecked == ()

    def test_unknown_contract_is_unknown(self):
        u = Pair.array("u", np.array([1.0, 4.0]))

        def fake(x):
            return np.asarray(x, dtype=float) * 0 + 7.0

        r = Pair._opaque_call(fake, (u,), {})
        assert r.formula == sympy.Function("fake")(
            sympy.IndexedBase("u")[sympy.Symbol("i", integer=True)]
        )
        from skverify.pair import _OPAQUE

        name, verdicts = _OPAQUE[-1]
        assert name == "fake"
        assert dict(verdicts)["contract"] == "unknown"

    def test_no_opaque_calls_means_empty(self):
        r = to_sympy(lambda x: x * 2.0, np.arange(3.0))
        assert r.unchecked == ()

    def test_unmapped_single_out_ufunc_goes_opaque(self):
        u = Pair.array("u", np.array([1.0, 4.0]))
        r = np.cbrt(u)
        U = sympy.IndexedBase("u")
        i = sympy.Symbol("i", integer=True)
        assert r.formula == sympy.Function("cbrt")(U[i])
        assert np.allclose(r.value, np.cbrt(u.value))

    def test_multi_output_ufunc_still_refuses(self):
        u = Pair.array("u", np.array([1.0, 4.0]))
        with pytest.raises(NotImplementedError):
            np.frexp(u)
