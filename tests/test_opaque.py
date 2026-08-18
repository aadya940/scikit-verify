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
        i = sympy.Symbol("i", integer=True)
        assert r.formula == sympy.IndexedBase("solve_0")[i]
        assert np.allclose(r.value, np.linalg.solve(A0, b0))
        name, verdicts, definition = r.unchecked[0]
        assert name == "solve"
        assert dict(verdicts)["square"] == "ok"
        assert dict(verdicts)["residual"] == "ok"
        assert definition == ("solve_0[i]", "solve(A[i, j], b[i])")

    def test_requires_violation_is_recorded(self):
        A0 = np.zeros((2, 3))
        with pytest.raises(Exception):
            to_sympy(lambda A, b: np.linalg.solve(A, b), A0, np.zeros(2))

    def test_opaque_composes_with_arithmetic(self):
        A0 = np.array([[2.0, 0.0], [0.0, 2.0]])
        b0 = np.array([1.0, 1.0])
        r = to_sympy(lambda A, b: np.linalg.solve(A, b) * 2.0, A0, b0)
        assert r.formula.has(sympy.IndexedBase("solve_0"))
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
        assert isinstance(r.formula, sympy.Indexed)
        from skverify.pair import _OPAQUE

        name, verdicts, definition = _OPAQUE[-1]
        assert name == "fake"
        assert dict(verdicts)["contract"] == "unknown"
        assert "fake(u[i])" in definition[1]

    def test_no_opaque_calls_means_empty(self):
        r = to_sympy(lambda x: x * 2.0, np.arange(3.0))
        assert r.unchecked == ()

    def test_unmapped_single_out_ufunc_goes_opaque(self):
        u = Pair.array("u", np.array([1.0, 4.0]))
        r = np.cbrt(u)
        assert isinstance(r.formula, sympy.Indexed)
        assert str(r.formula.base).startswith("cbrt")
        assert np.allclose(r.value, np.cbrt(u.value))

    def test_multi_output_ufunc_still_refuses(self):
        u = Pair.array("u", np.array([1.0, 4.0]))
        with pytest.raises(NotImplementedError):
            np.frexp(u)


class TestContractChecks:
    def test_eigh_symmetric_ok(self):
        from skverify.contracts import check_call

        A0 = np.array([[2.0, 1.0], [1.0, 2.0]])
        name, verdicts = check_call("eigh", (A0, None), None)
        assert dict(verdicts)["symmetric"] == "ok"

    def test_eigh_asymmetric_failed(self):
        from skverify.contracts import check_call

        A0 = np.array([[2.0, 1.0], [0.0, 2.0]])
        name, verdicts = check_call("eigh", (A0, None), None)
        assert dict(verdicts)["symmetric"] == "failed"

    def test_solve_residual_failed_on_wrong_result(self):
        from skverify.contracts import check_call

        A0 = np.array([[4.0, 1.0], [1.0, 3.0]])
        b0 = np.array([1.0, 2.0])
        wrong = np.array([9.0, 9.0])
        name, verdicts = check_call("solve", (A0, b0), wrong)
        assert dict(verdicts)["residual"] == "failed"

    def test_mutation_guard_fires(self):
        u = Pair.array("u", np.arange(4.0))

        def mutator(x):
            x[0] = -1.0
            return np.array([0.0])

        with pytest.raises(NotImplementedError):
            Pair._opaque_call(mutator, (u,), {})
