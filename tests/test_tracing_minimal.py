"""The minimal 'trace all Python' layer: coercion guards, duck facts,
and the __wrapped__ fallback that runs numpy's real bodies on Pairs."""

import numpy as np
import pytest
import sympy

from skverify import Pair

U = sympy.IndexedBase("u")


def make():
    return Pair.array("u", np.array([1.0, 2.0, 4.0, 7.0]))


class TestCoercionGuards:
    def test_float_refuses(self):
        x = Pair(3.0, sympy.Symbol("x"))
        with pytest.raises(NotImplementedError):
            float(x)

    def test_int_refuses(self):
        x = Pair(3.0, sympy.Symbol("x"))
        with pytest.raises(NotImplementedError):
            int(x)

    def test_complex_refuses(self):
        x = Pair(3.0, sympy.Symbol("x"))
        with pytest.raises(NotImplementedError):
            complex(x)

    def test_value_is_the_deliberate_exit(self):
        x = Pair(3.0, sympy.Symbol("x"))
        assert x.value == 3.0


class TestDuckFacts:
    def test_ndim_shape_dtype(self):
        u = Pair.array("u", np.zeros((4, 7)))
        assert u.ndim == 2
        assert u.shape == (4, 7)
        # object, deliberately: numpy cast branches (ret.dtype.type(x))
        # become passthroughs instead of float(Pair) deaths
        assert u.dtype == np.dtype(object)

    def test_scalar_pair_facts(self):
        x = Pair(3.0, sympy.Symbol("x"))
        assert x.ndim == 0
        assert x.shape == ()


class TestWrappedFallback:
    def test_diff_traces_through_numpy_source(self):
        u = make()
        r = np.diff(u)
        # numpy's real body ran; elements are scalar Pairs, unrolled
        assert [e.formula for e in r] == [
            U[1] - U[0],
            U[2] - U[1],
            U[3] - U[2],
        ]
        assert [e.value for e in r] == list(np.diff(u.value))

    def test_second_difference(self):
        u = make()
        r = np.diff(u, n=2)
        assert [e.formula for e in r] == [
            U[0] - 2 * U[1] + U[2],
            U[1] - 2 * U[2] + U[3],
        ]
        assert [e.value for e in r] == list(np.diff(u.value, n=2))

    def test_table_entries_still_win(self):
        # np.sum is curated: indexed Sum formula, not unrolled elements
        u = make()
        s = np.sum(u)
        assert isinstance(s.formula, sympy.Sum)

    def test_dot_traces_as_inner_product(self):
        # np.dot's python wrapper runs; decompression + dunders do the rest
        u = make()
        r = np.dot(u, u)
        assert r.formula == U[0] ** 2 + U[1] ** 2 + U[2] ** 2 + U[3] ** 2
        assert r.value == np.dot(u.value, u.value)

    def test_to_sympy_recompresses_unrolled_results(self):
        # fallback path returns ndarray-of-Pairs; repack folds the shift-
        # invariant pattern back into ONE indexed formula (proven, not guessed)
        from skverify import to_sympy
        from skverify.helpers import axis_idx

        x = np.linspace(0.0, 9.0, 10)
        s = to_sympy(lambda x: np.diff(x, 2), x)
        assert isinstance(s, Pair)
        X, I = sympy.IndexedBase("x"), axis_idx(0)
        assert sympy.expand(s.formula - (X[I] - 2 * X[I + 1] + X[I + 2])) == 0
        assert s.domain == (0, 8)
        assert np.allclose(s.value, np.diff(x, 2))

    def test_scalar_add_folds_to_sum(self):
        # dot: one big Add -> a pure Sum, proven term by term
        from skverify import to_sympy
        from skverify.helpers import axis_idx

        d = to_sympy(np.dot, np.arange(4.0), np.arange(4.0))
        assert d.formula.atoms(sympy.Sum)
        assert float(
            d.formula.doit().subs(
                {sympy.IndexedBase(n)[k]: float(k) for n in "ab" for k in range(4)}
            )
        ) == float(d.value)

    def test_boundary_terms_split_from_sum(self):
        # trapezoid: half-weight endpoints stay OUTSIDE the Sum
        from skverify import to_sympy
        from scipy.integrate import trapezoid

        y = np.linspace(0, 1, 8) ** 2
        t = to_sympy(lambda y: trapezoid(y, dx=0.1), y)
        Y = sympy.IndexedBase("y")
        assert t.formula.atoms(sympy.Sum)
        assert t.formula.has(Y[0]) and t.formula.has(Y[7])  # boundaries visible
        assert np.allclose(t.value, trapezoid(y, dx=0.1))

    def test_phased_fold_simpson(self):
        # simpson: alternating 4/3, 2/3 weights -> TWO stride-2 Sums
        from skverify import to_sympy
        from scipy.integrate import simpson

        y = np.linspace(0, 1, 9) ** 2
        s = to_sympy(lambda y: simpson(y, dx=0.125), y)
        assert len(s.formula.atoms(sympy.Sum)) == 2
        assert np.allclose(s.value, simpson(y, dx=0.125))

    def test_running_sums_fold_cumulatively(self):
        # growing elements fold as elem[0] + Sum(difference); values unchanged
        from skverify import to_sympy
        from scipy.integrate import cumulative_trapezoid

        b = to_sympy(cumulative_trapezoid, np.linspace(0, 1, 6) ** 2)
        assert b.formula.has(sympy.Sum)
        assert np.allclose(b.value, cumulative_trapezoid(np.linspace(0, 1, 6) ** 2))

    def test_to_sympy_passes_config_ints_through(self):
        # ints are config (n=, axis=), not math: no wrapping, no lambda needed
        from skverify import to_sympy

        x = np.linspace(0.0, 9.0, 10)
        s = to_sympy(np.diff, x, 2)
        from skverify.helpers import axis_idx

        A, I = sympy.IndexedBase("a"), axis_idx(0)  # named from diff's signature
        assert sympy.expand(s.formula - (A[I] - 2 * A[I + 1] + A[I + 2])) == 0
        assert np.allclose(s.value, np.diff(x, 2))

    def test_ufunc_methods_for_object_loop(self):
        # numpy's object-dtype ufunc loop calls elem.log() etc. instead of
        # dispatching; Pair grows one method per mapped ufunc (registry-driven)
        u = Pair.array("u", np.linspace(1.0, 2.0, 4))
        assert u.log().formula == sympy.log(U[sympy.Symbol("i", integer=True)])
        obj = np.asarray(u)  # decompression: object array of scalar Pairs
        r = np.log(obj)
        assert r[0].formula == sympy.log(U[0])
        assert float(r[0].value) == np.log(u.value[0])

    def test_unsupported_op_inside_body_is_loud(self):
        # np.median's body needs Pair < Pair (not supported yet):
        # dies mid-trace with a loud error, never silently
        u = make()
        with pytest.raises((NotImplementedError, TypeError)):
            np.median(u)
