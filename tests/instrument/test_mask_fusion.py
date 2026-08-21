"""The mask-gather -> reduction fusion, end to end.

A bool mask born from comparing traced arrays keeps its per-position
conditions; gathering by it carries (element, condition) provenance;
a reduction fuses the conditions INTO its formula instead of dumping
one guard per position. The payoff: counting metrics like precision
emerge as ratios of counting sums from generic tracing alone.
"""

import numpy as np
import pytest
import sympy

from skverify import to_sympy


def masked_total(x, t):
    m = x > t
    kept = x[m]
    return np.sum(kept)


class TestFusion:
    def test_reduction_fuses_mask_into_formula(self):
        x = np.array([1.0, -2.0, 3.0, -4.0])
        out = to_sympy(masked_total, x, 0.0)
        assert float(out.value) == 4.0
        X = sympy.IndexedBase("x")
        # the mask lives inside the formula, not in the preconditions
        assert out.formula.has(sympy.Piecewise) or out.formula.has(sympy.Sum)
        assert out.preconditions is sympy.true
        subs = {X[k]: v for k, v in enumerate(x)}
        subs[sympy.Symbol("t", real=True)] = 0.0
        assert float(out.formula.doit().subs(subs)) == 4.0
        # and the formula is a real function of the input: flip a sign
        flipped = dict(subs)
        flipped[X[2]] = -3.0
        assert float(out.formula.doit().subs(flipped)) == 1.0

    def test_precision_emerges_as_counting_ratio(self):
        sklearn = pytest.importorskip("sklearn")
        from sklearn.metrics import precision_score

        y_t = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=float)
        y_p = np.array([0, 1, 0, 0, 1, 1, 1, 1], dtype=float)
        out = to_sympy(precision_score, y_t, y_p)
        assert float(out.value) == precision_score(y_t, y_p)
        # not a constant: a ratio over per-sample conditions
        T = sympy.IndexedBase("y_true")
        P = sympy.IndexedBase("y_pred")
        assert out.formula.has(T) and out.formula.has(P)
        subs = {T[k]: v for k, v in enumerate(y_t)}
        subs.update({P[k]: v for k, v in enumerate(y_p)})
        got = out.formula.doit().subs(subs)
        assert float(got) == precision_score(y_t, y_p)


class TestNanMedian:
    def test_axis_median_with_ordering_preconditions(self):
        x = np.array([[3.0, 1.0], [1.0, 5.0], [2.0, 4.0]])
        out = to_sympy(lambda a: np.nanmedian(a, axis=0), x)
        assert np.allclose(
            np.asarray(out.value, dtype=float), np.nanmedian(x, axis=0)
        )
        X = sympy.IndexedBase("a")
        col0 = np.ravel(out.formula)[0]
        # odd count: the median IS one symbolic element, chosen
        # path-scoped under the recorded ordering (here both columns'
        # medians sit in row 2, so harvest recompresses to a[2, i])
        assert col0.subs(sympy.Symbol("i", integer=True), 0) == X[2, 0]
        assert out.preconditions.has(sympy.Le(X[1, 0], X[2, 0]))


class TestMutators:
    """In-place numpy mutators are assignments in disguise: both lanes
    must observe the write, or the call must refuse -- never no-op."""

    def test_add_at_accumulates_on_both_lanes(self):
        def f(x):
            y = x * 1.0
            np.add.at(y, [0, 0], 1.0)
            return y[0]

        out = to_sympy(f, np.array([1.5, -2.0, 3.25, 0.5]))
        assert float(out.value) == 3.5
        X = sympy.IndexedBase("x")
        assert out.formula.subs(X[0], 1.5) == 3.5

    def test_place_and_putmask_write_through(self):
        def g(x):
            y = x * 1.0
            np.place(y, y < 0, [0.0])
            return np.sum(y)

        x = np.array([1.5, -2.0, 3.25, 0.5])
        out = to_sympy(g, x)
        assert float(out.value) == 5.25

    def test_copyto_writes_both_lanes(self):
        def h(x):
            y = np.zeros(4)
            np.copyto(y, x * 2)
            return y[1]

        out = to_sympy(h, np.array([1.5, -2.0, 3.25, 0.5]))
        assert float(out.value) == -4.0
        X = sympy.IndexedBase("x")
        assert out.formula.subs(X[1], -2.0) == -4.0

    def test_fill_and_tolist(self):
        def k(x):
            y = x * 1.0
            y.fill(2.0)
            return y[0] + x.tolist()[1]

        out = to_sympy(k, np.array([1.5, -2.0, 3.25, 0.5]))
        assert float(out.value) == 0.0


class TestCoercionHonesty:
    def test_nan_to_num_clamps_like_numpy(self):
        def f(x):
            with np.errstate(divide="ignore"):
                return np.nan_to_num(x / 0.0)[1]

        x = np.array([1.5, -2.0, 3.25, 0.5])
        out = to_sympy(f, x)
        with np.errstate(divide="ignore"):
            assert float(out.value) == np.nan_to_num(x / 0.0)[1]

    def test_int_argument_assumption_is_disclosed(self):
        out = to_sympy(lambda a: a + 1, 3)
        assert int(out.value) == 4
        assert any("integer argument" in str(r) for r in out.unchecked)

    def test_pandas_series_decompresses(self):
        pd = pytest.importorskip("pandas")

        def f(x):
            return pd.Series(x).sum()

        x = np.array([1.5, -2.0, 3.25, 0.5])
        out = to_sympy(f, x)
        assert float(out.value) == pd.Series(x).sum()
        assert out.formula.has(sympy.Sum)

    def test_round_lifts_with_tie_guards(self):
        def f(x):
            return np.round(x, 1).sum()

        x = np.array([1.44, -2.07, 3.21, 0.58])
        out = to_sympy(f, x)
        X = sympy.IndexedBase("x")
        got = float(
            sympy.N(out.formula.doit().subs({X[k]: v for k, v in enumerate(x)}))
        )
        assert got == float(f(x))
        assert out.preconditions.has(sympy.Mod)

    def test_round_at_a_tie_refuses(self):
        def g(a):
            return np.round(a)

        with pytest.raises(NotImplementedError):
            to_sympy(g, np.array([0.5, 1.5]))
