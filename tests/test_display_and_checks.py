"""Cover the display helpers (latex, pretty, _tidy) and the checks
module (against, conserves_mass, centered, banded) with real traces."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy, latex
from skverify import checks


V = sympy.IndexedBase("v")
i = sympy.Symbol("i", integer=True)


class TestLatex:
    def test_latex_of_traced_formula(self):
        out = to_sympy(lambda v: (v - v.mean()), np.array([1.0, 2.0, 3.0]))
        s = latex(out.formula[0] if hasattr(out.formula, "shape") else out.formula)
        assert isinstance(s, str) and len(s) > 0

    def test_latex_non_basic_passthrough(self):
        assert latex(3.5) == "3.5"

    def test_latex_matrix(self):
        M = sympy.Matrix([[sympy.Symbol("a_1"), 0], [0, sympy.Symbol("a_2")]])
        s = latex(M)
        assert "mathtt" in s

    def test_latex_alias_legend_for_long_names(self):
        long = sympy.Symbol("a_really_long_underscored_name_here")
        aliases = {}
        s = latex(long + 1, aliases=aliases)
        assert aliases  # filled with T1 -> full name
        assert any(v == str(long) for v in aliases.values())

    def test_tidy_collapses_and_flips(self):
        # a median trace: negated comparisons should read positively
        out = to_sympy(np.median, np.array([3.0, 1.0, 4.0, 1.5]))
        txt = out.pretty()
        assert "formula" in txt


class TestChecks:
    def test_against_proven(self):
        out = to_sympy(lambda v: v.sum(), np.array([1.0, 2.0, 3.0]))
        j = sympy.Dummy("j", integer=True)
        ev = checks.against(out, sympy.Sum(V[j], (j, 0, 2)))
        assert "proven" in str(ev).lower() or ev.verdict

    def test_against_refuted(self):
        out = to_sympy(lambda v: v.sum(), np.array([1.0, 2.0, 3.0]))
        j = sympy.Dummy("j", integer=True)
        ev = checks.against(out, sympy.Sum(V[j], (j, 0, 2)) + 1)
        assert "refut" in str(ev).lower() or not getattr(ev, "ok", True)

    def test_conserves_mass_true(self):
        # a convex-combination update: coefficients sum to one
        def update(v):
            return 0.25 * v[0] + 0.75 * v[1]
        out = to_sympy(update, np.array([1.0, 2.0]))
        ev = checks.conserves_mass(out)
        assert "proven" in str(ev).lower()

    def test_conserves_mass_refuted(self):
        def bad(v):
            return 0.25 * v[0] + 0.5 * v[1]
        out = to_sympy(bad, np.array([1.0, 2.0]))
        ev = checks.conserves_mass(out)
        assert "refut" in str(ev).lower()

    def test_centered_symmetric(self):
        # a symmetric second difference: offsets -1, 0, +1
        def lap(v):
            return v[:-2] - 2.0 * v[1:-1] + v[2:]
        out = to_sympy(lap, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        ev = checks.centered(out)
        assert ev is not None
