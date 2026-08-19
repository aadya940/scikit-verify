"""to_sympy front door."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy


def f(u, c):
    return c * u[1:]


class TestToSympy:
    def test_names_from_signature(self):
        out = to_sympy(f, np.ones(5), 2.0)
        assert sympy.Symbol("c", real=True) in out.formula.free_symbols
        assert any(s.name == "u" for s in out.formula.atoms(sympy.IndexedBase))

    def test_too_many_args(self):
        with pytest.raises(TypeError):
            to_sympy(f, np.ones(5), 2.0, 3.0)

    def test_scalar_only_function(self):
        g = lambda a, b: a * b + 1
        out = to_sympy(g, 3.0, 4.0)
        assert out.value == 13.0
        assert out.domain is None
