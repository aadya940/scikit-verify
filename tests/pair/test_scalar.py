"""Scalar Pair behavior: arithmetic dunders, both lanes, refusals."""

import numpy as np
import pytest
import sympy

from skverify import Pair

x_ = sympy.Symbol("x")
y_ = sympy.Symbol("y")


def make():
    return Pair(3.0, x_), Pair(4.0, y_)


class TestBothLanes:
    def test_arithmetic_composition(self):
        x, y = make()
        r = 2 - x / 5 + y * x
        assert r.value == 2 - 3.0 / 5 + 4.0 * 3.0
        assert sympy.simplify(r.formula - (2 - x_ / 5 + y_ * x_)) == 0

    def test_differential(self):
        """The core invariant: formula evaluated == value computed."""
        x, y = make()
        cases = [
            x + y,
            x - y,
            x * y,
            x / y,
            x**y,
            -x,
            abs(x - y),
            2 + x,
            2 - x,
            2 * x,
            2 / x,
            2**x,
            (x + 1) * (y - 2) / (x**2 + 1),
        ]
        for r in cases:
            evaluated = float(r.formula.subs({x_: 3.0, y_: 4.0}))
            assert evaluated == pytest.approx(r.value, abs=1e-12)


class TestReflectedOps:
    """The __r*__ family: operand order is the classic bug."""

    def test_rsub_order(self):
        x, _ = make()
        assert (2 - x).value == -1.0
        assert (2 - x).formula == 2 - x_

    def test_rtruediv_order(self):
        x, _ = make()
        assert (6 / x).value == 2.0
        assert (6 / x).formula == 6 / x_

    def test_rpow_order(self):
        x, _ = make()
        assert (2**x).value == 8.0
        assert (2**x).formula == 2**x_

    def test_commutative_aliases(self):
        x, _ = make()
        assert (2 + x).formula == (x + 2).formula
        assert (2 * x).formula == (x * 2).formula


class TestUnary:
    def test_neg(self):
        x, _ = make()
        assert (-x).value == -3.0
        assert (-x).formula == -x_

    def test_abs(self):
        x, _ = make()
        assert abs(x).value == 3.0
        assert abs(x).formula == sympy.Abs(x_)


class TestScalarProperties:
    def test_scalar_has_no_domain(self):
        x, y = make()
        assert x.domain is None
        assert (x * y + 2).domain is None

    def test_scalar_not_subscriptable(self):
        x, _ = make()
        with pytest.raises(TypeError):
            x[0]


class TestRefusals:
    def test_bool_refuses(self):
        x, _ = make()
        with pytest.raises(NotImplementedError):
            bool(x)

    def test_bool_refuses_in_if(self):
        x, _ = make()
        with pytest.raises(NotImplementedError):
            if x:  # How it will be used.
                pass
