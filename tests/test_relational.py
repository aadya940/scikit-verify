"""Relational Pairs, mask algebra, the 0/1 bridge, all/any.
Built from the A-F case inventory; every positive case checks BOTH lanes."""

import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx

I, J = axis_idx(0), axis_idx(1)
JD = sympy.Symbol("j", integer=True)
U = sympy.IndexedBase("u")
V = sympy.IndexedBase("v")


def make(vals=(0.3, -0.1, 0.5, 2.0)):
    return Pair.array("u", np.array(vals))


# A. the six comparisons
class TestComparisons:
    @pytest.mark.parametrize(
        "op,sy",
        [
            (lambda a, b: a < b, sympy.Lt),
            (lambda a, b: a <= b, sympy.Le),
            (lambda a, b: a > b, sympy.Gt),
            (lambda a, b: a >= b, sympy.Ge),
            (lambda a, b: a == b, sympy.Eq),
            (lambda a, b: a != b, sympy.Ne),
        ],
    )
    def test_pair_scalar_all_six(self, op, sy):
        u = make()
        m = op(u, 0)
        assert m.formula == sy(U[I], 0)
        assert np.array_equal(m.value, op(u.value, 0))
        assert m.domain == (0, 4)

    def test_scalar_on_the_left_reflects(self):
        u = make()
        m = 0 < u  # Python mirrors to u.__gt__(0)
        assert m.formula == sympy.Gt(U[I], 0)
        assert np.array_equal(m.value, u.value > 0)

    def test_pair_pair_same_shape(self):
        u, v = make(), Pair.array("v", np.arange(4.0))
        m = u > v
        assert m.formula == sympy.Gt(U[I], V[I])
        assert np.array_equal(m.value, u.value > np.arange(4.0))

    def test_pair_pair_rank_broadcast(self):
        u = Pair.array("u", np.zeros((3, 4)))
        v = Pair.array("v", np.arange(4.0))
        m = u > v  # v aligns to the trailing axis
        assert m.formula == sympy.Gt(U[I, J], V[J])
        assert m.domain == ((0, 3), (0, 4))

    def test_uniform_raw_ndarray_operand(self):
        u = make()
        m = u > np.zeros(4)  # uniform -> constant folds in
        assert m.formula == sympy.Gt(U[I], 0.0)

    def test_scalar_pairs(self):
        x = Pair(3.0, sympy.Symbol("x"))
        m = x > 2
        assert m.formula == sympy.Gt(sympy.Symbol("x"), 2)
        assert bool(m.value) is True
        assert m.domain is None

    def test_steps_flow_through(self):
        u = make()
        m = np.exp(u) > 1
        assert sympy.exp(U[I]) in m.steps
        assert m.steps[-1] == m.formula


# B. mask algebra
class TestMaskAlgebra:
    def test_and_or_xor_invert(self):
        u = make()
        a, b = u > 0, u < 1
        assert (a & b).formula == sympy.And(sympy.Gt(U[I], 0), sympy.Lt(U[I], 1))
        assert (a | b).formula == sympy.Or(sympy.Gt(U[I], 0), sympy.Lt(U[I], 1))
        assert (a ^ b).formula == sympy.Xor(sympy.Gt(U[I], 0), sympy.Lt(U[I], 1))
        assert (~a).formula == sympy.Le(U[I], 0)  # Not(Gt) simplifies
        assert np.array_equal((a & b).value, (u.value > 0) & (u.value < 1))
        assert np.array_equal((~a).value, ~(u.value > 0))

    def test_cross_pair_masks_merge_domains(self):
        u, v = make(), Pair.array("v", np.arange(4.0))
        m = (u > 0) & (v < 3)
        assert m.domain == (0, 4)
        assert np.array_equal(m.value, (u.value > 0) & (np.arange(4.0) < 3))

    def test_python_and_or_refuse(self):
        u = make()
        with pytest.raises(NotImplementedError):
            (u > 0) and (u < 1)
        with pytest.raises(NotImplementedError):
            (u > 0) or (u < 1)

    def test_chained_comparison_refuses(self):
        u = make()
        with pytest.raises(NotImplementedError):
            0 < u < 1  # Python forces bool() between the two


# C. the 0/1 bridge
class TestBridge:
    def test_mask_sum_counts(self):
        u = make()
        c = np.sum(u > 0)
        expected = sympy.Sum(
            sympy.Piecewise((1, sympy.Gt(U[JD], 0)), (0, True)), (JD, 0, 3)
        )
        assert c.formula == expected
        assert c.value == np.sum(u.value > 0)

    def test_mask_times_array_selects(self):
        u = make()
        r = (u > 0) * u
        assert r.formula == sympy.Piecewise((1, sympy.Gt(U[I], 0)), (0, True)) * U[I]
        assert np.allclose(r.value, (u.value > 0) * u.value)

    def test_one_minus_mask(self):
        u = make()
        r = 1 - (u > 0)
        assert np.allclose(r.value, 1 - (u.value > 0))

    def test_masks_combine_raw_not_bridged(self):
        # & consumes Booleans raw: And(...), never And(Piecewise(...))
        u = make()
        m = (u > 0) & (u < 1)
        assert not m.formula.atoms(sympy.Piecewise)


# D. consumers
class TestConsumers:
    def test_where_carries_real_condition(self):
        u = make()
        w = np.where(u > 0, u, 0.0 * u)
        assert w.formula == sympy.Piecewise((U[I], sympy.Gt(U[I], 0)), (0, True))
        assert np.allclose(w.value, np.where(u.value > 0, u.value, 0.0))

    def test_where_nonrelational_cond_keeps_ne_fallback(self):
        u = make()
        w = np.where(u, u, 0.0 * u)
        assert w.formula.args[0][1] == sympy.Ne(U[I], 0)

    def test_all_is_count_equals_n(self):
        u = make((1.0, 2.0, 3.0))
        m = u > 0
        a = np.all(m)
        assert bool(a.value) is True
        count = sympy.Sum(
            sympy.Piecewise((1, sympy.Gt(U[JD], 0)), (0, True)), (JD, 0, 2)
        )
        assert a.formula == sympy.Eq(count, 3)

    def test_any_is_count_positive(self):
        u = make()
        m = u > 10
        a = np.any(m)
        assert bool(a.value) is False
        assert isinstance(a.formula, sympy.core.relational.StrictGreaterThan)

    def test_method_forms(self):
        u = make()
        assert bool((u > -100).all().value) is True
        assert bool((u > 100).any().value) is False

    def test_maximum_still_max(self):
        u = make()
        assert np.maximum(u, 0).formula == sympy.Max(U[I], 0)


# E. edge cases
class TestEdges:
    def test_unhashable(self):
        u = make()
        with pytest.raises(TypeError):
            hash(u)
        with pytest.raises(TypeError):
            {u: 1}

    def test_empty_all_is_vacuously_true(self):
        u = Pair.array("u", np.zeros(0))
        a = np.all(u > 0)
        assert bool(a.value) is True  # numpy's vacuous forall
        assert a.formula == sympy.true  # Eq(empty Sum -> 0, 0) evaluates true

    def test_comparing_a_mask_bridges_then_compares(self):
        u = make()
        m = (u > 0) > 0  # bools are 0/1 to numpy
        assert m.formula.atoms(sympy.Piecewise)
        assert np.array_equal(m.value, (u.value > 0) > 0)

    def test_float_eq_allowed(self):
        u = make()
        m = u == 0.5
        assert m.formula == sympy.Eq(U[I], 0.5)

    def test_bool_indexing_still_refused(self):
        u = make()
        with pytest.raises(NotImplementedError):
            u[u > 0]  # mask indexing is a later feature

    def test_isnan_dies_loudly(self):
        u = make()
        with pytest.raises((NotImplementedError, TypeError)):
            np.isnan(u)
