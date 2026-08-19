"""Two-lane sets: Python set semantics, sympy Set formulas."""

import numpy as np
import sympy

from skverify import Pair
from skverify.sets import TracedSet


def _elems(vals, name="y"):
    base = sympy.IndexedBase(name)
    return [Pair(v, base[k]) for k, v in enumerate(vals)]


def test_dedup_by_value_formula_keeps_algebra():
    ts = TracedSet(_elems([1.0, 0.0, 1.0]))
    assert len(ts) == 2                      # concrete lane deduped
    assert isinstance(ts.formula, sympy.FiniteSet)
    assert len(ts.formula.args) == 3         # symbols kept: equality is data


def test_union_intersection_difference():
    a = TracedSet(_elems([1.0, 2.0], "a"))
    b = TracedSet(_elems([2.0, 3.0], "b"))
    u = a | b
    assert u.value == {1.0, 2.0, 3.0}
    # sympy merges finite unions eagerly; the four symbols must survive
    A, B = sympy.IndexedBase("a"), sympy.IndexedBase("b")
    assert all(e in u.formula.args for e in (A[0], A[1], B[0], B[1]))
    i = a & b
    assert i.value == {2.0}
    assert isinstance(i.formula, sympy.Intersection)
    d = a - b
    assert d.value == {1.0}
    assert d.formula.func is sympy.Complement


def test_membership_and_iteration_concrete():
    ts = TracedSet(_elems([1.0, 0.0]))
    assert 1.0 in ts
    assert 5.0 not in ts
    assert sorted(p.value for p in ts) == [0.0, 1.0]


def test_pair_hash_buckets_by_value_and_discloses():
    from skverify.session import current

    current.reset()
    y = _elems([1.0, 0.0, 1.0])
    d = {y[0]: "one", y[1]: "zero"}
    assert d[y[2]] == "one"                  # equal value finds the key
    assert current.hashed                    # identity use recorded
