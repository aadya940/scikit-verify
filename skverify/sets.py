"""Two-lane sets: Python set semantics with sympy Set formulas.

Scientific code that does set arithmetic deserves certificates that
show the algebra. A :class:`TracedSet` carries both lanes, like a
Pair: the value lane is a real Python set of concrete values (it
drives control flow exactly as the untraced run would), and the
formula lane is a sympy ``Set`` expression (``FiniteSet``, ``Union``,
``Intersection``, ``Complement``) recording the operations performed.

Discrete queries resolve concretely, scoped to the query: ``len`` is
this trace's cardinality, iteration follows the concrete elements,
membership answers with the concrete truth. The formula stays exact
under the traced input, which is the library-wide contract.
"""

import numpy as np
import sympy

from .coercion import formula_of, value_of


def _scalar(v):
    """Concrete scalar from a value-lane entry."""
    return v.item() if hasattr(v, "item") else v


class TracedSet:
    """A set with a concrete lane and a sympy Set formula.

    Parameters
    ----------
    elements : iterable
        Elements; traced scalars contribute their formulas.
    formula : sympy.Set, optional
        The symbolic set. Defaults to a ``FiniteSet`` of the
        elements' formulas (duplicates kept symbolically -- whether
        two symbols coincide is data, not syntax).
    """

    def __init__(self, elements=(), formula=None):
        self._elements = []
        seen = set()
        for e in elements:
            v = _scalar(value_of(e))
            if v not in seen:
                seen.add(v)
                self._elements.append(e)
        self.value = seen
        if formula is None:
            formula = sympy.FiniteSet(
                *[formula_of(e) for e in elements]
            ) if elements else sympy.EmptySet
        self.formula = formula

    def __repr__(self):
        return f"TracedSet({self.formula})"

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self._elements)

    def __contains__(self, item):
        return _scalar(value_of(item)) in self.value

    def _combine(self, other, py_op, sy_op):
        other_set = other if isinstance(other, TracedSet) else TracedSet(list(other))
        result = TracedSet.__new__(TracedSet)
        values = py_op(self.value, other_set.value)
        result.value = values
        pool = {**{_scalar(value_of(e)): e for e in other_set._elements},
                **{_scalar(value_of(e)): e for e in self._elements}}
        result._elements = [pool[v] for v in values if v in pool]
        result.formula = sy_op(self.formula, other_set.formula)
        return result

    def __or__(self, other):
        return self._combine(other, set.union, sympy.Union)

    def __and__(self, other):
        return self._combine(other, set.intersection, sympy.Intersection)

    def __sub__(self, other):
        return self._combine(
            other, set.difference, lambda a, b: sympy.Complement(a, b)
        )

    def __xor__(self, other):
        return self._combine(
            other,
            set.symmetric_difference,
            lambda a, b: sympy.SymmetricDifference(a, b),
        )
