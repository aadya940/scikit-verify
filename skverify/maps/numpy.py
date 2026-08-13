"Module to map NumPy Ops to SymPy Ops."

import numpy as np
import sympy

from ..registry import (
    UFUNC_TABLE,
    FUNCTION_TABLE,
)
from ..helpers import axis_idx

from ..pair import Pair

# UFUNCs, Elementwise
_SAME = "sin cos tan sinh cosh tanh exp log sqrt floor sign".split()
_RENAMED = {
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "absolute": "Abs",
    "ceil": "ceiling",
}

UFUNC_TABLE.update({getattr(np, n): getattr(sympy, n) for n in _SAME})
UFUNC_TABLE.update({getattr(np, k): getattr(sympy, v) for k, v in _RENAMED.items()})

# Others
UFUNC_TABLE[np.maximum] = sympy.Max
UFUNC_TABLE[np.minimum] = sympy.Min
UFUNC_TABLE[np.arctan2] = sympy.atan2
UFUNC_TABLE[np.conjugate] = sympy.conjugate

# comparisons spelled as functions: np.less(u, 0) etc.
UFUNC_TABLE[np.less] = sympy.Lt
UFUNC_TABLE[np.less_equal] = sympy.Le
UFUNC_TABLE[np.greater] = sympy.Gt
UFUNC_TABLE[np.greater_equal] = sympy.Ge
UFUNC_TABLE[np.equal] = sympy.Eq
UFUNC_TABLE[np.not_equal] = sympy.Ne

# numpy's OBJECT-dtype ufunc loop does not dispatch through __array_ufunc__:
# it calls a same-named METHOD on each element (elem.log(), elem.exp(), ...).
# After decompression (an object array of scalar Pairs) that convention is the
# only way ufuncs reach us, so generate one method per mapped ufunc, each
# re-entering the normal traced path. Registry-driven: the two lists can't drift.


def _attach_ufunc_methods():
    for ufunc in UFUNC_TABLE:
        name = ufunc.__name__
        if hasattr(Pair, name):
            continue  # never clobber real Pair API

        def method(self, *args, _ufunc=ufunc):
            return _ufunc(self, *args)  # back through __array_ufunc__

        method.__name__ = name
        setattr(Pair, name, method)


_attach_ufunc_methods()


# FUNCTION TABLE (non-UFUNCS)


def _where(cond, a, b):
    domain = Pair._merge_domains(
        Pair._domain_of(cond), Pair._domain_of(a), Pair._domain_of(b)
    )
    cond_f = Pair._formula_of(cond)
    if not isinstance(
        cond_f, (sympy.logic.boolalg.Boolean, sympy.core.relational.Relational)
    ):
        cond_f = sympy.Ne(cond_f, 0)
    return Pair(
        np.where(Pair._value_of(cond), Pair._value_of(a), Pair._value_of(b)),
        sympy.Piecewise((Pair._formula_of(a), cond_f), (Pair._formula_of(b), True)),
        domain,
        steps=Pair._steps_of(cond, a, b),
    )


def _fresh_dummy(formula, n_axes, base="j"):
    """A summation dummy not colliding with any symbol already in the
    formula -- sum(sum(u, axis=0)) must not capture the inner Sum's j.

    Axis letters don't count as taken: the reduction substitutes ALL of
    them simultaneously, so reusing a letter's name cannot capture."""
    letters = {axis_idx(ax).name for ax in range(n_axes)}
    taken = {s.name for s in formula.atoms(sympy.Symbol)} - letters
    if base not in taken:
        return sympy.Symbol(base, integer=True)
    k = 2
    while f"{base}{k}" in taken:
        k += 1
    return sympy.Symbol(f"{base}{k}", integer=True)


def _sum(a, axis=None, **kwargs):
    if kwargs:
        raise NotImplementedError(f"np.sum kwargs {list(kwargs)} not supported")
    if not isinstance(a, Pair) or a.domain is None:
        return np.sum(a)  # plain input, not ours

    a = Pair(
        a.value, Pair._bridge_numeric(a.formula), a._axis_bounds, steps=a.steps
    )  # np.sum(u > 0) counts: Boolean -> 0/1 before the Sum
    bounds = a._axis_bounds
    if isinstance(axis, tuple):
        raise NotImplementedError("axis tuples not supported yet")
    if axis is not None and not (axis == 0 and len(bounds) == 1):
        # per-axis reduction: bind ONE letter, survivors renumber down
        # p (3x4), axis=0:  Sum(p[j, i], (j, 0, 2))   domain (0, 4)
        # p (3x4), axis=1:  Sum(p[i, j], (j, 0, 3))   domain (0, 3)
        k = axis % len(bounds)
        j = _fresh_dummy(a.formula, len(bounds))
        rename = {axis_idx(k): j}
        rename.update(
            {axis_idx(ax): axis_idx(ax - 1) for ax in range(k + 1, len(bounds))}
        )
        lo, hi = bounds[k]
        formula = sympy.Sum(a.formula.subs(rename, simultaneous=True), (j, lo, hi - 1))
        new_bounds = bounds[:k] + bounds[k + 1 :]
        return Pair(
            np.sum(a.value, axis=k),
            formula,
            new_bounds or None,
            steps=a.steps,
        )

    # one Sum per axis, innermost axis innermost:
    # 1-D: Sum(p[j], (j, 0, n-1))                      (unchanged output)
    # 2-D: Sum(Sum(p[j0, j1], (j1, 0, m-1)), (j0, 0, n-1))
    if len(bounds) == 1:
        dummies = [_fresh_dummy(a.formula, len(bounds))]
    else:
        dummies = [
            _fresh_dummy(a.formula, len(bounds), base=f"j{ax}")
            for ax in range(len(bounds))
        ]
    formula = a.formula.subs(
        {axis_idx(ax): d for ax, d in enumerate(dummies)}, simultaneous=True
    )
    for ax in reversed(range(len(bounds))):
        lo, hi = bounds[ax]
        formula = sympy.Sum(formula, (dummies[ax], lo, hi - 1))  # inclusive
    return Pair(np.sum(a.value), formula, None, steps=a.steps)


def _zeros_like(a, **kwargs):
    if kwargs:
        raise NotImplementedError(f"zeros_like kwargs {list(kwargs)} not supported")
    return Pair(np.zeros_like(Pair._value_of(a)), sympy.Integer(0), Pair._domain_of(a))


def _ones_like(a, **kwargs):
    if kwargs:
        raise NotImplementedError(f"ones_like kwargs {list(kwargs)} not supported")
    return Pair(np.ones_like(Pair._value_of(a)), sympy.Integer(1), Pair._domain_of(a))


def _full_like(a, fill_value, **kwargs):
    if kwargs:
        raise NotImplementedError(f"full_like kwargs {list(kwargs)} not supported")
    return Pair(
        np.full_like(Pair._value_of(a), Pair._value_of(fill_value)),
        sympy.sympify(Pair._value_of(fill_value)),
        Pair._domain_of(a),
    )


FUNCTION_TABLE[np.zeros_like] = _zeros_like
FUNCTION_TABLE[np.ones_like] = _ones_like
FUNCTION_TABLE[np.full_like] = _full_like


def _count(a):
    """Sum(Piecewise((1, cond), (0, True))) over the mask's domain."""
    bridged = Pair(
        a.value, Pair._bridge_numeric(a.formula), a._axis_bounds, steps=a.steps
    )
    return _sum(bridged)


def _all(a, axis=None, **kwargs):
    # m.all(): "cond holds at EVERY position" == count reached n
    if kwargs or axis is not None:
        raise NotImplementedError("all() kwargs/axis not supported yet")
    n = int(np.prod([hi - lo for lo, hi in a._axis_bounds]))
    return Pair(np.all(a.value), sympy.Eq(_count(a).formula, n), None, steps=a.steps)


def _any(a, axis=None, **kwargs):
    # m.any(): "cond holds SOMEWHERE" == count positive
    if kwargs or axis is not None:
        raise NotImplementedError("any() kwargs/axis not supported yet")
    return Pair(np.any(a.value), sympy.Gt(_count(a).formula, 0), None, steps=a.steps)


FUNCTION_TABLE[np.sum] = _sum
FUNCTION_TABLE[np.all] = _all
FUNCTION_TABLE[np.any] = _any
FUNCTION_TABLE[np.where] = _where
FUNCTION_TABLE[np.transpose] = lambda a, axes=None: (
    a.transpose(axes) if isinstance(a, Pair) else np.transpose(a, axes)
)
