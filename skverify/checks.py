"""Validators: pure functions formula -> Evidence.

Every check attempts a symbolic decision first and names the method that
decided it. Verdicts are three-valued; sympy's inability to decide maps
to "unknown", never to a verdict.
"""

from collections import namedtuple

import sympy

from .pair import Pair
from .helpers import _AXIS_SYMBOLS

Evidence = namedtuple("Evidence", "verdict method detail")

PROVEN = "proven"
REFUTED = "refuted"
UNKNOWN = "unknown"


def _formula(obj):
    return obj.formula if isinstance(obj, Pair) else sympy.sympify(obj)


def against(obj, reference):
    """Is the lifted formula equivalent to the reference expression?

    Decided by canonical difference; the residual names where they part.
    """
    diff = _formula(obj) - sympy.sympify(reference)
    residual = sympy.expand(diff.doit() if diff.has(sympy.Sum) else diff)
    if residual == 0:
        return Evidence(PROVEN, "canonical", 0)
    residual = sympy.simplify(residual)
    if residual == 0:
        return Evidence(PROVEN, "simplify", 0)
    if residual.is_number and residual != 0:
        return Evidence(REFUTED, "canonical", residual)
    if not residual.free_symbols:
        return Evidence(REFUTED, "canonical", residual)
    if residual.atoms(sympy.Indexed) or residual.atoms(sympy.Symbol):
        return Evidence(REFUTED, "residual", residual)
    return Evidence(UNKNOWN, "simplify", residual)


def conserves_mass(obj):
    """Do the update rule's coefficients sum to one?

    Setting every indexed value to 1 must leave exactly 1: the linear
    mass check for Markov/conservative updates, decided symbolically.
    """
    f = _formula(obj)
    ones = {a: sympy.Integer(1) for a in f.atoms(sympy.Indexed)}
    total = sympy.simplify(f.xreplace(ones))
    diff = sympy.simplify(total - 1)
    if diff == 0:
        return Evidence(PROVEN, "coefficient sum", 1)
    if diff.is_number or diff.free_symbols:
        # not identically one; the residual names the leak
        return Evidence(REFUTED, "coefficient sum", diff)
    return Evidence(UNKNOWN, "coefficient sum", total)


def centered(obj, at=0):
    """Is the stencil symmetric about offset `at` from the output point?

    Collects every index offset relative to the axis letters; the
    off-center KdV bug is the canonical refutation.
    """
    f = _formula(obj)
    letters = set(_AXIS_SYMBOLS)
    offsets = []
    for a in f.atoms(sympy.Indexed):
        for e in a.indices:
            free = e.free_symbols & letters
            if len(free) == 1:
                offsets.append(sympy.expand(e - free.pop()))
            elif not free and e.is_Integer:
                continue
            else:
                return Evidence(UNKNOWN, "offsets", e)
    if not offsets:
        return Evidence(UNKNOWN, "offsets", "no indexed terms with letters")
    if not all(o.is_Integer for o in offsets):
        return Evidence(UNKNOWN, "offsets", offsets)
    shifted = sorted(int(o) - at for o in offsets)
    if shifted == sorted(-v for v in shifted):
        return Evidence(PROVEN, "offset symmetry", shifted)
    return Evidence(REFUTED, "offset symmetry", shifted)
