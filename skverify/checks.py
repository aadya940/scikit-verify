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


def _scatter_positions(formula):
    """Concrete (row, col) writes read off a 2-D scatter formula.
    Returns None when the formula is not a concrete scatter."""
    from .helpers import axis_idx

    i, j = axis_idx(0), axis_idx(1)
    positions = []
    stack = [formula]
    while stack:
        f = stack.pop()
        if f == 0 or f == sympy.Integer(0):
            continue
        if isinstance(f, sympy.Piecewise):
            for expr, cond in f.args:
                if cond is sympy.true:
                    stack.append(expr)
                    continue
                eqs = cond.args if isinstance(cond, sympy.And) else (cond,)
                pos = {}
                ok = True
                for eq in eqs:
                    if (
                        isinstance(eq, sympy.Eq)
                        and eq.lhs in (i, j)
                        and eq.rhs.is_Integer
                    ):
                        pos[eq.lhs] = int(eq.rhs)
                    else:
                        ok = False
                if ok and i in pos and j in pos:
                    positions.append((pos[i], pos[j]))
                else:
                    return None
            continue
        return None  # a base layer that is not zero: unknown reach
    return positions


def banded(obj):
    """Is the traced 2-D array banded, and with what widths?

    The claim is read off the FORMULA's write conditions -- every
    position the scatter can touch -- not off the values, so it is a
    structural fact about the computation, proven for this trace."""
    formula = _formula(obj)
    positions = _scatter_positions(formula)
    if positions is None:
        return Evidence(UNKNOWN, "structure", "not a concrete scatter")
    if not positions:
        return Evidence(PROVEN, "structure", "zero matrix (band 0, 0)")
    kl = max((r - c for r, c in positions), default=0)
    ku = max((c - r for r, c in positions), default=0)
    return Evidence(
        PROVEN, "structure", f"banded ({max(kl, 0)}, {max(ku, 0)})"
    )
