"""The boundary between traced and raw values, in one place.

Every conversion where a :class:`~skverify.pair.Pair` meets plain
numpy, or plain numpy meets the formula lane, lives here. This is the
library's highest-risk surface: a conversion that silently drops a
formula produces a wrong certificate, and a conversion that leaks an
object-dtype array into compiled code produces the classic
"setting an array element with a sequence" wall. Consolidating the
rules makes them reviewable side by side.

Three directions of crossing:

``value_of``
    Anything -> the concrete numeric lane. Never fails; the value lane
    always exists.
``formula_of``
    Anything -> the symbolic lane. May refuse: a raw operand with no
    provable formula raises rather than guessing. Concrete constant
    arrays become disclosed ``const_n`` tables.
``numeric``
    Plain-number object arrays -> real float arrays. Lossless by
    definition (no Pairs inside means no formulas to lose), needed
    before any compiled entry point.
"""

import numpy as np
import sympy

from .helpers import axis_idx
from .session import current as _session


def value_of(x):
    """Concrete numeric content of any operand.

    Parameters
    ----------
    x : Pair, ndarray, list, tuple or scalar
        The operand as it appears at a boundary. Object arrays and
        sequences may hold Pairs at any positions.

    Returns
    -------
    ndarray or scalar
        Plain numeric data. Pairs contribute their value lane;
        containers are rebuilt with values in the Pairs' places.
    """
    import weakref

    from .pair import Pair

    if isinstance(x, Pair):
        v = x.value
        if isinstance(v, np.ndarray) and isinstance(x.formula, sympy.Basic):
            # remember which buffer carried which formula: an opaque
            # call receiving this exact array later can name its
            # operand symbolically instead of as an anonymous const.
            # Identity is verified through the weakref at lookup, so
            # a recycled id can never alias a different array.
            if len(_session.value_origins) < 4096:
                try:
                    _session.value_origins[id(v)] = (
                        weakref.ref(v),
                        x.formula,
                    )
                except TypeError:
                    pass
        return v
    if isinstance(x, np.ndarray) and x.dtype == object:
        elems = x.ravel()
        if any(isinstance(e, Pair) for e in elems):
            vals = [e.value if isinstance(e, Pair) else e for e in elems]
            return np.array(vals, dtype=float).reshape(x.shape)
    if isinstance(x, (list, tuple)) and any(isinstance(e, Pair) for e in x):
        return type(x)(value_of(e) for e in x)
    return x


def formula_of(x):
    """Symbolic content of any operand, or a loud refusal.

    Parameters
    ----------
    x : Pair, ndarray or scalar
        The operand. Arrays of Pairs must share one provable pattern;
        concrete numeric arrays become disclosed constant tables.

    Returns
    -------
    sympy.Expr
        The operand's formula.

    Raises
    ------
    NotImplementedError
        When no exact formula exists: a decompressed array with no
        provable pattern, or a raw non-uniform array too large to
        disclose as a table. Refusal is the contract; a guessed
        formula would be a wrong certificate.
    """
    from .pair import Pair

    if isinstance(x, Pair):
        return x.formula
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            elems = x.ravel()
            if all(isinstance(e, Pair) for e in elems):
                formulas = [e.formula for e in elems]
                if len(set(formulas)) == 1:
                    # a keepdims reduction: one scalar in a box
                    return formulas[0]
                from .api import _recompress

                rule = _recompress(formulas)
                if rule is not None:
                    return rule
                raise NotImplementedError(
                    "decompressed operand without a provable pattern"
                )
        vals = np.unique(x)
        if len(vals) == 1:
            # uniform field (zeros, ones, full): one clean constant
            item = vals.item()
            if isinstance(item, float) and np.isnan(item):
                return sympy.Symbol("NaN", real=True)
            return sympy.sympify(item)
        if x.dtype.kind in "fiub" and x.size <= 4096:
            # a concrete operand (filter kernels, weights): a named
            # table with its values disclosed, like gather tables
            name = f"const_{len(_session.opaque)}"
            table = sympy.IndexedBase(name)
            letters = tuple(axis_idx(ax) for ax in range(x.ndim))
            _session.opaque.append(
                (
                    name,
                    (("table", "concrete"),),
                    (str(table[letters]), f"{name} = {x.tolist()}"),
                )
            )
            return table[letters]
        raise NotImplementedError(
            "raw non-uniform ndarray operand, wrap it: Pair.array(name, x)"
        )
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        # IEEE NaN is sentinel semantics (empty-slice placeholders),
        # never mathematics. sympy's S.NaN detonates every relational
        # fold it later meets; an inert named symbol is honest, prints
        # as NaN, and only fails evaluation on branches that would
        # genuinely BE NaN -- where the value lane is NaN too.
        return sympy.Symbol("NaN", real=True)
    return sympy.sympify(x)


def repack(x):
    """One indexed Pair from a decompressed object array, or None.

    The inverse of decompression: when every element is a Pair and the
    element formulas share one provable pattern, the array becomes a
    single Pair carrying the recompressed indexed formula. Boundaries
    that need whole-array semantics (method-call reductions, compiled
    entries) try this FIRST -- succeeding keeps every downstream
    mechanism on the strong, indexed path.

    Parameters
    ----------
    x : ndarray
        Candidate object array.

    Returns
    -------
    Pair or None
        The repacked Pair, or None when no exact pattern exists
        (callers fall back to their own handling; refusal stays THEIR
        contract, not this helper's).
    """
    from .pair import Pair

    if not (
        isinstance(x, np.ndarray)
        and x.dtype == object
        and x.size
        and all(isinstance(e, Pair) for e in x.ravel())
    ):
        return None
    try:
        formula = formula_of(x)
    except NotImplementedError:
        return None
    return Pair(
        value_of(x),
        formula,
        tuple((0, int(n)) for n in x.shape),
        steps=tuple(x.ravel()),
    )


def numeric(v, copy=True):
    """Real float array from a plain-number object array.

    Parameters
    ----------
    v : ndarray or other
        Candidate array. Object dtype with no Pairs inside coerces;
        everything else passes through untouched.
    copy : bool, optional
        Whether non-object arrays are copied on the way through.

    Returns
    -------
    ndarray or other
        Numeric data safe to hand to compiled entry points.
    """
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            try:
                return np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                pass
        return np.array(v, copy=True) if copy else v
    return v
