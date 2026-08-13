"""Public API."""

import inspect

import numpy as np
import sympy

from .pair import _GUARDS, Pair
from .helpers import axis_idx


def to_sympy(fn, *args):
    """Run ``fn`` with tracing values.

    Array arguments become indexed formulas named after the function's
    parameters; float arguments become symbols of the same name. Ints,
    bools, strings and None pass through untraced: they are configuration
    (an ``n=``, an ``axis=``), not mathematics.
    Returns the traced result: ``.formula``, ``.value``, ``.domain``.
    """
    wrapped = [_wrap(name, val) for name, val in _infer_names(fn, args)]
    _GUARDS.clear()
    out = _repack(fn(*wrapped))
    if isinstance(out, Pair):
        # every branch taken during the trace, as one hypothesis: the
        # formula holds for inputs satisfying these preconditions
        out.preconditions = sympy.And(*_GUARDS) if _GUARDS else sympy.true
    return out


def _wrap(name, val):
    if val is None or isinstance(val, (bool, np.bool_, int, np.integer, str)):
        return val  # config, not math: np.diff(a, 2) keeps its plain 2
    if np.isscalar(val):
        return Pair(val, sympy.Symbol(name, real=True))
    return Pair.array(name, val)


def _repack(out):
    """Normalize the traced result to one object with .formula/.value/.domain.

    The fallback path (numpy's own bodies run on Pairs) returns an ndarray
    whose ELEMENTS are scalar Pairs, formulas unrolled per element. Repack
    into a single Pair: values as a real ndarray, formulas as a sympy.Array.
    """
    if isinstance(out, Pair):
        if out.domain is None and isinstance(out.formula, sympy.Add):
            folded = _fold_add(out.formula)
            if folded is not None:
                return Pair(out.value, folded, None)
        return out
    if isinstance(out, (int, float, complex, np.number)):
        # a guarded C algorithm (searchsorted's bisection, argmin, ...)
        # returned a plain number: a CONSTANT under the recorded branch
        # preconditions. Wrap it so .formula/.preconditions exist.
        return Pair(out, sympy.sympify(out), None)
    if not (isinstance(out, np.ndarray) and out.dtype == object):
        return out
    elements = out.ravel()
    if not all(isinstance(p, Pair) for p in elements):
        return out  # not ours: leave untouched
    values = np.array([p.value for p in elements]).reshape(out.shape)
    formulas = [p.formula for p in elements]
    if out.ndim == 1:
        general = _recompress(formulas)
        if general is not None:
            return Pair(values, general, domain=(0, len(formulas)))
    return Pair(
        values,
        sympy.Array(formulas, out.shape),
        domain=tuple((0, s) for s in out.shape),
    )


def _shift_indices(expr, offset):
    """u[0] - u[1]  ->  u[0 + offset] - u[1 + offset], every index shifted."""
    return expr.replace(
        lambda x: isinstance(x, sympy.Indexed),
        lambda x: x.base[tuple(e + offset for e in x.indices)],
    )


def _fold_add(expr):
    """Fold one big scalar Add back into Sum form. ONE generic algorithm:

        0.05*y[0] + 0.1*y[1] + ... + 0.1*y[6] + 0.05*y[7]
        -> 0.05*y[0] + Sum(0.1*y[j + 1], (j, 0, 5)) + 0.05*y[7]

    Terms are index-sorted, then scanned with three knobs: boundary
    budget (0..2 terms spared at each end), stride (1..3), and phases
    (stride-many interleaved subpatterns, e.g. Simpson's alternating
    4,2 weights). Every fold is PROVEN term by term; no proof, no fold.
    """
    terms = list(expr.args)
    keyed = []
    for t in terms:
        idxs = [e for a in t.atoms(sympy.Indexed) for e in a.indices]
        if not idxs:
            return None  # loose constants: out of scope, keep the Add
        if not all(e.is_Integer for e in idxs):
            return None  # already symbolic: nothing to fold
        keyed.append((min(idxs), t))
    keyed.sort(key=lambda kt: kt[0])
    terms = [t for _, t in keyed]
    n = len(terms)
    j = sympy.Symbol("j", integer=True)

    for lo in (0, 1, 2):
        for hi in (0, 1, 2):
            middle = terms[lo : n - hi]
            if len(middle) < 3:
                continue
            for stride in (1, 2, 3):
                phases = [middle[r::stride] for r in range(stride)]
                if any(len(ph) < 2 for ph in phases):
                    continue
                sums = []
                for ph in phases:
                    cand = _shift_indices(ph[0], stride * j)
                    if not all(
                        sympy.expand(cand.subs(j, k) - ph[k]) == 0
                        for k in range(len(ph))
                    ):
                        sums = None
                        break
                    sums.append(sympy.Sum(cand, (j, 0, len(ph) - 1)))
                if sums is not None:
                    return sympy.Add(*terms[:lo], *sums, *terms[n - hi :])
    return None


def _recompress(formulas):
    """Fold unrolled per-element formulas back into one indexed rule.

    [-u[0]+u[1], -u[1]+u[2], -u[2]+u[3]]  ->  -u[i] + u[i+1]

    The pattern is PROPOSED from element 0 and PROVEN by checking every
    element (exact sympy equality); no proof, no fold: returns None and
    the caller keeps the honest unrolled Array. Tries strides 1..3.
    """
    if len(formulas) < 2:
        return None
    i = axis_idx(0)
    for stride in (1, 2, 3):
        candidate = _shift_indices(formulas[0], stride * i)
        if all(
            sympy.expand(candidate.subs(i, k) - formulas[k]) == 0
            for k in range(len(formulas))
        ):
            return candidate

    # cumulative: elements that GROW (running sums) are prefix sums of a
    # shiftable difference.  cumtrapz: elem[k+1]-elem[k] = y[k+1]/2 + y[k+2]/2
    # folds by shift, so  elem(i) = elem[0] + Sum(difference(j), (j, 0, i-1)).
    diffs = [
        sympy.expand(formulas[k + 1] - formulas[k]) for k in range(len(formulas) - 1)
    ]
    if len(diffs) >= 2:
        j = sympy.Symbol("j", integer=True)
        candidate = _shift_indices(diffs[0], i)
        if all(
            sympy.expand(candidate.subs(i, k) - diffs[k]) == 0
            for k in range(len(diffs))
        ):
            rule = formulas[0] + sympy.Sum(_shift_indices(diffs[0], j), (j, 0, i - 1))
            # belt and braces: the assembled rule must reproduce EVERY element
            if all(
                sympy.expand(rule.subs(i, k).doit() - formulas[k]) == 0
                for k in range(len(formulas))
            ):
                return rule
    return None


def _infer_names(fn, args):
    """Pair each positional argument with its parameter name from fn's signature."""
    names = list(inspect.signature(fn).parameters)
    if len(args) > len(names):
        raise TypeError(f"{fn.__name__} takes {len(names)} arguments, got {len(args)}")
    return zip(names, args)
