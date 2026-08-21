"""Public API."""

import inspect

import numpy as np
import sympy

from .pair import _GUARDS, _OPAQUE, Pair
from .session import current as _session
from .helpers import axis_idx


def to_sympy(fn, *args, **kwargs):
    """Run ``fn`` with tracing values.

    Array arguments become indexed formulas named after the function's
    parameters; float arguments become symbols of the same name. Ints,
    bools, strings and None pass through untraced: they are configuration
    (an ``n=``, an ``axis=``), not mathematics.
    Returns the traced result: ``.formula``, ``.value``, ``.domain``.
    """
    import sys

    from .instrument import instrument

    # scatter formulas nest one Piecewise per element write; sympy
    # recurses per level, so real-size traces need headroom
    if sys.getrecursionlimit() < 20000:
        sys.setrecursionlimit(20000)

    _session.reset()
    _session.instrumented = False
    # reset FIRST: _wrap records disclosures (integer-as-config) into
    # the session, and they must survive to the harvest
    if kwargs:
        # keyword arguments wrap by their own names
        args = args + tuple(kwargs.values())
        kw_wrapped = {k: _wrap(k, v) for k, v in kwargs.items()}
    else:
        kw_wrapped = {}
    wrapped = [
        _wrap(name, val)
        for name, val in _infer_names(fn, args[: len(args) - len(kw_wrapped)])
    ]
    sites = ()
    try:
        out = _repack(fn(*wrapped, **kw_wrapped))
    except (NotImplementedError, ValueError, TypeError, AttributeError):
        # a wall the plain trace cannot pass; retry a semantically
        # identical instrumented copy (math-neutral calls replaced)
        fn_run, sites = instrument(fn)
        _session.instrumented = True
        if not sites:
            import inspect as _inspect

            try:
                _inspect.getsource(fn)
            except (OSError, TypeError):
                raise type(sys.exc_info()[1])(
                    f"{sys.exc_info()[1]} \n[skverify] the trace hit a wall and "
                    "your function's source is unavailable (interactive shells "
                    "without source caching), so the instrumented retry cannot "
                    "run. Define the function in a .py file or notebook cell."
                ) from sys.exc_info()[1]
            raise
        _session.reset()
        out = _repack(fn_run(*wrapped, **kw_wrapped))
        if any("decorator unwrapped" in site for site in sites):
            # names propose, runs dispose: the wrapper must have been
            # neutral FOR THIS CALL -- rerun the real function on the
            # concrete inputs and compare
            reference = fn(*args)
            traced_value = Pair._value_of(out) if isinstance(out, Pair) else None
            if traced_value is not None and not np.allclose(
                np.asarray(traced_value, dtype=float),
                np.asarray(reference, dtype=float),
                equal_nan=True,
            ):
                raise NotImplementedError(
                    "a decorator was unwrapped but changed this call's "
                    "result; the wrapper is not math-neutral here"
                )
    try:
        # every branch taken during the trace, as one hypothesis: the
        # formula holds for inputs satisfying these preconditions.
        # Attached to whatever came back -- a Pair, or a library object
        # (BSpline) whose attributes carry the traced Pairs
        for pending in _session.pending_mask_guards.values():
            # gathers no reduction fused: their selection facts are
            # preconditions after all
            _GUARDS.extend(pending)
        _session.pending_mask_guards.clear()
        if _session.recurrences:
            # folded loops traced through featherweight symbols whose
            # meanings live in the definitions map. Small certificates
            # inline (a toy loop should read as its Iterate directly);
            # large ones STAY FOLDED -- the lemma structure is the
            # readable form, and xreplace(out.definitions) rebuilds
            # the monolith for anyone who wants it
            from .recurrence import inline

            rec_map = dict(_session.recurrences)
            total = sum(sympy.count_ops(v) for v in rec_map.values())
            if hasattr(out, "formula") and isinstance(out.formula, sympy.Basic):
                total += sympy.count_ops(out.formula)
            if total < 2000:
                if hasattr(out, "formula") and isinstance(out.formula, sympy.Basic):
                    f = out.formula
                    if isinstance(f, sympy.NDimArray):
                        # per element: the container must be rebuilt
                        # EVALUATED or its loop size stays a held Mul
                        out.formula = sympy.ImmutableDenseNDimArray(
                            [inline(e, rec_map) for e in f], f.shape
                        )
                    else:
                        out.formula = inline(f, rec_map)
                for i, g in enumerate(_GUARDS):
                    if isinstance(g, sympy.Basic):
                        _GUARDS[i] = inline(g, rec_map)
                out.definitions = {}
            else:
                # only definitions the certificate actually REACHES:
                # the transitive closure from formula and guards. The
                # repair map holds every probe symbol ever planted;
                # unreferenced ones are internal bookkeeping, not
                # certificate content.
                roots = set()
                if hasattr(out, "formula"):
                    f = out.formula
                    elements = (
                        list(f) if isinstance(f, sympy.NDimArray) else [f]
                    )
                    for e in elements:
                        if isinstance(e, sympy.Basic):
                            roots |= e.free_symbols
                for g in _GUARDS:
                    if isinstance(g, sympy.Basic):
                        roots |= g.free_symbols
                needed = {}
                frontier = roots & set(rec_map)
                while frontier:
                    sym = frontier.pop()
                    if sym in needed:
                        continue
                    needed[sym] = rec_map[sym]
                    frontier |= rec_map[sym].free_symbols & set(rec_map) - set(needed)
                out.definitions = needed
        else:
            out.definitions = {}
        out.preconditions = sympy.And(*_GUARDS) if _GUARDS else sympy.true
        records = list(_OPAQUE)
        if _session.hashed:
            # traced values used as identities (dict keys, set members):
            # ONE disclosed line, not n^2 equality guards
            shown = sorted(_session.hashed)[:64]
            records.append(
                (
                    "used_as_identity",
                    (("identity", "concrete"),),
                    ("hashed values", "; ".join(f"{f} = {v}" for f, v in shown)),
                )
            )
        out.unchecked = tuple(records)
        out.instrumented = sites
    except (AttributeError, TypeError):
        pass  # slots-only/immutable results keep their trace in skverify.pair._OPAQUE
    return out


def _wrap(name, val):
    if val is None or isinstance(val, (bool, np.bool_, str)):
        return val
    if isinstance(val, (int, np.integer)):
        # config, not math: np.diff(a, 2) keeps its plain 2. But the
        # assumption must be DISCLOSED -- an integer that was really
        # data would otherwise certify a constant silently
        _session.opaque.append(
            (
                f"arg:{name}",
                (("integer argument", "traced as constant"),),
                (
                    f"{name} = {val}",
                    "integers are treated as configuration; pass a "
                    "float to trace symbolically",
                ),
            )
        )
        return val
    if np.isscalar(val):
        return Pair(val, sympy.Symbol(name, real=True))
    if isinstance(val, np.ndarray):
        # remembered by NAME: an opaque atom receiving a foreign copy
        # of this input can disclose the observed equality
        _session.inputs = getattr(_session, "inputs", {})
        _session.inputs[name] = np.array(val, copy=True)
    if hasattr(val, "to_numpy") and not isinstance(val, np.ndarray):
        # a DataFrame or Series: welcome it, keep the parameter's name
        val = val.to_numpy(dtype=float)
    return Pair.array(name, val)


def _repack(out):
    """Normalize the traced result to one object with .formula/.value/.domain.

    The fallback path (numpy's own bodies run on Pairs) returns an ndarray
    whose ELEMENTS are scalar Pairs, formulas unrolled per element. Repack
    into a single Pair: values as a real ndarray, formulas as a sympy.Array.
    """
    if isinstance(out, Pair):
        if out.domain is None and isinstance(out.formula, (sympy.Add, sympy.Mul)):
            folded = _fold_poly(out.formula)
            if folded is None and isinstance(out.formula, sympy.Add):
                folded = _fold_add(out.formula)
            if folded is None and sympy.count_ops(out.formula) < 2000:
                # expand is multinomial: a 4th power of a 50-term sum
                # would be millions of terms. Big formulas stay factored
                expanded = sympy.expand(out.formula)
                if isinstance(expanded, sympy.Add):
                    folded = _fold_add(expanded)
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
    if not all(
        isinstance(p, Pair) or isinstance(p, (int, float, np.number))
        for p in elements
    ):
        return out  # not ours: leave untouched
    if not any(isinstance(p, Pair) for p in elements):
        return out
    # plain numbers riding along (a column of ones from add_constant)
    # are constants of the formula, not foreign objects
    values = np.array(
        [p.value if isinstance(p, Pair) else p for p in elements]
    ).reshape(out.shape)
    formulas = [
        p.formula if isinstance(p, Pair) else sympy.sympify(p)
        for p in elements
    ]
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
    """u[0] - u[1] -> u[offset] - u[1 + offset]. Only concrete integer
    indices move; symbolic letters (a surviving row index) stay put."""
    return expr.replace(
        lambda x: isinstance(x, sympy.Indexed),
        lambda x: x.base[tuple(e + offset if e.is_Integer else e for e in x.indices)],
    )


def _shift_slot(expr, offset, slot):
    """Shift index position `slot` only: y[0, 3] -> y[0 + offset, 3]."""

    def shifted(x):
        idx = list(x.indices)
        if slot < len(idx):
            idx[slot] = idx[slot] + offset
        return x.base[tuple(idx)]

    return expr.replace(lambda x: isinstance(x, sympy.Indexed), shifted)


def _fold_poly(expr):
    """Horner nests fold through their polynomial coefficients.

    ((c[0]*x + c[1])*x + c[2])  ->  Sum(c[j]*x**(2 - j), (j, 0, 2))

    Proven, not guessed: sympy.Poly extracts the coefficient list and the
    fold happens only if it is exactly c[0], c[1], ..., c[n-1] of one base.
    """
    from .helpers import _AXIS_SYMBOLS

    indexed = list(expr.atoms(sympy.Indexed))
    if not indexed:
        return None
    bases = {a.base for a in indexed}
    if len(bases) != 1:
        return None
    base = bases.pop()
    labels = {sympy.Symbol(str(b.base)) for b in indexed}
    xs = [
        s
        for s in expr.free_symbols
        if isinstance(s, sympy.Symbol) and s not in _AXIS_SYMBOLS and s not in labels
    ]
    if len(xs) != 1:
        return None
    x = xs[0]
    try:
        coeffs = sympy.Poly(expr, x).all_coeffs()
    except sympy.PolynomialError:
        return None
    n = len(coeffs)
    if n < 3:
        return None
    if any(coeffs[k] != base[k] for k in range(n)):
        return None
    j = sympy.Symbol("j", integer=True)
    return sympy.Sum(base[j] * x ** (n - 1 - j), (j, 0, n - 1))


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
        idxs = [e for a in t.atoms(sympy.Indexed) for e in a.indices if e.is_Integer]
        if not idxs:
            return None  # loose constants or fully symbolic: nothing to fold
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
    if any(sympy.count_ops(f) > 1500 for f in formulas):
        # proof-by-expand is superlinear; on big elements (folded-loop
        # results embedding recurrence state) the honest Array wins
        return None
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

    # per-slot: elements that differ only in ONE index position, e.g. the
    # per-row integrals of a 2-D reduction: e_r = 0.05*y[r,0] + ...
    slots = {len(a.indices) for f in formulas for a in f.atoms(sympy.Indexed)}
    if slots and max(slots) > 1:
        for slot in range(max(slots)):
            candidate = _shift_slot(formulas[0], i, slot)
            if all(
                sympy.expand(candidate.subs(i, k) - formulas[k]) == 0
                for k in range(len(formulas))
            ):
                if isinstance(candidate, sympy.Add):
                    inner = _fold_add(candidate)
                    if inner is not None:
                        return inner
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
    params = inspect.signature(fn).parameters
    names = []
    for name, p in params.items():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            # *args: one name per remaining argument
            names.extend(f"{name}{k}" for k in range(len(args) - len(names)))
            break
        names.append(name)
    if len(args) > len(names):
        raise TypeError(f"{fn.__name__} takes {len(names)} arguments, got {len(args)}")
    return zip(names, args)
