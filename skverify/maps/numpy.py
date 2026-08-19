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
UFUNC_TABLE[np.mod] = sympy.Mod
UFUNC_TABLE[np.copysign] = lambda a, b: sympy.Abs(a) * sympy.Piecewise(
    (-1, b < 0), (1, True)
)

# comparisons spelled as functions: np.less(u, 0) etc.
UFUNC_TABLE[np.less] = sympy.Lt
UFUNC_TABLE[np.less_equal] = sympy.Le
UFUNC_TABLE[np.greater] = sympy.Gt
UFUNC_TABLE[np.greater_equal] = sympy.Ge
UFUNC_TABLE[np.equal] = sympy.Eq
UFUNC_TABLE[np.not_equal] = sympy.Ne

# clip(x, lo, hi) is a 3-input ufunc behind np.clip's dispatch
from numpy._core.umath import clip as _np_clip  # noqa: E402

UFUNC_TABLE[_np_clip] = lambda x, lo, hi: sympy.Min(sympy.Max(x, lo), hi)

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


def _where(cond, a=None, b=None):
    if a is None and b is None:
        # single-arg where is nonzero-position discovery: a concrete
        # fact about this trace, like searchsorted's bisection
        return np.where(np.asarray(Pair._value_of(cond), dtype=bool))
    domain = Pair._merge_domains(
        Pair._domain_of(cond), Pair._domain_of(a), Pair._domain_of(b)
    )
    cond_f = Pair._formula_of(cond)
    if not isinstance(
        cond_f, (sympy.logic.boolalg.Boolean, sympy.core.relational.Relational)
    ):
        cond_f = sympy.Ne(cond_f, 0)
    from ..pair import _piecewise_under_sum

    if _piecewise_under_sum(cond_f):
        # sympy's Piecewise ctor hoists such conditions through the Sum
        # bound (even with evaluate=False): the result leaks the bound
        # index and is WRONG. Loud refusal beats a wrong formula.
        raise NotImplementedError(
            "np.where over a Sum-of-Piecewise condition: sympy rewrites "
            "this incorrectly; restructure or apply the condition earlier"
        )
    return Pair(
        np.where(Pair._value_of(cond), Pair._value_of(a), Pair._value_of(b)),
        sympy.Piecewise((Pair._formula_of(a), cond_f), (Pair._formula_of(b), True)),
        domain,
        steps=Pair._steps_of(cond, a, b),
    )


def _held_sum(body, *limits):
    """Construct ``Sum(body, *limits)`` without hoisting re-evaluation.

    ``Sum.__new__`` runs ``piecewise_fold`` on the body, masking only
    Piecewise terms free of the NEW binder. A Piecewise whose condition
    references an INNER sum's bound variable passes that guard and gets
    hoisted through its own binder: the condition escapes and the
    formula is silently wrong (upstream sympy bug, caught by the
    two-lane fuzzer). For that hazardous shape the Sum is assembled by
    direct ``Expr.__new__``, skipping the constructor's preprocessing;
    everything else uses the normal constructor.
    """
    if body.has(sympy.Piecewise) and body.has(sympy.Sum):
        # Selector sums -- an inner Sum whose Piecewise conditions on
        # the inner sum's OWN dummy (Eq(k, 1) selectors from pinv-style
        # code) -- resolve exactly by unrolling: each term's condition
        # becomes concrete and collapses. Resolving them here removes
        # the shape sympy's ctor would corrupt.
        resolutions = {}
        for inner in body.atoms(sympy.Sum):
            dummies = {lim[0] for lim in inner.limits}
            if any(
                pw.has(*dummies)
                for pw in inner.function.atoms(sympy.Piecewise)
            ):
                try:
                    resolved = inner.doit()
                except Exception:
                    continue
                if not resolved.has(sympy.Piecewise):
                    resolutions[inner] = resolved
        if resolutions:
            body = body.xreplace(resolutions)
    built = sympy.Sum(body, *limits)
    if body.has(sympy.Piecewise) and body.has(sympy.Sum):
        # The precise corruption test: a hoist is only WRONG when it
        # frees a symbol that the input had bound (the condition
        # escaped its binder). Benign folds keep the free-symbol set.
        binders = {lim[0] for lim in built.limits} if hasattr(built, "limits") else set()
        expected_free = body.free_symbols - {l[0] for l in limits}
        escaped = (built.free_symbols | binders) - expected_free - {l[0] for l in limits}
        escaped = {e for e in escaped if e in built.free_symbols and e not in expected_free}
        if escaped:
            # a correct Sum could be BUILT by bypassing the ctor, but any
            # later doit/simplify re-runs the fold and corrupts it in the
            # user's hands. Refusal is the only safe output.
            raise NotImplementedError(
                "summing over a Piecewise bound inside an inner Sum: sympy's "
                f"piecewise_fold frees {sorted(map(str, escaped))} through "
                "its binder (upstream bug); restructure so the mask applies "
                "before the inner reduction"
            )
    return built


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
    if isinstance(a, Pair) and a.domain is None:
        return a  # the sum of a scalar is itself
    if not isinstance(a, Pair):
        if isinstance(a, np.ndarray) and a.dtype == object:
            total = a.ravel()[0]
            for e in a.ravel()[1:]:
                total = total + e  # element dunders keep the trace
            return total
        return np.sum(np.asarray(a))

    a = Pair(
        a.value, Pair._bridge_numeric(a.formula), a._axis_bounds, steps=(a,)
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
        formula = _held_sum(a.formula.xreplace(rename), (j, lo, hi - 1))
        new_bounds = bounds[:k] + bounds[k + 1 :]
        return Pair(
            np.sum(a.value, axis=k),
            formula,
            new_bounds or None,
            steps=(a,),
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
    formula = a.formula.xreplace(
        {axis_idx(ax): d for ax, d in enumerate(dummies)}
    )
    for ax in reversed(range(len(bounds))):
        lo, hi = bounds[ax]
        formula = _held_sum(formula, (dummies[ax], lo, hi - 1))  # inclusive
    return Pair(np.sum(a.value), formula, None, steps=(a,))


def _matmul(a, b):
    """Contraction as a Sum, numpy matmul semantics for every rank.

    A (n x m) @ B (m x p) -> Sum(A[i, k]*B[k, j], (k, 0, m-1))
    1-D operands lose their would-be axis; leading axes are batch dims
    and broadcast (an extent-1 batch axis indexes at 0).
    """
    bounds_a, bounds_b = Pair._domain_of(a), Pair._domain_of(b)
    if bounds_a is None or bounds_b is None:
        raise ValueError("matmul: both operands must be at least 1-D")
    na, nb = len(bounds_a), len(bounds_b)
    value = np.matmul(Pair._value_of(a), Pair._value_of(b))
    res_bounds = tuple((0, int(s)) for s in np.shape(value))
    a2, b2 = na >= 2, nb >= 2
    nbatch = len(res_bounds) - a2 - b2

    fa = Pair._bridge_numeric(Pair._formula_of(a))
    fb = Pair._bridge_numeric(Pair._formula_of(b))
    # the result letters SURVIVE in the formula (unlike _sum, which
    # substitutes every letter away), so the dummy must dodge them too
    taken = {s.name for s in (fa * fb).atoms(sympy.Symbol)}
    taken |= {axis_idx(ax).name for ax in range(len(res_bounds))}
    name, n = "k", 2
    while name in taken:
        name, n = f"k{n}", n + 1
    k = sympy.Symbol(name, integer=True)

    def batch_target(bounds, ax, pos):
        # operand batch axis -> result batch letter; extent 1 broadcasts,
        # so it indexes at 0 regardless of the result letter
        extent = bounds[ax][1] - bounds[ax][0]
        res_extent = res_bounds[pos][1] - res_bounds[pos][0]
        if extent == 1 and res_extent != 1:
            return sympy.Integer(0)
        return axis_idx(pos)

    sub_a = {}
    if a2:
        sub_a[axis_idx(na - 1)] = k
        sub_a[axis_idx(na - 2)] = axis_idx(nbatch)
        for ax in range(na - 2):
            sub_a[axis_idx(ax)] = batch_target(bounds_a, ax, nbatch - (na - 2) + ax)
    else:
        sub_a[axis_idx(0)] = k

    sub_b = {}
    if b2:
        sub_b[axis_idx(nb - 2)] = k
        sub_b[axis_idx(nb - 1)] = axis_idx(nbatch + a2)
        for ax in range(nb - 2):
            sub_b[axis_idx(ax)] = batch_target(bounds_b, ax, nbatch - (nb - 2) + ax)
    else:
        sub_b[axis_idx(0)] = k

    lo, hi = bounds_a[na - 1]
    formula = _held_sum(
        fa.xreplace(sub_a) * fb.xreplace(sub_b),
        (k, 0, hi - lo - 1),
    )
    return Pair(value, formula, res_bounds or None, steps=Pair._steps_of(a, b))


def _dot(a, b, out=None):
    if out is not None:
        raise NotImplementedError("np.dot out= not supported")
    if Pair._domain_of(a) is None or Pair._domain_of(b) is None:
        return a * b  # np.dot with a scalar multiplies
    if len(Pair._domain_of(a)) > 2 or len(Pair._domain_of(b)) > 2:
        raise NotImplementedError(
            "np.dot N-D contracts differently from matmul; use np.matmul or @"
        )
    return _matmul(a, b)


FUNCTION_TABLE[np.matmul] = _matmul
FUNCTION_TABLE[np.dot] = _dot


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


def _empty_like(a, dtype=None, **kwargs):
    bounds = Pair._domain_of(a)
    letters = tuple(axis_idx(ax) for ax in range(len(bounds)))
    return Pair(
        np.empty(tuple(hi - lo for lo, hi in bounds)),
        sympy.Function("uninitialized")(*letters),
        bounds,
    )


def _gradient(f, *varargs, axis=None, edge_order=1):
    if not isinstance(f, Pair) or len(f._axis_bounds) != 1:
        raise NotImplementedError("gradient: 1-D traced input only")
    if axis not in (None, 0, -1) or edge_order != 1:
        raise NotImplementedError("gradient: axis/edge_order not supported")
    if len(varargs) > 1 or (varargs and not np.isscalar(Pair._value_of(varargs[0]))):
        raise NotImplementedError("gradient: uniform scalar spacing only")
    dx = varargs[0] if varargs else 1.0
    out = np.zeros_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
    out[0] = (f[1] - f[0]) / dx
    out[-1] = (f[-1] - f[-2]) / dx
    return out


FUNCTION_TABLE[np.gradient] = _gradient


def _ascontiguousarray(a, dtype=None, **kwargs):
    if not isinstance(a, Pair):
        return np.ascontiguousarray(a, dtype=dtype)
    value = np.ascontiguousarray(a.value, dtype=dtype)
    return Pair(value, a.formula, a._axis_bounds, steps=(a,))


FUNCTION_TABLE[np.ascontiguousarray] = _ascontiguousarray
FUNCTION_TABLE[np.empty_like] = _empty_like
FUNCTION_TABLE[np.zeros_like] = _zeros_like
FUNCTION_TABLE[np.ones_like] = _ones_like
FUNCTION_TABLE[np.full_like] = _full_like


def _count(a):
    """Sum(Piecewise((1, cond), (0, True))) over the mask's domain."""
    bridged = Pair(
        a.value, Pair._bridge_numeric(a.formula), a._axis_bounds, steps=(a,)
    )
    return _sum(bridged)


def _all(a, axis=None, **kwargs):
    # m.all(): "cond holds at EVERY position" == count reached n
    if kwargs or axis is not None:
        raise NotImplementedError("all() kwargs/axis not supported yet")
    n = int(np.prod([hi - lo for lo, hi in a._axis_bounds]))
    return Pair(np.all(a.value), sympy.Eq(_count(a).formula, n), None, steps=(a,))


def _any(a, axis=None, **kwargs):
    # m.any(): "cond holds SOMEWHERE" == count positive
    if kwargs or axis is not None:
        raise NotImplementedError("any() kwargs/axis not supported yet")
    return Pair(np.any(a.value), sympy.Gt(_count(a).formula, 0), None, steps=(a,))


# closeness/finiteness are validation checks on the concrete lane,
# not mathematics: they return plain numpy results
def _concrete_check(np_fn):
    def check(*args, **kwargs):
        vals = [
            np.asarray(Pair._value_of(a), dtype=float)
            if isinstance(a, Pair)
            or (isinstance(a, np.ndarray) and a.dtype == object)
            else a
            for a in args
        ]
        return np_fn(*vals, **kwargs)

    return check


def _astype(a, dtype, copy=True, **kwargs):
    if isinstance(a, Pair):
        return a.astype(dtype)
    if (
        isinstance(a, np.ndarray)
        and a.dtype == object
        and any(isinstance(e, Pair) for e in a.ravel())
    ):
        return a  # traced elements: the cast is math-neutral
    return np.astype(np.asarray(a), dtype, copy=copy, **kwargs)


def _diag(v, k=0):
    if not isinstance(v, Pair) or k != 0:
        return np.diag(Pair._value_of(v), k)
    i, j = axis_idx(0), axis_idx(1)
    if v._axis_bounds is not None and len(v._axis_bounds) == 1:
        # vector -> diagonal matrix: D[i, j] = v[i] when i == j else 0
        n = v._axis_bounds[0]
        return Pair(
            np.diag(v.value),
            sympy.Piecewise((v.formula, sympy.Eq(i, j)), (0, True)),
            (n, n),
            steps=(v,),
        )
    if v._axis_bounds is not None and len(v._axis_bounds) == 2:
        # matrix -> its diagonal: d[i] = M[i, i]
        lo = min(hi - lo for lo, hi in v._axis_bounds)
        return Pair(
            np.diag(v.value),
            v.formula.subs(j, i),
            (0, lo),
            steps=(v,),
        )
    return np.diag(Pair._value_of(v), k)


def _mean(a, axis=None, **kwargs):
    if isinstance(a, Pair):
        return a.mean(axis=axis)
    if (
        isinstance(a, np.ndarray)
        and a.dtype == object
        and axis is None
        and any(isinstance(e, Pair) for e in a.ravel())
    ):
        return _sum(a) / a.size  # element dunders keep the trace
    return np.mean(np.asarray(Pair._value_of(a), dtype=float), axis=axis, **kwargs)


def _var(a, axis=None, ddof=0, correction=None, **kwargs):
    if correction is not None:
        ddof = correction  # the array-api spelling of ddof
    if isinstance(a, Pair) and axis == 0 and len(a._axis_bounds or ()) == 1:
        axis = None  # axis 0 of 1-D IS the whole array
    if isinstance(a, Pair) and axis is not None:
        raise NotImplementedError("var over one axis of N-D not supported yet")
    if not isinstance(a, Pair) or axis is not None:
        return np.var(np.asarray(Pair._value_of(a), dtype=float), axis=axis, ddof=ddof)
    n = int(np.prod([hi - lo for lo, hi in a._axis_bounds]))
    centered = a - a.mean()
    return _sum(centered * centered) / (n - ddof)


def _std(a, axis=None, ddof=0, correction=None, **kwargs):
    return _var(a, axis=axis, ddof=ddof, correction=correction, **kwargs) ** 0.5


def _median(a, axis=None, **kwargs):
    if not isinstance(a, Pair) or axis is not None or len(a._axis_bounds or ()) != 1:
        return np.median(np.asarray(Pair._value_of(a), dtype=float), axis=axis)
    from ..pair import _GUARDS

    vals = np.asarray(a.value, dtype=float)
    order = np.argsort(vals, kind="stable")
    sym = axis_idx(0)
    # the selection is path-scoped: the sorted order holds for THIS
    # input, recorded as explicit ordering preconditions
    for k in range(len(order) - 1):
        _GUARDS.append(
            sympy.Le(
                a.formula.subs(sym, int(order[k])),
                a.formula.subs(sym, int(order[k + 1])),
            )
        )
    mid = len(order) // 2
    if len(order) % 2:
        formula = a.formula.subs(sym, int(order[mid]))
    else:
        formula = (
            a.formula.subs(sym, int(order[mid - 1]))
            + a.formula.subs(sym, int(order[mid]))
        ) / 2
    return Pair(np.median(vals), formula, None, steps=(a,))


def _quantile_like(np_fn, scale):
    def entry(a, q, axis=None, **kwargs):
        if not isinstance(a, Pair) or axis is not None or len(a._axis_bounds or ()) != 1:
            return np_fn(np.asarray(Pair._value_of(a), dtype=float), q, axis=axis)
        from ..pair import _GUARDS

        vals = np.asarray(a.value, dtype=float)
        order = np.argsort(vals, kind="stable")
        sym = axis_idx(0)
        for k in range(len(order) - 1):
            _GUARDS.append(
                sympy.Le(
                    a.formula.subs(sym, int(order[k])),
                    a.formula.subs(sym, int(order[k + 1])),
                )
            )

        def one(qv):
            pos = (len(order) - 1) * float(qv) / scale
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            w = pos - lo
            f_lo = a.formula.subs(sym, int(order[lo]))
            f_hi = a.formula.subs(sym, int(order[hi]))
            return (1 - w) * f_lo + w * f_hi

        if np.ndim(q) == 0:
            return Pair(np_fn(vals, q), one(q), None, steps=(a,))
        formulas = [one(qv) for qv in np.asarray(q).ravel()]
        out = np.empty(len(formulas), dtype=object)
        values = np.asarray(np_fn(vals, q), dtype=float)
        for k, f in enumerate(formulas):
            out[k] = Pair(values[k], f, None, steps=(a,))
        return out

    return entry


FUNCTION_TABLE[np.percentile] = _quantile_like(np.percentile, 100.0)
FUNCTION_TABLE[np.quantile] = _quantile_like(np.quantile, 1.0)


def _round(a, decimals=0, **kwargs):
    if isinstance(a, Pair):
        raise NotImplementedError("rounding a traced value changes the math")
    return np.round(Pair._numeric(np.asarray(a), copy=False), decimals)


FUNCTION_TABLE[np.round] = _round
def _average(a, axis=None, weights=None, **kwargs):
    if not isinstance(a, Pair):
        arr = np.asarray(a)
        if (
            arr.dtype == object
            and axis is None
            and any(isinstance(e, Pair) for e in arr.ravel())
        ):
            elems = list(arr.ravel())
            if weights is None:
                return _sum(arr) / arr.size  # element dunders keep the trace
            wvals = np.asarray(Pair._value_of(weights)).ravel()
            num = elems[0] * float(wvals[0])
            for e, wv in zip(elems[1:], wvals[1:]):
                num = num + e * float(wv)
            return num / float(wvals.sum())
        return np.average(
            Pair._numeric(arr, copy=False), axis=axis, weights=weights
        )
    if weights is None:
        return a.mean(axis=axis)
    w = weights
    return _sum(a * w, axis=axis) / _sum(
        w if isinstance(w, Pair) else Pair(np.asarray(w), Pair._formula_of(w), a._axis_bounds),
        axis=axis,
    )


FUNCTION_TABLE[np.average] = _average
def _unique(ar, **kwargs):
    # which distinct values exist is a fact about THIS trace (label
    # sets, category counts): bookkeeping, not mathematics. Runs on
    # concrete values; downstream label handling gets plain numbers.
    return np.unique(np.asarray(Pair._value_of(ar), dtype=float), **kwargs)


def _concrete_inventory(np_fn):
    # membership and set-difference over label inventories: facts about
    # THIS trace, not mathematics (same doctrine as np.unique)
    def entry(*args, **kwargs):
        vals = [
            np.asarray(Pair._value_of(a), dtype=float)
            if not np.isscalar(a)
            else a
            for a in args
        ]
        return np_fn(*vals, **kwargs)

    return entry


FUNCTION_TABLE[np.isin] = _concrete_inventory(np.isin)
FUNCTION_TABLE[np.setdiff1d] = _concrete_inventory(np.setdiff1d)
FUNCTION_TABLE[np.union1d] = _concrete_inventory(np.union1d)
FUNCTION_TABLE[np.intersect1d] = _concrete_inventory(np.intersect1d)
FUNCTION_TABLE[np.unique] = _unique
FUNCTION_TABLE[np.median] = _median
FUNCTION_TABLE[np.mean] = _mean
FUNCTION_TABLE[np.var] = _var
FUNCTION_TABLE[np.std] = _std
def _broadcast_to(a, shape, **kwargs):
    if not isinstance(a, Pair):
        return np.broadcast_to(Pair._numeric(np.asarray(a), copy=False), shape)
    shape = tuple(int(n) for n in (shape if np.iterable(shape) else (shape,)))
    value = np.broadcast_to(a.value, shape).copy()
    bounds = a._axis_bounds or ()
    formula = Pair._shift_axes(a.formula, bounds, len(shape))
    merged = tuple((0, n) for n in shape)
    formula = Pair._pin_ones(formula, bounds, merged)
    return Pair(value, formula, merged, steps=(a,))


FUNCTION_TABLE[np.broadcast_to] = _broadcast_to
FUNCTION_TABLE[np.diag] = _diag
FUNCTION_TABLE[np.astype] = _astype
FUNCTION_TABLE[np.linalg.matrix_rank] = _concrete_check(np.linalg.matrix_rank)
FUNCTION_TABLE[np.linalg.svd] = lambda *a, **k: Pair._opaque_call(
    np.linalg.svd, a, k
)
FUNCTION_TABLE[np.linalg.pinv] = lambda *a, **k: Pair._opaque_call(
    np.linalg.pinv, a, k
)
FUNCTION_TABLE[np.allclose] = _concrete_check(np.allclose)
FUNCTION_TABLE[np.isclose] = _concrete_check(np.isclose)

FUNCTION_TABLE[np.sum] = _sum
FUNCTION_TABLE[np.clip] = lambda a, lo, hi, **kw: _np_clip(a, lo, hi)
FUNCTION_TABLE[np.all] = _all
FUNCTION_TABLE[np.any] = _any
FUNCTION_TABLE[np.where] = _where
FUNCTION_TABLE[np.transpose] = lambda a, axes=None: (
    a.transpose(axes) if isinstance(a, Pair) else np.transpose(a, axes)
)
