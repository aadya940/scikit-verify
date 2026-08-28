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
UFUNC_TABLE[np.fabs] = sympy.Abs
UFUNC_TABLE[np.log1p] = lambda x: sympy.log(1 + x)
UFUNC_TABLE[np.expm1] = lambda x: sympy.exp(x) - 1
UFUNC_TABLE[np.log2] = lambda x: sympy.log(x) / sympy.log(2)
UFUNC_TABLE[np.log10] = lambda x: sympy.log(x) / sympy.log(10)
UFUNC_TABLE[np.exp2] = lambda x: 2**x
UFUNC_TABLE[np.cbrt] = lambda x: sympy.real_root(x, 3)
UFUNC_TABLE[np.square] = lambda x: x**2
UFUNC_TABLE[np.reciprocal] = lambda x: 1 / x
UFUNC_TABLE[np.radians] = lambda x: x * sympy.pi / 180
UFUNC_TABLE[np.degrees] = lambda x: x * 180 / sympy.pi
UFUNC_TABLE[np.deg2rad] = lambda x: x * sympy.pi / 180
UFUNC_TABLE[np.rad2deg] = lambda x: x * 180 / sympy.pi
UFUNC_TABLE[np.trunc] = lambda x: sympy.sign(x) * sympy.floor(sympy.Abs(x))
UFUNC_TABLE[np.hypot] = lambda a, b: sympy.sqrt(a**2 + b**2)
UFUNC_TABLE[np.logaddexp] = lambda a, b: sympy.log(sympy.exp(a) + sympy.exp(b))
UFUNC_TABLE[np.logaddexp2] = lambda a, b: sympy.log(2**a + 2**b) / sympy.log(2)
UFUNC_TABLE[np.ldexp] = lambda x, n: x * 2**n
UFUNC_TABLE[np.fmod] = lambda a, b: sympy.sign(a) * sympy.Mod(sympy.Abs(a), sympy.Abs(b))
UFUNC_TABLE[np.positive] = lambda x: x
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
    if any(_is_bag(x) for x in (cond, a, b)):
        # decompressed operands have no single indexed pattern; the
        # selection is still exact per element
        cv = np.asarray(Pair._value_of(cond))
        shape = np.broadcast_shapes(
            np.shape(cv), np.shape(Pair._value_of(a)), np.shape(Pair._value_of(b))
        )

        def elem(x, idx):
            arr = np.broadcast_to(np.asarray(x, dtype=object) if _is_bag(x) or isinstance(x, np.ndarray) else np.asarray(Pair._value_of(x)), shape) if not isinstance(x, Pair) else None
            if isinstance(x, Pair):
                return x[idx] if x._axis_bounds else x
            return arr[idx]

        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            c_e, a_e, b_e = (elem(x, idx) for x in (cond, a, b))
            c_f = Pair._formula_of(c_e)
            if not isinstance(
                c_f,
                (sympy.logic.boolalg.Boolean, sympy.core.relational.Relational),
            ):
                c_f = sympy.Ne(c_f, 0)
            out[idx] = Pair(
                np.where(
                    Pair._value_of(c_e), Pair._value_of(a_e), Pair._value_of(b_e)
                )[()],
                sympy.Piecewise(
                    (Pair._formula_of(a_e), c_f), (Pair._formula_of(b_e), True)
                ),
                None,
            )
        return out
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


def _prune_dead_branches(expr):
    """Drop Piecewise branches whose condition is self-contradictory
    (Eq(k, 0) & Eq(k, 1) from products of sparse scatter rows). sympy
    keeps these dead branches, and chained matmuls multiply them
    combinatorially; pruning restores the sparsity the code had."""

    def contradictory(cond):
        if not isinstance(cond, sympy.And):
            return False
        pinned = {}
        for c in cond.args:
            if isinstance(c, sympy.Eq):
                lhs, rhs = c.args
                if rhs.is_number:
                    if lhs in pinned and pinned[lhs] != rhs:
                        return True
                    pinned[lhs] = rhs
        return False

    def prune(pw):
        kept = [
            (v, c) for v, c in pw.args if not contradictory(c)
        ]
        if len(kept) == len(pw.args):
            return pw
        if not kept:
            return sympy.Integer(0)
        return sympy.Piecewise(*kept)

    return expr.replace(
        lambda e: isinstance(e, sympy.Piecewise), prune
    )


def _held_sum(body, *limits):
    """Construct ``Sum(body, *limits)`` without hoisting re-evaluation.

    ``Sum.__new__`` runs ``piecewise_fold`` on the body, masking only
    Piecewise terms free of the NEW binder. A Piecewise whose condition
    references an INNER sum's bound variable passes that guard and gets
    hoisted through its own binder: the condition escapes and the
    formula is silently wrong (upstream sympy bug, caught by the
    two-lane fuzzer). Selector sums (inner Sum over its own dummy's
    Piecewise) are resolved exactly by unrolling (doit + piecewise_fold
    + prune) before construction; the Sum is then built via the normal
    constructor. Any residual binder escape is detected by free-symbol
    comparison and refused — a bypass via ``Expr.__new__`` would be
    momentarily correct but later doit/simplify would re-corrupt, so
    refusal is the only safe output.
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
                extents = [
                    lim[2] - lim[1] + 1 for lim in inner.limits
                    if lim[2].is_number and lim[1].is_number
                ]
                if len(extents) != len(inner.limits) or any(
                    e > 64 for e in extents
                ):
                    continue  # symbolic or large extent: unroll refused
                try:
                    # fold FIRST so branch conditions materialize as
                    # And-conjunctions, then prune the contradictions:
                    # each entry collapses to its true sparsity before
                    # the next chain level can multiply dead branches
                    resolved = _prune_dead_branches(
                        sympy.piecewise_fold(inner.doit())
                    )
                except Exception:
                    continue
                if not resolved.has(sympy.Sum):
                    # the inner binder is gone: any Piecewise left
                    # conditions only on outer indices, the single-Sum
                    # shape the guarded constructor handles
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


def _held_prod(body, *limits):
    """Construct ``Product(body, *limits)`` without hoisting re-evaluation.

    ``Product.__new__`` may run ``piecewise_fold`` on the body, masking only
    Piecewise terms free of the NEW binder. A Piecewise whose condition
    references an INNER product's bound variable passes that guard and gets
    hoisted through its own binder: the condition escapes and the
    formula is silently wrong (upstream sympy bug, same family as the Sum
    case). Selector products (inner Product over its own dummy's Piecewise)
    are resolved exactly by unrolling (doit + piecewise_fold + prune) before
    construction; the Product is then built via the normal constructor. Any
    residual binder escape is detected by free-symbol comparison and refused
    — a bypass via ``Expr.__new__`` would be momentarily correct but later
    doit/simplify would re-corrupt, so refusal is the only safe output.
    """
    if body.has(sympy.Piecewise) and body.has(sympy.Product):
        # Selector products -- an inner Product whose Piecewise conditions on
        # the inner product's OWN dummy -- resolve exactly by unrolling.
        resolutions = {}
        for inner in body.atoms(sympy.Product):
            dummies = {lim[0] for lim in inner.limits}
            if any(
                pw.has(*dummies)
                for pw in inner.function.atoms(sympy.Piecewise)
            ):
                extents = [
                    lim[2] - lim[1] + 1 for lim in inner.limits
                    if lim[2].is_number and lim[1].is_number
                ]
                if len(extents) != len(inner.limits) or any(
                    e > 64 for e in extents
                ):
                    continue  # symbolic or large extent: unroll refused
                try:
                    resolved = _prune_dead_branches(
                        sympy.piecewise_fold(inner.doit())
                    )
                except Exception:
                    continue
                if not resolved.has(sympy.Product):
                    resolutions[inner] = resolved
        if resolutions:
            body = body.xreplace(resolutions)
    built = sympy.Product(body, *limits)
    if body.has(sympy.Piecewise) and body.has(sympy.Product):
        binders = {lim[0] for lim in built.limits} if hasattr(built, "limits") else set()
        expected_free = body.free_symbols - {l[0] for l in limits}
        escaped = (built.free_symbols | binders) - expected_free - {l[0] for l in limits}
        escaped = {e for e in escaped if e in built.free_symbols and e not in expected_free}
        if escaped:
            raise NotImplementedError(
                "multiplying over a Piecewise bound inside an inner Product: sympy's "
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


def _masked_fuse(a):
    """(source, mask) provenance of a mask gather, popping its lazy
    guards: the reduction that calls this absorbs the mask into the
    formula, so the guards are not needed."""
    prov = getattr(a, "_mask_prov", None)
    if prov is None:
        return None
    from ..session import current as _session

    _session.pending_mask_guards.pop(id(a), None)
    return prov


def _prod(a, axis=None, **kwargs):
    prov = _masked_fuse(a) if isinstance(a, Pair) else None
    if prov is not None and axis is None:
        src, mask = prov
        bounds = src._axis_bounds
        if bounds is not None and len(bounds) == 1:
            lo, hi = bounds[0]
            i0 = axis_idx(0)
            j = _fresh_dummy(sympy.Tuple(src.formula, mask.formula), 1)
            body = sympy.Piecewise(
                (src.formula.xreplace({i0: j}), mask.formula.xreplace({i0: j})),
                (1, True),
            )
            return Pair(
                np.prod(np.asarray(Pair._value_of(a.value), dtype=float)),
                sympy.Product(body, (j, lo, hi - 1)),
                None,
                steps=(a,),
            )
    return _prod_plain(a, axis=axis, **kwargs)


def _prod_plain(a, axis=None, **kwargs):
    if isinstance(axis, tuple) and len(axis) == 1:
        axis = axis[0]  # scipy normalizes axis to tuples internally
    # numpy passes np._NoValue sentinels for unset options
    kwargs = {
        k: v for k, v in kwargs.items() if v is not np._NoValue
    }
    out = kwargs.pop("out", None)
    if out is not None:
        r = _prod_plain(a, axis=axis, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, Pair):
            if isinstance(out.value, np.ndarray):
                out.value[...] = Pair._value_of(r)
            else:
                out.value = Pair._value_of(r)
            out.formula = Pair._formula_of(r)
            out._axis_bounds = getattr(r, "_axis_bounds", None)
            return out
        raise NotImplementedError(
            "reduction with out= into an untraced buffer: later reads "
            "of the buffer would silently lose the formula"
        )
    where = kwargs.pop("where", True)
    if where is not True:
        # a masked product IS the product of the masked selection, with
        # identity 1 for excluded elements (contrast sum's 0)
        if not isinstance(where, (Pair, np.ndarray)):
            where = np.asarray(where)
        return _prod_plain(
            _where(where, a, 1.0), axis=axis, **kwargs
        )
    keepdims = kwargs.pop("keepdims", False)
    dt = kwargs.pop("dtype", None)
    if dt is not None and np.dtype(dt).kind not in "fc" and np.dtype(dt) != object:
        raise NotImplementedError("np.prod with a non-float dtype changes the math")
    if kwargs:
        raise NotImplementedError(f"np.prod kwargs {list(kwargs)} not supported")
    if keepdims:
        r = _prod_plain(a, axis=axis)  # refuses first on unsupported axes
        if isinstance(axis, tuple):
            raise NotImplementedError("np.prod keepdims with axis tuples")
        if isinstance(r, Pair):
            nd = np.ndim(Pair._value_of(a))
            shape = [1] * nd if axis is None else [
                1 if ax == (axis % nd) else n
                for ax, n in enumerate(np.shape(Pair._value_of(a)))
            ]
            v = np.reshape(np.asarray(r.value), shape)
            return Pair(v, r.formula, tuple((0, int(n)) for n in shape), steps=(r,))
        return np.reshape(r, [1] * np.ndim(Pair._value_of(a))) if axis is None else r

    if isinstance(a, Pair) and a.domain is None:
        return a  # the product of a scalar is itself
    if not isinstance(a, Pair):
        if isinstance(a, np.ndarray) and a.dtype == object:
            if axis is None or a.ndim == 1:
                elems = list(a.ravel())
                if not elems:
                    return np.prod(a, axis=axis)
                out = elems[0]
                for e in elems[1:]:
                    out = out * e  # element dunders keep the trace
                return out
            # per-axis on a bag: multiply each 1-D slice along the axis
            ax = axis % a.ndim
            out_shape = a.shape[:ax] + a.shape[ax + 1 :]
            out = np.empty(out_shape, dtype=object)
            for oidx in np.ndindex(out_shape):
                idx = oidx[:ax] + (slice(None),) + oidx[ax:]
                lane = a[idx]
                if len(lane) == 0:
                    out[oidx] = np.prod(lane)
                    continue
                product = lane[0]
                for e in lane[1:]:
                    product = product * e
                out[oidx] = product
            return out
        return np.prod(a, axis=axis)

    a = Pair(
        a.value, Pair._bridge_numeric(a.formula), a._axis_bounds, steps=(a,)
    )  # np.prod(u > 0) counts with 0/1 via Product's Piecewise bridge
    bounds = a._axis_bounds
    if isinstance(axis, tuple):
        raise NotImplementedError("axis tuples not supported yet")
    if axis is not None and not (axis == 0 and len(bounds) == 1):
        # per-axis reduction: bind ONE letter, survivors renumber down
        # p (3x4), axis=0:  Product(p[j, i], (j, 0, 2))   domain (0, 4)
        # p (3x4), axis=1:  Product(p[i, j], (j, 0, 3))   domain (0, 3)
        k = axis % len(bounds)
        j = _fresh_dummy(a.formula, len(bounds))
        rename = {axis_idx(k): j}
        rename.update(
            {axis_idx(ax): axis_idx(ax - 1) for ax in range(k + 1, len(bounds))}
        )
        lo, hi = bounds[k]
        formula = _held_prod(a.formula.xreplace(rename), (j, lo, hi - 1))
        new_bounds = bounds[:k] + bounds[k + 1 :]
        return Pair(
            np.prod(a.value, axis=k),
            formula,
            new_bounds or None,
            steps=(a,),
        )

    # one Product per axis, innermost axis innermost:
    # 1-D: Product(p[j], (j, 0, n-1))                      (unchanged output)
    # 2-D: Product(Product(p[j0, j1], (j1, 0, m-1)), (j0, 0, n-1))
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
        formula = _held_prod(formula, (dummies[ax], lo, hi - 1))  # inclusive
    return Pair(np.prod(a.value), formula, None, steps=(a,))


def _clip_entry(a, a_min=None, a_max=None, **kwargs):
    """np.clip is Max(lo, Min(x, hi)), exactly."""
    kwargs = {k: v for k, v in kwargs.items() if v is not np._NoValue}
    if kwargs:
        raise NotImplementedError(f"clip kwargs {list(kwargs)} not supported")
    out = a
    if a_max is not None:
        out = np.minimum(out, a_max)
    if a_min is not None:
        out = np.maximum(out, a_min)
    return out


def _sum(a, axis=None, **kwargs):
    prov = _masked_fuse(a) if isinstance(a, Pair) else None
    if prov is not None and axis is None:
        src, mask = prov
        bounds = src._axis_bounds
        if bounds is not None and len(bounds) == 1:
            lo, hi = bounds[0]
            i0 = axis_idx(0)
            j = _fresh_dummy(sympy.Tuple(src.formula, mask.formula), 1)
            body = sympy.Piecewise(
                (src.formula.xreplace({i0: j}), mask.formula.xreplace({i0: j})),
                (0, True),
            )
            return Pair(
                np.sum(np.asarray(Pair._value_of(a.value), dtype=float)),
                sympy.Sum(body, (j, lo, hi - 1)),
                None,
                steps=(a,),
            )
    return _sum_plain(a, axis=axis, **kwargs)


def _sum_plain(a, axis=None, **kwargs):
    if isinstance(axis, tuple) and len(axis) == 1:
        axis = axis[0]  # scipy normalizes axis to tuples internally
    # numpy passes np._NoValue sentinels for unset options
    kwargs = {
        k: v for k, v in kwargs.items() if v is not np._NoValue
    }
    out = kwargs.pop("out", None)
    if out is not None:
        r = _sum_plain(a, axis=axis, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, Pair):
            if isinstance(out.value, np.ndarray):
                out.value[...] = Pair._value_of(r)
            else:
                out.value = Pair._value_of(r)
            out.formula = Pair._formula_of(r)
            out._axis_bounds = getattr(r, "_axis_bounds", None)
            return out
        raise NotImplementedError(
            "reduction with out= into an untraced buffer: later reads "
            "of the buffer would silently lose the formula"
        )
    where = kwargs.pop("where", True)
    if where is not True:
        # a masked sum IS the sum of the masked selection, exactly
        if not isinstance(where, (Pair, np.ndarray)):
            where = np.asarray(where)
        return _sum_plain(
            _where(where, a, 0.0), axis=axis, **kwargs
        )
    keepdims = kwargs.pop("keepdims", False)
    dt = kwargs.pop("dtype", None)
    if dt is not None and np.dtype(dt).kind not in "fc" and np.dtype(dt) != object:
        raise NotImplementedError("np.sum with a non-float dtype changes the math")
    if kwargs:
        raise NotImplementedError(f"np.sum kwargs {list(kwargs)} not supported")
    if keepdims:
        r = _sum_plain(a, axis=axis)  # refuses first on unsupported axes
        if isinstance(axis, tuple):
            raise NotImplementedError("np.sum keepdims with axis tuples")
        if isinstance(r, Pair):
            nd = np.ndim(Pair._value_of(a))
            shape = [1] * nd if axis is None else [
                1 if ax == (axis % nd) else n
                for ax, n in enumerate(np.shape(Pair._value_of(a)))
            ]
            v = np.reshape(np.asarray(r.value), shape)
            return Pair(v, r.formula, tuple((0, int(n)) for n in shape), steps=(r,))
        return np.reshape(r, [1] * np.ndim(Pair._value_of(a))) if axis is None else r

    if isinstance(a, Pair) and a.domain is None:
        return a  # the sum of a scalar is itself
    if not isinstance(a, Pair):
        if isinstance(a, np.ndarray) and a.dtype == object:
            if axis is None or a.ndim == 1:
                total = a.ravel()[0]
                for e in a.ravel()[1:]:
                    total = total + e  # element dunders keep the trace
                return total
            # per-axis on a bag: sum each 1-D slice along the axis
            ax = axis % a.ndim
            out_shape = a.shape[:ax] + a.shape[ax + 1 :]
            out = np.empty(out_shape, dtype=object)
            for oidx in np.ndindex(out_shape):
                idx = oidx[:ax] + (slice(None),) + oidx[ax:]
                lane = a[idx]
                total = lane[0]
                for e in lane[1:]:
                    total = total + e
                out[oidx] = total
            return out
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


def _is_bag(x):
    return (
        isinstance(x, np.ndarray)
        and x.dtype == object
        and any(isinstance(e, Pair) for e in x.ravel())
    )


def _bag_matmul(a, b):
    """matmul on decompressed operands: explicit per-element sums.

    A bag has no single indexed pattern (rows may mix unrelated
    formulas), but the contraction is still exact arithmetic on the
    element Pairs -- Pair dunders carry both lanes."""
    A = np.atleast_2d(np.asarray(a, dtype=object))
    B = np.asarray(b, dtype=object)
    one_d_b = B.ndim == 1
    if one_d_b:
        B = B[:, None]
    n, m = A.shape
    m2, p = B.shape
    out = np.empty((n, p), dtype=object)
    for i in range(n):
        for j in range(p):
            acc = A[i, 0] * B[0, j]
            for kk in range(1, m):
                acc = acc + A[i, kk] * B[kk, j]
            out[i, j] = acc
    if np.ndim(a) == 1:
        out = out[0]
    if one_d_b:
        out = out[..., 0] if out.ndim else out
    return out


def _matmul(a, b):
    """Contraction as a Sum, numpy matmul semantics for every rank.

    A (n x m) @ B (m x p) -> Sum(A[i, k]*B[k, j], (k, 0, m-1))
    1-D operands lose their would-be axis; leading axes are batch dims
    and broadcast (an extent-1 batch axis indexes at 0).
    """
    if (_is_bag(a) or _is_bag(b)) and np.ndim(a) <= 2 and np.ndim(b) <= 2:
        return _bag_matmul(
            a if _is_bag(a) else np.asarray(a, dtype=object),
            b if _is_bag(b) else np.asarray(b, dtype=object),
        )
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
    if not isinstance(f, Pair):
        raise NotImplementedError("gradient: traced input only")
    if edge_order != 1:
        raise NotImplementedError("gradient: edge_order 2 not supported")
    nd = len(f._axis_bounds)
    if varargs and any(
        not np.isscalar(Pair._value_of(v)) for v in varargs
    ):
        raise NotImplementedError("gradient: uniform scalar spacing only")
    if len(varargs) == 0:
        spacing = [1.0] * nd
    elif len(varargs) == 1:
        spacing = [varargs[0]] * nd
    elif len(varargs) == nd:
        spacing = list(varargs)
    else:
        raise NotImplementedError("gradient: spacing arity mismatch")
    axes = list(range(nd)) if axis is None else [
        (axis + nd) % nd if np.isscalar(axis) else None
    ]
    if axes[0] is None:
        raise NotImplementedError("gradient: axis tuples not supported")

    def along(ax, dx):
        def key(s):
            return tuple(
                s if d == ax else slice(None) for d in range(nd)
            )

        out = np.zeros_like(f)
        out[key(slice(1, -1))] = (
            f[key(slice(2, None))] - f[key(slice(None, -2))]
        ) / (2.0 * dx)
        out[key(0)] = (f[key(1)] - f[key(0)]) / dx
        out[key(-1)] = (f[key(-1)] - f[key(-2)]) / dx
        return out

    outs = [along(ax, spacing[ax]) for ax in axes]
    return outs[0] if len(outs) == 1 else outs


FUNCTION_TABLE[np.gradient] = _gradient


def _linspace(start, stop, num=50, endpoint=True, retstep=False, **kwargs):
    if not (isinstance(start, Pair) or isinstance(stop, Pair)):
        # nothing traced: numpy's own linspace, untouched
        return np.linspace(
            Pair._value_of(start),
            Pair._value_of(stop),
            int(num),
            endpoint=endpoint,
            retstep=retstep,
            **kwargs,
        )
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if not (v is None or (k == "axis" and v == 0))
    }
    if kwargs or retstep:
        raise NotImplementedError("linspace: retstep/kwargs not supported")
    if np.size(Pair._value_of(start)) != 1 or np.size(Pair._value_of(stop)) != 1:
        raise NotImplementedError("linspace: scalar endpoints only")
    num = int(num)
    lo, hi = Pair._formula_of(start), Pair._formula_of(stop)
    n_steps = (num - 1) if endpoint else num
    i = axis_idx(0)
    formula = lo + i * (hi - lo) / sympy.Integer(max(n_steps, 1))
    value = np.linspace(
        float(np.asarray(Pair._value_of(start)).ravel()[0]),
        float(np.asarray(Pair._value_of(stop)).ravel()[0]),
        num,
        endpoint=endpoint,
    )
    steps = tuple(p for p in (start, stop) if isinstance(p, Pair))
    return Pair(value, formula, ((0, num),), steps=steps)


FUNCTION_TABLE[np.linspace] = _linspace


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
    if isinstance(a, np.ndarray) and a.ndim == 1 and axis in (0, -1):
        axis = None  # the only axis of a 1-D array IS the flatten axis
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
        # per-axis variance: mean of squares minus square of mean,
        # both per-axis reductions we already have
        m = _mean(a, axis=axis)
        m2 = _mean(a * a, axis=axis)
        r = m2 - m * m
        if ddof:
            n = np.shape(Pair._value_of(a))[axis % np.ndim(Pair._value_of(a))]
            r = r * (float(n) / (n - ddof))
        return r
    if not isinstance(a, Pair) or axis is not None:
        return np.var(np.asarray(Pair._value_of(a), dtype=float), axis=axis, ddof=ddof)
    n = int(np.prod([hi - lo for lo, hi in a._axis_bounds]))
    centered = a - a.mean()
    return _sum(centered * centered) / (n - ddof)


def _std(a, axis=None, ddof=0, correction=None, **kwargs):
    return _var(a, axis=axis, ddof=ddof, correction=correction, **kwargs) ** 0.5


def _median(a, axis=None, **kwargs):
    if np.ndim(Pair._value_of(a)) == 1 and axis in (0, -1):
        # the only axis of a 1-D array IS the flatten axis
        axis = None
    bag = (
        isinstance(a, np.ndarray)
        and a.dtype == object
        and a.ndim == 1
        and any(isinstance(e, Pair) for e in a.ravel())
    )
    if bag and axis is None:
        # decompressed operand: same selection, element formulas
        from ..pair import _GUARDS

        vals = np.asarray(Pair._value_of(a), dtype=float)
        order = np.argsort(vals, kind="stable")
        forms = [Pair._formula_of(e) for e in a]
        for k in range(len(order) - 1):
            _GUARDS.append(
                sympy.Le(forms[order[k]], forms[order[k + 1]])
            )
        mid = len(order) // 2
        if len(order) % 2:
            f = forms[order[mid]]
        else:
            f = (forms[order[mid - 1]] + forms[order[mid]]) / 2
        return Pair(np.median(vals), f, None, steps=Pair._steps_of(*a))
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
        # floor(x*10^d + 1/2)/10^d is exact everywhere EXCEPT half-way
        # ties, where numpy rounds half to even. WHICH values sit on a
        # tie is a trace fact: refuse only when one actually does, and
        # record the tie-free hypotheses like median's ordering guards.
        from ..pair import _GUARDS

        vals = np.asarray(Pair._value_of(a), dtype=float)
        scale = 10 ** int(decimals)
        scaled = vals * scale
        on_tie = np.isclose(np.mod(scaled + 0.5, 1.0), 0.0) & ~np.isclose(
            scaled, np.round(scaled)
        )
        if np.any(on_tie):
            raise NotImplementedError(
                "rounding at an exact half-way tie: numpy rounds half "
                "to even, which has no small exact formula"
            )
        sym = axis_idx(0)
        n = int(np.size(vals))
        f = a.formula
        if a._axis_bounds is not None and len(a._axis_bounds) == 1:
            for k in range(n):
                _GUARDS.append(
                    sympy.Ne(
                        sympy.Mod(f.subs(sym, k) * scale + sympy.Rational(1, 2), 1), 0
                    )
                )
        elif a._axis_bounds is None:
            _GUARDS.append(
                sympy.Ne(sympy.Mod(f * scale + sympy.Rational(1, 2), 1), 0)
            )
        else:
            raise NotImplementedError("np.round on >1-D traced arrays")
        formula = sympy.floor(f * scale + sympy.Rational(1, 2)) / scale
        return Pair(
            np.round(vals, decimals), formula, a._axis_bounds, steps=(a,)
        )
    return np.round(Pair._numeric(np.asarray(a), copy=False), decimals)


FUNCTION_TABLE[np.round] = _round
FUNCTION_TABLE[np.around] = _round
def _average(a, axis=None, weights=None, returned=False, **kwargs):
    if returned:
        # (average, sum_of_weights): cov-style callers unpack both
        avg = _average(a, axis=axis, weights=weights)
        if weights is None:
            n = np.shape(Pair._value_of(a))[axis] if axis is not None else np.size(
                Pair._value_of(a)
            )
            scl_val, scl_f = float(n), sympy.Integer(int(n))
        else:
            w_total = float(np.sum(Pair._value_of(weights)))
            scl_val, scl_f = w_total, Pair._formula_of(weights) if isinstance(
                weights, Pair
            ) else sympy.sympify(w_total)
        # numpy returns the weight sum BROADCAST to the average's shape
        # (cov unpacks w_sum[0]); match it
        avg_shape = np.shape(Pair._value_of(avg))
        scl = Pair(
            np.full(avg_shape, scl_val) if avg_shape else scl_val,
            scl_f,
            tuple((0, int(d)) for d in avg_shape) or None,
        )
        return avg, scl
    return _average_impl(a, axis=axis, weights=weights, **kwargs)


def _average_impl(a, axis=None, weights=None, **kwargs):
    traced_1d = (
        isinstance(a, Pair) and np.ndim(a.value) == 1
    ) or (
        isinstance(a, np.ndarray)
        and a.dtype == object
        and a.ndim == 1
        and any(isinstance(e, Pair) for e in a)
    )
    if traced_1d and axis in (0, -1):
        axis = None  # the only axis of a 1-D array IS the flatten axis
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
        if arr.dtype == object and any(
            isinstance(e, Pair) for e in arr.ravel()
        ):
            # per-axis bag average: element dunders keep the trace;
            # falling to np.average would dispatch straight back here
            if weights is None:
                n = arr.shape[axis % arr.ndim]
                return _sum_plain(arr, axis=axis) / float(n)
            w = np.asarray(Pair._value_of(weights), dtype=float)
            ax = axis % arr.ndim
            if w.ndim == 1 and w.size == arr.shape[ax]:
                shape = [1] * arr.ndim
                shape[ax] = -1
                w = w.reshape(shape)  # np.average's weight alignment
            return _sum_plain(arr * w, axis=axis) / float(w.sum())
        if isinstance(weights, Pair) or (
            isinstance(weights, np.ndarray)
            and weights.dtype == object
            and any(isinstance(e, Pair) for e in weights.ravel())
        ):
            # numeric data, TRACED weights: np.average would dispatch
            # straight back here on the weights. Wrap the data as a
            # disclosed constant and take the Pair path
            arr_f = np.asarray(Pair._numeric(arr, copy=False), dtype=float)
            ap = Pair(
                arr_f,
                Pair._formula_of(arr_f),
                tuple((0, int(n)) for n in arr_f.shape),
            )
            return _average_impl(ap, axis=axis, weights=weights)
        return np.average(
            Pair._numeric(arr, copy=False), axis=axis, weights=weights
        )
    if weights is None:
        return a.mean(axis=axis)
    w = weights
    if not isinstance(w, Pair):
        w_arr = np.asarray(Pair._value_of(w), dtype=float)
        w = Pair(
            w_arr,
            Pair._formula_of(w_arr),
            tuple((0, int(n)) for n in w_arr.shape),
        )
    a_nd = len(a._axis_bounds or ())
    w_nd = len(w._axis_bounds or ())
    if axis is not None and a_nd > 1 and w_nd == 1:
        # np.average aligns 1-D weights with the REDUCED axis
        ax = axis % a_nd
        extents = [hi - lo for lo, hi in a._axis_bounds]
        if (w._axis_bounds[0][1] - w._axis_bounds[0][0]) == extents[ax]:
            shape = [1] * a_nd
            shape[ax] = extents[ax]
            w = w.reshape(tuple(shape))
    return _sum(a * w, axis=axis) / _sum(w, axis=axis)


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


def _bincount(x, weights=None, minlength=0):
    """bincount as counting mathematics.

    result[k] = Sum(Piecewise((w_j, Eq(x[j], k)), (0, True)), j) --
    occurrence counting (or weighted counting) as an explicit Sum for
    each bin. The bins themselves come from the concrete values (how
    many bins exist is a trace fact); the counts stay symbolic.
    """
    xv = np.asarray(Pair._value_of(x)).astype(int)
    n = len(xv)
    n_bins = max(int(xv.max()) + 1 if n else 0, int(minlength))
    concrete = np.bincount(xv, weights=Pair._value_of(weights) if weights is not None else None, minlength=minlength)
    if not (isinstance(x, Pair) or (
        isinstance(x, np.ndarray) and x.dtype == object
    ) or isinstance(weights, Pair)):
        return concrete
    def elem_formulas(arr, m):
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            return [Pair._formula_of(e) for e in arr.ravel()]
        f = Pair._formula_of(arr)
        i0 = axis_idx(0)
        return [f.xreplace({i0: sympy.Integer(jj)}) for jj in range(m)]

    bag = getattr(x, "skv_pairs", None)
    if bag is not None:
        out = np.empty(n_bins, dtype=object)
        for k in range(n_bins):
            terms = []
            for elem, cond in bag:
                e_f = Pair._formula_of(elem)
                c_f = Pair._formula_of(cond) if isinstance(cond, Pair) else sympy.sympify(bool(cond))
                terms.append(
                    sympy.Piecewise(
                        (1, sympy.And(c_f, sympy.Eq(e_f, k))), (0, True)
                    )
                )
            out[k] = Pair(float(concrete[k]), sympy.Add(*terms), None)
        return out
    prov = _masked_fuse(x) if isinstance(x, Pair) else None
    if prov is not None:
        src, mask = prov
        bounds = src._axis_bounds
        if bounds is not None and len(bounds) == 1:
            lo, hi = bounds[0]
            i0 = axis_idx(0)
            out = np.empty(n_bins, dtype=object)
            for k in range(n_bins):
                j = sympy.Symbol("j", integer=True)
                body = sympy.Piecewise(
                    (
                        1,
                        sympy.And(
                            mask.formula.xreplace({i0: j}),
                            sympy.Eq(src.formula.xreplace({i0: j}), k),
                        ),
                    ),
                    (0, True),
                )
                out[k] = Pair(
                    float(concrete[k]),
                    sympy.Sum(body, (j, lo, hi - 1)),
                    None,
                )
            return out
    x_fs = elem_formulas(x, n)
    if weights is None:
        w_fs = [sympy.Integer(1)] * n
    else:
        w_fs = elem_formulas(weights, n)
    out = np.empty(n_bins, dtype=object)
    for k in range(n_bins):
        count = sympy.Add(
            *[
                sympy.Piecewise((w_fs[jj], sympy.Eq(x_fs[jj], k)), (0, True))
                for jj in range(n)
            ]
        )
        out[k] = Pair(float(concrete[k]), count, None)
    return out


def _searchsorted(a, v, side="left", sorter=None):
    """searchsorted as counting mathematics.

    For ascending bins, the insertion index IS a count:
    side='left'  -> index = Sum_k [bins[k] < v]
    side='right' -> index = Sum_k [bins[k] <= v]
    Bins concrete (an inventory), the searched values symbolic: each
    result element carries its counting formula.
    """
    bins = np.asarray(Pair._value_of(a), dtype=float)
    concrete = np.searchsorted(
        bins, np.asarray(Pair._value_of(v), dtype=float), side=side, sorter=sorter
    )
    bins_traced = isinstance(a, Pair) or (
        isinstance(a, np.ndarray) and a.dtype == object
    )
    v_traced = isinstance(v, Pair) or (
        isinstance(v, np.ndarray) and v.dtype == object
    )
    if sorter is not None or not (bins_traced or v_traced):
        return concrete
    if bins_traced:
        # symbolic bins: index = Sum_k [bins[k] rel v], valid only for
        # sorted bins -- the ordering is recorded as preconditions
        from ..session import current as _session

        i0 = axis_idx(0)
        n = bins.size
        b_f = [Pair._formula_of(a).subs(i0, k) if isinstance(a, Pair)
               else Pair._formula_of(a.ravel()[k]) for k in range(n)]
        for k in range(n - 1):
            _session.guards.append(sympy.Le(b_f[k], b_f[k + 1]))
        rel = sympy.Lt if side == "left" else sympy.Le

        def count_for(target_f):
            return sympy.Add(
                *[sympy.Piecewise((1, rel(bf, target_f)), (0, True))
                  for bf in b_f]
            )

        if np.ndim(concrete) == 0:
            return Pair(
                concrete, count_for(Pair._formula_of(v)), None,
                steps=Pair._steps_of(a, v),
            )
        v_f = Pair._formula_of(v)
        out = np.empty(np.shape(concrete), dtype=object)
        for j in range(out.size):
            out.ravel()[j] = Pair(
                concrete.ravel()[j], count_for(v_f.subs(i0, j)), None,
                steps=Pair._steps_of(a, v),
            )
        return out
    v_f = Pair._bridge_numeric(Pair._formula_of(v))
    i0 = axis_idx(0)
    rel = sympy.Lt if side == "left" else sympy.Le

    def index_formula(elem_formula):
        return sympy.Add(
            *[
                sympy.Piecewise(
                    (1, rel(sympy.Float(b), elem_formula)), (0, True)
                )
                for b in bins
            ]
        )

    if np.ndim(concrete) == 0:
        return Pair(int(concrete), index_formula(v_f), None)
    out = np.empty(len(concrete), dtype=object)
    for jj in range(len(concrete)):
        out[jj] = Pair(
            int(concrete[jj]), index_formula(v_f.xreplace({i0: jj})), None
        )
    return out


def _nan_reduce(np_fn, kind):
    """nan-aware reductions: WHICH entries are nan is a trace fact;
    the surviving elements reduce symbolically."""

    def entry(a, axis=None, **kwargs):
        vals = np.asarray(Pair._value_of(a), dtype=float)
        if axis is not None or not (
            isinstance(a, Pair)
            or (isinstance(a, np.ndarray) and a.dtype == object)
        ):
            return np_fn(vals, axis=axis, **kwargs)
        if isinstance(a, np.ndarray):
            elems = list(a.ravel())
        else:
            elems = [a[k] for k in range(len(vals.ravel()))]
        keep = [e for e, v in zip(elems, vals.ravel()) if not np.isnan(v)]
        if not keep:
            return np_fn(vals, **kwargs)
        total = keep[0]
        for e in keep[1:]:
            total = total + e
        if kind == "mean":
            return total / len(keep)
        if kind == "prod":
            out = keep[0]
            for e in keep[1:]:
                out = out * e
            return out
        if kind in ("var", "std"):
            ddof = kwargs.get("ddof", 0)
            ddof = 0 if ddof is np._NoValue else ddof
            k = len(keep)
            mean = total / k
            sq = (keep[0] - mean) ** 2
            for e in keep[1:]:
                sq = sq + (e - mean) ** 2
            out = sq / (k - ddof)
            return np.sqrt(out) if kind == "std" else out
        return total

    return entry


def _nanmedian(a, axis=None, **kwargs):
    """nan-aware median: WHICH entries are nan is a trace fact; the
    survivors' median is a path-scoped selection under explicit
    ordering preconditions (same contract as np.median)."""
    vals = np.asarray(Pair._value_of(a), dtype=float)
    traced = isinstance(a, Pair) or (
        isinstance(a, np.ndarray) and a.dtype == object
    )
    if not traced or kwargs:
        return np.nanmedian(vals, axis=axis, **kwargs)
    from ..pair import _GUARDS

    def elem_formula(idx):
        if isinstance(a, Pair):
            f = a.formula
            for ax, k in enumerate(idx):
                f = f.subs(axis_idx(ax), int(k))
            return f
        return Pair._formula_of(a[idx])

    def median_of(cells):
        # cells: list of index tuples into vals, nans already dropped
        cells = sorted(cells, key=lambda idx: vals[idx])
        for k in range(len(cells) - 1):
            _GUARDS.append(
                sympy.Le(elem_formula(cells[k]), elem_formula(cells[k + 1]))
            )
        mid = len(cells) // 2
        if len(cells) % 2:
            f = elem_formula(cells[mid])
        else:
            f = (elem_formula(cells[mid - 1]) + elem_formula(cells[mid])) / 2
        v = np.nanmedian([vals[c] for c in cells])
        return Pair(v, f, None, steps=Pair._steps_of(a))

    live = [idx for idx in np.ndindex(vals.shape) if not np.isnan(vals[idx])]
    if not live:
        return np.nanmedian(vals, axis=axis)
    if axis is None:
        return median_of(live)
    axis = axis % vals.ndim
    out_shape = vals.shape[:axis] + vals.shape[axis + 1 :]
    out = np.empty(out_shape, dtype=object)
    for oidx in np.ndindex(out_shape):
        cells = [
            idx for idx in live
            if idx[:axis] + idx[axis + 1 :] == oidx
        ]
        out[oidx] = (
            median_of(cells) if cells else np.nan
        )
    return out


def _mutating_write(np_fn):
    """copyto/place/putmask as the assignments they secretly are.

    Their whole output is a side effect on the destination; treating
    them as calls loses the write. Route both lanes through Pair's
    setitem machinery instead, or refuse when the write cannot be
    represented -- never pass through silently."""

    def entry(dst, *args, **kwargs):
        if not isinstance(dst, Pair):
            raise NotImplementedError(
                f"{np_fn.__name__} into a non-traced destination holding "
                "traced values; assign with dst[...] = src instead"
            )
        kwargs.pop("casting", None)  # dtype bookkeeping, not math
        w = kwargs.pop("where", True)
        if kwargs:
            raise NotImplementedError(
                f"{np_fn.__name__} with options is not supported"
            )
        if np_fn is np.copyto:
            (src,) = args
            if w is True:
                dst[...] = src
            else:
                dst[w] = src  # masked copy IS a masked write
            return None
        mask, vals = args
        v = np.asarray(Pair._value_of(vals))
        if np_fn is np.place and v.size != 1 and v.size != int(
            np.count_nonzero(np.asarray(Pair._value_of(mask), dtype=bool))
        ):
            # np.place CYCLES a short vals list through the mask; a
            # cycled write has no honest single formula
            raise NotImplementedError(
                "np.place with a cycled values list is not supported"
            )
        if isinstance(mask, Pair):
            # np.place/putmask select by TRUTHINESS: a numeric mask
            # ((1-cond)+isnan(x) in scipy's distributions) means
            # nonzero, exactly Ne(f, 0)
            mf = mask.formula
            if not Pair._is_condition(mf):
                mf = sympy.Ne(mf, 0)
            mask = Pair(
                np.asarray(Pair._value_of(mask)) != 0,
                mf,
                mask._axis_bounds,
                steps=(mask,),
            )
        elif not (
            isinstance(mask, np.ndarray) and mask.dtype == np.bool_
        ):
            mask = np.asarray(Pair._value_of(mask)) != 0
        dst[mask] = vals if np.ndim(vals) == 0 or v.size != 1 else (
            vals[0] if not isinstance(vals, Pair) else vals
        )
        return None

    return entry


def _nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """nan_to_num exactly: WHICH entries are nan/inf is a trace fact.

    Finite positions keep their formulas; non-finite positions become
    the replacement constants numpy would write (the huge finfo bounds
    for inf unless overridden)."""
    vals = np.asarray(Pair._value_of(x), dtype=float)
    fixed = np.nan_to_num(vals, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
    if not isinstance(x, Pair):
        return fixed
    if np.all(np.isfinite(vals)):
        return Pair(fixed, x.formula, x._axis_bounds, steps=(x,))
    if vals.ndim == 0:
        return Pair(fixed, sympy.Float(float(fixed)), None, steps=(x,))
    sym = axis_idx(0)
    out = np.empty(vals.shape, dtype=object)
    for idx in np.ndindex(vals.shape):
        if np.isfinite(vals[idx]):
            f = x.formula
            for ax, k in enumerate(idx):
                f = f.subs(axis_idx(ax), int(k))
        else:
            f = sympy.Float(float(fixed[idx]))
        out[idx] = Pair(fixed[idx], f, None, steps=(x,))
    return out


def _select(condlist, choicelist, default=0):
    """np.select is chained np.where; its body only adds a dtype gate
    that rejects traced masks. Same Piecewise, built directly."""
    out = _where(condlist[-1], choicelist[-1], default)
    for cond, choice in zip(condlist[-2::-1], choicelist[-2::-1]):
        out = _where(cond, choice, out)
    return out


def _interp(x, xp, fp, left=None, right=None, period=None):
    """Piecewise-linear interpolation, exactly.

    The table (xp, fp) concrete and the query traced: the exact
    Piecewise over the table's intervals. Compiled internals make a
    table entry the honest route; a traced TABLE has no closed
    branch-free form and refuses."""
    if period is not None:
        raise NotImplementedError("np.interp with period is not supported")
    concrete = np.interp(
        np.asarray(Pair._value_of(x), dtype=float),
        np.asarray(Pair._value_of(xp), dtype=float),
        np.asarray(Pair._value_of(fp), dtype=float),
        left=left,
        right=right,
    )
    if isinstance(xp, Pair) or isinstance(fp, Pair) or _is_bag(xp) or _is_bag(fp):
        raise NotImplementedError(
            "np.interp with a traced table: the interval selection has "
            "no single exact formula; interpolate explicitly instead"
        )
    if not (isinstance(x, Pair) or _is_bag(x)):
        return concrete
    xs = np.asarray(xp, dtype=float)
    fs = np.asarray(fp, dtype=float)
    lo = float(fs[0]) if left is None else float(left)
    hi = float(fs[-1]) if right is None else float(right)

    def branchwise(q):
        pieces = [(sympy.Float(lo), q < float(xs[0]))]
        for k in range(len(xs) - 1):
            x0, x1 = float(xs[k]), float(xs[k + 1])
            f0, f1 = float(fs[k]), float(fs[k + 1])
            slope = (f1 - f0) / (x1 - x0) if x1 != x0 else 0.0
            pieces.append((f0 + slope * (q - x0), q <= x1))
        pieces.append((sympy.Float(hi), True))
        return sympy.Piecewise(*pieces)

    if np.ndim(Pair._value_of(x)) == 0:
        return Pair(
            float(concrete), branchwise(Pair._formula_of(x)), None,
            steps=Pair._steps_of(x),
        )
    sym = axis_idx(0)
    out = np.empty(np.shape(concrete), dtype=object)
    for k in range(out.size):
        qf = (
            Pair._formula_of(x).subs(sym, k)
            if isinstance(x, Pair)
            else Pair._formula_of(x.ravel()[k])
        )
        out.ravel()[k] = Pair(
            float(np.ravel(concrete)[k]), branchwise(qf), None,
            steps=Pair._steps_of(x),
        )
    return out


FUNCTION_TABLE[np.select] = _select
FUNCTION_TABLE[np.interp] = _interp
def _trace(a, offset=0, **kwargs):
    if not isinstance(a, Pair) or kwargs or len(a._axis_bounds or ()) != 2:
        return np.trace(np.asarray(Pair._value_of(a)), offset=offset)
    (r0, r1), (c0, c1) = a._axis_bounds
    n = min(r1, c1 - offset) - max(r0, -offset)
    k = _fresh_dummy(a.formula, 2)
    body = a.formula.xreplace({axis_idx(0): k, axis_idx(1): k + offset})
    return Pair(
        np.trace(np.asarray(a.value), offset=offset),
        _held_sum(body, (k, max(r0, -offset), max(r0, -offset) + n - 1)),
        None,
        steps=(a,),
    )


FUNCTION_TABLE[np.trace] = _trace
FUNCTION_TABLE[np.nan_to_num] = _nan_to_num
def _fill_diagonal(a, val, wrap=False):
    if not isinstance(a, Pair):
        raise NotImplementedError(
            "fill_diagonal into a non-traced destination holding traced "
            "values; assign per element instead"
        )
    if wrap or len(a._axis_bounds or ()) != 2:
        raise NotImplementedError("fill_diagonal: 2-D unwrapped only")
    n = min(hi - lo for lo, hi in a._axis_bounds)
    vals = np.broadcast_to(np.asarray(Pair._value_of(val)), (n,))
    for k in range(n):
        a[k, k] = vals[k] if not isinstance(val, Pair) else val[k] if val._axis_bounds else val
    return None


FUNCTION_TABLE[np.fill_diagonal] = _fill_diagonal
FUNCTION_TABLE[np.copyto] = _mutating_write(np.copyto)
FUNCTION_TABLE[np.place] = _mutating_write(np.place)
FUNCTION_TABLE[np.putmask] = _mutating_write(np.putmask)
FUNCTION_TABLE[np.nanmedian] = _nanmedian
FUNCTION_TABLE[np.nanmean] = _nan_reduce(np.nanmean, "mean")
FUNCTION_TABLE[np.nansum] = _nan_reduce(np.nansum, "sum")
FUNCTION_TABLE[np.nanstd] = _nan_reduce(np.nanstd, "std")
FUNCTION_TABLE[np.nanvar] = _nan_reduce(np.nanvar, "var")
FUNCTION_TABLE[np.nanprod] = _nan_reduce(np.nanprod, "prod")
FUNCTION_TABLE[np.searchsorted] = _searchsorted
FUNCTION_TABLE[np.bincount] = _bincount
FUNCTION_TABLE[np.digitize] = _concrete_inventory(np.digitize)
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
FUNCTION_TABLE[np.prod] = _prod
FUNCTION_TABLE[np.clip] = _clip_entry


def _modf(x, **kwargs):
    # frac and whole parts: x - trunc(x), trunc(x)
    whole = np.trunc(x)
    return x - whole, whole


def _divmod_entry(x, y, **kwargs):
    return np.floor_divide(x, y), np.mod(x, y)


def _frexp(x, **kwargs):
    # x = m * 2**e with 0.5 <= |m| < 1: e = floor(log2|x|) + 1
    e = np.floor(np.log2(np.abs(x))) + 1.0
    return x / 2.0**e, e


FUNCTION_TABLE[np.modf] = _modf
FUNCTION_TABLE[np.divmod] = _divmod_entry
FUNCTION_TABLE[np.frexp] = _frexp
FUNCTION_TABLE[np.clip] = lambda a, lo, hi, **kw: _np_clip(a, lo, hi)
FUNCTION_TABLE[np.all] = _all
FUNCTION_TABLE[np.any] = _any
FUNCTION_TABLE[np.where] = _where
FUNCTION_TABLE[np.transpose] = lambda a, axes=None: (
    a.transpose(axes) if isinstance(a, Pair) else np.transpose(a, axes)
)
