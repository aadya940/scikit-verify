"""Runtime twins: the traced replacements for math-neutral calls.

Some numpy/scipy calls add nothing to the mathematics but destroy the
trace: raw allocation (np.zeros picks float64 before any traced value
exists), dtype-forcing coercion, and compiled scipy callables outside
numpy's dispatch. to_sympy runs a semantically identical copy of the
function with exactly those calls replaced by traced twins. Every
replacement site is recorded and disclosed on the result.

Only math-neutral calls are ever rewritten; nothing that computes is
touched.
"""


import inspect

import operator

import numpy as np
import sympy

from ..helpers import axis_idx
from ..pair import Pair
from .registries import OPAQUE_OUT

def _bounds_of(shape):
    if isinstance(shape, (int, np.integer)):
        return ((0, int(shape)),)
    return tuple((0, int(s)) for s in shape)


def _skv_zeros(shape, dtype=None, *args, **kwargs):
    # the value lane honors the requested dtype (bool masks, int
    # counters); the formula is 0 regardless
    return Pair(np.zeros(shape, dtype=dtype), sympy.Integer(0), _bounds_of(shape))


def _skv_ones(shape, dtype=None, *args, **kwargs):
    return Pair(np.ones(shape, dtype=dtype), sympy.Integer(1), _bounds_of(shape))


def _skv_full(shape, fill_value, dtype=None, *args, **kwargs):
    # parameter names mirror np.full: callers pass fill_value by keyword
    if not isinstance(fill_value, Pair):
        # a concrete fill stays a concrete buffer: lifting it would pin
        # a formula that later compiled writes (np.place, putmask)
        # cannot update -- correct value lane, silently wrong formula
        return np.full(shape, fill_value, dtype=dtype)
    return Pair(
        np.full(shape, Pair._value_of(fill_value), dtype=dtype),
        Pair._formula_of(fill_value),
        _bounds_of(shape),
        steps=Pair._steps_of(fill_value),
    )


def _skv_empty(shape, dtype=None, *args, **kwargs):
    bounds = _bounds_of(shape)
    letters = tuple(axis_idx(ax) for ax in range(len(bounds)))
    return Pair(
        np.empty(shape, dtype=dtype),
        sympy.Function("uninitialized")(*letters),
        bounds,
    )


def _skv_neutral(a, dtype=None, **kwargs):
    if isinstance(a, Pair):
        value = a.value
        if isinstance(value, np.ndarray) and value.dtype == object:
            value = Pair._value_of(value)  # value lane holding Pairs
        value = np.ascontiguousarray(value, dtype=dtype)
        ndmin = kwargs.get("ndmin", 0)
        bounds = a._axis_bounds
        if ndmin and np.ndim(value) < ndmin:
            value = np.array(value, ndmin=ndmin)
            bounds = tuple((0, n) for n in value.shape)
        return Pair(value, a.formula, bounds, steps=(a,))
    if isinstance(a, np.ndarray) and a.dtype == object:
        unwrapped = Pair._value_of(a)
        if unwrapped is not a:
            return a  # object array OF Pairs: keep the traced elements
    return np.asarray(a, dtype=dtype)


def _skv_method(name, obj, *args, **kwargs):
    if inspect.ismodule(obj):
        # xp.astype(a, dt): a module function, not a method -- route
        # through the doorman like any other call
        from .triage import _skv_maybe  # late: runtime and triage are mutually recursive

        return _skv_maybe(getattr(obj, name))(*args, **kwargs)
    if name == "type" and args and isinstance(args[0], Pair):
        return args[0]  # ret.dtype.type(x): a cast on a traced scalar
    if (
        isinstance(obj, np.ndarray)
        and obj.dtype == object
        and any(isinstance(e, Pair) for e in obj.ravel())
    ):
        if name in ("astype", "copy"):
            return obj  # traced scalars; the cast/copy is math-neutral
        from ..coercion import repack

        from .triage import _BAG_REDUCTIONS

        rp = repack(obj) if name in _BAG_REDUCTIONS else None
        np_fn = getattr(np, name, None) if name in _BAG_REDUCTIONS else None
        if rp is not None and callable(np_fn):
            # a provable pattern: back on the indexed path, where the
            # dispatch protocol routes to the registered entry
            return np_fn(rp, *args, **kwargs)
        from ..registry import FUNCTION_TABLE

        entry = FUNCTION_TABLE.get(np_fn)
        if entry is not None:
            # no pattern (a permuted bag): the registered entry works
            # elementwise from the object array
            return entry(obj, *args, **kwargs)
    if not isinstance(obj, Pair):
        if (
            isinstance(obj, np.ndarray)
            and obj.dtype == object
            and not any(isinstance(e, Pair) for e in obj.ravel())
        ):
            obj = Pair._numeric(obj, copy=False)
        return getattr(obj, name)(*args, **kwargs)
    if name == "view":
        dtype = args[0] if args else kwargs.get("dtype")
        if dtype is not None and np.dtype(dtype).kind not in "f":
            raise NotImplementedError("view reinterpretation would change the math")
        return Pair(obj.value, obj.formula, obj._axis_bounds, steps=(obj,))
    if name == "astype":
        dtype = args[0] if args else kwargs.get("dtype")
        return obj.astype(dtype)
    if name == "toarray":
        return Pair(
            np.asarray(obj.value.toarray()),
            obj.formula,
            obj._axis_bounds,
            steps=(obj,),
        )
    if name == "copy":
        value = obj.value.copy() if hasattr(obj.value, "copy") else obj.value
        return Pair(value, obj.formula, obj._axis_bounds, steps=(obj,))
    return getattr(obj.value, name)(*args, **kwargs)


class _SkvAtIndexed:
    def __init__(self, pair, idx):
        self.pair, self.idx = pair, idx

    def _updated(self, op, v):
        out = Pair(
            np.array(self.pair.value, copy=True),
            self.pair.formula,
            self.pair._axis_bounds,
            steps=(self.pair,),
        )
        out[self.idx] = op(out[self.idx], v) if op else v
        return out

    def set(self, v, **kw):
        return self._updated(None, v)

    def add(self, v, **kw):
        return self._updated(lambda a, b: a + b, v)

    def subtract(self, v, **kw):
        return self._updated(lambda a, b: a - b, v)

    def multiply(self, v, **kw):
        return self._updated(lambda a, b: a * b, v)

    def divide(self, v, **kw):
        return self._updated(lambda a, b: a / b, v)


class _SkvAt:
    def __init__(self, pair):
        self.pair = pair

    def __getitem__(self, idx):
        return _SkvAtIndexed(self.pair, idx)


def _skv_at(real_at, x, *args, **kwargs):
    # two callables share the name: ufunc.at (in-place, duplicate
    # indices accumulate) and array_api_extra's functional at(x, idx)
    if isinstance(getattr(real_at, "__self__", None), np.ufunc):
        ufunc = real_at.__self__
        if not isinstance(x, Pair):
            return real_at(x, *args, **kwargs)
        idx = np.ravel(np.asarray(args[0]))
        if len(args) > 1:
            vals = np.broadcast_to(np.asarray(args[1]), idx.shape)
            # sequential per-position updates: exactly ufunc.at's
            # accumulation semantics, on both lanes via Pair setitem
            for i, v in zip(idx.tolist(), vals.tolist()):
                x[int(i)] = ufunc(x[int(i)], v)
        else:
            for i in idx.tolist():
                x[int(i)] = ufunc(x[int(i)])
        return None
    # array_api_extra's functional update: on a Pair it is setitem on
    # a copy; anything else goes to the real helper
    if isinstance(x, Pair):
        if args:  # at(x, idx) form
            return _SkvAtIndexed(x, args[0])
        return _SkvAt(x)
    return real_at(x, *args, **kwargs)


def _skv_concrete(name, a, *args, **kwargs):
    return getattr(np, name)(np.asarray(Pair._value_of(a)), *args, **kwargs)


def _skv_scalarize(kind, x):
    v = np.asarray(Pair._value_of(x))
    if v.ndim and v.size == 1:
        return kind(v.item())  # numpy 2 forbids float() on size-1 arrays
    return kind(Pair._value_of(x))


def _skv_set(iterable=()):
    """Guarded set: dedup traced scalars by == (guards recorded).

    Parameters
    ----------
    iterable : iterable
        Elements. Plain values build a real ``set``; traced elements
        dedup through Pair's guarded equality, so every merge decision
        lands in the trace's preconditions and the resulting control
        flow matches the untraced run exactly.

    Returns
    -------
    set or list
        A real set for plain elements; an order-preserving deduped
        list when traced elements are present (list supports the
        same iteration/len/membership uses downstream code makes).
    """
    items = list(iterable)
    if not any(isinstance(e, Pair) for e in items):
        return set(items)
    from ..sets import TracedSet

    return TracedSet(items)


class MaskedElems(np.ndarray):
    """Survivors of a bag-form mask gather, carrying provenance.

    ``skv_pairs`` holds (element, condition) for EVERY original
    position; reductions fuse the conditions into their formulas
    instead of needing per-position guards.
    """

    def __array_finalize__(self, obj):
        self.skv_pairs = getattr(obj, "skv_pairs", None)


_CMP_OPS = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
    "le": operator.le, "gt": operator.gt, "ge": operator.ge,
}


def _skv_cmp(op, left, right):
    """Comparison that survives object arrays of Pairs.

    Plain operands compare natively. When either side is an object
    array holding Pairs, numpy would bool() each element's condition
    away; instead compare elementwise and keep the condition Pairs.
    """
    def bag(x):
        return (
            isinstance(x, np.ndarray)
            and x.dtype == object
            and any(isinstance(e, Pair) for e in x.ravel())
        )

    if bag(left) or bag(right):
        from ..coercion import value_of

        l = np.asarray(left, dtype=object)
        r = np.asarray(right, dtype=object)
        l, r = np.broadcast_arrays(l, r)
        conds = np.empty(l.shape, dtype=object)
        truths = np.empty(l.shape, dtype=bool)
        f = _CMP_OPS[op]
        for idx in np.ndindex(l.shape):
            c = f(l[idx], r[idx])
            conds[idx] = c
            truths[idx] = bool(value_of(c))
        # concrete bool lane keeps every downstream numpy op working;
        # the conditions ride along for gather sites that can use them
        out = truths.view(MaskedElems)
        out.skv_pairs = tuple(conds.ravel())
        return out
    return _CMP_OPS[op](left, right)


def _skv_float(x):
    """float() that keeps a traced scalar traced."""
    if isinstance(x, Pair):
        return Pair(float(np.asarray(x.value).item()), x.formula, None, steps=(x,))
    if isinstance(x, np.ndarray) and x.size == 1:
        e = x.ravel()[0]
        if isinstance(e, Pair):
            return _skv_float(e)
        # numpy 2 forbids float() on size-1 arrays; traced shims can
        # produce (1,) where numpy gives 0-d, so unwrap explicitly
        return float(np.asarray(e).item())
    return float(x)


def _skv_getitem(obj, key):
    """Subscript with selection semantics for traced keys.

    A mapping looked up by a traced scalar is a SELECTION: the result
    carries the Piecewise over the table. Everything else (arrays,
    lists, plain keys) subscripts normally.
    """
    import collections.abc

    if (
        isinstance(obj, np.ndarray)
        and obj.dtype == object
        and isinstance(key, np.ndarray)
        and key.shape == obj.shape
        and getattr(key, "skv_pairs", None) is not None
    ):
        # bag-form mask gather: y[m] where m is a concrete bool mask
        # still carrying its per-position condition Pairs. Select by
        # the truths, keep (element, condition) provenance so a later
        # reduction can fuse the conditions into its formula
        truths = np.asarray(key, dtype=bool).ravel()
        survivors = np.asarray(obj.ravel())[truths]
        out = np.empty(len(survivors), dtype=object).view(MaskedElems)
        for jj, e in enumerate(survivors):
            out[jj] = e
        out.skv_pairs = tuple(zip(obj.ravel(), key.skv_pairs))
        return out
    if isinstance(key, Pair) and isinstance(obj, collections.abc.Mapping):
        import sympy

        from ..coercion import formula_of, value_of

        v = key.value.item() if hasattr(key.value, "item") else key.value
        stored = obj[v]
        pieces = []
        for k, val in obj.items():
            if isinstance(val, Pair) or isinstance(k, Pair):
                k = value_of(k)
                val_f = formula_of(val)
            else:
                val_f = formula_of(val)
            pieces.append((val_f, sympy.Eq(key.formula, formula_of(k))))
        formula = sympy.Piecewise(*pieces, (sympy.Symbol("NaN", real=True), True))
        return Pair(
            value_of(stored) if isinstance(stored, Pair) else stored,
            formula,
            None,
            steps=(key,),
        )
    return obj[key]


def _skv_dict(mapping):
    """Guarded dict: traced keys anywhere make it a TracedDict, whose
    lookups by traced keys return Piecewise selections."""
    items = dict(mapping)
    if any(isinstance(k, Pair) for k in items) or True:
        # keys may be concrete while LOOKUPS are traced; a TracedDict
        # costs nothing and preserves selection formulas either way
        from ..sets import TracedDict

        return TracedDict(items)
    return items


def _skv_isinstance(obj, types):
    # inside instrumented code a Pair IS an ndarray for gate purposes
    if isinstance(obj, Pair):
        tt = types if isinstance(types, tuple) else (types,)
        if any(t is np.ndarray for t in tt):
            return True
    return isinstance(obj, types)


def _skv_namespace(*args, **kwargs):
    # array-api-compat is dispatch plumbing, not mathematics: the
    # concrete lane is numpy, so the namespace IS numpy and xp.mean
    # etc. reach Pairs through the normal protocol
    return np


def _skv_finfo(fn):
    def wrapper(dt, *args, **kwargs):
        if getattr(dt, "kind", None) == "O" or dt is object:
            return np.finfo(np.float64)  # the concrete lane is float64
        return fn(dt, *args, **kwargs)

    return wrapper


def _skv_concrete_call(fn):
    def wrapper(*args, **kwargs):
        return fn(*[Pair._value_of(a) for a in args], **kwargs)

    return wrapper


def _skv_opaque_out(fn, out_idxs, transposed, *args, **kwargs):
    """Run an out-parameter Cython routine on the concrete lane and
    return the filled buffers as fresh opaque atoms (the traced twin
    of `fn(a, out1, out2)` rewritten to `out1, out2 = ...`)."""
    from ..pair import _OPAQUE
    from ..contracts import check_call

    values = [Pair._value_of(a) for a in args]
    for pos, idx in enumerate(out_idxs):
        out_val = np.asarray(values[idx], dtype=float)
        values[idx] = np.ascontiguousarray(out_val.T if transposed[pos] else out_val)
    fn(*values, **kwargs)
    bufs = {
        idx: (values[idx].T.copy() if transposed[pos] else values[idx])
        for pos, idx in enumerate(out_idxs)
    }

    name = fn.__name__.lstrip("_")
    operands = []
    for i, a in enumerate(args):
        if i in bufs:
            continue
        if isinstance(a, Pair):
            operands.append(a.formula)
        elif np.isscalar(a):
            operands.append(sympy.sympify(a))
        else:
            operands.append(sympy.Symbol(f"const{i}"))
    call = sympy.Function(name)(*operands)

    record = len(_OPAQUE)
    outs = []
    for pos, idx in enumerate(out_idxs):
        buf = bufs[idx]
        suffix = f"_{pos}" if len(out_idxs) > 1 else ""
        base = sympy.IndexedBase(f"{name}_{record}{suffix}")
        letters = tuple(axis_idx(ax) for ax in range(buf.ndim))
        formula = base[letters] if letters else sympy.Symbol(f"{name}_{record}")
        outs.append(
            Pair(
                buf,
                formula,
                tuple((0, int(n)) for n in buf.shape),
                steps=Pair._steps_of(*args),
            )
        )
    _OPAQUE.append(
        check_call(name, values, tuple(bufs.values()))
        + ((f"{name}_{record}", str(call)),)
    )
    return outs[0] if len(outs) == 1 else tuple(outs)


