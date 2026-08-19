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
    return Pair(np.zeros(shape), sympy.Integer(0), _bounds_of(shape))


def _skv_ones(shape, dtype=None, *args, **kwargs):
    return Pair(np.ones(shape), sympy.Integer(1), _bounds_of(shape))


def _skv_full(shape, fill, dtype=None, *args, **kwargs):
    return Pair(
        np.full(shape, Pair._value_of(fill)),
        Pair._formula_of(fill),
        _bounds_of(shape),
        steps=Pair._steps_of(fill),
    )


def _skv_empty(shape, dtype=None, *args, **kwargs):
    bounds = _bounds_of(shape)
    letters = tuple(axis_idx(ax) for ax in range(len(bounds)))
    return Pair(np.empty(shape), sympy.Function("uninitialized")(*letters), bounds)


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
        if dtype is not None and np.dtype(dtype).kind not in "fc":
            vals = np.asarray(Pair._value_of(obj.value))
            integral = np.dtype(dtype).kind in "iub" and np.all(vals == np.floor(vals))
            if not integral:
                raise NotImplementedError("astype to non-float would change the math")
        return Pair(obj.value, obj.formula, obj._axis_bounds, steps=(obj,))
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


