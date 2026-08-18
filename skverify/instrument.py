"""Trace-time instrumentation.

Some numpy/scipy calls add nothing to the mathematics but destroy the
trace: raw allocation (np.zeros picks float64 before any traced value
exists), dtype-forcing coercion, and compiled scipy callables outside
numpy's dispatch. to_sympy runs a semantically identical copy of the
function with exactly those calls replaced by traced twins. Every
replacement site is recorded and disclosed on the result.

Only math-neutral calls are ever rewritten; nothing that computes is
touched.
"""

import ast
import inspect
import textwrap

import numpy as np
import sympy

from .helpers import axis_idx
from .pair import Pair, _loop_end, _loop_iter

ALLOC = {"zeros", "empty", "ones", "full"}
NEUTRAL = {
    "asarray",
    "asanyarray",
    "ascontiguousarray",
    "asfortranarray",
    "asarray_chkfinite",
}
OPAQUE_CALLABLES = {
    "solve_banded",
    "solveh_banded",
    "cho_solve",
    "design_matrix",
    "gbsv",
    "data_matrix",
    "fpback",
    "evaluate_all_bspl",
    "solve",
    "lstsq",
    "_lstsq",
    "svd",
    "pinv",
    "r2c",
    "c2c",
    "c2r",
    "rfft",
    "irfft",
    "fft",
    "ifft",
    "rfftn",
    "fftn",
}
NEUTRAL_METHODS = {"toarray", "astype", "copy", "view", "type"}
CONCRETE = {"isfinite", "isnan", "isinf"}  # validation checks, not math
SCALARIZE = {"float", "int"}  # scalar coercion at a compiled boundary
# compiled lookups whose result is bookkeeping (an interval index),
# not mathematics: run on values, return the plain result
CONCRETE_CALLABLES = {"find_interval"}
# compiled routines that RETURN through array out-parameters (scipy's
# Cython convention); value = argument positions of the out arrays
OPAQUE_OUT = {"_coloc": (3,), "qr_reduce": (0, 3)}

_SITES = []


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
    from .pair import _OPAQUE
    from .contracts import check_call

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


def runtime_twin(fn):
    """Memoized instrumented copy of a plain function, or the original
    when instrumentation finds nothing to rewrite."""
    if fn in _FN_MEMO:
        sub, sites = _FN_MEMO[fn]
        return sub if sites else fn
    try:
        sub, sites = _instrument(fn, 0, set())
    except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
        sub, sites = fn, ()
    _FN_MEMO[fn] = (sub, sites)
    return sub if sites else fn


def _skv_maybe(fn):
    """Pass Python functions, methods, classes, ufuncs and builtins
    through; wrap anything COMPILED so traced arguments turn the call
    into an opaque atom instead of a crash. Curated boundaries match
    the RESOLVED callable's name, so aliases and attribute lookups
    cannot dodge them."""
    if getattr(fn, "__name__", None) in OPAQUE_CALLABLES:
        return _skv_opaque(fn)
    if inspect.isclass(fn):
        # classes reached through variables (dispatch tables) twin at
        # runtime; the cache makes this once per class
        try:
            twin, _ = _instrument_class(fn, 3, set())
            return twin
        except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
            return fn
    if inspect.ismethod(fn) and inspect.isfunction(fn.__func__):
        # a bound method (norm.pdf): twin the function, rebind to the
        # instance; self.method calls inside chain through the doorman
        inner = fn.__func__
        if not getattr(inner, "__closure__", None):
            sub = runtime_twin(inner)
            if sub is not inner:
                return sub.__get__(fn.__self__)
        return fn
    if inspect.isfunction(fn) and not getattr(fn, "__closure__", None):
        mod = getattr(fn, "__module__", "") or ""
        if not mod.startswith(("builtins", "skverify")) and "__skv" not in fn.__name__:
            if fn in _FN_MEMO:
                sub, sub_sites = _FN_MEMO[fn]
            else:
                try:
                    sub, sub_sites = _instrument(fn, 0, set())
                except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
                    sub, sub_sites = fn, ()
                _FN_MEMO[fn] = (sub, sub_sites)
            return sub if sub_sites else fn
        return fn
    if hasattr(fn, "__wrapped__") and not inspect.isfunction(fn):
        # numpy dispatcher: the protocol handles Pair args, but OBJECT
        # arrays of Pairs bypass it -- route those through our table
        from .registry import FUNCTION_TABLE

        entry = FUNCTION_TABLE.get(fn)
        if entry is None:
            def plain_shim(*args, **kwargs):
                if any(
                    isinstance(a, np.ndarray)
                    and a.dtype == object
                    and not any(isinstance(e, Pair) for e in a.ravel())
                    for a in args
                ):
                    # plain-number object arrays carry no formulas:
                    # coercing loses nothing and numpy internals need it
                    args = [Pair._numeric(a, copy=False) for a in args]
                return fn(*args, **kwargs)

            return plain_shim

        def dispatcher_shim(*args, **kwargs):
            if any(
                (_traced(a) and not isinstance(a, Pair))
                or (isinstance(a, np.ndarray) and a.dtype == object)
                for a in args
            ):
                return entry(*args, **kwargs)
            return fn(*args, **kwargs)

        return dispatcher_shim
    if isinstance(fn, np.ufunc):
        def ufunc_shim(*args, **kwargs):
            if any(
                isinstance(a, np.ndarray)
                and a.dtype == object
                and not any(isinstance(e, Pair) for e in a.ravel())
                for a in args
            ):
                # plain-number object arrays: no object loop, no Pairs
                args = [Pair._numeric(a, copy=False) for a in args]
            return fn(*args, **kwargs)

        return ufunc_shim
    if (
        inspect.isfunction(fn)
        or inspect.ismethod(fn)
        or getattr(fn, "__module__", None) == "builtins"
        or hasattr(fn, "__wrapped__")  # numpy dispatchers: the protocol handles them
    ):
        return fn

    def wrapper(*args, **kwargs):
        if any(_traced(a) for a in args) or any(
            _traced(v) for v in kwargs.values()
        ):
            return Pair._opaque_call(fn, args, kwargs)
        return fn(*args, **kwargs)

    return wrapper


def _traced(a):
    return isinstance(a, Pair) or (
        isinstance(a, np.ndarray)
        and a.dtype == object
        and any(isinstance(e, Pair) for e in a.ravel())
    )


def _skv_opaque(fn):
    def wrapper(*args, **kwargs):
        if any(_traced(a) for a in args):
            return Pair._opaque_call(fn, args, kwargs)
        return fn(*args, **kwargs)

    return wrapper


class _Rewriter(ast.NodeTransformer):
    def __init__(self, fn_globals, tag=""):
        self.fn_globals = fn_globals
        self.tag = tag
        self.sites = []
        self.callees = set()
        self.wrapped = 0

    def _tag_loop(self, node):
        self.generic_visit(node)
        loop_id = f"{self.tag}:{node.lineno}"
        marker = ast.Expr(
            ast.Call(
                func=ast.Name(id="__skv_loop_iter__", ctx=ast.Load()),
                args=[ast.Constant(value=loop_id)],
                keywords=[],
            )
        )
        end = ast.Expr(
            ast.Call(
                func=ast.Name(id="__skv_loop_end__", ctx=ast.Load()),
                args=[ast.Constant(value=loop_id)],
                keywords=[],
            )
        )
        node.body.insert(0, marker)
        self.sites.append(f"loop {loop_id} tagged")
        return [node, end]

    def visit_Expr(self, node):
        # `fn(x, t, k, ab.T, n)` filling ab in place -> `ab = twin(...)`;
        # multiple out-params become tuple assignments. Out-parameter
        # Cython calls turn into assignments of opaque atoms
        call = node.value if isinstance(node.value, ast.Call) else None
        name = self._target_name(call.func) if call else None
        if name in OPAQUE_OUT and not call.keywords:
            idxs = OPAQUE_OUT[name]
            targets, flags, ok = [], [], max(idxs) < len(call.args)
            if ok:
                for idx in idxs:
                    arg = call.args[idx]
                    transposed = (
                        isinstance(arg, ast.Attribute)
                        and arg.attr == "T"
                        and isinstance(arg.value, ast.Name)
                    )
                    base = (
                        arg.value
                        if transposed
                        else arg if isinstance(arg, ast.Name) else None
                    )
                    if base is None:
                        ok = False
                        break
                    targets.append(base.id)
                    flags.append(transposed)
            if ok:
                # visit args only: generic_visit would let the doorman
                # wrap the callee and rename the atom to 'wrapper'
                args = [self.visit(a) for a in call.args]
                for idx, tid in zip(idxs, targets):
                    args[idx] = ast.Name(id=tid, ctx=ast.Load())
                self.sites.append(f"{name} -> opaque out-parameter call")
                names = [ast.Name(id=t, ctx=ast.Store()) for t in targets]
                target = (
                    names[0]
                    if len(names) == 1
                    else ast.Tuple(elts=names, ctx=ast.Store())
                )
                return ast.Assign(
                    targets=[target],
                    value=ast.Call(
                        func=ast.Name(id="__skv_opaque_out__", ctx=ast.Load()),
                        args=[
                            call.func,
                            ast.Constant(value=tuple(idxs)),
                            ast.Constant(value=tuple(flags)),
                        ]
                        + args,
                        keywords=[],
                    ),
                )
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        return self._tag_loop(node)

    def visit_While(self, node):
        return self._tag_loop(node)

    def _target_name(self, func):
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return None

    def visit_Call(self, node):
        self.generic_visit(node)
        # a neutral function passed by REFERENCE (map(np.asarray_chkfinite,
        # ...)) must be swapped too, or Pairs decompress inside it
        for pos, arg in enumerate(node.args):
            ref = self._target_name(arg)
            if ref in NEUTRAL and isinstance(arg, (ast.Attribute, ast.Name)):
                self.sites.append(f"{ref} (reference) -> pair-preserving")
                node.args[pos] = ast.Name(id="__skv_neutral__", ctx=ast.Load())
        name = self._target_name(node.func)
        if name in ALLOC:
            self.sites.append(f"{name} -> traced allocation")
            node.func = ast.Name(id=f"__skv_{name}__", ctx=ast.Load())
        elif name in NEUTRAL:
            self.sites.append(f"{name} -> pair-preserving")
            node.func = ast.Name(id="__skv_neutral__", ctx=ast.Load())
        elif name in CONCRETE:
            self.sites.append(f"{name} -> concrete-lane check")
            node = ast.Call(
                func=ast.Name(id="__skv_concrete__", ctx=ast.Load()),
                args=[ast.Constant(value=name)] + node.args,
                keywords=node.keywords,
            )
        elif name == "isinstance" and isinstance(node.func, ast.Name):
            self.sites.append("isinstance -> Pair counts as ndarray")
            node.func = ast.Name(id="__skv_isinstance__", ctx=ast.Load())
        elif name in SCALARIZE and isinstance(node.func, ast.Name):
            self.sites.append(f"{name} -> concrete scalar")
            node = ast.Call(
                func=ast.Name(id="__skv_scalarize__", ctx=ast.Load()),
                args=[node.func] + node.args,
                keywords=[],
            )
        elif name == "super" and not node.args:
            # re-exec'd methods lose the __class__ cell zero-arg super()
            # needs; the twin class is injected as __skv_class__
            node.args = [
                ast.Name(id="__skv_class__", ctx=ast.Load()),
                ast.Name(id="self", ctx=ast.Load()),
            ]
            self.sites.append("super() -> explicit twin super")
        elif name == "at":
            self.sites.append("at -> traced functional update")
            node = ast.Call(
                func=ast.Name(id="__skv_at__", ctx=ast.Load()),
                args=[node.func] + node.args,
                keywords=node.keywords,
            )
        elif name == "array_namespace":
            self.sites.append("array_namespace -> numpy (compat layer skipped)")
            node.func = ast.Name(id="__skv_namespace__", ctx=ast.Load())
        elif name == "finfo":
            self.sites.append("finfo -> concrete-lane precision")
            node.func = ast.Call(
                func=ast.Name(id="__skv_finfo__", ctx=ast.Load()),
                args=[node.func],
                keywords=[],
            )
        elif name in CONCRETE_CALLABLES:
            self.sites.append(f"{name} -> concrete lookup")
            node.func = ast.Call(
                func=ast.Name(id="__skv_concrete_call__", ctx=ast.Load()),
                args=[node.func],
                keywords=[],
            )
        elif name in OPAQUE_CALLABLES:
            self.sites.append(f"{name} -> opaque contract call")
            node.func = ast.Call(
                func=ast.Name(id="__skv_opaque__", ctx=ast.Load()),
                args=[node.func],
                keywords=[],
            )
        elif name in NEUTRAL_METHODS and isinstance(node.func, ast.Attribute):
            self.sites.append(f".{name}() -> pair-preserving")
            node = ast.Call(
                func=ast.Name(id="__skv_method__", ctx=ast.Load()),
                args=[ast.Constant(value=name), node.func.value] + node.args,
                keywords=node.keywords,
            )
        elif isinstance(node.func, ast.Name):
            self.callees.add(node.func.id)
            node = self._maybe_opaque(node)
        elif isinstance(node.func, ast.Attribute):
            node = self._maybe_opaque(node)
        return node

    def visit_Name(self, node):
        if node.id == "__class__":
            node.id = "__skv_class__"  # re-exec'd methods lose the cell
        return node

    def _maybe_opaque(self, node):
        # the doorman: unregistered COMPILED callables auto-seal into
        # opaque atoms at runtime; Python functions, ufuncs, builtins
        # and classes pass through untouched
        if isinstance(node.func, ast.Name) and node.func.id.startswith("__skv"):
            return node
        self.wrapped += 1
        node.func = ast.Call(
            func=ast.Name(id="__skv_maybe__", ctx=ast.Load()),
            args=[node.func],
            keywords=[],
        )
        return node


def instrument(fn, depth=3):
    """A semantically identical copy of fn with math-neutral calls
    replaced by traced twins. Returns (fn_copy, sites); on any failure
    returns (fn, ()) so tracing proceeds uninstrumented."""
    try:
        if getattr(fn, "__closure__", None):
            inner = fn
            while getattr(inner, "__wrapped__", None) is not None:
                inner = inner.__wrapped__  # peel stacked decorators
            if inner is fn:
                inner = None
            if inspect.isfunction(inner) and not inner.__closure__:
                # a wrapped function is just two functions: trace the
                # inner one; to_sympy verifies the wrapper was neutral
                # for this call by rerunning the real thing on values
                sub, sites = _instrument(inner, depth, seen=set())
                return sub, (f"{fn.__name__}: decorator unwrapped",) + sites
            return fn, ()  # opaque closures do not survive re-exec
        return _instrument(fn, depth, seen=set())
    except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
        return fn, ()


def _instrument(fn, depth, seen):
    source = textwrap.dedent(inspect.getsource(fn))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # getsource of a lambda inside a larger expression can return a
        # fragment; grab the lambda subexpression instead
        tree = None
    if tree is None or not isinstance(tree.body[0], (ast.FunctionDef,)):
        lam = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Lambda):
                    lam = node
                    break
        if lam is None:
            raise TypeError("no function definition found in source")
        fdef = ast.FunctionDef(
            name="__skv_lambda__",
            args=lam.args,
            body=[ast.Return(value=lam.body)],
            decorator_list=[],
            type_params=[],
        )
        tree = ast.Module(body=[fdef], type_ignores=[])
        ast.fix_missing_locations(tree)
    else:
        fdef = tree.body[0]
    if fdef.decorator_list:
        # transparent decorators (metadata-only, returned the original
        # function: no closure, no __wrapped__) are safe to strip; a
        # wrapping decorator changes semantics, so bail. A __class__
        # cell alone is fine: super() is rewritten explicit
        only_class_cell = all(
            v == "__class__" for v in fn.__code__.co_freevars
        )
        if getattr(fn, "__wrapped__", None) is not None or (
            fn.__closure__ and not only_class_cell
        ):
            raise TypeError("wrapped functions are not instrumented")
        fdef.decorator_list = []
    rewriter = _Rewriter(fn.__globals__, tag=fn.__name__)
    rewriter.visit(tree)
    ast.fix_missing_locations(tree)

    namespace = dict(fn.__globals__)
    namespace["__skv_zeros__"] = _skv_zeros
    namespace["__skv_empty__"] = _skv_empty
    namespace["__skv_ones__"] = _skv_ones
    namespace["__skv_full__"] = _skv_full
    namespace["__skv_neutral__"] = _skv_neutral
    namespace["__skv_opaque__"] = _skv_opaque
    namespace["__skv_method__"] = _skv_method
    namespace["__skv_maybe__"] = _skv_maybe
    namespace["__skv_finfo__"] = _skv_finfo
    namespace["__skv_namespace__"] = _skv_namespace
    namespace["__skv_at__"] = _skv_at
    namespace["__skv_concrete__"] = _skv_concrete
    namespace["__skv_scalarize__"] = _skv_scalarize
    namespace["__skv_isinstance__"] = _skv_isinstance
    namespace["__skv_concrete_call__"] = _skv_concrete_call
    namespace["__skv_opaque_out__"] = _skv_opaque_out
    namespace["__skv_loop_iter__"] = _loop_iter
    namespace["__skv_loop_end__"] = _loop_end

    sites = list(rewriter.sites)
    if rewriter.wrapped and not sites:
        sites.append(f"auto-opaque guard on {rewriter.wrapped} calls")
    if depth > 0:
        for name in rewriter.callees:
            callee = fn.__globals__.get(name)
            if getattr(callee, "__name__", None) in OPAQUE_CALLABLES:
                continue  # will be sealed at the boundary, not entered
            _inner = callee
            while getattr(_inner, "__wrapped__", None) is not None:
                _inner = _inner.__wrapped__  # peel stacked decorators
            if (
                callable(callee)
                and getattr(callee, "__closure__", None)
                and _inner is not callee
                and inspect.isfunction(_inner)
                and not _inner.__closure__
                and callee.__name__ not in seen
            ):
                # decorator-wrapped callee: instrument the inner function
                seen.add(callee.__name__)
                try:
                    sub, sub_sites = _instrument(_inner, depth - 1, seen)
                except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
                    continue
                namespace[name] = sub
                sites.append(f"{name}: decorator unwrapped")
                sites.extend(f"{name}: {t}" for t in sub_sites)
                continue
            if inspect.isclass(callee):
                try:
                    twin, twin_sites = _instrument_class(callee, depth, seen)
                except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
                    continue
                if twin_sites:
                    namespace[name] = twin
                    sites.extend(f"{name}: {t}" for t in twin_sites)
                continue
            if (
                inspect.isfunction(callee)
                and not getattr(callee, "__closure__", None)
                and callee.__module__ not in (None, "builtins")
            ):
                if callee in _FN_MEMO:
                    sub, sub_sites = _FN_MEMO[callee]
                elif callee.__name__ in seen:
                    continue  # cycle guard only; memo handles reuse
                else:
                    seen.add(callee.__name__)
                    try:
                        sub, sub_sites = _instrument(callee, depth - 1, seen)
                    except (
                        OSError,
                        TypeError,
                        SyntaxError,
                        KeyError,
                        AttributeError,
                    ):
                        continue
                    _FN_MEMO[callee] = (sub, sub_sites)
                if sub_sites:
                    namespace[name] = sub
                    sites.extend(f"{name}: {s}" for s in sub_sites)

    code = compile(tree, filename=f"<instrumented {fn.__name__}>", mode="exec")
    exec(code, namespace)
    return namespace[fdef.name], tuple(sites)


_CLASS_TWINS = {}
_FN_MEMO = {}  # instrumented-callee reuse across namespaces (class methods)


def _instrument_class(C, depth, seen):
    """A parallel twin of the class: every method instrumented, the
    base twinned too so super() chains stay instrumented. Returns
    (twin_or_C, sites)."""
    if C in _CLASS_TWINS:
        return _CLASS_TWINS[C]
    if (
        not inspect.isclass(C)
        or C is object
        or C.__module__ in (None, "builtins")
        or len(C.__bases__) != 1
    ):
        return C, ()
    key = f"class:{C.__module__}.{C.__qualname__}"
    if key in seen:
        return C, ()
    seen.add(key)

    base, base_sites = (
        _instrument_class(C.__bases__[0], depth, seen)
        if C.__bases__[0] is not object
        else (object, ())
    )
    members = {
        k: v
        for k, v in vars(C).items()
        if k not in ("__dict__", "__weakref__")
    }
    slots = members.pop("__slots__", None)
    if slots is not None:
        # the twin gets a plain __dict__; slot descriptors would clash
        for slot in ((slots,) if isinstance(slots, str) else slots):
            members.pop(slot, None)
    sites = [f"{C.__name__}(base): {t}" for t in base_sites]
    patch_namespaces = []
    for name, raw in list(members.items()):
        if name in (
            "__getattribute__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
        ):
            continue  # attribute plumbing: instrumenting it recurses
        wrap, target = None, None
        if isinstance(raw, staticmethod):
            target, wrap = raw.__func__, staticmethod
        elif isinstance(raw, classmethod):
            target, wrap = raw.__func__, classmethod
        elif inspect.isfunction(raw):
            target = raw
        elif callable(raw) and getattr(raw, "__wrapped__", None) is not None:
            # decorator-wrapped method (xp_capabilities): peel to the
            # inner function, same as module-level callees
            inner = raw
            while getattr(inner, "__wrapped__", None) is not None:
                inner = inner.__wrapped__
            if inspect.isfunction(inner):
                target = inner
        if target is None or not inspect.isfunction(target):
            continue  # classmethod(GenericAlias) and similar non-functions
        freevars = getattr(target.__code__, "co_freevars", ())
        if any(v != "__class__" for v in freevars):
            # a decorated method: the wrapper is a closure, but the
            # real function underneath may not be -- peel it
            inner = target
            while getattr(inner, "__wrapped__", None) is not None:
                inner = inner.__wrapped__
            if inner is target and target.__closure__:
                # wrappers without __wrapped__ (deprecation shims) hide
                # the real method in a closure cell: find it by name
                for cell in target.__closure__:
                    try:
                        held = cell.cell_contents
                    except ValueError:
                        continue
                    if inspect.isfunction(held) and held.__name__ == name:
                        inner = held
                        break
            if (
                inner is not target
                and inspect.isfunction(inner)
                and all(
                    v == "__class__"
                    for v in getattr(inner.__code__, "co_freevars", ())
                )
            ):
                target = inner
            else:
                chain = [target]
                while getattr(chain[-1], "__wrapped__", None) is not None:
                    chain.append(chain[-1].__wrapped__)
                if any(
                    "__class__"
                    in getattr(getattr(f, "__code__", None), "co_freevars", ())
                    for f in chain
                ):
                    # an unpeelable super()-using member: a parallel twin
                    # would break its super chain. No twin for this class
                    _CLASS_TWINS[C] = (C, ())
                    return C, ()
                continue  # real closures don't survive re-exec
        try:
            tree_fn = _instrument(target, depth - 1, seen)
        except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
            continue
        sub, sub_sites = tree_fn
        if sub_sites:
            members[name] = wrap(sub) if wrap else sub
            patch_namespaces.append(sub.__globals__)
            sites.extend(f"{C.__name__}.{name}: {t}" for t in sub_sites)
    if not sites:
        _CLASS_TWINS[C] = (C, ())
        return C, ()
    # methods carried over UNinstrumented but holding the original
    # class in their __class__ cell would break super(): clone them
    # with a fresh cell and rebind it to the twin
    import types

    rebind = []
    for name, raw in list(members.items()):
        target = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        if (
            inspect.isfunction(target)
            and getattr(target.__code__, "co_freevars", ()) == ("__class__",)
            and target.__closure__ is not None
        ):
            cell = types.CellType(None)
            clone = types.FunctionType(
                target.__code__,
                target.__globals__,
                target.__name__,
                target.__defaults__,
                (cell,),
            )
            clone.__kwdefaults__ = target.__kwdefaults__
            wrapper_kind = type(raw) if isinstance(raw, (staticmethod, classmethod)) else None
            members[name] = wrapper_kind(clone) if wrapper_kind else clone
            rebind.append(cell)
    try:
        twin = type(C.__name__, (base,), members)
    except TypeError:
        _CLASS_TWINS[C] = (C, ())
        return C, ()
    for cell in rebind:
        cell.cell_contents = twin
    for ns in patch_namespaces:
        ns["__skv_class__"] = twin
    _CLASS_TWINS[C] = (twin, tuple(sites))
    return twin, tuple(sites)
