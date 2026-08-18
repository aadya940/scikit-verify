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
}
NEUTRAL_METHODS = {"toarray", "astype", "copy", "view"}
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


def _skv_zeros(shape, dtype=None, **kwargs):
    return Pair(np.zeros(shape), sympy.Integer(0), _bounds_of(shape))


def _skv_ones(shape, dtype=None, **kwargs):
    return Pair(np.ones(shape), sympy.Integer(1), _bounds_of(shape))


def _skv_full(shape, fill, dtype=None, **kwargs):
    return Pair(
        np.full(shape, Pair._value_of(fill)),
        Pair._formula_of(fill),
        _bounds_of(shape),
        steps=Pair._steps_of(fill),
    )


def _skv_empty(shape, dtype=None, **kwargs):
    bounds = _bounds_of(shape)
    letters = tuple(axis_idx(ax) for ax in range(len(bounds)))
    return Pair(np.empty(shape), sympy.Function("uninitialized")(*letters), bounds)


def _skv_neutral(a, dtype=None, **kwargs):
    if isinstance(a, Pair):
        value = np.ascontiguousarray(a.value, dtype=dtype)
        return Pair(value, a.formula, a._axis_bounds, steps=(a,))
    return np.asarray(a, dtype=dtype)


def _skv_method(name, obj, *args, **kwargs):
    if (
        isinstance(obj, np.ndarray)
        and obj.dtype == object
        and all(isinstance(e, Pair) for e in obj.ravel())
    ):
        if name in ("astype", "copy"):
            return obj  # traced scalars; the cast/copy is math-neutral
    if not isinstance(obj, Pair):
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


def _skv_concrete(name, a, *args, **kwargs):
    return getattr(np, name)(np.asarray(Pair._value_of(a)), *args, **kwargs)


def _skv_scalarize(kind, x):
    return kind(Pair._value_of(x))


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


def _skv_maybe(fn):
    """Pass Python functions, methods, classes, ufuncs and builtins
    through; wrap anything COMPILED so traced arguments turn the call
    into an opaque atom instead of a crash."""
    if (
        inspect.isfunction(fn)
        or inspect.ismethod(fn)
        or isinstance(fn, (type, np.ufunc))
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
        elif name in SCALARIZE and isinstance(node.func, ast.Name):
            self.sites.append(f"{name} -> concrete scalar")
            node = ast.Call(
                func=ast.Name(id="__skv_scalarize__", ctx=ast.Load()),
                args=[node.func] + node.args,
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
            return fn, ()  # closures do not survive re-exec
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
        # wrapping decorator changes semantics, so bail
        if getattr(fn, "__wrapped__", None) is not None or fn.__closure__:
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
    namespace["__skv_concrete__"] = _skv_concrete
    namespace["__skv_scalarize__"] = _skv_scalarize
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
            if (
                inspect.isfunction(callee)
                and not getattr(callee, "__closure__", None)
                and callee.__module__ not in (None, "builtins")
                and callee.__name__ not in seen
            ):
                seen.add(callee.__name__)
                try:
                    sub, sub_sites = _instrument(callee, depth - 1, seen)
                except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
                    continue
                if sub_sites:
                    namespace[name] = sub
                    sites.extend(f"{name}: {s}" for s in sub_sites)

    code = compile(tree, filename=f"<instrumented {fn.__name__}>", mode="exec")
    exec(code, namespace)
    return namespace[fdef.name], tuple(sites)
