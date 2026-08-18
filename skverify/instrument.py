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
from .pair import Pair

ALLOC = {"zeros", "empty", "ones", "full"}
NEUTRAL = {"asarray", "asanyarray", "ascontiguousarray", "asfortranarray"}
OPAQUE_CALLABLES = {"solve_banded", "solveh_banded", "cho_solve", "design_matrix"}
NEUTRAL_METHODS = {"toarray", "astype", "copy"}

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
    if not isinstance(obj, Pair):
        return getattr(obj, name)(*args, **kwargs)
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


def _skv_opaque(fn):
    def wrapper(*args, **kwargs):
        if any(isinstance(a, Pair) for a in args):
            return Pair._opaque_call(fn, args, kwargs)
        return fn(*args, **kwargs)

    return wrapper


class _Rewriter(ast.NodeTransformer):
    def __init__(self, fn_globals):
        self.fn_globals = fn_globals
        self.sites = []
        self.callees = set()

    def _target_name(self, func):
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return None

    def visit_Call(self, node):
        self.generic_visit(node)
        name = self._target_name(node.func)
        if name in ALLOC:
            self.sites.append(f"{name} -> traced allocation")
            node.func = ast.Name(id=f"__skv_{name}__", ctx=ast.Load())
        elif name in NEUTRAL:
            self.sites.append(f"{name} -> pair-preserving")
            node.func = ast.Name(id="__skv_neutral__", ctx=ast.Load())
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
    rewriter = _Rewriter(fn.__globals__)
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

    sites = list(rewriter.sites)
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
