"""Triage: the runtime dispatch policy for every call.

Instrumented code routes each call through :func:`_skv_maybe`, which
examines the resolved callable and assigns a treatment, in order:
curated opaque boundary, class (twin it), bound method (twin and
rebind), plain Python function (twin), numpy dispatcher (protocol or
table shim), ufunc/builtin (pass through), anything compiled left
over (seal into an atom when traced values arrive). Nothing receives
treatment before classification, and misclassification -- not
missing coverage -- is the failure mode this module guards.
"""

import inspect
import sys

import numpy as np

from ..pair import Pair
from ..session import current as _session
from .registries import OPAQUE_CALLABLES

# The instrumented-function cache is the session's (blank per trace).
_FN_MEMO = _session.fn_twins


def _twin_builders():
    """Late import of the twin builders; triage and twins are
    mutually recursive subsystems (a twin's calls route back through
    triage), so the reference resolves at call time."""
    from .twins import _instrument, _instrument_class

    return _instrument, _instrument_class

def runtime_twin(fn):
    """Memoized instrumented copy of a plain function, or the original
    when instrumentation finds nothing to rewrite."""
    if fn in _FN_MEMO:
        sub, sites = _FN_MEMO[fn]
        return sub if sites else fn
    try:
        sub, sites = _twin_builders()[0](fn, 0, set())
    except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
        sub, sites = fn, ()
    _FN_MEMO[fn] = (sub, sites)
    return sub if sites else fn


def _twinnable(fn):
    # stdlib code is never mathematics; re-exec also breaks its
    # name-mangled privates (__marker) and C-adjacent idioms
    mod = (getattr(fn, "__module__", "") or "").split(".")[0]
    return mod not in sys.stdlib_module_names and mod not in ("builtins", "skverify")


def _skv_maybe(fn):
    """Pass Python functions, methods, classes, ufuncs and builtins
    through; wrap anything COMPILED so traced arguments turn the call
    into an opaque atom instead of a crash. Curated boundaries match
    the RESOLVED callable's name, so aliases and attribute lookups
    cannot dodge them."""
    if getattr(fn, "__name__", None) in OPAQUE_CALLABLES:
        return _skv_opaque(fn)
    if inspect.isbuiltin(fn) and not isinstance(
        getattr(fn, "__self__", None), (np.ndarray, Pair, type(np))
    ):
        # builtin METHODS (list.append, dict.get) have __module__ None
        # and must never opaque: they would swallow Pairs into values
        return fn
    if inspect.isclass(fn):
        # classes reached through variables (dispatch tables) twin at
        # runtime; the cache makes this once per class
        try:
            twin, _ = _twin_builders()[1](fn, 3, set())
            return twin
        except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
            return fn
    if inspect.ismethod(fn) and inspect.isfunction(fn.__func__):
        # a bound method (norm.pdf): twin the function, rebind to the
        # instance; self.method calls inside chain through the doorman
        inner = fn.__func__
        if isinstance(fn.__self__, Pair) or not _twinnable(inner):
            return fn
        if not getattr(inner, "__closure__", None):
            sub = runtime_twin(inner)
            if sub is not inner:
                return sub.__get__(fn.__self__)
        return fn
    if inspect.isfunction(fn) and not getattr(fn, "__closure__", None):
        if _twinnable(fn) and "__skv" not in fn.__name__:
            if fn in _FN_MEMO:
                sub, sub_sites = _FN_MEMO[fn]
            else:
                try:
                    sub, sub_sites = _twin_builders()[0](fn, 0, set())
                except (OSError, TypeError, SyntaxError, KeyError, AttributeError):
                    sub, sub_sites = fn, ()
                _FN_MEMO[fn] = (sub, sub_sites)
            return sub if sub_sites else fn
        return fn
    if hasattr(fn, "__wrapped__") and not inspect.isfunction(fn):
        # numpy dispatcher: the protocol handles Pair args, but OBJECT
        # arrays of Pairs bypass it -- route those through our table
        from ..registry import FUNCTION_TABLE

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


