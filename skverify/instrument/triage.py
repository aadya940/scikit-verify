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
from .registries import CONCRETE_BY_NAME, NEUTRAL, OPAQUE_CALLABLES

# ndarray methods whose signature matches their np.* function: only
# these may route a decompressed array through the function's entry
_BAG_REDUCTIONS = {"mean", "sum", "prod", "min", "max", "median", "var", "std", "ptp"}

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
    # name-mangled privates (__marker) and C-adjacent idioms. numpy is
    # VOCABULARY, not user code: its meanings live in the registries
    # and the dispatch protocol, and twinning its buffer-trick bodies
    # (eye's strided flat write) walks straight into aliasing walls
    mod = (getattr(fn, "__module__", "") or "").split(".")[0]
    return mod not in sys.stdlib_module_names and mod not in (
        "builtins",
        "skverify",
        "numpy",
    )


def _skv_maybe(fn):
    """Pass Python functions, methods, classes, ufuncs and builtins
    through; wrap anything COMPILED so traced arguments turn the call
    into an opaque atom instead of a crash. Curated boundaries match
    the RESOLVED callable's name, so aliases and attribute lookups
    cannot dodge them."""
    if getattr(fn, "__name__", None) in OPAQUE_CALLABLES:
        return _skv_opaque(fn)
    if (
        getattr(fn, "__name__", None) in NEUTRAL
        and (getattr(fn, "__module__", "") or "").split(".")[0] == "numpy"
    ):
        # the asarray family reached by REFERENCE (toarray = np.asarray;
        # toarray(a)): neutral by resolved identity, not spelling --
        # numpy bodies are never twinned, so this is the only doorman
        from .runtime import _skv_neutral

        return _skv_neutral
    if getattr(fn, "__name__", None) in ("diags_array", "diags"):
        # sparse diagonal constructors: on traced data the matrix
        # lives dense in the value lane (structured allocation, the
        # eye family); untraced calls go to the real constructor
        from .runtime import _traced_diags

        def diags_shim(diagonals, offsets=0, shape=None, format=None,
                       dtype=None):
            return _traced_diags(fn, diagonals, offsets=offsets,
                                 shape=shape, format=format, dtype=dtype)

        return diags_shim
    if getattr(fn, "__name__", None) in CONCRETE_BY_NAME:
        # inventory routines run on concrete values: their results are
        # facts about this trace, and their bodies (sorting, boolean
        # index tricks) are hostile to traced operands
        def concrete_inventory(*args, **kwargs):
            from ..coercion import value_of

            def deep(a):
                if isinstance(a, (list, tuple)):
                    return type(a)(deep(e) for e in a)
                if isinstance(a, np.ndarray) and a.dtype == object:
                    # let numpy INFER the dtype: an int-valued index
                    # array must stay integer (scipy's sparse ctor
                    # silently zeroes on float indices)
                    vals = [value_of(e) for e in a.ravel()]
                    return np.array(vals).reshape(a.shape)
                v = value_of(a)
                if isinstance(v, np.ndarray) and v.dtype == object:
                    vals = [value_of(e) for e in v.ravel()]
                    return np.array(vals).reshape(v.shape)
                return v

            out_args = [deep(a) for a in args]
            if fn.__name__.startswith(("coo_", "csr_", "csc_")) and out_args:
                # sparse constructors: the index structure must be
                # integer -- scipy silently builds a ZERO matrix from
                # float indices (even integral ones). Data stays as-is.
                def intify(x):
                    if isinstance(x, tuple):
                        return tuple(intify(e) for e in x)
                    if (
                        isinstance(x, np.ndarray)
                        and x.dtype.kind == "f"
                        and np.all(x == np.floor(x))
                    ):
                        return x.astype(np.int64)
                    return x

                dt = kwargs.get("dtype")
                if dt is not None and np.dtype(dt) == object:
                    # an object dtype (derived from traced weights)
                    # makes scipy build an object matrix that folds to
                    # zeros; the concretized data's own dtype is right
                    kwargs = dict(kwargs)
                    del kwargs["dtype"]
                first = out_args[0]
                if isinstance(first, tuple) and len(first) in (2, 3):
                    out_args[0] = (first[0],) + intify(first[1:])
            return fn(*out_args, **kwargs)

        return concrete_inventory
    self_arr = getattr(fn, "__self__", None)
    if isinstance(self_arr, (np.random.Generator, np.random.RandomState)):
        from ..atoms import RNG_DISTS, _base_draw, rng_draw

        if fn.__name__ in RNG_DISTS:
            # a random draw: sealed as a distribution-tagged atom, the
            # generator consumed exactly as in an untraced run
            # unreachable: under trace_rng generators are TracedGenerator
            # instances whose draw methods already route through rng_draw,
            # so this shim is shadowed
            base = _base_draw(self_arr, fn.__name__)  # pragma: no cover

            def rng_shim(*args, **kwargs):  # pragma: no cover
                return rng_draw(base, args, kwargs)

            return rng_shim  # pragma: no cover
        return fn
    if (
        isinstance(self_arr, np.ndarray)
        and self_arr.dtype == object
        and getattr(fn, "__name__", None) in _BAG_REDUCTIONS
        and any(isinstance(e, Pair) for e in self_arr.ravel())
    ):
        # a bound ndarray method on a decompressed traced array: the
        # traced data is __self__, invisible to argument checks. Route
        # through the method twin, which repacks when a pattern exists
        from .runtime import _skv_method

        name = fn.__name__

        def bag_method(*args, **kwargs):
            return _skv_method(name, self_arr, *args, **kwargs)

        return bag_method
    if inspect.isbuiltin(fn) and not isinstance(
        getattr(fn, "__self__", None), (np.ndarray, Pair, type(np))
    ):
        owner = getattr(fn, "__self__", None)
        mod = (
            fn.__module__
            or (type(owner).__module__ if owner is not None else "")
            or ""
        ).split(".")[0]
        name = getattr(fn, "__name__", "") or ""
        if (
            mod
            and mod not in sys.stdlib_module_names
            and mod not in ("builtins", "numpy")
            and not (name.startswith("__") and name.endswith("__"))
        ):
            # dunders are object protocol (constructors set state,
            # never return values), not mathematical routines
            # compiled extension code (SWIG, C API): ANY non-Python
            # call receiving traced data seals into an atom -- the
            # mechanism is universal, not curated
            return _skv_opaque(fn)
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
    if inspect.isfunction(fn) and getattr(fn, "__closure__", None):
        # a decorated function (validate_params-style): peel the
        # wrapper chain to the plain inner function and triage THAT --
        # the same treatment the static callee path gives decorators
        inner = fn
        while getattr(inner, "__wrapped__", None) is not None:
            inner = inner.__wrapped__
        if inner is fn:
            # decorators built without functools.wraps (scipy's
            # _axis_nan_policy) hide the inner function in a CLOSURE
            # CELL. A cell holding a function of the wrapper's own
            # name is that inner function -- exact identification by
            # the name-preserving convention, not a guess.
            named = [
                c.cell_contents
                for c in fn.__closure__
                if inspect.isfunction(getattr(c, "cell_contents", None))
                and c.cell_contents.__name__ == fn.__name__
            ]
            if len(named) == 1:
                inner = named[0]
        if (
            inner is not fn
            and inspect.isfunction(inner)
            and not getattr(inner, "__closure__", None)
            and _twinnable(inner)
        ):
            sub = runtime_twin(inner)
            if sub is not inner:
                wrapper = fn
                try:
                    inner_sig = inspect.signature(inner)
                except (ValueError, TypeError):
                    inner_sig = None

                try:
                    inspect.signature(wrapper)
                except (ValueError, TypeError):
                    pass

                def peeled(*args, **kwargs):
                    # some wrappers INJECT arguments into the inner
                    # call: decide by SIGNATURE, before calling -- a
                    # text-sniff on TypeError misattributes nested
                    # errors and silently reverts to the raw wrapper
                    if inner_sig is None:
                        return sub(*args, **kwargs)
                    try:
                        inner_sig.bind(*args, **kwargs)
                        return sub(*args, **kwargs)
                    except TypeError:
                        pass
                    # kwargs the wrapper accepts but the inner lacks
                    # (keepdims, _no_deco, nan_policy variants): drop
                    # them and peel anyway -- then VERIFY, per call, by
                    # running the wrapper on the concrete values and
                    # comparing. Names propose, runs dispose: a peel
                    # that changes this call's numbers loses to the
                    # untouched wrapper.
                    inner_params = set(inner_sig.parameters)
                    kept = {k: v for k, v in kwargs.items() if k in inner_params}
                    try:
                        inner_sig.bind(*args, **kept)
                        r = sub(*args, **kept)
                    except Exception:
                        return wrapper(*args, **kwargs)
                    from ..coercion import value_of

                    try:
                        ref = wrapper(
                            *[value_of(a) for a in args],
                            **{k: value_of(v) for k, v in kwargs.items()},
                        )
                        if _values_match(r, ref):
                            return r
                    except Exception:
                        pass
                    return wrapper(*args, **kwargs)

                return peeled
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
            if fn.__name__ in ("isnan", "isinf", "isfinite") and any(
                _traced(a) for a in args
            ):
                # validation checks are concrete facts about this
                # trace, not math (the CONCRETE registry doctrine);
                # numpy's float-only loops reject bags outright
                from ..coercion import value_of

                def conc(a):
                    if isinstance(a, np.ndarray) and a.dtype == object:
                        return np.asarray(
                            [value_of(e) for e in a.ravel()], dtype=float
                        ).reshape(a.shape)
                    return value_of(a)

                return fn(*[conc(a) for a in args], **kwargs)
            if any(_traced(a) and not isinstance(a, Pair) for a in args):
                # an object array HOLDING Pairs: numpy's object loop
                # cannot dispatch a mapped ufunc. Apply elementwise;
                # each scalar call re-enters the traced path
                target = kwargs.pop("out", None)
                if isinstance(target, tuple):
                    target = target[0]
                bcast = np.broadcast_arrays(
                    *[np.asarray(a, dtype=object) for a in args]
                )
                out = np.empty(bcast[0].shape, dtype=object)
                for idx in np.ndindex(bcast[0].shape):
                    out[idx] = fn(*[b[idx] for b in bcast], **kwargs)
                if target is not None:
                    # out= on an object array: element replacement is
                    # the in-place semantics callers rely on
                    target[...] = out
                    return target
                return out
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


def _values_match(r, ref):
    from ..coercion import value_of

    def flat(x):
        if isinstance(x, tuple):
            out = []
            for e in x:
                out.extend(flat(e))
            return out
        return [np.asarray(value_of(x), dtype=float)]

    try:
        a, b = flat(r), flat(ref)
        return len(a) == len(b) and all(
            np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
            for x, y in zip(a, b)
        )
    except Exception:
        return False


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


