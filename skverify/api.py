"""Public API."""

import inspect

import numpy as np
import sympy

from .pair import Pair


def to_sympy(fn, *args):
    """Run ``fn`` with tracing values.

    Array arguments become indexed formulas named after the function's
    parameters; float arguments become symbols of the same name. Ints,
    bools, strings and None pass through untraced: they are configuration
    (an ``n=``, an ``axis=``), not mathematics.
    Returns the traced result: ``.formula``, ``.value``, ``.domain``.
    """
    wrapped = [_wrap(name, val) for name, val in _infer_names(fn, args)]
    return _repack(fn(*wrapped))


def _wrap(name, val):
    if val is None or isinstance(val, (bool, np.bool_, int, np.integer, str)):
        return val  # config, not math: np.diff(a, 2) keeps its plain 2
    if np.isscalar(val):
        return Pair(val, sympy.Symbol(name, real=True))
    return Pair.array(name, val)


def _repack(out):
    """Normalize the traced result to one object with .formula/.value/.domain.

    The fallback path (numpy's own bodies run on Pairs) returns an ndarray
    whose ELEMENTS are scalar Pairs — formulas unrolled per element. Repack
    into a single Pair: values as a real ndarray, formulas as a sympy.Array.
    """
    if not (isinstance(out, np.ndarray) and out.dtype == object):
        return out
    elements = out.ravel()
    if not all(isinstance(p, Pair) for p in elements):
        return out  # not ours: leave untouched
    values = np.array([p.value for p in elements]).reshape(out.shape)
    formulas = sympy.Array([p.formula for p in elements], out.shape)
    return Pair(
        values,
        formulas,
        domain=tuple((0, s) for s in out.shape),
    )


def _infer_names(fn, args):
    """Pair each positional argument with its parameter name from fn's signature."""
    names = list(inspect.signature(fn).parameters)
    if len(args) > len(names):
        raise TypeError(f"{fn.__name__} takes {len(names)} arguments, got {len(args)}")
    return zip(names, args)
