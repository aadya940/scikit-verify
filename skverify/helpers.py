"""Stateless utilities for skverify."""

import sympy

_AXIS_SYMBOLS = [sympy.Symbol(sym, integer=True) for sym in "i j k l m".split(" ")]


def axis_idx(ax):
    try:
        return _AXIS_SYMBOLS[ax]
    except IndexError:
        raise NotImplementedError(
            f"arrays beyond {len(_AXIS_SYMBOLS)}-D are not supported"
        )


def normalize_slice(key, length):
    """Resolve a step-1 slice to concrete (start, stop) in [0, length]."""
    if key.step not in (None, 1):
        raise NotImplementedError("only step-1 slices are supported")
    start = key.start or 0
    stop = length if key.stop is None else key.stop
    if start < 0:
        start += length
    if stop < 0:
        stop += length
    return start, stop
