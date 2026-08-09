"""Stateless utilities for skverify."""

import numpy as np
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


def normalize_key(key, lengths):
    """Canonicalize an nd-getitem key: one entry per axis.

    Returns (start, stop) for slice axes, a non-negative int for
    dropped axes. Handles, for example:
        1. Negative indices -> positive.
        2. `...` Ellipsis -> explicit full slices.
        3. Short keys -> padded with full slices.
    """
    ndim = len(lengths)
    if not isinstance(key, tuple):
        key = (key,)

    # refuse newaxis before the length check: `None` does not consume an axis
    if any(k is None for k in key):
        raise NotImplementedError("newaxis indexing not supported")

    # expand `...` into the full slices it stands for
    # (identity checks, not ==/count: an ndarray in the key breaks equality)
    ellipsis_positions = [p for p, k in enumerate(key) if k is Ellipsis]
    if len(ellipsis_positions) > 1:
        raise IndexError("an index can only have a single `...`")
    if ellipsis_positions:
        pos = ellipsis_positions[0]
        fill = (slice(None),) * (ndim - len(key) + 1)
        key = key[:pos] + fill + key[pos + 1 :]

    if len(key) > ndim:
        raise IndexError(f"too many indices: {len(key)} for {ndim}-D.")
    key = key + (slice(None),) * (ndim - len(key))  # pad short keys

    normalized = []
    for k, length in zip(key, lengths):
        if isinstance(k, slice):
            normalized.append(normalize_slice(k, length))
        elif k is None or isinstance(k, (bool, np.bool_)):
            raise NotImplementedError("newaxis/boolean indexing not supported")
        elif isinstance(k, (int, np.integer)):
            k = int(k) + length if k < 0 else int(k)
            if not 0 <= k < length:
                raise IndexError(f"index {k} out of bounds for length {length}")
            normalized.append(k)
        else:
            raise NotImplementedError("fancy indexing not supported")
    return tuple(normalized)
