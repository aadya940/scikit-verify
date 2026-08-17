"""The differential fuzzer: random op chains, one invariant.

King's commutativity square: evaluating the lifted formula at the concrete
inputs must reproduce the concrete result. Hypothesis invents the programs;
loud refusals (NotImplementedError/ValueError) are legal outcomes; silent
disagreement between the two lanes is the only failure.
"""

import numpy as np
import sympy
from hypothesis import given, settings, strategies as st

from skverify import Pair
from skverify.helpers import axis_idx


def evaluate(pair, arrays):
    """Numerically evaluate pair.formula against the named input arrays."""
    mapping = {}
    for name, arr in arrays.items():
        base = sympy.IndexedBase(name)
        arr = np.atleast_1d(arr)
        for idx in np.ndindex(arr.shape):
            key = base[idx] if len(idx) > 1 else base[idx[0]]
            mapping[key] = float(arr[idx])

    def eval_expr(expr):
        v = expr.doit().xreplace(mapping)
        if isinstance(v, (int, float)):
            return float(v)  # xreplace fully collapsed to a python number
        return float(sympy.N(v.doit()))

    if pair.domain is None:
        return eval_expr(pair.formula)
    if isinstance(pair.formula, sympy.Array):
        flat = [eval_expr(e) for e in pair.formula]
        return np.array(flat).reshape(np.shape(pair.value))
    bounds = pair._axis_bounds
    out = np.empty(tuple(hi - lo for lo, hi in bounds))
    for pos in np.ndindex(out.shape):
        letters = {axis_idx(ax): pos[ax] for ax in range(len(bounds))}
        out[pos] = eval_expr(pair.formula.subs(letters, simultaneous=True))
    return out


# the op menu: each entry is (name, applies_fn); refusals are legal
OPS_1D = st.sampled_from(
    [
        ("slice_head", lambda u, k: u[k:]),
        ("slice_tail", lambda u, k: u[: -k or None]),
        ("stride2", lambda u, k: u[::2]),
        ("flip", lambda u, k: u[::-1]),
        ("exp", lambda u, k: np.exp(u * 0.3)),
        ("sin", lambda u, k: np.sin(u)),
        ("mul", lambda u, k: u * (0.5 + k)),
        ("add_scalar", lambda u, k: u + k),
        ("rsub", lambda u, k: k - u),
        ("square", lambda u, k: u**2),
        ("neg", lambda u, k: -u),
        ("self_stencil", lambda u, k: u[1:] - u[:-1]),
        ("self_add", lambda u, k: u + u),
        ("mask_select", lambda u, k: np.where(u > 0.5, u, 0.0 * u)),
        ("mask_count_times", lambda u, k: (u > 0.5) * u),
        ("sum_full", lambda u, k: np.sum(u)),
        ("mask_count", lambda u, k: np.sum(u > 0.5)),
        ("write_int", lambda u, k: _written(u, k)),
        ("write_slice", lambda u, k: _slice_written(u, k)),
        ("write_mask", lambda u, k: _mask_written(u, k)),
    ]
)


def _written(u, k):
    r = u * 1.0
    r[k] = 0.25
    return r


def _slice_written(u, k):
    r = u * 1.0
    r[k:] = u[k:] * 2.0
    return r


def _mask_written(u, k):
    r = u * 1.0
    r[r > 0.5] = 0.5
    return r

VALUES = st.floats(-2.0, 2.0, allow_nan=False, width=32)


@settings(max_examples=200, deadline=None)
@given(
    data=st.lists(VALUES, min_size=3, max_size=7),
    chain=st.lists(st.tuples(OPS_1D, st.integers(1, 2)), min_size=1, max_size=4),
)
def test_two_lanes_agree_1d(data, chain):
    arr = np.array(data)
    u = Pair.array("u", arr)
    r = u
    for (name, op), k in chain:
        if not isinstance(r, Pair) or r.domain is None:
            break  # chain reduced to a scalar; nothing left to do
        try:
            r = op(r, k)
        except (NotImplementedError, ValueError, IndexError):
            return  # loud refusal is a legal outcome
    if not isinstance(r, Pair):
        return
    got = evaluate(r, {"u": arr})
    want = np.atleast_1d(r.value).astype(float)
    got = np.atleast_1d(got)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9), (
        f"lanes disagree after {[c[0][0] for c in chain]}: "
        f"formula gives {got}, execution gave {want}"
    )


OPS_2D = st.sampled_from(
    [
        ("row_slice", lambda u, k: u[k:, :]),
        ("col_slice", lambda u, k: u[:, k:]),
        ("transpose", lambda u, k: u.T),
        ("row_drop", lambda u, k: u[k]),
        ("stencil0", lambda u, k: u[1:, :] - u[:-1, :]),
        ("exp", lambda u, k: np.exp(u * 0.2)),
        ("mul", lambda u, k: u * (0.5 + k)),
        ("sum_axis0", lambda u, k: np.sum(u, axis=0)),
        ("sum_axis1", lambda u, k: np.sum(u, axis=1)),
        ("sum_full", lambda u, k: np.sum(u)),
        ("where", lambda u, k: np.where(u > 0.0, u, 0.0 * u)),
    ]
)


@settings(max_examples=200, deadline=None)
@given(
    data=st.lists(VALUES, min_size=12, max_size=12),
    chain=st.lists(st.tuples(OPS_2D, st.integers(1, 2)), min_size=1, max_size=4),
)
def test_two_lanes_agree_2d(data, chain):
    arr = np.array(data).reshape(3, 4)
    u = Pair.array("u", arr)
    r = u
    for (name, op), k in chain:
        if not isinstance(r, Pair) or r.domain is None:
            break
        try:
            r = op(r, k)
        except (NotImplementedError, ValueError, IndexError):
            return
    if not isinstance(r, Pair):
        return
    got = np.atleast_1d(evaluate(r, {"u": arr}))
    want = np.atleast_1d(r.value).astype(float)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9), (
        f"lanes disagree after {[c[0][0] for c in chain]}: "
        f"formula gives {got}, execution gave {want}"
    )


@settings(max_examples=100, deadline=None)
@given(
    data=st.lists(VALUES, min_size=4, max_size=6),
    other=st.lists(VALUES, min_size=4, max_size=6),
)
def test_two_pair_interactions(data, other):
    # cross-Pair broadcasting: 2-D against 1-D, the letter-bug territory
    n = min(len(data), len(other))
    a2 = np.array(data[: n * 2] + data[:n] * max(0, 2 * n - len(data)))[: 2 * n]
    arr2 = np.abs(a2).reshape(2, n) + 0.1
    arr1 = np.array(other[:n])
    u = Pair.array("u", arr2)
    v = Pair.array("v", arr1)
    r = u * v + v
    got = evaluate(r, {"u": arr2, "v": arr1})
    assert np.allclose(got, r.value.astype(float), rtol=1e-9)
