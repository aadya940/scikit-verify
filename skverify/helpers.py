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
    """Resolve a slice to concrete (start, stop, step).

    slice(1, None).indices(5)      -> (1, 5, 1)
    slice(None, None, -1).indices(5) -> (4, -1, -1)   # walks 4, 3, 2, 1, 0
    Negatives and out-of-range stops are clamped by Python itself.
    """
    return key.indices(length)


def normalize_key(key, lengths):
    """Canonicalize an nd-getitem key: one entry per axis.

    Returns (start, stop, step) for slice axes, a non-negative int for
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


def ops_capped(expr, limit, node_factor=8):
    """``count_ops(expr)`` when provably under ``limit``, else None.

    Certificates can be DAGs: a scatter formula nests the previous
    state in BOTH Piecewise branches, and sympy walks that shape as a
    tree -- count_ops on it is exponential in the write count. This
    walks at most ``node_factor * limit`` tree nodes first; a tree
    that finishes the walk is genuinely small and count_ops answers
    exactly and cheaply. A tree that trips the cap is declared over
    the limit: measured certificates carry 4-5 tree nodes per op, so
    the margin holds for every shape the tracer builds. A node-heavy
    op-light misclassification only routes to the conservative side
    (wall instead of grow, fold instead of inline) -- never to wrong
    mathematics.
    """
    if not isinstance(expr, sympy.Basic):
        return 0
    cap = node_factor * limit + 64
    n = 0
    stack = [expr]
    while stack:
        e = stack.pop()
        n += 1
        if n > cap:
            return None
        stack.extend(a for a in e.args if isinstance(a, sympy.Basic))
    total = int(sympy.count_ops(expr))
    return total if total < limit else None


def has_probe(expr, keys):
    """Do any ``keys`` symbols occur in ``expr``, as bare Symbols or as
    Indexed bases? Memoized scan: shared subtrees are visited once, so
    DAG-shaped formulas cost their DISTINCT size (free_symbols would
    walk them as exponential trees)."""
    if not isinstance(expr, sympy.Basic):
        return False
    seen = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if not isinstance(e, sympy.Basic) or id(e) in seen:
            continue
        seen.add(id(e))
        if e in keys:
            return True
        if isinstance(e, sympy.Indexed) and getattr(e.base, "label", None) in keys:
            return True
        stack.extend(e.args)
    return False


def swap_probes(expr, meanings, axis0):
    """Substitute probe Symbols and probe Indexed slots in one
    memoized post-order pass.

    ``meanings`` maps plain Symbols to expressions and IndexedBase
    labels to 1-D indexed formulas; ``slot[k]`` becomes the meaning
    with ``axis0`` replaced by ``k``. sympy's xreplace/replace re-walk
    shared subtrees on every visit, which is exponential on DAG-shaped
    scatter formulas; the memo makes this linear in distinct
    subexpressions. Scope-aware like subs: a binder's bound symbols
    (Sum/Product/Integral dummies) shadow map entries for that
    subtree, so a bound variable is never rewritten into its limits.
    Indexed slots substitute as whole nodes and IndexedBase args are
    never descended into, so an array probe's label cannot be
    rewritten into a malformed base. Rebuilds go through
    ``e.func(*args)`` exactly like xreplace, so the ambient
    ``sympy.evaluate`` flag is respected -- call under
    ``sympy.evaluate(False)`` when meanings hold Iterate structures.
    """
    if not isinstance(expr, sympy.Basic):
        return expr
    all_keys = frozenset(meanings)
    memos = {all_keys: {}}
    stack = [(expr, False, all_keys)]
    while stack:
        e, done, scope = stack.pop()
        memo = memos[scope]
        if not isinstance(e, sympy.Basic) or e in memo:
            continue
        if done:
            child_scope = scope
            bound = getattr(e, "bound_symbols", None)
            if bound:
                shadowed = scope - set(bound)
                if shadowed != scope:
                    child_scope = shadowed
            cmemo = memos.get(child_scope, memo)
            args = tuple(
                cmemo.get(a, a) if isinstance(a, sympy.Basic) else a
                for a in e.args
            )
            memo[e] = (
                e
                if all(a is b for a, b in zip(args, e.args))
                else e.func(*args)
            )
            continue
        if e in scope:
            memo[e] = meanings[e]
            continue
        if isinstance(e, sympy.Indexed):
            label = getattr(e.base, "label", None)
            if label in scope:
                # scope-aware recursion, not xreplace: a meaning that
                # binds the axis symbol in a Sum must keep its binder
                memo[e] = swap_probes(
                    meanings[label], {axis0: e.indices[0]}, axis0
                )
                continue
        if isinstance(e, sympy.tensor.indexed.IndexedBase) or not e.args:
            memo[e] = e
            continue
        child_scope = scope
        bound = getattr(e, "bound_symbols", None)
        if bound:
            shadowed = scope - set(bound)
            if shadowed != scope:
                child_scope = shadowed
                if child_scope not in memos:
                    memos[child_scope] = {}
                if not child_scope:
                    # every map entry shadowed: subtree is untouched
                    memo[e] = e
                    continue
        stack.append((e, True, scope))
        for a in e.args:
            if isinstance(a, sympy.Basic):
                stack.append((a, False, child_scope))
    return memos[all_keys].get(expr, expr)


def tree_nodes_capped(expr, cap):
    """Tree-node count of ``expr``, or None past ``cap``.

    Pure structural walk, no per-node sympy machinery: the growth
    tripwire needs a size measure, not an exact op census, and
    count_ops pays fraction()/signsimp-adjacent work on every node.
    Capped, so DAG shapes that expand to exponential trees cost at
    most ``cap`` visits.
    """
    if not isinstance(expr, sympy.Basic):
        return 0
    n = 0
    stack = [expr]
    while stack:
        e = stack.pop()
        n += 1
        if n > cap:
            return None
        stack.extend(a for a in e.args if isinstance(a, sympy.Basic))
    return n


def reevaluated(expr, cap=200_000):
    """Rebuild ``expr`` bottom-up under normal evaluation, restoring
    the evaluated normal form after evaluate(False) construction.

    Held structures -- Sum, Product, Iterate-style Functions with a
    ``__skv_held__`` marker -- and every ANCESTOR of one pass through
    verbatim: re-running a guarded reduction constructor is the
    piecewise_fold hoist hazard, and evaluating a parent (Abs.eval ->
    expand) with a held child as argument is the hang inline() guards
    against. Only subtrees fully free of held nodes re-evaluate: an
    unevaluated Add(2, 3) becomes 5, an unevaluated Abs collapses.
    Memoized (DAG-linear) and capped: past ``cap`` tree nodes the
    expression returns unchanged -- at that size it is wall territory
    and cosmetic normal form is moot.
    """
    if not isinstance(expr, sympy.Basic):
        return expr
    if tree_nodes_capped(expr, cap) is None:
        return expr
    held = (sympy.Sum, sympy.Product)

    def is_held(e):
        return isinstance(e, held) or getattr(e, "__skv_held__", False)

    memo = {}  # e -> (result, tainted)
    stack = [(expr, False)]
    while stack:
        e, done = stack.pop()
        if not isinstance(e, sympy.Basic) or e in memo:
            continue
        if done:
            tainted = any(
                memo[a][1]
                for a in e.args
                if isinstance(a, sympy.Basic) and a in memo
            )
            args = tuple(
                memo[a][0] if isinstance(a, sympy.Basic) and a in memo else a
                for a in e.args
            )
            unchanged = all(a is b for a, b in zip(args, e.args))
            try:
                if tainted:
                    # clean siblings normalize, but the node itself
                    # must not re-evaluate with a held child as
                    # argument (Abs.eval on an Iterate is the hang;
                    # a reduction ctor re-run is the hoist hazard).
                    # Add/Mul are exempt: flattening runs no .eval
                    if e.func in (sympy.Add, sympy.Mul):
                        memo[e] = (e.func(*args), True)
                    elif unchanged:
                        memo[e] = (e, True)
                    else:
                        with sympy.evaluate(False):
                            memo[e] = (e.func(*args), True)
                else:
                    memo[e] = (e.func(*args), False)
            except (TypeError, ValueError):
                memo[e] = (e, tainted)  # exotic node: keep the original
            continue
        if is_held(e):
            memo[e] = (e, True)
            continue
        if not e.args:
            memo[e] = (e, False)
            continue
        stack.append((e, True))
        for a in e.args:
            if isinstance(a, sympy.Basic):
                stack.append((a, False))
    return memo.get(expr, (expr, False))[0]
