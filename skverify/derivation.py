"""The derivation subsystem: from a Pair's provenance DAG to prose.

A traced result remembers its parents. This module turns that memory
into something a person (or a model) can read, in three passes:

1. Delta abstraction (:func:`_delta_steps`): each step's formula with
   earlier steps' formulas replaced by ``step[m]`` references, so
   repetition becomes visible as index arithmetic.
2. Structure recovery: loop-provenance events recorded by the
   instrumentation group steps by the program's own (loop, iteration)
   context (:func:`_group_tree`); iteration bodies generalize against
   each other by anti-unification (:func:`_generalize`), and
   accumulator slots earn verified Sum/Product closed forms
   (:func:`_close_form`). Without provenance, flat period-scanning
   (:func:`_fold_runs`) is the fallback.
3. Rendering: shared subexpressions get cse names; pure-reference
   noise lines are dropped; the result is always the last line.

Every fold is verified member-by-member by exact substitution before
it replaces anything: the pretty view can never drift from the truth.
"""

import sympy

from .session import current as _session

# The moved code speaks in the historical alias names; they remain the
# session's own collections.
_GUARDS = _session.guards
_OPAQUE = _session.opaque
_LOOP_EVENTS = _session.loop_events
_LOOP_STACK = _session.loop_stack


def _loop_iter(loop_id):
    if _LOOP_STACK and _LOOP_STACK[-1][0] == loop_id:
        _LOOP_STACK[-1][1] += 1
    else:
        _LOOP_STACK.append([loop_id, 0])
    _LOOP_EVENTS.append((_session.seq, tuple((l, i) for l, i in _LOOP_STACK)))
    from .recurrence import on_loop_iter

    on_loop_iter(loop_id, _LOOP_STACK[-1][1])


def _loop_end(loop_id):
    if _LOOP_STACK and _LOOP_STACK[-1][0] == loop_id:
        _LOOP_STACK.pop()
    _LOOP_EVENTS.append((_session.seq, tuple((l, i) for l, i in _LOOP_STACK)))
    from .recurrence import on_loop_end

    on_loop_end(loop_id)


def _context_of(seq):
    """The loop context stack active when the node was created."""
    from bisect import bisect_left

    pos = bisect_left(_LOOP_EVENTS, (seq,)) - 1
    return _LOOP_EVENTS[pos][1] if pos >= 0 else ()


def _fresh_name(base, exprs):
    taken = set()
    for e in exprs:
        taken |= {s.name for s in e.free_symbols}
    name, c = base, 2
    while name in taken:
        name, c = f"{base}{c}", c + 1
    return sympy.Symbol(name, integer=True)


def _generalize(e1, e2, k):
    """The template both expressions instantiate: equal parts kept,
    Integers that differ become linear in k (k=0 gives e1, k=1 gives
    e2). None when the trees differ in any non-Integer way."""
    if e1 == e2:
        return e1
    if isinstance(e1, sympy.Integer) and isinstance(e2, sympy.Integer):
        return e1 + k * (e2 - e1)
    if isinstance(e1, sympy.Float) and isinstance(e2, sympy.Float):
        # float constants drift linearly too (float(k) casts); both
        # consumers verify every instantiation exactly afterwards, so
        # a wrong lift can only fail a fold, never corrupt a formula
        return e1 + k * (e2 - e1)
    if e1.func is not e2.func or len(e1.args) != len(e2.args) or not e1.args:
        return None
    args = [_generalize(a, b, k) for a, b in zip(e1.args, e2.args)]
    if any(a is None for a in args):
        return None
    return e1.func(*args)


_STEP = sympy.IndexedBase("step")


def _delta_steps(steps, nodes=None):
    """Steps with earlier steps' formulas abstracted to step[m]
    references, at any distance. Cross-iteration references then
    generalize as linear index expressions (step[9*n + 37]) instead of
    breaking the fold.

    sympy flattens Add/Mul, so an accumulator's previous value is not
    an exact subtree of the next (u0*u1*u2 does not contain u0*u1).
    With nodes given, the DAG's parents guide a multiset factoring:
    the parent's args are removed and replaced by its step reference."""
    index = {id(n): m for m, n in enumerate(nodes)} if nodes else {}
    mapping = {}
    deltas = []
    for m, expr in enumerate(steps):
        original = expr
        if (
            nodes is not None
            and expr not in mapping
            and isinstance(expr, (sympy.Add, sympy.Mul))
        ):
            for parent in nodes[m]._parents:
                pi = index.get(id(parent))
                if pi is None or pi >= m:
                    continue
                f = steps[pi]
                if f.is_Atom or f == expr:
                    continue
                args = list(expr.args)
                if f.func is expr.func:
                    fargs = list(f.args)
                    if all(fargs.count(a) <= args.count(a) for a in set(fargs)):
                        for a in fargs:
                            args.remove(a)
                        expr = expr.func(_STEP[pi], *args)
                        continue
                if f in args:
                    args[args.index(f)] = _STEP[pi]
                    expr = expr.func(*args)
        if mapping:
            expr = expr.xreplace(mapping)
        deltas.append(expr)
        if not original.is_Atom and original not in mapping:
            mapping[original] = _STEP[m]
    return deltas


def _fold_runs(deltas, k, min_blocks=3, max_period=24):
    """Consecutive expressions repeating as one block of templates
    under k collapse to (templates, start, blocks); period-p blocks
    cover alternating patterns (gather then write). Every member is
    verified by exact subs equality before folding. Unfolded
    expressions stay ((expr,), m, 1)."""
    items, m = [], 0
    n = len(deltas)
    while m < n:
        folded = False
        for p in range(1, max_period + 1):
            if m + 2 * p > n:
                break
            ts = [
                _generalize(deltas[m + j], deltas[m + p + j], k) for j in range(p)
            ]
            if any(t is None for t in ts) or not any(t.has(k) for t in ts):
                continue
            blocks = 0
            while m + (blocks + 1) * p <= n and all(
                ts[j].subs(k, blocks) == deltas[m + blocks * p + j]
                for j in range(p)
            ):
                blocks += 1
            if blocks >= min_blocks:
                items.append((tuple(ts), m, blocks))
                m += blocks * p
                folded = True
                break
        if not folded:
            items.append(((deltas[m],), m, 1))
            m += 1
    return items


def _group_tree(entries, depth):
    """[(context_stack, delta_index)] -> nested items following the
    program's loop structure: ('step', idx) or
    ('loop', loop_id, [items per iteration])."""
    items, i = [], 0
    while i < len(entries):
        stack = entries[i][0]
        if len(stack) <= depth:
            items.append(("step", entries[i][1]))
            i += 1
            continue
        lid = stack[depth][0]
        iters, cur, cur_iter = [], [], stack[depth][1]
        j = i
        while (
            j < len(entries)
            and len(entries[j][0]) > depth
            and entries[j][0][depth][0] == lid
        ):
            if entries[j][0][depth][1] != cur_iter:
                iters.append(cur)
                cur, cur_iter = [], entries[j][0][depth][1]
            cur.append(entries[j])
            j += 1
        iters.append(cur)
        items.append(("loop", lid, [_group_tree(g, depth + 1) for g in iters]))
        i = j
    return items


# item model for hierarchical folding: (layout, exprs, span, positions).
# layout is a tuple of ("text", indent, str) and ("expr", indent) slots;
# exprs fill the expr slots in order; span counts the original units
# (steps or iterations) the item covers; positions holds each expr's
# absolute delta index (used to detect accumulator self-references).
# Two items fold together only when their layouts are IDENTICAL and
# their exprs generalize -- a fold can never merge structurally
# different regions.


_CLOSED_DUMMY = sympy.Dummy("j", integer=True)


def _self_reference(expr, k, base_pos, stride):
    """The step-reference in expr pointing at THIS slot one iteration
    back (position stride*k + base_pos - stride), or None."""
    target = stride * k + base_pos - stride
    for ref in expr.atoms(sympy.Indexed):
        if ref.base == _STEP and sympy.expand(ref.indices[0] - target) == 0:
            return ref
    return None


def _close_form(template, k, selfref, members, stride, base_pos):
    """A verified closed form for an accumulator template.

    acc + g(k) -> init + Sum(g, ...);  acc * g(k) -> init * Product;
    a(k)*acc + b(k) -> rsolve when sympy can. Verified by exact doit
    equality against the unrolled chain for EVERY member; None when
    the pattern or the proof fails. Function-agnostic: only the
    template's structure is inspected."""
    init = members[0]
    j = _CLOSED_DUMMY  # shared: closed forms must compare equal across folds
    closed = None
    if isinstance(template, sympy.Add) and selfref in template.args:
        g = (template - selfref).xreplace({k: j})
        closed = init + sympy.Sum(g, (j, 1, k))
    elif isinstance(template, sympy.Mul) and selfref in template.args:
        g = (template / selfref).xreplace({k: j})
        closed = init * sympy.Product(g, (j, 1, k))
    else:
        a = template.coeff(selfref)
        b = sympy.expand(template - a * selfref)
        if a != 0 and not a.has(selfref, _STEP) and not b.has(selfref):
            y = sympy.Function("y")
            try:
                closed = sympy.rsolve(
                    y(k) - a * y(k - 1) - b, y(k), {y(0): init}
                )
            except (ValueError, NotImplementedError):
                closed = None
    if closed is None:
        return None
    expected = init
    for r in range(1, len(members)):
        prev_ref = _STEP[stride * r + base_pos - stride]
        expected = members[r].xreplace({prev_ref: expected})
        got = closed.subs(k, r).doit()
        if sympy.expand(got - expected) != 0:
            return None
    return closed


def _fold_seq(items, k, min_run=3, unit="items"):
    """Fold runs of consecutive items instantiating one template;
    accumulator slots in a fold get verified closed forms."""
    out, r, total = [], 0, len(items)
    while r < total:
        lay = items[r][0]
        cand = None
        if r + 1 < total and items[r + 1][0] == lay:
            cand = [
                _generalize(a, b, k)
                for a, b in zip(items[r][1], items[r + 1][1])
            ]
            if any(c is None for c in cand) or not any(c.has(k) for c in cand):
                cand = None
        run = 0
        if cand is not None:
            while (
                r + run < total
                and items[r + run][0] == lay
                and all(
                    c.subs(k, run) == e
                    for c, e in zip(cand, items[r + run][1])
                )
            ):
                run += 1
        if cand is not None and run >= min_run:
            pos0 = items[r][3]
            strides = [b - a for a, b in zip(pos0, items[r + 1][3])]
            aligned = all(
                items[r + q][3][j] == pos0[j] + q * strides[j]
                for q in range(run)
                for j in range(len(pos0))
            )
            exprs = list(cand)
            if aligned and run <= 200:
                for j, t in enumerate(cand):
                    ref = _self_reference(t, k, pos0[j], strides[j])
                    if ref is None:
                        continue
                    closed = _close_form(
                        t,
                        k,
                        ref,
                        [items[r + q][1][j] for q in range(run)],
                        strides[j],
                        pos0[j],
                    )
                    if closed is not None:
                        exprs[j] = closed
            header = ("text", 0, f"repeat {run} {unit}, {k} = 0..{run - 1}:")
            layout = (header,) + tuple(
                (slot[0], slot[1] + 1) + tuple(slot[2:]) for slot in lay
            )
            out.append(
                (
                    layout,
                    exprs,
                    sum(items[r + q][2] for q in range(run)),
                    pos0,
                )
            )
            r += run
            continue
        out.append(items[r])
        r += 1
    return out


def _merge_items(items, indent=0):
    """Concatenate items into one (layout, exprs, positions) triple."""
    layout, exprs, positions = [], [], []
    for lay, ex, _, pos in items:
        layout.extend((slot[0], slot[1] + indent) + tuple(slot[2:]) for slot in lay)
        exprs.extend(ex)
        positions.extend(pos)
    return tuple(layout), exprs, positions


def _items_of(tree, deltas, ks, depth):
    """Grouped tree -> folded item list, bottom-up. Loop iterations
    fold against each other; the loop becomes ONE item so repeated
    loop instances (comprehensions, per-row calls) fold at the level
    above."""
    items = []
    for it in tree:
        if it[0] == "step":
            items.append(((("expr", 0),), [deltas[it[1]]], 1, [it[1]]))
            continue
        _, lid, iters = it
        iter_items = []
        for group in iters:
            sub = _items_of(group, deltas, ks, depth + 1)
            sub = _fold_seq(sub, ks[depth + 1], unit="items")
            lay, ex, pos = _merge_items(sub)
            iter_items.append((lay, ex, 1, pos))
        folded = _fold_seq(iter_items, ks[depth], unit="iterations")
        layout, exprs, positions = [], [], []
        r = 0
        for lay, ex, span, pos in folded:
            title = (
                f"loop {lid}, iteration {r}:"
                if span == 1
                else f"loop {lid}, iterations {r}..{r + span - 1}:"
            )
            layout.append(("text", 0, title))
            layout.extend((slot[0], slot[1] + 1) + tuple(slot[2:]) for slot in lay)
            exprs.extend(ex)
            positions.extend(pos)
            r += span
        items.append((tuple(layout), exprs, 1, positions))
    return items

