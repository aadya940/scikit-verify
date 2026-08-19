import numpy as np
import sympy

from .registry import (
    UFUNC_TABLE,
    FUNCTION_TABLE,
)
from .helpers import (
    axis_idx,
    _AXIS_SYMBOLS,
    normalize_key,
)

IDX = axis_idx(0)  # `i`

from .session import current as _session

# Historical aliases into the active TraceSession. These are the SAME
# list objects the session owns (cleared in place, never rebound), so
# long-standing imports keep observing live trace state.
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


def _loop_end(loop_id):
    if _LOOP_STACK and _LOOP_STACK[-1][0] == loop_id:
        _LOOP_STACK.pop()
    _LOOP_EVENTS.append((_session.seq, tuple((l, i) for l, i in _LOOP_STACK)))


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


class Pair:
    """Convert math and array style operations to SymPy
    expressions.
    """

    def __init__(self, value, formula, domain=None, steps=None):
        self.value = value  # the real ndarray/scalar, what executes
        self.formula = formula  # the sympy Expr, what it means
        # provenance is a DAG of parent Pairs; .steps flattens it on
        # demand, deduplicating shared ancestors. Two runs taking
        # different branches produce different steps, because different
        # ops ran.
        self._parents = tuple(steps or ())
        self._seq = _session.next_seq()  # creation order; keys into the loop event log

        if domain is not None and len(domain) == 0:
            domain = None  # 0-d allocation: a scalar
        if domain is not None and not isinstance(domain[0], tuple):
            domain = (domain,)

        self._axis_bounds = domain  # for ndarray

    def _repr_latex_(self):
        # Jupyter renders the formula as mathematics
        text = sympy.latex(self.formula)
        return f"$\\displaystyle {text}$"

    def __repr__(self):
        text = str(self.formula)
        if len(text) > 60:
            text = text[:57] + "..."
        if self._axis_bounds is None:
            return f"Pair({text})"
        return f"Pair({text}, domain={self.domain})"

    def _step_nodes(self):
        seen = set()
        out = []
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                out.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in reversed(node._parents):
                stack.append((parent, False))
        return out

    @property
    def steps(self):
        return [n.formula for n in self._step_nodes()]

    def cse_steps(self):
        """The derivation with shared subexpressions named: a list of
        (t_k, expression) assignments and the steps rewritten in terms
        of them. Substituting the assignments back (in reverse order)
        reproduces .steps exactly; nothing is simplified away."""
        assignments, steps = sympy.cse(
            self.steps, symbols=sympy.numbered_symbols("t"), order="none"
        )
        return assignments, steps

    def derivation(self):
        """Human-readable derivation, complete under expansion.

        Steps referencing their predecessor show it as `prev`; runs of
        consecutive steps that are one template under an integer index
        fold to a single rule line, every member verified against the
        template (exact subs equality) before folding. Shared
        subexpressions are cse-named. Expanding rules and names back
        reproduces .steps exactly."""
        nodes = self._step_nodes()
        if _LOOP_EVENTS:
            # chronological order: execution order, and still topological
            # (parents are always stamped before their consumers)
            nodes = sorted(nodes, key=lambda n: n._seq)
            contexts = [_context_of(n._seq) for n in nodes]
            if any(contexts):
                deltas = _delta_steps([n.formula for n in nodes], nodes)
                return self._derivation_by_loops(deltas, contexts)
        deltas = _delta_steps([n.formula for n in nodes])
        k = _fresh_name("n", deltas)
        items = _fold_runs(deltas, k)  # (templates, start, blocks)

        flat = [t for ts, _, _ in items for t in ts]
        assignments, reduced = sympy.cse(
            flat, symbols=sympy.numbered_symbols("t"), order="none"
        )
        named = {sym for sym, _ in assignments}
        lines = [f"{sym} = {expr}" for sym, expr in assignments]
        pos = 0
        last_line = None
        for ts, start, blocks in items:
            body = reduced[pos : pos + len(ts)]
            pos += len(ts)
            if blocks > 1:
                stop = start + blocks * len(ts) - 1
                lines.append(f"steps {start}-{stop}, {k} = 0..{blocks - 1}:")
                lines.extend(f"  {expr}" for expr in body)
                last_line = None
                continue
            if body[0] in named or body[0] == last_line:
                continue
            if (
                isinstance(body[0], sympy.Indexed)
                and body[0].base == _STEP
            ):
                continue  # `step 6: step[5]` says nothing
            last_line = body[0]
            lines.append(f"step {start}: {body[0]}")
        final = reduced[-1]
        if items[-1][2] > 1:
            final = final.subs(k, items[-1][2] - 1)
        lines.append(f"result: {final}")
        return "\n".join(lines)

    def _derivation_by_loops(self, deltas, contexts):
        """Fold by the program's own loop structure: steps grouped by
        the (loop, iteration) context recorded at trace time, iteration
        bodies generalized against each other -- alignment comes from
        the AST, not pattern search."""
        ks = [_fresh_name("n", deltas)] + [
            sympy.Symbol(f"n{d}", integer=True) for d in range(2, 10)
        ]
        tree = _group_tree(list(zip(contexts, range(len(deltas)))), 0)
        items = _items_of(tree, deltas, ks, 0)
        items = _fold_seq(items, ks[0], unit="items")
        layout, exprs, _ = _merge_items(items)

        assignments, reduced = sympy.cse(
            exprs, symbols=sympy.numbered_symbols("t"), order="none"
        )
        lines = [f"{sym} = {expr}" for sym, expr in assignments]
        out = iter(reduced)
        for slot in layout:
            pad = "  " * slot[1]
            if slot[0] == "text":
                lines.append(pad + slot[2])
            else:
                lines.append(pad + str(next(out)))
        lines.append(f"result: {reduced[-1] if exprs else deltas[-1]}")
        return "\n".join(lines)

    @staticmethod
    def _steps_of(*operands):
        return tuple(x for x in operands if isinstance(x, Pair))

    @staticmethod
    def _formula_of(x):
        if isinstance(x, Pair):
            return x.formula
        if isinstance(x, np.ndarray):
            if x.dtype == object:
                elems = x.ravel()
                if all(isinstance(e, Pair) for e in elems):
                    formulas = [e.formula for e in elems]
                    if len(set(formulas)) == 1:
                        # keepdims mean and friends: one scalar in a box
                        return formulas[0]
                    from .api import _recompress

                    rule = _recompress(formulas)
                    if rule is not None:
                        return rule
                    raise NotImplementedError(
                        "decompressed operand without a provable pattern"
                    )
            vals = np.unique(x)
            if len(vals) == 1:  # uniform: zeros, ones, full
                return sympy.sympify(vals.item())  # constant field, clean
            if x.dtype.kind in "fiub" and x.size <= 4096:
                # a concrete operand (filter kernels, weights): a named
                # table, values disclosed -- same treatment as gathers
                name = f"const_{len(_OPAQUE)}"
                table = sympy.IndexedBase(name)
                letters = tuple(axis_idx(ax) for ax in range(x.ndim))
                _OPAQUE.append(
                    (
                        name,
                        (("table", "concrete"),),
                        (str(table[letters]), f"{name} = {x.tolist()}"),
                    )
                )
                return table[letters]
            raise NotImplementedError(
                "raw non-uniform ndarray operand, wrap it: Pair.array(name, x)"
            )
        return sympy.sympify(x)

    @staticmethod
    def _domain_of(x):
        if isinstance(x, Pair):
            return x._axis_bounds  # canonical, always
        if isinstance(x, np.ndarray):
            return tuple((0, s) for s in x.shape)  # ndim > 1 handled too
        return None

    @property
    def domain(self):
        if self._axis_bounds is None:
            return None
        if len(self._axis_bounds) == 1:
            return self._axis_bounds[0]  # the 60's flat face
        return self._axis_bounds

    @staticmethod
    def _merge_domains(*domains):
        """Merge canonical axis tuples. None = scalar. Non-None domains
        must agree per axis; differing ranks are refused."""
        result = None
        for d in domains:
            if d is None:
                continue
            if result is None:
                result = d
                continue
            if len(d) != len(result):
                # (4, 7) meeting (7,): line the short one up against the
                # END of the long one and merge those axes
                short, long = sorted((d, result), key=len)
                head = long[: len(long) - len(short)]
                tail = long[len(long) - len(short) :]
                result = head + Pair._merge_axes(short, tail)
                continue
            result = Pair._merge_axes(d, result)
        return result

    @staticmethod
    def _merge_axes(d, result):
        merged = []
        for ax, (result_item, d_item) in enumerate(zip(result, d)):
            if d_item == result_item:
                merged.append(result_item)
            elif d_item == (0, 1):  # numpy broadcasting: extent 1 stretches
                merged.append(result_item)
            elif result_item == (0, 1):
                merged.append(d_item)
            else:
                raise ValueError(
                    f"domain mismatch at axis {ax}: {result_item} vs {d_item}"
                )
        return tuple(merged)


    @staticmethod
    def _defers(other):
        """Pair op object-array-of-Pairs: defer to numpy's object loop
        (elementwise dunders), the honest per-element route."""
        return (
            isinstance(other, np.ndarray)
            and other.dtype == object
            and any(isinstance(e, Pair) for e in other.ravel())
        )

    @staticmethod
    def _binary(inputs, fwd, rev, self):
        a, b = inputs
        return fwd(a, b) if a is self else rev(b, a)

    @staticmethod
    def _numeric(v, copy=True):
        """Object-dtype numeric arrays coerce to float (the dtype duck
        leaks into allocations); anything else passes through."""
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                try:
                    return np.asarray(v, dtype=float)
                except (TypeError, ValueError):
                    pass
            return np.array(v, copy=True) if copy else v
        return v

    @staticmethod
    def _value_of(x):
        if isinstance(x, Pair):
            return x.value
        if isinstance(x, np.ndarray) and x.dtype == object:
            elems = x.ravel()
            if any(isinstance(e, Pair) for e in elems):
                vals = [e.value if isinstance(e, Pair) else e for e in elems]
                return np.array(vals, dtype=float).reshape(x.shape)
        if isinstance(x, (list, tuple)) and any(isinstance(e, Pair) for e in x):
            return type(x)(Pair._value_of(e) for e in x)
        return x

    @staticmethod
    def _shift_axes(formula, bounds, ndim):
        """Rename a lower-rank operand's letters to the trailing axes.

        v[i] (1 axis) meeting a 2-axis result: v runs along the last
        axis, so v[i] -> v[j].
        """
        if bounds is None:
            return formula  # scalar: no letters to move
        offset = ndim - len(bounds)
        if offset == 0:
            return formula
        index_map = {axis_idx(a): axis_idx(a + offset) for a in range(len(bounds))}
        return formula.subs(index_map, simultaneous=True)

    @staticmethod
    def _is_condition(formula):
        # NOT isinstance(..., Boolean): sympy Symbols subclass Boolean
        return isinstance(
            formula,
            (
                sympy.core.relational.Relational,
                sympy.logic.boolalg.BooleanFunction,
                sympy.logic.boolalg.BooleanAtom,
            ),
        )

    @staticmethod
    def _bridge_numeric(formula):
        """Boolean entering arithmetic means 0/1, as in numpy:
        (u > 0).sum() counts, mask * u selects."""
        if Pair._is_condition(formula):
            if _piecewise_under_sum(formula):
                # sympy's Piecewise ctor hoists conditions through Sum
                # bounds (even with evaluate=False), leaking the bound
                # index -- a wrong formula. Refuse until fixed upstream.
                raise NotImplementedError(
                    "condition over a Sum of Piecewise: sympy rewrites "
                    "this incorrectly; restructure or use .value"
                )
            return sympy.Piecewise((1, formula), (0, True))
        return formula

    @staticmethod
    def _broadcast(a, b, bridge=True):
        """Formulas + merged bounds for a binary op.

        u (4x7) + v (7):  u[i, j] + v[j],  bounds ((0, 4), (0, 7))
        bridge=False (mask algebra, comparisons) keeps Booleans raw.
        """
        bounds_a, bounds_b = Pair._domain_of(a), Pair._domain_of(b)
        merged = Pair._merge_domains(bounds_a, bounds_b)
        formula_a = Pair._formula_of(a)
        formula_b = Pair._formula_of(b)
        if merged is not None:
            formula_a = Pair._shift_axes(formula_a, bounds_a, len(merged))
            formula_b = Pair._shift_axes(formula_b, bounds_b, len(merged))
            formula_a = Pair._pin_ones(formula_a, bounds_a, merged)
            formula_b = Pair._pin_ones(formula_b, bounds_b, merged)
        if bridge:
            formula_a = Pair._bridge_numeric(formula_a)
            formula_b = Pair._bridge_numeric(formula_b)
        return formula_a, formula_b, merged

    @staticmethod
    def _pin_ones(formula, bounds, merged):
        """A broadcast extent-1 axis has one valid index: after axis
        alignment its letter reads 0, whatever the merged extent is."""
        if bounds is None:
            return formula
        offset = len(merged) - len(bounds)
        subs = {}
        for ax, (lo, hi) in enumerate(bounds):
            m_lo, m_hi = merged[ax + offset]
            if hi - lo == 1 and m_hi - m_lo != 1:
                subs[axis_idx(ax + offset)] = sympy.Integer(0)
        return formula.subs(subs, simultaneous=True) if subs else formula

    def _remap(self, value, index_map, axis_bounds):
        """Implements infrastructure for array methods where
        indexes are remapped from old to new using `index_map`.

        For example:
        index_map = {i: j, j: i} for a transpose operation.

        _axis_bounds are reversed in order the transpose operation for example.
        """
        for old_sym, expr in index_map.items():
            expr = sympy.sympify(expr)
            if not expr.free_symbols <= set(_AXIS_SYMBOLS):
                raise NotImplementedError(
                    f"index map {old_sym} -> {expr} is not an index expression"
                )
            if not expr.is_integer and not all(
                expr.diff(s).is_Integer for s in expr.free_symbols
            ):
                raise NotImplementedError(
                    f"index map {old_sym} -> {expr} is not affine"
                )
        formula = self.formula.subs(index_map, simultaneous=True)
        return Pair(value, formula, domain=axis_bounds or None, steps=(self,))

    def _getitem_nd(self, key):
        # u[1:, 2] on a 4x7 array -> entries ((1, 4), 2)
        entries = normalize_key(key, tuple(hi - lo for lo, hi in self._axis_bounds))
        index_map, new_bounds = {}, []
        for ax, entry in enumerate(entries):
            sym = axis_idx(ax)
            if isinstance(entry, tuple):
                # u[1:]   : u[i] -> u[i + 1]      4 rows -> 3 rows
                # u[::2]  : u[i] -> u[2*i]        5 -> 3
                # u[::-1] : u[i] -> u[4 - i]      5 -> 5
                # Survivors take the letter of their RESULT position, so that
                # letter k always means axis k:  u[2, 1:] -> u[2, i + 1], not j
                start, stop, step = entry
                new_sym = axis_idx(len(new_bounds))
                index_map[sym] = step * new_sym + start
                count = max(0, -(-(stop - start) // step))  # ceil for +/- step
                new_bounds.append((0, count))
            else:
                # u[2] : u[i, j] -> u[2, i], row axis gone, survivor renamed
                index_map[sym] = sympy.Integer(entry)
        value = self.value[key]
        if isinstance(value, np.ndarray):
            value = value.copy()  # slices are value-semantic: no view aliasing
        result = self._remap(
            value=value,  # raw key: numpy interprets it independently
            index_map=index_map,
            axis_bounds=tuple(new_bounds),  # u[2, 3] -> (), remap makes it scalar
        )
        if all(isinstance(e, tuple) for e in entries):
            # chained writes (dk[1:-1][mask] = v) must reach the parent:
            # remember where this slice came from
            result._slice_of = (self, entries)
        return result

    def __setitem__(self, key, val):
        # u[2:5] = v: the formula becomes a scatter, old rule outside the
        # written region, the value's rule (re-indexed) inside it
        if self._axis_bounds is None:
            raise TypeError("scalar Pair does not support item assignment")

        origin = getattr(self, "_slice_of", None)
        if origin is not None:
            # write-through: translate this write onto the parent so the
            # chained idiom dk[1:-1][mask] = v stays correct
            parent, entries = origin
            parent[Pair._compose_key(entries, key, self)] = val
            self.__dict__["_slice_of"] = None  # local write follows below

        def is_idx_arr(k):
            return (
                isinstance(k, (list, np.ndarray))
                and np.asarray(k).dtype.kind in "iu"
                and np.asarray(k).ndim >= 1
            )

        parts = key if isinstance(key, tuple) else (key,)
        if (
            len(parts) == 1
            and isinstance(parts[0], np.ndarray)
            and parts[0].dtype == bool
        ):
            # concrete boolean mask with any value shape: the selected
            # positions are trace facts; decompose into scalar writes
            positions = np.nonzero(parts[0])
            parts = (positions[0],) if len(positions) == 1 else positions
        lengths_all = tuple(hi - lo for lo, hi in self._axis_bounds)

        def is_full_slice(k, ax):
            return isinstance(k, slice) and k.indices(lengths_all[ax]) == (
                0,
                lengths_all[ax],
                1,
            )

        if (
            sum(is_idx_arr(k) for k in parts) == 1
            and any(
                is_full_slice(k, i) for i, k in enumerate(parts)
            )
            and all(
                is_idx_arr(k) or is_full_slice(k, i)
                for i, k in enumerate(parts)
            )
        ):
            # one index array among full slices: decompose per element,
            # each row-write handled by the ordinary scatter machinery
            ax = next(i for i, k in enumerate(parts) if is_idx_arr(k))
            arr = np.asarray(parts[ax]).ravel()
            scalar_val = np.ndim(Pair._value_of(val)) == 0
            for k_pos, p in enumerate(arr):
                sub = tuple(
                    int(p) if i == ax else parts[i] for i in range(len(parts))
                )
                self[sub if len(sub) > 1 else sub[0]] = (
                    val if scalar_val else val[k_pos]
                )
            return
        if any(is_idx_arr(k) for k in parts) and all(
            is_idx_arr(k) or isinstance(k, (int, np.integer)) for k in parts
        ):
            # scatter through index arrays: concrete positions, so
            # decompose into scalar writes (p[rows, cols] = v)
            arrs = [np.asarray(k) for k in parts if is_idx_arr(k)]
            arrays = np.broadcast_arrays(*arrs)
            flat = [a.ravel() for a in arrays]
            n = flat[0].size
            for pos in range(n):
                it = iter(flat)
                target = tuple(
                    int(k) if isinstance(k, (int, np.integer)) else int(next(it)[pos])
                    for k in parts
                )
                target = target if len(target) > 1 else target[0]
                if np.ndim(Pair._value_of(val)) == 0:
                    self[target] = val
                else:
                    self[target] = val[pos]
            return
        if isinstance(key, Pair):
            if Pair._is_condition(key.formula):
                if isinstance(val, (Pair, np.ndarray)) and np.ndim(Pair._value_of(val)):
                    # concrete mask, array value: positions are trace
                    # facts, conditions become per-position guards, and
                    # the write decomposes scalar by scalar
                    mask = np.asarray(key.value, dtype=bool)
                    selected = np.nonzero(mask)[0]
                    n_val = np.size(Pair._value_of(val))
                    if n_val != selected.size:
                        raise ValueError(
                            f"NumPy boolean array indexing assignment "
                            f"cannot assign {n_val} input values to the "
                            f"{selected.size} output values where the mask is true"
                        )
                    sym = axis_idx(0)
                    for pos in range(mask.size):
                        cond = key.formula.subs(sym, pos)
                        _GUARDS.append(cond if mask[pos] else sympy.Not(cond))
                    for k, pos in enumerate(selected):
                        self[int(pos)] = val[k]
                    return
                self.value[key.value] = Pair._value_of(val)
                prior = self.formula
                self.formula = sympy.Piecewise(
                    (Pair._formula_of(val), key.formula), (prior, True)
                )
                self._record_write(prior, key, val)
                return
            kv = np.asarray(key.value)
            if kv.dtype.kind in "iu" or (
                kv.dtype.kind == "f" and np.all(kv == np.round(kv))
            ):
                # a traced index (vector or scalar): the positions are
                # concrete trace facts (argsort outputs and friends)
                if kv.ndim == 0:
                    self[int(kv)] = val
                else:
                    self[kv.astype(int)] = val
                return
            raise NotImplementedError("only boolean Pair keys are supported")

        lengths = tuple(hi - lo for lo, hi in self._axis_bounds)
        entries = normalize_key(key, lengths)
        seq = None
        if isinstance(val, (tuple, list)):
            seq = list(val)
        elif isinstance(val, np.ndarray) and val.dtype == object:
            seq = list(val.ravel())
        if seq is not None:
            # a sequence of traced scalars: decompose into scalar writes
            axes_ranges = []
            for ax, entry in enumerate(entries):
                if isinstance(entry, tuple):
                    start, stop, step = entry
                    if step != 1:
                        raise NotImplementedError(
                            "strided assignment not supported yet"
                        )
                    axes_ranges.append(range(start, stop))
                else:
                    axes_ranges.append(range(entry, entry + 1))
            positions = list(np.ndindex(*[len(r) for r in axes_ranges]))
            if len(positions) != len(seq):
                raise NotImplementedError(
                    "assigned sequence length must match the region"
                )
            for pos, elem in zip(positions, seq):
                concrete = tuple(axes_ranges[ax][p] for ax, p in enumerate(pos))
                self[concrete if len(concrete) > 1 else concrete[0]] = elem
            return
        val_formula = Pair._formula_of(val)
        condition = []
        val_map = {}
        val_rank = 0
        for ax, entry in enumerate(entries):
            sym = axis_idx(ax)
            if isinstance(entry, tuple):
                start, stop, step = entry
                if step != 1:
                    raise NotImplementedError("strided assignment not supported yet")
                if not (start == 0 and stop == lengths[ax]):
                    condition.append(sympy.Ge(sym, start))
                    condition.append(sympy.Lt(sym, stop))
                val_map[axis_idx(val_rank)] = sym - start
                val_rank += 1
            else:
                condition.append(sympy.Eq(sym, entry))
        if isinstance(val, Pair) and val._axis_bounds is not None:
            if len(val._axis_bounds) != val_rank:
                raise NotImplementedError(
                    "assigned value rank must match the sliced region"
                )
            val_formula = val_formula.subs(
                {k: v for k, v in val_map.items()}, simultaneous=True
            )
        self.value[key] = Pair._value_of(val)
        prior = self.formula
        if condition:
            self.formula = sympy.Piecewise(
                (val_formula, sympy.And(*condition)), (prior, True)
            )
        else:
            self.formula = val_formula
        self._record_write(prior, val)

    @staticmethod
    def _compose_key(entries, key, view):
        """Map a key on the sliced view to the parent's coordinates:
        entry (start, stop, step) turns index i into start + step*i."""
        parts = key if isinstance(key, tuple) else (key,)
        parts = parts + (slice(None),) * (len(entries) - len(parts))
        out = []
        for ax, (entry, k) in enumerate(zip(entries, parts)):
            start, stop, step = entry
            length = len(range(start, stop, step))
            if isinstance(k, (int, np.integer)):
                out.append(start + step * int(k))
            elif isinstance(k, slice):
                a, b, c = k.indices(length)
                out.append(slice(start + step * a, start + step * b, step * c))
            elif isinstance(k, np.ndarray) and k.dtype == bool:
                out.append(start + step * np.nonzero(k)[0])
            elif isinstance(k, Pair) and Pair._is_condition(k.formula):
                mask = np.asarray(k.value, dtype=bool)
                out.append(start + step * np.nonzero(mask)[0])
            elif isinstance(k, (list, np.ndarray)):
                out.append(start + step * np.asarray(k, dtype=int))
            else:
                raise NotImplementedError(
                    "write-through slice: unsupported sub-key"
                )
        return tuple(out) if len(out) > 1 else out[0]

    def _record_write(self, prior_formula, *operands):
        # an in-place write mutates formula; the pre-write state becomes
        # a parent node so the DAG keeps the whole derivation
        prior = Pair(self.value, prior_formula, self._axis_bounds, steps=self._parents)
        self._parents = (prior,) + Pair._steps_of(*operands)
        self._seq = _session.next_seq()  # the write happened NOW; creation seq is stale

    def transpose(self, axes=None):
        # u (4x7), u.T: u[i, j] -> u[j, i], bounds ((0,4),(0,7)) -> ((0,7),(0,4))
        # axes=(2,0,1) on 3-D: result position k reads old axis axes[k]
        if self._axis_bounds is None or len(self._axis_bounds) == 1:
            return self  # scalars and 1-D: transpose is a no-op, like numpy
        ndim = len(self._axis_bounds)
        if axes is None:
            axes = tuple(reversed(range(ndim)))
        index_map = {
            axis_idx(old_ax): axis_idx(new_pos) for new_pos, old_ax in enumerate(axes)
        }
        return self._remap(
            value=self.value.transpose(axes),
            index_map=index_map,
            axis_bounds=tuple(self._axis_bounds[old_ax] for old_ax in axes),
        )

    @property
    def T(self):
        return self.transpose()

    def reshape(self, shape, *rest):
        if rest:
            shape = (shape, *rest)
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        shape = tuple(int(n) for n in shape)
        current = tuple(hi - lo for lo, hi in (self._axis_bounds or ()))
        target = np.reshape(np.empty(current), shape).shape  # resolves -1
        if target == current:
            return self
        if tuple(n for n in target if n != 1) == tuple(
            n for n in current if n != 1
        ):
            # only extent-1 axes inserted/removed: layout preserved.
            # y (n,) -> (n, 1): old letters move to the surviving axes,
            # dropped extent-1 axes pin to index 0
            survivors = iter(
                ax for ax, n in enumerate(target) if n != 1
            )
            index_map = {}
            for old_ax, n in enumerate(current):
                if n == 1:
                    index_map[axis_idx(old_ax)] = sympy.Integer(0)
                else:
                    index_map[axis_idx(old_ax)] = axis_idx(next(survivors))
            return self._remap(
                value=self.value.reshape(shape).copy(),
                index_map=index_map,
                axis_bounds=tuple((0, n) for n in target),
            )
        raise NotImplementedError(
            "reshape that changes the layout is not supported yet"
        )

    def __add__(self, other):
        if Pair._defers(other):
            return NotImplemented
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value + Pair._value_of(other),
            formula=mine + theirs,  # sympy dunder does the rest
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __radd__(self, other):
        if Pair._defers(other):
            return NotImplemented  # handles  2 + u
        return self.__add__(other)

    def __sub__(self, other):
        if Pair._defers(other):
            return NotImplemented
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value - Pair._value_of(other),
            formula=mine - theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __rsub__(self, other):
        if Pair._defers(other):
            return NotImplemented  # handles  2 - u   (order matters!)
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=Pair._value_of(other) - self.value,
            formula=theirs - mine,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __mul__(self, other):
        if Pair._defers(other):
            return NotImplemented
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value * Pair._value_of(other),
            formula=mine * theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    __rmul__ = __mul__

    def __mod__(self, other):
        if Pair._defers(other):
            return NotImplemented
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value % Pair._value_of(other),
            formula=sympy.Mod(mine, theirs),
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __rmod__(self, other):
        if Pair._defers(other):
            return NotImplemented
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=Pair._value_of(other) % self.value,
            formula=sympy.Mod(theirs, mine),
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __matmul__(self, other):
        from .maps.numpy import _matmul

        return _matmul(self, other)

    def __rmatmul__(self, other):
        from .maps.numpy import _matmul

        return _matmul(other, self)

    def __abs__(self):
        return Pair(
            abs(self.value),
            sympy.Abs(self.formula),
            domain=self._axis_bounds,
            steps=(self,),
        )

    def __bool__(self):
        # Branch capture: a scalar condition (if x > 0:) is decided by the
        # concrete lane and RECORDED -- the branch taken becomes a hypothesis
        # on the certificate. to_sympy harvests _GUARDS into .preconditions.
        if Pair._is_condition(self.formula) and np.ndim(self.value) == 0:
            outcome = bool(self.value)
            _GUARDS.append(self.formula if outcome else sympy.Not(self.formula))
            return outcome
        # arrays: ambiguous, like numpy's own error -- use .all()/.any().
        # non-conditions: silent truthiness on math is never meaningful.
        raise NotImplementedError(
            "data-dependent branch on a traced value; "
            "for masks use .all()/.any(), for combining use & | ~"
        )

    def __array__(self, dtype=None, copy=None):
        """Coercion is deliberate decompression: an indexed Pair becomes an
        object array of per-element scalar Pairs, so the trace survives
        element by element. Forced numeric dtypes would discard formulas."""
        if dtype is not None and np.dtype(dtype) != object:
            raise NotImplementedError(
                f"coercion to dtype={dtype} would discard the formula"
            )
        if self._axis_bounds is None:
            out = np.empty((), dtype=object)
            out[()] = self
            return out
        shape = tuple(hi - lo for lo, hi in self._axis_bounds)
        n = int(np.prod(shape))
        if n > 4096:
            raise NotImplementedError(
                f"coercion would unroll {n} elements; the indexed form is lost"
            )
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            out[idx] = self[idx if len(idx) > 1 else idx[0]]
        return out

    # bare-number conversions discard the formula: the trace silently dies
    # otherwise. Use .value for a deliberate exit.
    def __float__(self):
        raise NotImplementedError(
            "float() on a traced value discards the formula; use .value"
        )

    def __int__(self):
        raise NotImplementedError(
            "int() on a traced value discards the formula; use .value"
        )

    def __complex__(self):
        raise NotImplementedError(
            "complex() on a traced value discards the formula; use .value"
        )

    # facts about the concrete lane; some library bodies read these
    # before doing any math
    @property
    def ndim(self):
        return np.ndim(self.value)

    @property
    def shape(self):
        return np.shape(self.value)

    @property
    def preconditions(self):
        """Branch conditions recorded during the trace this Pair came
        from (path-scoped: the formula holds for inputs satisfying
        them). Trace-global; to_sympy pins a snapshot on its result."""
        stored = self.__dict__.get("_preconditions")
        if stored is not None:
            return stored
        return sympy.And(*_GUARDS) if _GUARDS else sympy.true

    @preconditions.setter
    def preconditions(self, value):
        self.__dict__["_preconditions"] = value

    @property
    def unchecked(self):
        """Opaque-call records for the atoms appearing in THIS Pair's
        derivation: what the formula assumes rather than derives."""
        stored = self.__dict__.get("_unchecked")
        if stored is not None:
            return stored
        names = set()
        for formula in self.steps:
            for base in formula.atoms(sympy.IndexedBase):
                names.add(str(base.label))
            for fn in formula.atoms(sympy.Function):
                names.add(type(fn).__name__)
        records = []
        for entry in _OPAQUE:
            key = entry[-1][0].split("[")[0].rstrip("*").rstrip("_")
            if any(n == key or n.startswith(key + "_") for n in names):
                records.append(entry)
        return tuple(records)

    @unchecked.setter
    def unchecked(self, value):
        self.__dict__["_unchecked"] = value

    @property
    def size(self):
        return int(np.size(self.value))

    @property
    def flags(self):
        return np.asarray(self.value).flags

    def mean(self, axis=None, **kwargs):
        from .maps.numpy import _sum

        if self._axis_bounds is None:
            return self
        if axis is None:
            n = int(np.prod([hi - lo for lo, hi in self._axis_bounds]))
            return _sum(self) / n
        k = axis % len(self._axis_bounds)
        lo, hi = self._axis_bounds[k]
        return _sum(self, axis=k) / (hi - lo)

    def sum(self, axis=None, **kwargs):
        from .maps.numpy import _sum

        return _sum(self, axis=axis)

    def max(self, axis=None, **kwargs):
        return np.maximum.reduce(self, axis=axis, **kwargs)

    def min(self, axis=None, **kwargs):
        return np.minimum.reduce(self, axis=axis, **kwargs)

    @property
    def real(self):
        value = np.real(self.value)
        formula = self.formula if np.isrealobj(self.value) else sympy.re(self.formula)
        return Pair(value, formula, self._axis_bounds, steps=(self,))

    @property
    def imag(self):
        if np.isrealobj(self.value):
            return Pair(
                np.imag(self.value), sympy.Integer(0), self._axis_bounds, steps=(self,)
            )
        return Pair(
            np.imag(self.value), sympy.im(self.formula), self._axis_bounds, steps=(self,)
        )

    def conj(self):
        if np.isrealobj(self.value):
            return self
        return Pair(
            np.conj(self.value),
            sympy.conjugate(self.formula),
            self._axis_bounds,
            steps=(self,),
        )

    def setflags(self, **kwargs):
        pass  # writeability bookkeeping; the traced copy is ours

    def __index__(self):
        # a Pair used AS AN INDEX is bookkeeping (positions are trace
        # facts, like find_interval); non-integral values refuse
        v = self.value
        if np.ndim(v) == 0 and float(v) == int(v):
            return int(v)
        raise TypeError("only integral scalar Pairs can index")

    def copy(self, order="C"):
        value = self.value.copy() if hasattr(self.value, "copy") else self.value
        return Pair(value, self.formula, self._axis_bounds, steps=(self,))

    def var(self, axis=None, ddof=0, **kwargs):
        from .maps.numpy import _var

        return _var(self, axis=axis, ddof=ddof)

    def std(self, axis=None, ddof=0, **kwargs):
        from .maps.numpy import _std

        return _std(self, axis=axis, ddof=ddof)

    @property
    def flat(self):
        # numpy's flat iterator: our flattened view suffices for the
        # read patterns library code uses
        return self.ravel()

    def ravel(self, order="C"):
        # flattening is a layout-preserving reshape
        return self.reshape((int(np.size(self.value)),))

    def flatten(self, order="C"):
        return self.ravel()

    def squeeze(self, axis=None):
        # removing extent-1 axes is a layout-preserving reshape
        return self.reshape(np.squeeze(np.asarray(self.value), axis=axis).shape)

    def astype(self, dtype=None, copy=True, **kwargs):
        # float cast is math-neutral; a real reinterpretation would lie
        if dtype is not None and np.dtype(dtype).kind not in "fc":
            raise NotImplementedError("astype to non-float would change the math")
        return Pair(self.value, self.formula, self._axis_bounds, steps=(self,))

    @property
    def device(self):
        # array-API bookkeeping; the concrete lane lives wherever numpy is
        return getattr(np.asarray(self.value), "device", "cpu")

    @property
    def dtype(self):
        # the TRUTH: the concrete lane's numeric dtype. (An object duck
        # lived here once, for numpy's mean cast branch -- Pair.mean made
        # it obsolete, and the lie leaked into every finfo/kind gate.)
        value = np.asarray(self.value)
        if value.dtype == object:
            return np.dtype(float)
        return value.dtype

    def __truediv__(self, other):
        if Pair._defers(other):
            return NotImplemented  # self / other
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value / Pair._value_of(other),
            formula=mine / theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __rtruediv__(self, other):
        if Pair._defers(other):
            return NotImplemented  # other / self
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=Pair._value_of(other) / self.value,
            formula=theirs / mine,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __pow__(self, other):
        if Pair._defers(other):
            return NotImplemented  # self ** other
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value ** Pair._value_of(other),
            formula=mine**theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __rpow__(self, other):
        if Pair._defers(other):
            return NotImplemented  # other ** self
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=Pair._value_of(other) ** self.value,
            formula=theirs**mine,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __neg__(self):  # -self
        return Pair(
            -self.value, -self.formula, domain=self._axis_bounds, steps=(self,)
        )

    @classmethod
    def array(cls, name, value):
        value = np.asarray(value)
        if value.ndim > 5:
            raise NotImplementedError(
                "arrays beyond 5D are not supported.",
            )
        if value.ndim == 0:
            return cls(value[()], sympy.Symbol(name, real=True))
        idxs = tuple([axis_idx(idx) for idx in range(value.ndim)])
        formula = sympy.IndexedBase(name)[idxs]
        return cls(value, formula, domain=tuple((0, s) for s in value.shape))

    def __len__(self):
        n = self.value.shape[0]  # truth: the real array
        lo, hi = self._axis_bounds[0]
        assert n == hi - lo, "domain drifted from value"
        return n

    def __getitem__(self, key):
        """Slicing and integer indexing; 1-D is just the N=1 case."""
        if isinstance(key, tuple) and key == ():
            return self  # numpy's 0-d unwrap idiom, vals[()]
        if self._axis_bounds is None:
            raise TypeError("scalar Pair is not subscriptable")
        if isinstance(key, Pair) and Pair._is_condition(key.formula):
            # mask gather u[u > 0]: the mask's VALUE fixes the selected
            # positions (trace facts); the mask's CONDITIONS become
            # per-position preconditions, so the certificate is honest
            # about the path
            mask = np.asarray(key.value, dtype=bool)
            sym = axis_idx(0)
            for pos in range(mask.size):
                cond = key.formula.subs(sym, pos)
                _GUARDS.append(cond if mask[pos] else sympy.Not(cond))
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                value = np.asarray(self.value)[mask]
                return Pair(
                    value,
                    self.formula,
                    ((0, 0),) + tuple(self._axis_bounds[1:]),
                    steps=(self,),
                )
            return self._axis_gather(0, idx, mask)
        parts = key if isinstance(key, tuple) else (key,)
        if any(k is None for k in parts):
            # w[:, None]: newaxis only inserts extent-1 axes -- apply the
            # rest of the key, then reshape to numpy's resulting shape
            rest = tuple(k for k in parts if k is not None)
            base = self[rest] if rest else self
            return base.reshape(np.shape(self.value[key]))
        gathered = self._fancy_gather(key)
        if gathered is not None:
            return gathered
        return self._getitem_nd(key)

    def _fancy_gather(self, key):
        """Concrete integer-array indexing (diag_indices and friends):
        the positions are compile-time facts, so gather scalar Pairs per
        position and return the decompressed object array."""

        def is_index_array(k):
            return (
                isinstance(k, (list, np.ndarray)) and np.asarray(k).dtype.kind in "iu"
            )

        parts = key if isinstance(key, tuple) else (key,)
        if not any(is_index_array(k) for k in parts):
            return None
        if any(
            is_index_array(k) and np.asarray(k).ndim == 0 for k in parts
        ):  # 0-d index arrays are ints in disguise
            norm = tuple(
                int(np.asarray(k))
                if is_index_array(k) and np.asarray(k).ndim == 0
                else k
                for k in parts
            )
            return self[norm if len(norm) > 1 else norm[0]]

        def is_full(k):
            return isinstance(k, slice) and k == slice(None)

        if not all(is_index_array(k) for k in parts):
            if sum(is_index_array(k) for k in parts) == 1 and all(
                is_index_array(k) or is_full(k) for k in parts
            ):
                # X[:, cols]: gather along ONE axis, others untouched
                ax = next(
                    i for i, k in enumerate(parts) if is_index_array(k)
                )
                idx = np.asarray(parts[ax])
                if idx.ndim == 1 and 1 <= idx.size <= 4096:
                    return self._axis_gather(ax, idx, key)
            raise NotImplementedError("mixed fancy/slice indexing not supported")
        if len(parts) == 1:
            idx = np.asarray(parts[0])
            if idx.ndim == 1 and idx.size == 1:
                return self._axis_gather(0, idx, key)
            if idx.ndim == 1 and 2 <= idx.size <= 4096:
                strides = np.diff(idx)
                if len(set(strides.tolist())) != 1:
                    # irregular gather (pivot permutations, argsort):
                    # ONE rule through a recorded index table,
                    # u[[0, 1, 3]] -> u[gather_0[i]], table disclosed
                    name = f"gather_{len(_OPAQUE)}"
                    table = sympy.IndexedBase(name)
                    sym = axis_idx(0)
                    formula = self.formula.subs(sym, table[sym])
                    _OPAQUE.append(
                        (
                            name,
                            (("table", "concrete"),),
                            (str(table[sym]), f"{name} = {idx.tolist()}"),
                        )
                    )
                    value = self.value[idx]
                    return Pair(
                        value.copy() if isinstance(value, np.ndarray) else value,
                        formula,
                        ((0, int(idx.size)),) + tuple(self._axis_bounds[1:]),
                        steps=(self,),
                    )
                if True:
                    # affine gather: u[[3,2,1,0]] is the remap i -> 3 - i,
                    # same machinery as strided slices, ONE indexed formula
                    d, start = int(strides[0]), int(idx[0])
                    sym = axis_idx(0)
                    index_map = {sym: d * sym + start}
                    index_map.update(
                        {
                            axis_idx(ax): axis_idx(ax)
                            for ax in range(1, len(self._axis_bounds))
                        }
                    )
                    value = self.value[np.asarray(parts[0])]
                    return self._remap(
                        value=value.copy() if isinstance(value, np.ndarray) else value,
                        index_map=index_map,
                        axis_bounds=((0, int(idx.size)),)
                        + tuple(self._axis_bounds[1:]),
                    )
        arrays = np.broadcast_arrays(*[np.asarray(k) for k in parts])
        out = np.empty(arrays[0].shape, dtype=object)
        for pos in np.ndindex(arrays[0].shape):
            idx = tuple(int(a[pos]) for a in arrays)
            out[pos] = self[idx if len(idx) > 1 else idx[0]]
        return out

    def _axis_gather(self, ax, idx, key):
        """Gather along one axis: affine indices remap, irregular ones
        go through a recorded table."""
        sym = axis_idx(ax)
        strides = np.diff(idx) if idx.size >= 2 else np.array([1])
        value = self.value[key]
        if isinstance(value, np.ndarray):
            value = value.copy()
        bounds = (
            tuple(self._axis_bounds[:ax])
            + ((0, int(idx.size)),)
            + tuple(self._axis_bounds[ax + 1 :])
        )
        if len(set(strides.tolist())) == 1:
            d, start = int(strides[0]), int(idx[0])
            index_map = {sym: d * sym + start}
            index_map.update(
                {
                    axis_idx(a): axis_idx(a)
                    for a in range(len(self._axis_bounds))
                    if a != ax
                }
            )
            return self._remap(value, index_map, bounds)
        name = f"gather_{len(_OPAQUE)}"
        table = sympy.IndexedBase(name)
        formula = self.formula.subs(sym, table[sym])
        _OPAQUE.append(
            (
                name,
                (("table", "concrete"),),
                (str(table[sym]), f"{name} = {idx.tolist()}"),
            )
        )
        return Pair(value, formula, bounds, steps=(self,))

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError("out= is not supported (mutation)")
        if method == "reduce" and ufunc in (np.maximum, np.minimum, np.add):
            a = inputs[0]
            axis = kwargs.get("axis", 0)
            full = axis is None or (
                isinstance(a, Pair)
                and a._axis_bounds is not None
                and len(a._axis_bounds) == 1
                and axis == 0
            )
            if isinstance(a, Pair) and a._axis_bounds is not None and full:
                if ufunc is np.add and len(a._axis_bounds) == 1:
                    from .maps.numpy import _sum

                    return _sum(a)
                extents = [hi - lo for lo, hi in a._axis_bounds]
                if ufunc is not np.add and int(np.prod(extents)) <= 4096:
                    letters = [
                        axis_idx(ax) for ax in range(len(a._axis_bounds))
                    ]
                    op = sympy.Max if ufunc is np.maximum else sympy.Min
                    elems = [
                        a.formula.subs(
                            dict(zip(letters, pos)), simultaneous=True
                        )
                        for pos in np.ndindex(*extents)
                    ]
                    formula = op(
                        *elems,
                        evaluate=False,  # canonical sorting of n large args is quadratic
                    )
                    return Pair(
                        ufunc.reduce(a.value, axis=None),
                        formula,
                        None,
                        steps=(a,),
                    )
            if (
                isinstance(a, Pair)
                and a._axis_bounds is not None
                and isinstance(axis, int)
                and ufunc is not np.add
            ):
                bounds = a._axis_bounds
                k_ax = axis % len(bounds)
                lo, hi = bounds[k_ax]
                if hi - lo <= 4096:
                    # per-axis Max/Min: bind one letter over its concrete
                    # range, survivors renumber down (the letter invariant)
                    rename = {
                        axis_idx(ax): axis_idx(ax - 1)
                        for ax in range(k_ax + 1, len(bounds))
                    }
                    op = sympy.Max if ufunc is np.maximum else sympy.Min
                    elems = [
                        a.formula.subs(
                            {axis_idx(k_ax): v, **rename}, simultaneous=True
                        )
                        for v in range(lo, hi)
                    ]
                    formula = op(*elems, evaluate=False)
                    new_bounds = bounds[:k_ax] + bounds[k_ax + 1 :]
                    return Pair(
                        ufunc.reduce(a.value, axis=k_ax),
                        formula,
                        new_bounds or None,
                        steps=(a,),
                    )
            if (
                isinstance(a, np.ndarray)
                and a.dtype == object
                and a.size <= 4096
                and all(isinstance(e, Pair) for e in a.ravel())
                and ufunc is not np.add
            ):
                op = sympy.Max if ufunc is np.maximum else sympy.Min
                formula = op(
                    *[e.formula for e in a.ravel()], evaluate=False
                )
                return Pair(
                    ufunc.reduce(Pair._value_of(a), axis=None),
                    formula,
                    None,
                    steps=tuple(a.ravel()),
                )
        if method != "__call__" or kwargs.get("out") is not None:
            raise NotImplementedError(f"{ufunc.__name__}.{method} not supported")

        if any(
            isinstance(x, np.ndarray)
            and x.dtype == object
            and any(isinstance(e, Pair) for e in x.ravel())
            for x in inputs
        ):
            # a decompressed operand: decompress the Pair side too and
            # let numpy's object loop run element by element through
            # the attached methods -- honest per-element formulas
            conv = [
                np.asarray(x) if isinstance(x, Pair) else x for x in inputs
            ]
            return ufunc(*conv, **kwargs)

        if ufunc is np.add:
            return Pair._binary(inputs, Pair.__add__, Pair.__radd__, self)
        if ufunc is np.subtract:
            return Pair._binary(inputs, Pair.__sub__, Pair.__rsub__, self)
        if ufunc is np.multiply:
            return Pair._binary(inputs, Pair.__mul__, Pair.__rmul__, self)
        if ufunc is np.true_divide:
            return Pair._binary(inputs, Pair.__truediv__, Pair.__rtruediv__, self)
        if ufunc is np.power:
            return Pair._binary(inputs, Pair.__pow__, Pair.__rpow__, self)
        if ufunc is np.negative:
            return -inputs[0]
        if ufunc is np.matmul:
            from .maps.numpy import _matmul

            return _matmul(*inputs)

        target = UFUNC_TABLE.get(ufunc)
        if target is None:
            if ufunc.nout != 1:
                raise NotImplementedError(
                    f"ufunc {ufunc.__name__} has {ufunc.nout} outputs"
                )
            return Pair._opaque_call(ufunc, inputs, kwargs)

        values = [Pair._numeric(Pair._value_of(x), copy=False) for x in inputs]
        domain = Pair._merge_domains(*(Pair._domain_of(x) for x in inputs))
        formulas = [
            Pair._shift_axes(
                Pair._formula_of(x),
                Pair._domain_of(x),
                0 if domain is None else len(domain),
            )
            for x in inputs
        ]

        return Pair(
            ufunc(*values),
            target(*formulas),
            domain,
            steps=Pair._steps_of(*inputs),
        )

    @staticmethod
    def _opaque_call(func, args, kwargs):
        """A compiled routine the trace cannot enter: run it on the values,
        name it in the formula, snapshot inputs against hidden mutation,
        and record the call with its contract verdicts."""
        from .contracts import check_call

        pair_args = [a for a in args if isinstance(a, Pair)]
        snapshots = [
            np.asarray(a.value).tobytes()
            for a in pair_args
            if isinstance(a.value, np.ndarray)
        ]
        # the routine gets COPIES: overwrite_ab-style scribbling stays
        # off the traced values, and the snapshot guard keeps everyone
        # honest about it
        values = [Pair._numeric(Pair._value_of(a)) for a in args]
        kwvalues = {
            k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
            for k, v in ((k, Pair._value_of(v)) for k, v in kwargs.items())
        }
        # contracts must judge the INPUTS, not overwrite_*-mutated buffers
        pristine = [
            np.array(v, copy=True) if isinstance(v, np.ndarray) else v
            for v in values
        ]
        result = func(*values, **kwvalues)
        after = [
            np.asarray(a.value).tobytes()
            for a in pair_args
            if isinstance(a.value, np.ndarray)
        ]
        if snapshots != after:
            raise NotImplementedError(
                f"{func.__name__} mutated a traced input in place"
            )
        formulas = []
        n_const = 0
        for a in args:
            if isinstance(a, Pair):
                formulas.append(a.formula)
            elif np.isscalar(a):
                formulas.append(sympy.sympify(a))
            elif isinstance(a, np.ndarray):
                # a concrete operand: named, so the formula never hides it
                formulas.append(sympy.Symbol(f"const{n_const}"))
                n_const += 1
        # f2py fortran objects report __name__ as "function dgbsv":
        # keep the identifier part only
        fname = getattr(func, "__name__", "opaque").split()[-1]
        call = sympy.Function(fname)(*formulas)
        if isinstance(result, tuple):
            # multi-output routine (LAPACK gbsv: lu, piv, x, info): each
            # float-array output becomes its own atom; integer bookkeeping
            # (pivots, status) passes through concrete
            outs = []
            for pos, res in enumerate(result):
                if isinstance(res, np.ndarray) and res.dtype.kind in "fc":
                    base = sympy.IndexedBase(
                        f"{fname}_{len(_OPAQUE)}_{pos}"
                    )
                    letters = tuple(axis_idx(ax) for ax in range(res.ndim))
                    outs.append(
                        Pair(
                            res,
                            base[letters],
                            tuple((0, int(n)) for n in res.shape),
                            steps=Pair._steps_of(*args),
                        )
                    )
                else:
                    outs.append(res)
            _OPAQUE.append(
                check_call(fname, pristine, result)
                + ((f"{fname}_{len(_OPAQUE)}_*", str(call)),)
            )
            return tuple(outs)
        shape = np.shape(result) if hasattr(result, "shape") else ()
        if shape:
            # array output: a fresh indexed symbol, so downstream slicing
            # and arithmetic work; the definition rides in the record
            base = sympy.IndexedBase(f"{fname}_{len(_OPAQUE)}")
            letters = tuple(axis_idx(ax) for ax in range(len(shape)))
            formula = base[letters]
            domain = tuple((0, int(n)) for n in shape)
        else:
            formula = call
            domain = None
        _OPAQUE.append(
            check_call(fname, pristine, result) + ((str(formula), str(call)),)
        )
        return Pair(result, formula, domain=domain, steps=Pair._steps_of(*args))

    def __array_function__(self, func, types, args, kwargs):
        fn = FUNCTION_TABLE.get(func)
        if fn is not None:
            return fn(*args, **kwargs)  # curated: indexed formulas
        from .contracts import CONTRACTS

        if func.__name__ in CONTRACTS:
            return Pair._opaque_call(func, args, kwargs)
        inner = getattr(func, "__wrapped__", None)
        if inner is not None:
            # pure-Python numpy: run its real body on the Pairs; slices and
            # arithmetic inside dispatch back here, formulas unrolled per
            # element. A body that walls (finfo on the object dtype, type
            # gates) retries as an instrumented twin carrying the rewrites
            try:
                return inner(*args, **kwargs)
            except (TypeError, ValueError, AttributeError, NotImplementedError):
                from .instrument import runtime_twin

                twin = runtime_twin(inner)
                if twin is inner:
                    raise
                return twin(*args, **kwargs)
        return Pair._opaque_call(func, args, kwargs)


# relational & mask layer
# u > 0        -> Pair([True,...], u[i] > 0)     one rule, letters/domains reused
# (a) & (b)    -> And(a, b)                       raw Booleans, no bridge
# (u > 0).sum()-> Sum(Piecewise((1, cond), (0, True)))   via the bridge

_RELATIONALS = {
    "__lt__": (np.less, sympy.Lt),
    "__le__": (np.less_equal, sympy.Le),
    "__gt__": (np.greater, sympy.Gt),
    "__ge__": (np.greater_equal, sympy.Ge),
    "__eq__": (np.equal, sympy.Eq),
    "__ne__": (np.not_equal, sympy.Ne),
}

_MASK_OPS = {
    "__and__": (np.bitwise_and, sympy.And),
    "__or__": (np.bitwise_or, sympy.Or),
    "__xor__": (np.bitwise_xor, sympy.Xor),
}


def _piecewise_under_sum(expr):
    return isinstance(expr, sympy.Basic) and any(
        s.function.has(sympy.Piecewise) for s in expr.atoms(sympy.Sum)
    )


def _make_binary(np_op, sy_op, bridge):
    def op(self, other):
        mine, theirs, merged = Pair._broadcast(self, other, bridge=bridge)
        # sympy's relational constructor hoists Piecewise conditions OUT
        # of a Sum, leaking the bound index. Build unevaluated when that
        # hazard is present; the formula is the honest raw relation.
        if bridge and (_piecewise_under_sum(mine) or _piecewise_under_sum(theirs)):
            formula = sy_op(mine, theirs, evaluate=False)
        else:
            formula = sy_op(mine, theirs)
        return Pair(
            np_op(self.value, Pair._value_of(other)),
            formula,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    return op


for _name, (_np_op, _sy_op) in _RELATIONALS.items():
    setattr(Pair, _name, _make_binary(_np_op, _sy_op, bridge=True))
for _name, (_np_op, _sy_op) in _MASK_OPS.items():
    setattr(Pair, _name, _make_binary(_np_op, _sy_op, bridge=False))
    setattr(Pair, "__r" + _name[2:], _make_binary(_np_op, _sy_op, bridge=False))


def _invert(self):
    return Pair(
        np.bitwise_not(self.value),
        sympy.Not(self.formula),
        domain=self._axis_bounds,
        steps=(self,),
    )


Pair.__invert__ = _invert
Pair.all = lambda self, axis=None: np.all(self, axis=axis)
Pair.any = lambda self, axis=None: np.any(self, axis=axis)
Pair.__hash__ = None  # elementwise __eq__ (numpy semantics): unhashable, like ndarray
