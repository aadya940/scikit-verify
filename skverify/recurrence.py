"""Loops as domains: fold iteration into a sympy-native recurrence.

An array axis is a domain bound by ``Sum``; an iteration index is the
same object in time, bound by a recurrence. Unrolled, an iterative
solver's formula contains its whole history and grows without bound
(sympy op cost grows with expression size: the snowball). Folded, the
formula is one constant-size held object -- :class:`Iterate` -- that
sympy infrastructure traverses, substitutes into, and unrolls on
demand via ``doit``.

Capture is by PLANTING: before a probe body runs, every scalar carried
Pair's formula becomes a fresh Dummy, so the body's own execution
writes the step template directly in terms of those symbols
(reverse-engineering templates from eager formulas dies on sympy's
eager Number*Add distribution -- float products are not bit-stable
across association orders). The carried STATE is discovered, not
assumed: whichever planted dummies the next body actually references
are the state slots; one slot folds to a bare ``Iterate``, several
fold to a Tuple state with per-slot :class:`Nth` accessors. Two probe
bodies give two templates; integer and float drift between them
generalizes to the iteration index (``_generalize``).

Verification is by PATH, not by formula: the body is deterministic
code, so an iteration that fires the same branch guards, seals the
same opaque calls, and builds the same operation sequence as the
probe iteration computed the same template. Each subsequent body's
signature is compared to the probe's; on a match the carried values
(identified by their positions in the deterministic op sequence)
advance to ``Iterate(step, init, count+1)``. A body whose signature
differs stops the fold; eager exact formulas resume.

Folding engages only past ``FOLD_START`` iterations: short loops keep
today's fully unrolled formulas, which existing certificates pin.
"""

import re
import weakref

import sympy

from .session import current as _session

# iterations before the fold engages: small loops stay unrolled
FOLD_START = 8

# parent-sum size estimate at which plain-path blowup counts as a wall
GROWTH_LIMIT = 50_000

# body size at which the fold engages early: for fast-growing bodies
# every eager iteration multiplies the init formulas the fold must
# embed, so waiting is strictly worse
PLANT_TRIGGER = 500

# most state slots the folder will track (a body carrying more than
# this many independent values is not a foldable recurrence in v1)
MAX_SLOTS = 16


class Iterate(sympy.Function):
    """``Iterate(step, init, count)``: count-fold application of step.

    ``step`` is ``Lambda((s1, .., sk, n), expr)`` -- state slots and
    iteration index; ``init`` is the state before the first folded
    body (a ``Tuple`` when k > 1). The object stays held under
    symbolic manipulation; ``doit`` unrolls exactly.
    """

    @classmethod
    def eval(cls, step, init, count):
        return None  # always hold: unrolling is the caller's choice

    def doit(self, **hints):
        step, state, count = self.args
        if not (count.is_Integer and count >= 0):
            return self
        if hints.get("deep", True):
            state = state.doit(**hints)
        slot_syms = step.variables[:-1]
        n_sym = step.variables[-1]
        for n in range(int(count)):
            vals = state if isinstance(state, sympy.Tuple) else (state,)
            expr = step.expr
            for sym, val in zip(slot_syms, vals):
                expr = _subst_slot(expr, sym, val)
            expr = expr.xreplace({n_sym: sympy.Integer(n)})
            state = expr if not isinstance(state, sympy.Tuple) else expr
        return state


class Nth(sympy.Function):
    """``Nth(tuple_expr, i)``: held component access, unrolled by doit."""

    @classmethod
    def eval(cls, expr, i):
        if isinstance(expr, sympy.Tuple) and i.is_Integer:
            return expr[int(i)]
        return None

    def doit(self, **hints):
        expr, i = self.args
        inner = expr.doit(**hints) if hints.get("deep", True) else expr
        if isinstance(inner, sympy.Tuple) and i.is_Integer:
            return inner[int(i)]
        return Nth(inner, i)


def register_pair(pair):
    """Track Pairs born inside a loop body (called from Pair.__init__).

    When the fold is broken (body not one template) growth continues
    eagerly; past the limit the honest output is a refusal, not a
    hang."""
    _session.loop_new.append(weakref.ref(pair))
    if pair._fsize > 200 * GROWTH_LIMIT:
        # generous: the pre-probe iterations of a fast-growing body
        # must survive long enough for the fold to engage at
        # FOLD_START; this only fires when the fold then FAILS
        raise NotImplementedError(
            "loop body is not one template under the iteration index; "
            "the unrolled formula grows without bound"
        )


def _live(refs):
    return [p for p in (r() for r in refs) if p is not None]


_ATOM_INDEX = re.compile(r"_\d+")


def _axis0():
    from .helpers import axis_idx

    return axis_idx(0)


def _signature(body, guards_from, opaque_from):
    """The body's path fingerprint: operation sequence, branch guard
    shapes, and opaque-call names (iteration indices stripped). Equal
    fingerprints on deterministic code mean equal dataflow."""
    ops = tuple(
        p.formula.func.__name__
        if isinstance(p.formula, sympy.Basic)
        else type(p.formula).__name__
        for p in body
    )
    guards = tuple(
        g.func.__name__ if isinstance(g, sympy.Basic) else str(g)
        for g in _session.guards[guards_from:]
    )
    atoms = tuple(
        _ATOM_INDEX.sub("_n", str(rec[0])) for rec in _session.opaque[opaque_from:]
    )
    return (ops, guards, atoms)


def inline(expr, mapping):
    """Substitute recurrence symbols into an expression at harvest:
    plain symbols by xreplace, IndexedBase labels elementwise (a bare
    xreplace of an array-slot label would malform the IndexedBase)."""
    if not isinstance(expr, sympy.Basic):
        return expr
    keys = set(mapping)
    # substitution must not pay evaluation: rebuilding the combined
    # tree runs factor_terms/flatten over every node, and the fully
    # inlined certificate can be large. The held form is the same
    # mathematics; the user evaluates when and if they choose.
    with sympy.evaluate(False):
        return _inline_passes(expr, mapping, keys)


def _inline_passes(expr, mapping, keys):
    for _ in range(3):  # meanings may reference other recurrence symbols
        present = expr.free_symbols & keys
        if not present:
            break
        idx_labels = {
            e.base.label
            for e in expr.atoms(sympy.Indexed)
            if getattr(e.base, "label", None) in present
        }
        scalars = present - idx_labels
        if scalars:
            expr = expr.xreplace({d: mapping[d] for d in scalars})
        if idx_labels:
            expr = expr.replace(
                lambda e: isinstance(e, sympy.Indexed)
                and getattr(e.base, "label", None) in idx_labels,
                lambda e: mapping[e.base.label].xreplace({_axis0(): e.indices[0]}),
            )
    return expr


def _plantable(pair):
    """Which probe stands in for this pair: a plain Dummy for scalars,
    an IndexedBase over a Dummy label for 1-D arrays (the template
    then holds ``state[i]`` and reductions over it become Sums)."""
    f = pair.formula
    if not (isinstance(f, sympy.Basic) and f.free_symbols):
        return None
    if pair._axis_bounds is None:
        return "scalar"
    if len(pair._axis_bounds) == 1:
        return "array"
    return None


def _probe_for(kind, pos):
    from .helpers import axis_idx

    label = sympy.Dummy(f"state{pos}")
    if kind == "scalar":
        return label, label
    return label, sympy.IndexedBase(label)[axis_idx(0)]


def _subst_slot(expr, label, value):
    """Substitute one state slot into an expression: scalars by
    xreplace; array slots elementwise -- ``state[k]`` becomes the
    slot's indexed formula evaluated at ``k``."""
    from .helpers import axis_idx

    i0 = axis_idx(0)

    def is_slot(e):
        return (
            isinstance(e, sympy.Indexed)
            and getattr(e.base, "label", None) == label
        )

    if expr.has(sympy.Indexed):
        expr = expr.replace(
            is_slot, lambda e: value.xreplace({i0: e.indices[0]})
        )
    return expr.xreplace({label: value})


def on_loop_iter(loop_id, index):
    """Fold hook at the top of each body; the Pairs recorded since the
    previous marker belong to iteration ``index - 1``."""
    if index == 0:
        _session.loop_fold[loop_id] = {
            "phase": "watch",
            "planted": [],  # pairs since first plant, for repair
            "repairs": {},  # probe Dummy -> its eager meaning
            "guard_mark": len(_session.guards),
            "opaque_mark": len(_session.opaque),
        }
        _session.loop_new.clear()
        return
    rec = _session.loop_fold.get(loop_id)
    if rec is None or rec["phase"] == "broken":
        _session.loop_new.clear()
        return
    body = _live(_session.loop_new)
    _session.loop_new.clear()
    if rec["phase"] != "watch":
        rec["planted"].extend(weakref.ref(p) for p in body)
    sig = _signature(body, rec["guard_mark"], rec["opaque_mark"])
    rec["adv_guard_mark"] = rec["guard_mark"]
    rec["guard_mark"] = len(_session.guards)
    rec["opaque_mark"] = len(_session.opaque)
    phase = rec["phase"]
    if phase == "watch":
        # engage at FOLD_START, or immediately when the body is
        # already snowballing (fast-growing solvers never survive
        # eight eager iterations)
        if index >= FOLD_START - 1 or any(
            p._fsize > PLANT_TRIGGER for p in body
        ):
            _plant(rec, body, sig)
    elif phase == "probe1":
        _extract_first(rec, body, sig)
    elif phase == "probe2":
        _extract_second(rec, body, sig)
    elif phase == "carry":
        _advance(rec, body, sig)


def on_loop_end(loop_id):
    rec = _session.loop_fold.pop(loop_id, None)
    body = _live(_session.loop_new)
    _session.loop_new.clear()
    if rec is None:
        return
    if rec["phase"] == "carry":
        # the final body has no following marker: collapse it here
        sig = _signature(body, rec["guard_mark"], rec["opaque_mark"])
        rec["adv_guard_mark"] = rec["guard_mark"]
        _advance(rec, body, sig)
        # a convergence break leaves a PARTIAL final body: its slot
        # values fail the signature (correctly) and stay eager at
        # template size. Everything downstream of the loop composes
        # with them, so they leave as light symbols too -- the real
        # expressions inline at harvest with everything else.
        tag = len(_session.recurrences)
        for j, pos in enumerate(rec.get("positions", ())):
            if pos >= len(body):
                continue
            f = body[pos].formula
            if (
                isinstance(f, sympy.Basic)
                and f.free_symbols
                and not f.is_Symbol
                and sympy.count_ops(f) > 64
            ):
                sym = sympy.Symbol(f"loop{tag}_exit{j}", real=True)
                _session.recurrences[sym] = f
                body[pos].formula = sym
                body[pos]._fsize = 8
    elif rec["phase"] in ("probe1", "probe2"):
        _repair(rec)  # loop ended mid-probe: restore eager formulas


def _broken(rec):
    """A fold attempt failed. The path pattern often shifts ONCE (a
    convergence branch flips as iterates settle) and is stable again
    after, so the folder goes back to watching and may re-plant: the
    loop folds as segments, a later segment's init referencing the
    earlier segment's light symbols. A cap prevents thrashing."""
    _repair(rec)
    rec["segments"] = rec.get("segments", 0) + 1
    rec["phase"] = "watch" if rec["segments"] <= 8 else "broken"
    for key in ("plants", "b_plants", "positions", "head_syms"):
        rec.pop(key, None)


def _plant(rec, body, sig):
    """Plant a probe Dummy on every plantable pair of this body; the
    next body reveals which of them are actually carried."""
    plants = []
    for pos, p in enumerate(body):
        kind = _plantable(p)
        if kind:
            label, probe = _probe_for(kind, pos)
            plants.append((pos, weakref.ref(p), label, p.formula, kind))
            rec["repairs"][label] = p.formula
            p.formula = probe
            p._fsize = 4
    if not plants:
        return  # nothing carried yet; keep watching
    if len(plants) > 512:
        # pathological body size; planting is cheap but not free
        _broken(rec)
        return
    # NOTE: this body's own signature is NOT the reference. Its eager
    # formulas distribute (Number*Add flattens Mul into Add), so its
    # op sequence differs from every symbol-carrying body after it.
    # The reference is taken from the first planted body instead.
    rec.update(
        phase="probe1",
        plants=plants,
        plant_guard_mark=len(_session.guards),
    )
    rec["planted"] = [r for _, r, _, _, _ in plants]


def _extract_first(rec, body, sig):
    """The body ran on planted dummies: the referenced dummies are the
    state; read each slot's template off the same positions."""
    rec["sig_probe"] = sig  # first symbol-carrying body: the reference
    dummies = {d for _, _, d, _, _ in rec["plants"]}
    used = set()
    for p in body:
        if isinstance(p.formula, sympy.Basic):
            used |= p.formula.free_symbols & dummies
    slots = [pl for pl in rec["plants"] if pl[2] in used]
    if not slots or len(slots) > MAX_SLOTS or any(
        pl[0] >= len(body) for pl in slots
    ):
        # the CARRIED set is what must stay small: a body genuinely
        # consuming more than MAX_SLOTS prior values is not a
        # recurrence the certificate can hold
        _broken(rec)
        return
    positions = [pl[0] for pl in slots]
    kinds = [pl[4] for pl in slots]
    t_a = [body[pos].formula for pos in positions]
    if not all(isinstance(t, sympy.Basic) for t in t_a):
        _broken(rec)
        return
    # plant round two on the SAME positions of this body
    b_dummies = []
    for j, pos in enumerate(positions):
        label, probe = _probe_for(kinds[j], pos)
        # eager meaning of this body's slot IS its template; the
        # round-one dummies inside resolve recursively at repair time
        # (materializing them here would rebuild the giant pre-plant
        # formulas the fold exists to avoid)
        rec["repairs"][label] = t_a[j]
        b_dummies.append(label)
        rec.setdefault("b_plants", []).append(
            (weakref.ref(body[pos]), t_a[j])
        )
        body[pos].formula = probe
        body[pos]._fsize = 4
    rec.update(
        phase="probe2",
        positions=positions,
        kinds=kinds,
        a_dummies=[pl[2] for pl in slots],
        b_dummies=b_dummies,
        t_a=t_a,
        init=sympy.Tuple(*(pl[3] for pl in slots)),
    )


def _extract_second(rec, body, sig):
    from .derivation import _generalize

    if sig != rec["sig_probe"] or any(
        pos >= len(body) for pos in rec["positions"]
    ):
        _broken(rec)
        return
    remap = dict(zip(rec["b_dummies"], rec["a_dummies"]))
    # for array slots the Dummy is an IndexedBase LABEL: xreplace on
    # the label rewrites the base too, so one map serves both kinds
    a_set = set(rec["a_dummies"])
    b_set = set(rec["b_dummies"])
    n = sympy.Dummy("n", integer=True, nonnegative=True)
    templates = []
    for j, pos in enumerate(rec["positions"]):
        f = body[pos].formula
        if not isinstance(f, sympy.Basic) or f.free_symbols & a_set:
            # a stale round-one symbol here means a distance-2
            # reference (state from two iterations back): not a
            # first-order recurrence, not foldable in v1
            _broken(rec)
            return
        t = _generalize(rec["t_a"][j], f.xreplace(remap), n)
        if t is None or t.free_symbols & b_set:
            _broken(rec)
            return
        templates.append(t)
    s_syms = [sympy.Dummy(f"s{j}") for j in range(len(templates))]
    slot_map = dict(zip(rec["a_dummies"], s_syms))
    exprs = [t.xreplace(slot_map) for t in templates]
    scalar = len(templates) == 1
    if scalar:
        step = sympy.Lambda((s_syms[0], n), exprs[0])
        init = rec["init"][0]
    else:
        step = sympy.Lambda(tuple(s_syms) + (n,), sympy.Tuple(*exprs))
        init = rec["init"]
    held = Iterate(step, init, sympy.Integer(2))
    loop_tag = len(_session.recurrences)
    head_syms = [
        sympy.Symbol(f"loop{loop_tag}_{j}", real=True)
        for j in range(len(templates))
    ]
    for j, pos in enumerate(rec["positions"]):
        # the trace works with a FEATHERWEIGHT symbol: dragging the
        # held Iterate through every downstream sympy ctor (Abs.eval,
        # signsimp, assumption deduction) costs minutes per body. The
        # real object is inlined at harvest.
        _session.recurrences[head_syms[j]] = (
            held if scalar else Nth(held, sympy.Integer(j))
        )
        body[pos].formula = head_syms[j]
        body[pos]._fsize = 8
    rec.update(
        phase="carry",
        step=step,
        init_expr=init,
        count=2,
        scalar=scalar,
        head_syms=head_syms,
    )
    # the fold SUCCEEDED: round-two probe symbols have an exact light
    # meaning -- the state after one folded transition. Repairing them
    # to the eager trees instead would hand every later consumer
    # (convergence checks reading copies of probe-body values) a giant
    # expression to drag through sympy's eval machinery.
    after_one = Iterate(step, init, sympy.Integer(1))
    for j, label in enumerate(rec["b_dummies"]):
        rec["repairs"][label] = (
            after_one if scalar else Nth(after_one, sympy.Integer(j))
        )
    rec["b_plants"] = [
        (r, rec["repairs"][label])
        for (r, _), label in zip(rec.get("b_plants", ()), rec["b_dummies"])
    ]
    # earlier pairs may still hold probe symbols. Eagerly rewriting
    # all of them costs ~20s of xreplace for pairs that are mostly
    # provenance-only; instead their meanings enter the recurrence
    # map and inline at harvest IF anything still references them.
    keep = {id(body[pos]) for pos in rec["positions"]}
    _repair(rec, keep_ids=keep, lazy=True)
    _session.recurrences.update(rec["repairs"])


def _advance(rec, body, sig):
    """Same path fingerprint as the probe body => same deterministic
    dataflow => the templates apply; the carried values sit at the
    same positions. No formula matching."""
    for p in body:
        # carry-phase bodies build from head symbols: formula-small by
        # construction whether or not this body extends the fold. The
        # provenance-sum estimate would otherwise snowball and trip
        # the growth wall on post-loop consumers (a convergence break
        # makes the final body partial, failing the signature).
        p._fsize = min(p._fsize, 32)
    if sig != rec["sig_probe"] or any(
        pos >= len(body) for pos in rec["positions"]
    ):
        # the path changed (a convergence branch flipped): this body
        # stays eager and exact, and the folder goes back to watching
        # -- the loop folds as segments. Its slot values leave as
        # light symbols so downstream composition stays cheap.
        tag = len(_session.recurrences)
        for j, pos in enumerate(rec.get("positions", ())):
            if pos >= len(body):
                continue
            f = body[pos].formula
            if (
                isinstance(f, sympy.Basic)
                and f.free_symbols
                and not f.is_Symbol
                and sympy.count_ops(f) > 64
            ):
                sym = sympy.Symbol(f"loop{tag}_seg{j}", real=True)
                _session.recurrences[sym] = f
                body[pos].formula = sym
                body[pos]._fsize = 8
        rec["segments"] = rec.get("segments", 0) + 1
        rec["phase"] = "watch" if rec["segments"] <= 8 else "broken"
        for key in ("plants", "b_plants", "positions", "head_syms"):
            rec.pop(key, None)
        return
    m = rec["count"]
    # guards recorded during this body reference the head symbols,
    # which meant the state BEFORE this body ran: pin them to that
    # iteration's held object now, or the harvest inline would
    # wrongly give every guard the final state
    prev = {}
    for sym in rec["head_syms"]:
        # per-iteration SYMBOL, not the held object: guards stay one
        # line each, the definitions map carries each state once
        at = sympy.Symbol(f"{sym.name}_at{m}", real=True)
        if at not in _session.recurrences:
            _session.recurrences[at] = _session.recurrences[sym]
        prev[sym] = at
    for i in range(rec["adv_guard_mark"], len(_session.guards)):
        g = _session.guards[i]
        if isinstance(g, sympy.Basic) and g.free_symbols & set(prev):
            _session.guards[i] = g.xreplace(prev)
    held = Iterate(rec["step"], rec["init_expr"], sympy.Integer(m + 1))
    for j, sym in enumerate(rec["head_syms"]):
        _session.recurrences[sym] = (
            held if rec["scalar"] else Nth(held, sympy.Integer(j))
        )
    for j, pos in enumerate(rec["positions"]):
        body[pos].formula = rec["head_syms"][j]
    rec["count"] = m + 1


def _repair(rec, keep_ids=(), lazy=False):
    """Substitute probe symbols back to their eager meanings in every
    pair created since the first plant: no Dummy may outlive the fold
    attempt. Pairs in ``keep_ids`` (the new Iterate heads) are exempt.
    Guards recorded during probe bodies carry the symbols too -- a
    Dummy in .preconditions would be an unbound symbol in a
    certificate."""
    subs = rec.get("repairs", {})
    if not subs:
        return
    keys = set(subs)
    done = set()
    # planted pairs repair by DIRECT assignment: substituting a probe
    # symbol inside its own bare formula would drag the (potentially
    # giant) original through xreplace, and for array probes would
    # rewrite the IndexedBase label into a malformed base
    for pl in rec.get("plants", ()):
        p = pl[1]()
        if p is None or id(p) in keep_ids:
            continue
        f = p.formula
        is_probe = f == pl[2] or (
            isinstance(f, sympy.Indexed)
            and getattr(f.base, "label", None) == pl[2]
        )
        if is_probe:
            p.formula = pl[3]
            done.add(id(p))
    for ref, eager in rec.get("b_plants", ()):
        p = ref()
        if p is not None and id(p) not in keep_ids:
            p.formula = eager  # may hold round-one symbols: fix below
            # NOT marked done: fix() must resolve those round-one refs

    def fix(f):
        for _ in range(2):  # round-two meanings may hold round-one symbols
            present = f.free_symbols & keys
            if not present:
                break
            # one xreplace for scalar occurrences of every present
            # symbol, one replace pass for all Indexed slots together:
            # cost is two traversals of f, not one per dummy
            f = f.xreplace({d: subs[d] for d in present})
            if f.has(sympy.Indexed):
                f = f.replace(
                    lambda e: isinstance(e, sympy.Indexed)
                    and getattr(e.base, "label", None) in keys,
                    lambda e: subs[e.base.label].subs(
                        _axis0(), e.indices[0]
                    ),
                )
        return f

    for ref in () if lazy else rec.get("planted", ()):
        p = ref()
        if p is None or id(p) in keep_ids or id(p) in done:
            # direct-assigned pairs hold pre-plant formulas: re-walking
            # them (free_symbols is uncached) would traverse the giant
            # eager trees the fold exists to avoid
            continue
        f = p.formula
        if isinstance(f, sympy.Basic) and f.free_symbols & keys:
            p.formula = fix(f)
    if lazy:
        # fold SUCCEEDED: guards keep their probe symbols and the
        # definitions map carries every meaning -- eager insertion
        # here would rebuild exactly the giant trees the fold avoids
        return
    # only guards recorded since the plant can hold probe symbols
    start = rec.get("plant_guard_mark", 0)
    for i in range(start, len(_session.guards)):
        g = _session.guards[i]
        if isinstance(g, sympy.Basic) and g.free_symbols & keys:
            _session.guards[i] = fix(g)
