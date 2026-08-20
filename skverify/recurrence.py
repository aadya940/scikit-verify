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
        for n in range(int(count)):
            if isinstance(state, sympy.Tuple):
                state = step(*state, sympy.Integer(n))
            else:
                state = step(state, sympy.Integer(n))
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
    if pair._fsize > 4 * GROWTH_LIMIT:
        raise NotImplementedError(
            "loop body is not one template under the iteration index; "
            "the unrolled formula grows without bound"
        )


def _live(refs):
    return [p for p in (r() for r in refs) if p is not None]


_ATOM_INDEX = re.compile(r"_\d+")


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


def _plantable(pair):
    """Scalar Pairs with real formulas take a probe Dummy; array state
    (an IndexedBase-shaped plant) is future work and simply is not
    planted -- a loop carrying only arrays never folds, loudly."""
    f = pair.formula
    return (
        isinstance(f, sympy.Basic)
        and f.free_symbols
        and pair._axis_bounds is None
    )


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
    rec["guard_mark"] = len(_session.guards)
    rec["opaque_mark"] = len(_session.opaque)
    phase = rec["phase"]
    if phase == "watch":
        if index >= FOLD_START - 1:
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
        _advance(rec, body, sig)
    elif rec["phase"] in ("probe1", "probe2"):
        _repair(rec)  # loop ended mid-probe: restore eager formulas


def _broken(rec):
    _repair(rec)
    rec["phase"] = "broken"


def _plant(rec, body, sig):
    """Plant a probe Dummy on every plantable pair of this body; the
    next body reveals which of them are actually carried."""
    plants = []
    for pos, p in enumerate(body):
        if _plantable(p):
            d = sympy.Dummy(f"state{pos}")
            plants.append((pos, weakref.ref(p), d, p.formula))
            rec["repairs"][d] = p.formula
            p.formula = d
            p._fsize = 4
    if not plants:
        return  # nothing carried yet; keep watching
    if len(plants) > MAX_SLOTS:
        _broken(rec)
        return
    # NOTE: this body's own signature is NOT the reference. Its eager
    # formulas distribute (Number*Add flattens Mul into Add), so its
    # op sequence differs from every symbol-carrying body after it.
    # The reference is taken from the first planted body instead.
    rec.update(phase="probe1", plants=plants)
    rec["planted"] = [r for _, r, _, _ in plants]


def _extract_first(rec, body, sig):
    """The body ran on planted dummies: the referenced dummies are the
    state; read each slot's template off the same positions."""
    rec["sig_probe"] = sig  # first symbol-carrying body: the reference
    dummies = {d for _, _, d, _ in rec["plants"]}
    used = set()
    for p in body:
        if isinstance(p.formula, sympy.Basic):
            used |= p.formula.free_symbols & dummies
    slots = [pl for pl in rec["plants"] if pl[2] in used]
    if not slots or any(pl[0] >= len(body) for pl in slots):
        _broken(rec)
        return
    positions = [pl[0] for pl in slots]
    t_a = [body[pos].formula for pos in positions]
    if not all(isinstance(t, sympy.Basic) for t in t_a):
        _broken(rec)
        return
    # plant round two on the SAME positions of this body
    b_dummies = []
    for j, pos in enumerate(positions):
        d = sympy.Dummy(f"state{pos}")
        # eager meaning of this body's slot: its template with round-
        # one dummies substituted back
        rec["repairs"][d] = t_a[j].xreplace(rec["repairs"])
        b_dummies.append(d)
        body[pos].formula = d
        body[pos]._fsize = 4
    rec.update(
        phase="probe2",
        positions=positions,
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
    for j, pos in enumerate(rec["positions"]):
        body[pos].formula = held if scalar else Nth(held, sympy.Integer(j))
        body[pos]._fsize = 64
    rec.update(phase="carry", step=step, init_expr=init, count=2, scalar=scalar)
    # earlier pairs may still hold probe symbols: restore their eager
    # meaning so nothing outside the fold ever sees a Dummy
    keep = {id(body[pos]) for pos in rec["positions"]}
    _repair(rec, keep_ids=keep)


def _advance(rec, body, sig):
    """Same path fingerprint as the probe body => same deterministic
    dataflow => the templates apply; the carried values sit at the
    same positions. No formula matching."""
    if sig != rec["sig_probe"] or any(
        pos >= len(body) for pos in rec["positions"]
    ):
        rec["phase"] = "broken"  # eager formulas remain exact; fold ends
        return
    m = rec["count"]
    held = Iterate(rec["step"], rec["init_expr"], sympy.Integer(m + 1))
    for j, pos in enumerate(rec["positions"]):
        body[pos].formula = (
            held if rec["scalar"] else Nth(held, sympy.Integer(j))
        )
        body[pos]._fsize = 64
    rec["count"] = m + 1


def _repair(rec, keep_ids=()):
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
    for ref in rec.get("planted", ()):
        p = ref()
        if p is None or id(p) in keep_ids:
            continue
        f = p.formula
        if isinstance(f, sympy.Basic) and f.free_symbols & keys:
            p.formula = f.xreplace(subs).xreplace(subs)
    for i, g in enumerate(_session.guards):
        if isinstance(g, sympy.Basic) and g.free_symbols & keys:
            _session.guards[i] = g.xreplace(subs).xreplace(subs)
