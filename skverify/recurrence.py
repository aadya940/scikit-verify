"""Loops as domains: fold iteration into a sympy-native recurrence.

An array axis is a domain bound by ``Sum``; an iteration index is the
same object in time, bound by a recurrence. Unrolled, an iterative
solver's formula contains its whole history and grows without bound
(sympy op cost grows with expression size: the snowball). Folded, the
formula is one constant-size held object -- :class:`Iterate` -- that
sympy infrastructure traverses, substitutes into, and unrolls on
demand via ``doit``.

How the template is captured: reverse-engineering it from eager
formulas fails on sympy's eager ``Number*Add`` distribution (float
products are not bit-stable across association orders). Instead the
folder PLANTS an opaque state symbol: before a probe body runs, the
carried Pair's formula becomes a fresh Dummy, so the body's own
execution writes the step template directly in terms of that symbol.
Two probe bodies give two templates; integer drift between them
generalizes to the iteration index (``_generalize``). From then on
each body must reproduce ``template(state=Iterate(...), n=count)``
EXACTLY -- structural equality, no tolerance; ``Iterate`` is an
opaque Function node, so no distribution can smear it. A body that
stops matching stops the fold (formulas stay exact, merely eager).

Folding engages only past ``FOLD_START`` iterations: short loops keep
today's fully unrolled formulas, which existing certificates pin.
"""

import weakref

import sympy

from .session import current as _session

# iterations before the fold engages: small loops stay unrolled
FOLD_START = 8

# parent-sum size estimate at which plain-path blowup counts as a wall
GROWTH_LIMIT = 50_000


class Iterate(sympy.Function):
    """``Iterate(step, init, count)``: count-fold application of step.

    ``step`` is ``Lambda((s, n), expr)`` -- state and iteration index;
    ``init`` is the state before the first folded body. The object
    stays held under symbolic manipulation; ``doit`` unrolls exactly.
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
            state = step(state, sympy.Integer(n))
        return state


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


def _last_symbolic(pairs, needs=None):
    """The newest pair carrying a real formula (optionally one that
    references ``needs``): the presumed loop-carried value."""
    for p in reversed(pairs):
        f = p.formula
        if not isinstance(f, sympy.Basic) or not f.free_symbols:
            continue
        if needs is not None and not f.has(needs):
            continue
        return p
    return None


def on_loop_iter(loop_id, index):
    """Fold hook at the top of each body; the Pairs recorded since the
    previous marker belong to iteration ``index - 1``."""
    if index == 0:
        _session.loop_fold[loop_id] = {
            "phase": "watch",
            "planted": [],  # pairs since first plant, for repair
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
    phase = rec["phase"]
    if phase == "watch":
        if index >= FOLD_START - 1:
            _plant_first(rec, body)
    elif phase == "probe1":
        _extract_first(rec, body)
    elif phase == "probe2":
        _extract_second(rec, body)
    elif phase == "carry":
        _advance(rec, body)


def on_loop_end(loop_id):
    rec = _session.loop_fold.pop(loop_id, None)
    body = _live(_session.loop_new)
    _session.loop_new.clear()
    if rec is None:
        return
    if rec["phase"] == "carry":
        # the final body has no following marker: collapse it here
        _advance(rec, body)
    elif rec["phase"] in ("probe1", "probe2"):
        _repair(rec)  # loop ended mid-probe: restore eager formulas


def _plant_first(rec, body):
    p = _last_symbolic(body)
    if p is None:
        return  # nothing carried yet; try again next marker
    s_a = sympy.Dummy("state")
    rec.update(
        phase="probe1",
        orig=p.formula,     # the true init of the folded segment
        s_a=s_a,
        planted_pair=weakref.ref(p),
    )
    rec["planted"] = [weakref.ref(p)]
    p.formula = s_a
    p._fsize = 4  # the swap severs the provenance-size estimate


def _extract_first(rec, body):
    q = _last_symbolic(body, needs=rec["s_a"])
    if q is None:
        _repair(rec)
        rec["phase"] = "broken"
        return
    s_b = sympy.Dummy("state")
    rec.update(
        phase="probe2",
        t_a=q.formula,      # template in terms of s_a
        probe2_pair=weakref.ref(q),
        s_b=s_b,
    )
    q.formula = s_b
    q._fsize = 4


def _extract_second(rec, body):
    from .derivation import _generalize

    r = _last_symbolic(body, needs=rec["s_b"])
    if r is None:
        _repair(rec)
        rec["phase"] = "broken"
        return
    t_b = r.formula.xreplace({rec["s_b"]: rec["s_a"]})
    n = sympy.Dummy("n", integer=True, nonnegative=True)
    template = _generalize(rec["t_a"], t_b, n)
    if template is None:
        _repair(rec)
        rec["phase"] = "broken"
        return
    s = sympy.Dummy("s")
    step = sympy.Lambda((s, n), template.xreplace({rec["s_a"]: s}))
    r.formula = Iterate(step, rec["orig"], sympy.Integer(2))
    r._fsize = 64
    rec.update(
        phase="carry",
        step=step,
        template=template,
        n_sym=n,
        count=2,
        head=weakref.ref(r),
    )
    # earlier pairs may still hold probe symbols: restore their eager
    # meaning so nothing outside the fold ever sees a Dummy
    _repair(rec, keep_head=r)


def _advance(rec, body):
    head = rec["head"]()
    if head is None:
        rec["phase"] = "broken"
        return
    m = rec["count"]
    expected = rec["template"].xreplace(
        {rec["s_a"]: head.formula, rec["n_sym"]: sympy.Integer(m)}
    )
    p_new = None
    for p in reversed(body):
        if isinstance(p.formula, sympy.Basic) and p.formula == expected:
            p_new = p
            break
    if p_new is None:
        rec["phase"] = "broken"  # eager formulas remain exact; fold ends
        return
    p_new.formula = Iterate(rec["step"], rec["orig"], sympy.Integer(m + 1))
    p_new._fsize = 64
    rec["count"] = m + 1
    rec["head"] = weakref.ref(p_new)


def _repair(rec, keep_head=None):
    """Substitute probe symbols back to their eager meanings in every
    pair created since the first plant: no Dummy may outlive the fold
    attempt. ``keep_head`` (the new Iterate) is exempt."""
    subs = {}
    if "s_b" in rec:
        subs[rec["s_b"]] = rec["t_a"]           # state after probe 1
    if "s_a" in rec:
        subs[rec["s_a"]] = rec["orig"]
    if not subs:
        return
    for ref in rec.get("planted", ()):
        p = ref()
        if p is None or p is keep_head:
            continue
        f = p.formula
        if isinstance(f, sympy.Basic) and f.free_symbols & set(subs):
            p.formula = f.xreplace(subs).xreplace(subs)
    # guards recorded during probe bodies carry the probe symbols too:
    # a Dummy in .preconditions would be an unbound symbol in a
    # certificate. Rewrite them to their eager meanings in place.
    for i, g in enumerate(_session.guards):
        if isinstance(g, sympy.Basic) and g.free_symbols & set(subs):
            _session.guards[i] = g.xreplace(subs).xreplace(subs)
