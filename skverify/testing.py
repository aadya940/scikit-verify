"""Check code against the mathematics you believe it implements.

The spec comes from OUTSIDE the code -- a paper, a docstring, your
derivation. skverify traces the function into a formula and compares
the two symbolically, entry by entry. The spec must not be derived
from the trace itself: checking the code against its own output would
always pass.

Minimal scaffold: scalar and entrywise equality specs, the four-tier
verdict, the pytest decorator. Piecewise seams, property rungs beyond
a bare callable, assume-driven normalization and coverage proofs land
on top of this skeleton.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy

from .api import to_sympy
from .helpers import axis_idx

_FORMULA_CHAR_LIMIT = 400


def _truncate_formula(expr):
    """Return a display string for *expr*, truncating with a visible
    marker when the text representation exceeds the character limit.
    The original SymPy expression is never modified."""
    text = str(expr)
    if len(text) <= _FORMULA_CHAR_LIMIT:
        return text
    truncated = text[:_FORMULA_CHAR_LIMIT]
    remaining = len(text) - _FORMULA_CHAR_LIMIT
    return (
        f"{truncated} ... [cut: {remaining} more characters; "
        f"the full formula is in verdict.traced] ..."
    )


@dataclass
class Verdict:
    """The outcome of one spec check. Never a bare boolean: the tier
    and the shape it was decided at are part of the result."""

    tier: str  # exact | float-constant | sampled | differs | undecided | incomplete
    shape: tuple
    spec: object = None
    traced: object = None
    counterexample: dict = field(default_factory=dict)
    detail: str = ""

    @property
    def matches(self):
        return self.tier in ("exact", "float-constant", "sampled")

    def message(self):
        head = f"verdict: {self.tier} (at shape {self.shape})"
        if self.matches or self.tier == "incomplete":
            return head + (f"\n  {self.detail}" if self.detail else "")
        lines = [head]
        lines.append(f"  your spec:  {_truncate_formula(self.spec)}")
        lines.append(f"  the code:   {_truncate_formula(self.traced)}")
        if self.counterexample:
            lines.append("  counterexample:")
            for k, v in self.counterexample.items():
                lines.append(f"      {k} = {v}")
        if self.detail:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)


def _param_names_of(out):
    """Names of the traced inputs, from the wrapped bases present
    anywhere in formulas, guards or sealed-call definitions."""
    names = set()
    pre = getattr(out, "preconditions", sympy.true)
    pool = [pre] if isinstance(pre, sympy.Basic) else []
    for rec in getattr(out, "unchecked", ()):
        if isinstance(rec, tuple) and len(rec) >= 3:
            names.update(
                part.split("[")[0].split("(")[-1].strip()
                for part in str(rec[-1][-1]).replace(")", ",").split(",")
                if part.strip()
            )
    for e in pool:
        for a in e.atoms(sympy.Indexed):
            names.add(str(a.base.label))
    return names


def check_formula(fn, args, spec, indices=(), assume=(), samples=3,
                  explore=False):
    """Trace ``fn`` and compare its formula against ``spec``, per entry.

    The spec must come from outside the code -- a paper, a docstring,
    a derivation. The verdict is decided at the traced shape, the same
    honesty concolic testing states about fixed inputs.

    Parameters
    ----------
    fn : callable
        The function under test. Python + NumPy only; compiled calls
        inside it produce an ``incomplete`` verdict.
    args : tuple
        Concrete arguments to trace on. They fix shapes and the
        branch taken; the numbers never decide the verdict.
    spec : sympy.Expr
        The claimed mathematics, written over IndexedBase symbols
        named after ``fn``'s parameters (an array argument ``v``
        appears as ``IndexedBase("v")``).
    indices : tuple of sympy.Symbol, optional
        The spec's index symbols, bound to output axes in order:
        ``indices=(i,)`` makes ``i`` mean "position in the result".
        Omit for scalar results.
    assume : iterable of sympy relations, optional
        The derivation's preconditions, e.g. ``[v[0] > 0]``. Sample
        points for numeric arbitration are drawn to satisfy them: a
        spec is only claimed on its stated domain.
    samples : int, optional
        Exact rational sample points used when the symbolic
        difference does not vanish (the float-constant tier).
    explore : bool, optional
        Check the spec on EVERY reachable branch, not just the one
        ``args`` takes: branch guards are negated and solved for
        inputs on the other side (see :func:`skverify.explore.explore`).
        The verdict then reports coverage: "coverage proven" when
        every unvisited region was proven infeasible, or an honest
        qualifier when regions stay undecided.

    Returns
    -------
    Verdict
        ``tier`` is one of ``"exact"``, ``"float-constant"``,
        ``"differs"``, ``"undecided"``, ``"incomplete"``; see
        :class:`Verdict`. ``differs`` carries both formulas and a
        concrete counterexample.

    Examples
    --------
    >>> import numpy as np, sympy
    >>> from skverify.testing import check_formula
    >>> V = sympy.IndexedBase("v")
    >>> i = sympy.Symbol("i", integer=True)
    >>> def affine(v):
    ...     return 3.0 * v + 1.0
    >>> v = check_formula(affine, (np.array([1.0, 2.0]),),
    ...                   3 * V[i] + 1, indices=(i,))
    >>> v.tier, v.shape
    ('exact', (2,))

    A wrong implementation returns ``differs`` with a counterexample:

    >>> def wrong(v):
    ...     return 3.0 * v + 1.5
    >>> check_formula(wrong, (np.array([1.0, 2.0]),),
    ...               3 * V[i] + 1, indices=(i,)).tier
    'differs'
    """
    if explore:
        return _check_everywhere(fn, args, spec, indices, assume, samples)
    try:
        out = to_sympy(fn, *args)
    except NotImplementedError as e:
        return Verdict(
            tier="incomplete",
            shape=tuple(np.shape(args[0])),
            detail=f"the tracer refused: {e} (a tracer limit, not a code bug)",
        )
    return _check_traced(out, spec, indices, assume, samples)


def _check_everywhere(fn, args, spec, indices, assume, samples):
    """The explored form: check the spec on EVERY reachable path.
    The verdict is the weakest per-path tier; coverage status is part
    of the detail, so "all paths" is claimed only when proven."""
    from .explore import explore as _explore

    ex = _explore(fn, args, constraints=tuple(assume))
    if not ex.paths:
        return Verdict(
            tier="incomplete",
            shape=tuple(np.shape(args[0])),
            detail="no path traced: " + "; ".join(ex.refusals[:2]),
        )
    order = {"exact": 0, "float-constant": 1, "sampled": 2}
    worst = None
    for path in ex.paths:
        v = _check_traced(path.out, spec, indices, assume, samples)
        if not v.matches:
            cond = " & ".join(str(a) for a in path.condition) or "True"
            v.detail = (v.detail + f"\n  on the path where: {cond}").strip()
            return v
        if worst is None or order.get(v.tier, 9) > order.get(worst.tier, 9):
            worst = v
    worst.detail = (
        f"all {len(ex.paths)} path(s): " + worst.detail + "; " + ex.summary()
    )
    if not ex.complete:
        worst.detail += " (matches on explored paths only)"
    return worst


def _check_traced(out, spec, indices, assume, samples):
    shape = tuple(np.shape(out.value))
    bound = {sym: axis_idx(k) for k, sym in enumerate(indices)}
    # traced scalar symbols carry real=True; a user's plain
    # Symbol("u") must still mean the same thing. Bind by name.
    def _elements(f):
        if isinstance(f, sympy.Basic):
            return [f]
        if isinstance(f, sympy.NDimArray):
            return [e for e in f if isinstance(e, sympy.Basic)]
        if isinstance(f, np.ndarray) and f.dtype == object:
            # traces of some array results come back as numpy object
            # arrays of expressions rather than NDimArray
            return [e for e in f.ravel() if isinstance(e, sympy.Basic)]
        return []

    traced_syms = set().union(
        *(e.free_symbols for e in _elements(out.formula))
    ) if _elements(out.formula) else set()
    by_name = {t.name: t for t in traced_syms if isinstance(t, sympy.Symbol)}
    user_syms = set(spec.free_symbols)
    for f in assume:
        if isinstance(f, sympy.Basic):
            user_syms |= f.free_symbols
    for sym in user_syms:
        if (
            isinstance(sym, sympy.Symbol)
            and sym not in bound
            and sym.name in by_name
            and sym != by_name[sym.name]
        ):
            bound[sym] = by_name[sym.name]
    traced_bases = {
        e.base.label.name
        for x in _elements(out.formula)
        for e in x.atoms(sympy.Indexed)
    }
    unknown = sorted(
        s.name for s in spec.free_symbols
        if isinstance(s, sympy.Symbol)
        and s not in bound
        and not isinstance(s, sympy.tensor.indexed.IndexedBase)
        and s.name not in by_name
        and s.name not in traced_bases
        and not any(s in f.free_symbols for f in bound)
    )
    if unknown:
        sealed = [str(r[0]) for r in getattr(out, "unchecked", ())
                  if isinstance(r, tuple)]
        if sealed and any(u in _param_names_of(out) for u in unknown):
            # the input exists but the computation routed through
            # compiled calls: the formula is in sealed atoms, and a
            # formula-level spec cannot reach the inputs behind them
            return Verdict(
                tier="incomplete",
                shape=shape,
                spec=spec,
                detail=(
                    "the input(s) " + ", ".join(unknown)
                    + " do not appear in the traced formula: the "
                    "computation routed through sealed compiled calls ("
                    + ", ".join(sealed[:3])
                    + "). Formula-level specs stop at that boundary; "
                    "contracts on the sealed calls still apply."
                ),
            )
        return Verdict(
            tier="undecided",
            shape=shape,
            spec=spec,
            detail=(
                "the spec references symbols the trace does not: "
                + ", ".join(unknown)
                + " (a typo, or a constant that should be a number?)"
            ),
        )
    spec_b = spec.xreplace(bound) if bound else spec
    if bound and assume:
        assume = [
            f.xreplace(bound) if isinstance(f, sympy.Basic) else f
            for f in assume
        ]

    entries = list(np.ndindex(shape)) if shape else [()]
    sampled = False
    is_array = isinstance(out.formula, sympy.NDimArray) or (
        isinstance(out.formula, np.ndarray)
        and out.formula.dtype == object
    )
    for entry in entries:
        at = {axis_idx(k): int(v) for k, v in enumerate(entry)}
        if is_array:
            t = out.formula[entry]  # one expression per element
        else:
            t = out.formula.subs(at) if at else out.formula
        s = spec_b.subs(at) if at else spec_b
        verdict, used_sampling = _entry_equal(
            t, s, entry, samples, assume, getattr(out, "preconditions", ())
        )
        sampled = sampled or used_sampling
        if verdict is not None:
            verdict.shape = shape
            return verdict
    # the tier states HOW the decision was made: exact means every
    # entry's difference vanished symbolically. When at least one
    # entry needed arbitration at exact sample points, the tier says
    # WHY symbolic zero was out of reach: float-constant when the
    # trace carries rounded constants (a zscore's 1/sqrt(5) can never
    # equal an exact spec), sampled when the reason is structural
    # (path guards, or a difference simplify cannot close in budget)
    if not sampled:
        tier = "exact"
        why = ""
    elif isinstance(out.formula, sympy.Basic) and any(
        isinstance(a, sympy.Float) for a in out.formula.atoms(sympy.Number)
    ):
        tier = "float-constant"
        why = ("; code computes with rounded float constants, agreement "
               f"verified at {samples} exact rational points")
    else:
        tier = "sampled"
        why = (f"; agreement verified at {samples} exact rational points "
               "within the traced path's guards")
    detail = f"{len(entries)}/{len(entries)} entries agree" + why
    return Verdict(tier=tier, shape=shape, spec=spec, detail=detail)


def _rebuilt(e):
    """Unconditional bottom-up rebuild under normal evaluation.
    evaluate(False) construction freezes Add/Mul structure so hard
    that even evalf keeps the shape; a fully substituted, Sum-free
    sample expression is small enough to just rebuild outright."""
    if not isinstance(e, sympy.Basic) or not e.args:
        return e
    return e.func(*[_rebuilt(a) for a in e.args])


def _to_float(expr):
    """Numeric value of a fully substituted expression. Min/Max will
    not order a Float against an exact radical even fully rebuilt, so
    they resolve numerically here, innermost first."""
    e = _rebuilt(expr)
    while True:
        clamps = [
            m
            for m in e.atoms(sympy.Min, sympy.Max)
            if not any(a.has(sympy.Min, sympy.Max) for a in m.args)
        ]
        if not clamps:
            break
        pick = min if isinstance(clamps[0], sympy.Min) else max
        val = pick(float(sympy.N(a)) for a in clamps[0].args)
        e = _rebuilt(e.xreplace({clamps[0]: sympy.Float(val)}))
    v = sympy.N(e)
    try:
        return float(v)
    except TypeError:
        # complex-valued formulas (FFT lane) compare by modulus of
        # the difference; complex() is the honest numeric value
        return complex(v)


def _draw_point(slots, syms, rng, assume, tries=64):
    """A rational sample point satisfying every assumption, or None.
    Rejection sampling: cheap for the sign/ordering constraints real
    specs state, honest (undecided) when the domain is too thin."""
    for _ in range(tries):
        subs = {
            e: sympy.Rational(int(rng.integers(-300, 300)), 100)
            for e in slots
        }
        for sy in syms:
            if isinstance(sy, sympy.Symbol):
                subs[sy] = sympy.Rational(int(rng.integers(-300, 300)), 100)
        ok = True
        for cond in assume:
            if not isinstance(cond, sympy.Basic):
                continue
            v = cond.doit().xreplace(subs)
            # noqa-worthy on purpose: `not v` would call
            # Relational.__bool__, which RAISES on partially
            # substituted conditions; == False is a quiet non-match
            if v is sympy.false or v == False:  # noqa: E712
                ok = False
                break
        if ok:
            return subs
    return None


def _zero_within_budget(d, seconds=10):
    """Is ``d`` symbolically zero, giving simplify a bounded slice of
    time? Cheap closers run first (expand, together/cancel); the full
    simplify runs under a POSIX alarm where available -- a CI tool may
    never hang on one stubborn difference. On timeout or off-POSIX the
    answer is False and arbitration decides at sample points."""
    try:
        e = sympy.expand(d.doit())
        if e == 0:
            return True
        e2 = sympy.cancel(sympy.together(e))
        if e2 == 0:
            return True
    except Exception:
        return False
    import signal
    import threading

    if (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    ):
        def _timeout(signum, frame):
            raise TimeoutError

        old = signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return sympy.simplify(e2) == 0
        except (TimeoutError, Exception):
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    from .helpers import ops_capped

    if ops_capped(e2, 2000) is not None:  # small enough to risk
        try:
            return sympy.simplify(e2) == 0
        except Exception:
            return False
    return False




def _apply_assumptions(expr, assume):
    """Simplify ``expr`` using the facts in ``assume``, so stated
    domain knowledge closes proofs instead of only steering sample
    points.

    Rules, each fired only when a fact matches the arguments exactly
    or up to a cheap ``expand``:

    - ``a < b`` or ``a <= b``: ``Min(a, b) -> a``, ``Max(a, b) -> b``
    - ``x > 0``:  ``Abs(x) -> x``,  ``sign(x) -> 1``
    - ``x >= 0``: ``Abs(x) -> x``
    - ``x < 0``:  ``Abs(x) -> -x``, ``sign(x) -> -1``
    - ``Eq(lhs, rhs)``: substitute ``lhs -> rhs`` (a stated identity
      holds everywhere on the claimed domain)

    A fact that is too weak fires nothing: ``Ne(a, b)`` leaves
    ``Min(a, b)`` alone. Facts apply to BOTH the spec and the traced
    formula, so neither side is privileged.
    """
    if not assume or not isinstance(expr, sympy.Basic):
        return expr

    less = []      # (small, big) from a < b and a <= b
    pos, neg, nonneg = [], [], []
    eqs = {}
    for fact in assume:
        if not isinstance(fact, sympy.Basic):
            continue
        if isinstance(fact, sympy.Eq):
            eqs[fact.lhs] = fact.rhs
            continue
        if isinstance(fact, (sympy.Gt, sympy.Ge)):
            small, big = fact.rhs, fact.lhs
            strict = isinstance(fact, sympy.Gt)
        elif isinstance(fact, (sympy.Lt, sympy.Le)):
            small, big = fact.lhs, fact.rhs
            strict = isinstance(fact, sympy.Lt)
        else:
            continue
        less.append((small, big))
        if small == 0:
            (pos if strict else nonneg).append(big)
        if big == 0 and strict:
            neg.append(small)

    def _same(x, y):
        if x == y:
            return True
        try:
            return sympy.expand(x - y) == 0
        except Exception:
            return False

    for _ in range(3):  # nested Min/Abs resolve over a few rounds
        m = dict(eqs)
        for node in expr.atoms(sympy.Min, sympy.Max):
            if len(node.args) != 2:
                continue
            x, y = node.args
            for small, big in less:
                if (_same(x, small) and _same(y, big)) or (
                    _same(y, small) and _same(x, big)
                ):
                    m[node] = small if isinstance(node, sympy.Min) else big
                    break
        for node in expr.atoms(sympy.Abs):
            # sympy canonicalizes the argument's sign, so a fact about
            # b - a must also match an argument stored as a - b
            x = node.args[0]
            if any(_same(x, p) for p in pos + nonneg) or any(
                _same(-x, n) for n in neg
            ):
                m[node] = x
            elif any(_same(-x, p) for p in pos + nonneg) or any(
                _same(x, n) for n in neg
            ):
                m[node] = -x
        for node in expr.atoms(sympy.sign):
            x = node.args[0]
            if any(_same(x, p) for p in pos) or any(_same(-x, n) for n in neg):
                m[node] = sympy.Integer(1)
            elif any(_same(-x, p) for p in pos) or any(_same(x, n) for n in neg):
                m[node] = sympy.Integer(-1)
        m = {k: v for k, v in m.items() if k in expr.atoms(type(k)) or k in eqs}
        new = expr.xreplace(m) if m else expr
        if new == expr:
            break
        expr = new
    return expr

def _entry_equal(t, s, entry, samples, assume=(), guards=()):
    """(verdict, used_sampling): verdict is None when the entry
    agrees. Exact tier first, sample-point arbitration second; sample
    points are drawn to satisfy ``assume`` AND the traced path's own
    guards -- a per-path formula is only claimed on its path (a
    chebyshev trace that picked element 2 as the max must not be
    sampled where element 0 wins)."""
    t = _apply_assumptions(t, assume)
    s = _apply_assumptions(s, assume)
    if _zero_within_budget(t - s):
        return None, False
    rng = np.random.default_rng(0)
    # unroll reductions FIRST: a spec's Sum binds a dummy, and only
    # after doit() do its terms carry concrete indices a sample point
    # can bind
    td, sd = t.doit(), s.doit()
    slots = sorted(
        {
            e
            for x in (td, sd)
            for e in x.atoms(sympy.Indexed)
            if all(ix.is_Integer for ix in e.indices)
        },
        key=str,
    )
    labels = {
        e.base.label
        for x in (td, sd)
        for e in x.atoms(sympy.Indexed)
    }
    syms = sorted(
        (td - sd).free_symbols
        - set(sympy.symbols("i j k l m"))
        - labels,  # an Indexed's base label is not an assignable input
        key=str,
    )
    agree = True
    point = {}
    for _ in range(samples):
        conds = list(assume)
        if isinstance(guards, sympy.Basic) and guards not in (sympy.true,):
            conds.extend(
                guards.args if isinstance(guards, sympy.And) else [guards]
            )
        subs = _draw_point(slots, syms, rng, conds)
        if subs is None:
            return Verdict(
                tier="undecided",
                shape=(),
                spec=s,
                traced=t,
                detail=f"entry {entry}: no sample point satisfies the assumptions",
            ), True
        try:
            tv = _to_float(td.xreplace(subs))
            sv = _to_float(sd.xreplace(subs))
        except (TypeError, ValueError):
            return Verdict(
                tier="undecided",
                shape=(),
                spec=s,
                traced=t,
                detail=f"entry {entry}: could not decide symbolically or numerically",
            ), True
        if abs(tv - sv) > 1e-10 * max(1.0, abs(sv)):
            agree = False
            point = {str(k): v for k, v in subs.items()}
            point["spec value"] = sv
            point["code value"] = tv
            break
    if agree:
        return None, True  # float-constant tier, labeled by the caller
    return Verdict(
        tier="differs",
        shape=(),
        spec=s,
        traced=t,
        counterexample=point,
        detail=f"first disagreement at entry {entry}",
    ), True


def specifies(spec, indices=(), assume=(), explore=True):
    """Assert that a function implements a formula, as a pytest test.

    The decorated test RETURNS ``(fn, args)`` instead of calling
    anything -- the trace stays under skverify's control. A matching
    spec passes; a mismatch fails with both formulas and a concrete
    counterexample; a tracer refusal skips (a tracer limit is not a
    code bug).

    Parameters
    ----------
    spec : sympy.Expr
        The claimed mathematics; see :func:`check_formula`.
    indices : tuple of sympy.Symbol, optional
        Index symbols bound to output axes in order.
    assume : iterable of sympy relations, optional
        The derivation's preconditions; sample points respect them.
    explore : bool, optional
        True by default: the spec is checked on EVERY reachable
        branch (Z3 finds inputs for the paths the test data never
        took), so a green checkmark cannot hide an unchecked branch.
        Pass ``explore=False`` for the faster single-path check when
        a branchy function makes exploration slow.

    Examples
    --------
    The docstring of ``scipy.integrate.trapezoid`` for five samples
    with unit spacing, as a test::

        import numpy as np, sympy
        from scipy.integrate import trapezoid
        from skverify.testing import specifies

        Y = sympy.IndexedBase("y")

        @specifies(Y[0]/2 + Y[1] + Y[2] + Y[3] + Y[4]/2)
        def test_trapezoid_is_its_docstring():
            return (lambda y: trapezoid(y),
                    (np.array([0.7, 1.2, 2.5, 0.3, 0.4]),))

    An entrywise spec binds an index symbol::

        i = sympy.Symbol("i", integer=True)
        V = sympy.IndexedBase("v")

        @specifies(3 * V[i] + 1, indices=(i,))
        def test_affine():
            return my_affine, (np.array([1.0, 2.0]),)

    A precondition-carrying spec (harmonic mean is only claimed for
    positive data; scipy returns nan otherwise)::

        j = sympy.Dummy("j", integer=True)
        spec = 5 / sympy.Sum(1 / V[j], (j, 0, 4))

        @specifies(spec, assume=[V[k] > 0 for k in range(5)])
        def test_hmean():
            return (lambda v: scipy.stats.hmean(v),
                    (np.array([0.7, 1.2, 2.5, 0.3, 0.4]),))
    """

    def deco(test_fn):
        def wrapper():
            fn, args = test_fn()
            v = check_formula(fn, args, spec, indices=indices,
                              assume=assume, explore=explore)
            if v.tier == "incomplete":
                import pytest

                pytest.skip(v.message())
            assert v.matches, v.message()

        wrapper.__name__ = test_fn.__name__
        wrapper.__doc__ = test_fn.__doc__
        return wrapper

    return deco


def check_property(fn, args, prop, assume=(), explore=True, samples=3):
    """Check a FACT about what the code computes, on every reachable
    path: the property rung. Where check_formula needs the closed
    form, this needs only what the paper proves about it (symmetry,
    normalization, a null space).

    Parameters
    ----------
    fn, args
        As in :func:`check_formula`.
    prop : callable
        Receives the traced formula, returns a sympy relation (or
        boolean). Checked per path.
    assume : iterable of sympy relations, optional
        The property's domain; constrains exploration and sampling.
    explore : bool, optional
        True by default: the property is checked on every reachable
        branch, since a fact that holds on the traced path can fail
        on another.

    Returns
    -------
    Verdict
    """
    if explore:
        from .explore import explore as _explore

        ex = _explore(fn, args, constraints=tuple(assume))
        outs = [p.out for p in ex.paths]
        cover_note = "; " + ex.summary()
    else:
        try:
            outs = [to_sympy(fn, *args)]
        except NotImplementedError as e:
            return Verdict(
                tier="incomplete",
                shape=tuple(np.shape(args[0])),
                detail=f"the tracer refused: {e} (a tracer limit, not a code bug)",
            )
        cover_note = " (single traced path)"
    if not outs:
        return Verdict(tier="incomplete", shape=tuple(np.shape(args[0])),
                       detail="no path traced" + cover_note)
    sampled = False

    def check_claim(claim, out):
        """None when the claim holds on this path; (verdict, used)
        otherwise. And decomposes; Eq goes by symbolic residual;
        inequalities by entailment (guards and assume imply claim,
        a solver model of the negation is the failing input)."""
        if claim in (True, sympy.true):
            return None, False
        sealed = {
            str(r[0]) for r in getattr(out, "unchecked", ())
            if isinstance(r, tuple)
        }
        if sealed and isinstance(claim, sympy.Basic):
            atom_bases = {
                str(e.base.label)
                for e in claim.atoms(sympy.Indexed)
            } | {s_.name for s_ in claim.free_symbols
                 if isinstance(s_, sympy.Symbol)}
            touched = {
                b for b in atom_bases
                if any(b.startswith(nm) for nm in sealed)
            }
            if touched:
                # entries of a sealed call are DETERMINED by the real
                # inputs, not free: arbitrating over them invents
                # impossible counterexamples. The honest tier is
                # incomplete, boundary named.
                return Verdict(
                    tier="incomplete",
                    shape=tuple(np.shape(out.value)),
                    spec=claim,
                    detail=(
                        "the fact ranges over sealed compiled outputs ("
                        + ", ".join(sorted(touched)[:3])
                        + "); their entries are determined by the "
                        "inputs, so symbolic arbitration cannot "
                        "decide it"
                    ),
                ), False
        if claim in (False, sympy.false):
            return Verdict(
                tier="differs", shape=tuple(np.shape(out.value)),
                spec=claim, traced=out.formula,
                detail="property is False on a path" + cover_note,
            ), False
        if isinstance(claim, sympy.And):
            any_used = False
            for part in claim.args:
                v, used = check_claim(part, out)
                any_used = any_used or used
                if v is not None:
                    return v, any_used
            return None, any_used
        if isinstance(claim, sympy.Eq):
            verdict, used = _entry_equal(
                claim.lhs - claim.rhs, sympy.Integer(0), (), samples,
                assume, getattr(out, "preconditions", ()),
            )
            return verdict, used
        # inequality: guards & assume must ENTAIL it
        from .explore import _refuted, _witness, _witness_z3

        guards = getattr(out, "preconditions", sympy.true)
        guard_atoms = (
            list(guards.args) if isinstance(guards, sympy.And)
            else [] if guards in (sympy.true, True) else [guards]
        )
        target = (
            [a for a in assume if isinstance(a, sympy.Basic)]
            + guard_atoms + [sympy.Not(claim)]
        )
        wit = _witness_z3(target)
        if wit is None:
            wit = _witness(target, np.random.default_rng(0))
        if wit is not None:
            point = {str(k): v for k, v in wit.items()}
            return Verdict(
                tier="differs", shape=tuple(np.shape(out.value)),
                spec=claim, traced=out.formula,
                counterexample=point,
                detail="property fails at this input" + cover_note,
            ), True
        if _refuted(target):
            return None, False  # entailment proven
        return Verdict(
            tier="undecided", shape=tuple(np.shape(out.value)),
            spec=claim, traced=out.formula,
            detail="could not prove or refute the property" + cover_note,
        ), True

    for out in outs:
        claim = prop(out.formula)
        verdict, used = check_claim(claim, out)
        sampled = sampled or used
        if verdict is not None:
            verdict.spec = claim
            verdict.traced = out.formula
            if verdict.tier != "incomplete" and "property" not in verdict.detail:
                verdict.detail = (
                    "property does not hold: " + str(claim)[:160]
                    + cover_note
                )
            verdict.shape = tuple(np.shape(out.value))
            return verdict
    tier = "sampled" if sampled else "exact"
    return Verdict(
        tier=tier, shape=tuple(np.shape(outs[0].value)),
        detail=f"property holds on all {len(outs)} path(s)" + cover_note,
    )


def _property(prop, assume=(), explore=True):
    """Assert a property of the traced certificate, no closed form
    needed.

    Where :func:`specifies` checks *what* the code computes, this
    checks a fact the paper proves about it -- the stronger rung when
    nobody knows the entries.

    Parameters
    ----------
    prop : callable
        Receives the traced formula, returns a sympy relation (or a
        plain boolean). The relation must simplify to true.
    assume : iterable of sympy relations, optional
        Reserved for domain-restricted properties.

    Examples
    --------
    Centered data sums to exactly zero, whatever the input::

        import sympy
        from skverify.testing import specifies
        i = sympy.Symbol("i", integer=True)

        @specifies.property(
            lambda F: sympy.Eq(sum(F.subs(i, k) for k in range(5)), 0)
        )
        def test_centering_kills_the_mean():
            return (lambda v: v - v.mean(), (data,))
    """

    def deco(test_fn):
        def wrapper():
            fn, args = test_fn()
            v = check_property(fn, args, prop, assume=assume,
                               explore=explore)
            if v.tier == "incomplete":
                import pytest

                pytest.skip(v.message())
            assert v.matches, v.message()

        wrapper.__name__ = test_fn.__name__
        return wrapper

    return deco


specifies.property = _property
