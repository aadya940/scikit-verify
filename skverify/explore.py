"""Explore every branch a function has, and say so honestly.

One trace covers one path: the verdict "matches" holds on that path
only. explore() closes the gap the way DART (Godefroid, Klarlund and
Sen, 2005) does: take the branch conditions the trace recorded, negate
one, find an input on the other side, trace again, repeat. When every
unexplored side is either visited or PROVEN infeasible, coverage is a
theorem, not a sample.

Witnesses and refutations both come from Z3 (a required dependency):
a model gives the input for an unexplored branch as exact rationals,
unsat is the proof a region is empty. Models are still VERIFIED by
substitution before use -- the solver proposes, the check disposes.
Guards outside Z3's polynomial fragment (exp, log) fall back to
constrained exact-rational sampling. Anything neither witnessed nor
refuted is an UNDECIDED region, named in the result -- covers() is
then False and the caller knows exactly why.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy
import z3

from .api import to_sympy

MAX_PATHS = 32
MAX_TRIES = 512


@dataclass
class Path:
    args: tuple
    condition: tuple  # ordered guard atoms
    out: object = None


@dataclass
class Exploration:
    paths: list = field(default_factory=list)
    infeasible: list = field(default_factory=list)  # refuted regions
    undecided: list = field(default_factory=list)   # neither witnessed nor refuted
    refusals: list = field(default_factory=list)    # paths the tracer refused
    capped: bool = False  # stopped at max_paths with work remaining

    @property
    def complete(self):
        """True when the visited paths provably cover every input of
        this shape: no undecided regions, no refused paths, nothing
        left unexplored at the path cap, and every other region
        refuted."""
        return (
            not self.undecided
            and not self.refusals
            and not self.capped
            and bool(self.paths)
        )

    def summary(self):
        parts = [f"{len(self.paths)} path(s) explored"]
        if self.infeasible:
            parts.append(f"{len(self.infeasible)} region(s) proven infeasible")
        if self.undecided:
            parts.append(f"{len(self.undecided)} region(s) UNDECIDED")
        if self.refusals:
            parts.append(f"{len(self.refusals)} path(s) refused by the tracer")
        if self.capped:
            parts.append("stopped at the path cap with work remaining")
        head = ", ".join(parts)
        return head + (" -- coverage proven" if self.complete else
                       " -- coverage NOT proven")


def _atoms_of(pre):
    if pre in (sympy.true, True):
        return ()
    if isinstance(pre, sympy.And):
        return tuple(pre.args)
    return (pre,)


def _negate(atom):
    n = getattr(atom, "negated", None)
    return n if n is not None else sympy.Not(atom)


def _slots_of(atoms):
    """Concrete input slots a witness must assign: Indexed with integer
    indices, plus plain (non-axis) Symbols."""
    slots, syms, labels = set(), set(), set()
    axis_names = set("ijklm")
    for a in atoms:
        if not isinstance(a, sympy.Basic):
            continue
        for e in a.atoms(sympy.Indexed):
            if all(ix.is_Integer for ix in e.indices):
                slots.add(e)
            labels.add(e.base.label)
    for a in atoms:
        if not isinstance(a, sympy.Basic):
            continue
        for s in a.free_symbols:
            # an Indexed's free_symbols include its base LABEL, which
            # is not an assignable slot -- witnessing it would shadow
            # the real a0[k] slots
            if isinstance(s, sympy.Symbol) and s.name not in axis_names \
                    and s not in labels \
                    and not isinstance(s, sympy.tensor.indexed.IndexedBase):
                syms.add(s)
    return sorted(slots, key=str), sorted(syms, key=str)


def _holds(atom, subs):
    try:
        v = atom.xreplace(subs)
        if v in (sympy.true, True):
            return True
        if v in (sympy.false, False):
            return False
        return bool(sympy.simplify(v))
    except Exception:
        return False


def _unrolled(target):
    """Guards can bind reduction dummies (Sum(v[j], ...) > 0); doit()
    turns them into concrete-slot expressions a witness can assign."""
    out = []
    for a in target:
        try:
            out.append(a.doit())
        except Exception:
            out.append(a)
    return out


def _forced_ties(atoms):
    """Pairs bounded from both sides are equalities in disguise:
    a >= b together with a <= b is a tie region, which rejection
    sampling hits with probability zero. Detect them so the draw can
    assign the pair EQUAL constructively (sort and median tie paths,
    variance-zero branches)."""
    seen = {}
    ties = []
    for a in atoms:
        if not isinstance(a, (sympy.Le, sympy.Ge)):
            continue
        lo, hi = (a.lhs, a.rhs) if isinstance(a, sympy.Le) else (a.rhs, a.lhs)
        key = tuple(sorted((lo, hi), key=sympy.default_sort_key))
        direction = "le" if (lo, hi) == key else "ge"
        prev = seen.get(key)
        if prev is not None and prev != direction:
            ties.append(key)
        seen[key] = direction
    return ties


def _witness(target, rng):
    """An exact rational assignment satisfying every atom in target,
    or None. Equalities assign constructively (Eq(a, b) forces the
    slots equal, opposite inequalities force ties); plain inequalities
    go by rejection sampling. The last quarter of the budget draws
    ALL-EQUAL per array (one value for every slot of a base): the
    constructive route into variance-zero and all-tied regions."""
    target = _unrolled(target)
    eqs = [a for a in target if isinstance(a, sympy.Eq)]
    rest = [a for a in target if not isinstance(a, sympy.Eq)]
    ties = _forced_ties(rest)
    slots, syms = _slots_of(target)
    for trial in range(MAX_TRIES):
        # escalate the range every quarter of the budget: a guard like
        # v[0] > 100 lives far outside the default window
        span = 400 * 10 ** (4 * trial // MAX_TRIES)
        subs = {
            e: sympy.Rational(int(rng.integers(-span, span)), 100)
            for e in slots
        }
        subs.update({
            s: sympy.Rational(int(rng.integers(-span, span)), 100)
            for s in syms
        })
        if trial > 3 * MAX_TRIES // 4 and slots:
            # all-equal mode: one value per array base
            per_base = {}
            for e in slots:
                base = str(e.base.label)
                if base not in per_base:
                    per_base[base] = subs[e]
                subs[e] = per_base[base]
        for x, y in ties:
            if x in subs and y in subs:
                subs[x] = subs[y]
            elif x in subs and not (
                isinstance(y, sympy.Basic) and y.free_symbols
            ):
                subs[x] = sympy.nsimplify(y)
        ok = True
        for eq in eqs:
            l, r = eq.lhs, eq.rhs
            if l in subs and not (isinstance(r, sympy.Basic) and r.free_symbols):
                subs[l] = sympy.nsimplify(r)
            elif r in subs and not (isinstance(l, sympy.Basic) and l.free_symbols):
                subs[r] = sympy.nsimplify(l)
            elif l in subs and r in subs:
                subs[l] = subs[r]
            else:
                # one unknown slot: SOLVE the equality instead of
                # hoping to sample it (sinc's Eq(pi*x, 0) branch)
                free = [e for e in subs if isinstance(e, sympy.Basic)
                        and (l - r).has(e)]
                solved = False
                if len(free) >= 1 and trial < 8:
                    tgt = free[0]
                    others = {e: v for e, v in subs.items() if e != tgt}
                    try:
                        # solve() rejects Indexed unknowns: go through
                        # a Dummy stand-in
                        d = sympy.Dummy("w", real=True)
                        expr = (l - r).xreplace(others).xreplace({tgt: d})
                        sol = sympy.solve(expr, d, rational=True)
                        if sol:
                            subs[tgt] = sympy.nsimplify(sol[0])
                            solved = True
                    except Exception:
                        pass
                if not solved and not _holds(eq, subs):
                    ok = False
                    break
        if not ok:
            continue
        if all(_holds(a, subs) for a in eqs + rest):
            return subs
    return None


def _refuted(target):
    """True only when Z3 PROVES the region empty (unsat over the
    reals, nlsat decides the polynomial fragment). Atoms outside the
    fragment make refutation impossible here: the region stays
    undecided rather than guessed."""
    try:
        target = _unrolled(target)
        varmap = {}
        constraints = []
        for a in target:
            c = _to_z3(a, z3, varmap)
            if c is None:
                return False  # outside the fragment: cannot refute
            constraints.append(c)
        solver = z3.Solver()
        solver.set("timeout", 1000)
        solver.add(*constraints)
        return solver.check() == z3.unsat
    except Exception:
        return False


def _param_names(fn, n):
    """Positional parameter names, the same way the tracer names
    wrapped arguments; falls back to arg0.. when introspection
    fails (builtins, some callables)."""
    import inspect

    try:
        params = list(inspect.signature(fn).parameters)
        if len(params) >= n:
            return params[:n]
    except (TypeError, ValueError):
        pass
    return [f"arg{k}" for k in range(n)]


def _rebuild_args(fn, base_args, subs):
    """Concrete arguments realizing a witness: copies of the originals
    with every assigned slot written in, matched to parameters BY NAME
    (guard bases are named after the function's parameters)."""
    out = [np.array(a, dtype=float, copy=True) if isinstance(a, np.ndarray)
           else a for a in base_args]
    names = _param_names(fn, len(out))
    position = {name: k for k, name in enumerate(names)}
    for e, val in subs.items():
        if isinstance(e, sympy.Indexed):
            k = position.get(str(e.base.label))
            if k is not None and isinstance(out[k], np.ndarray):
                idx = tuple(int(ix) for ix in e.indices)
                try:
                    out[k][idx] = float(val)
                except (IndexError, ValueError):
                    pass
        elif isinstance(e, sympy.Symbol):
            k = position.get(e.name)
            if k is not None and not isinstance(out[k], np.ndarray):
                out[k] = float(val)
    return tuple(out)




def _to_z3(expr, z3, varmap):
    """sympy relational/arithmetic -> z3, real semantics. Returns None
    for anything outside the polynomial fragment (exp, log, ...)."""
    import sympy as sp

    def conv(e):
        if isinstance(e, sp.Indexed) or isinstance(e, sp.Symbol):
            if e not in varmap:
                varmap[e] = z3.Real(f"v{len(varmap)}")
            return varmap[e]
        if isinstance(e, sp.Integer):
            return z3.RealVal(int(e))
        if isinstance(e, sp.Rational):
            return z3.RealVal(f"{e.p}/{e.q}")
        if isinstance(e, sp.Float):
            r = sp.Rational(e)
            return z3.RealVal(f"{r.p}/{r.q}")
        if isinstance(e, sp.Add):
            parts = [conv(a) for a in e.args]
            if any(p is None for p in parts):
                return None
            out = parts[0]
            for p in parts[1:]:
                out = out + p
            return out
        if isinstance(e, sp.Mul):
            parts = [conv(a) for a in e.args]
            if any(p is None for p in parts):
                return None
            out = parts[0]
            for p in parts[1:]:
                out = out * p
            return out
        if isinstance(e, sp.Pow):
            base = conv(e.base)
            if base is None or not e.exp.is_Integer or e.exp < 0:
                return None
            out = base
            for _ in range(int(e.exp) - 1):
                out = out * base
            return out
        if isinstance(e, sp.Abs):
            inner = conv(e.args[0])
            if inner is None:
                return None
            return z3.If(inner >= 0, inner, -inner)
        return None

    rel = {sp.Gt: lambda a, b: a > b, sp.Ge: lambda a, b: a >= b,
           sp.Lt: lambda a, b: a < b, sp.Le: lambda a, b: a <= b,
           sp.Eq: lambda a, b: a == b, sp.Ne: lambda a, b: a != b}
    for cls, mk in rel.items():
        if isinstance(expr, cls):
            l, r = conv(expr.lhs), conv(expr.rhs)
            if l is None or r is None:
                return None
            return mk(l, r)
    return None


def _witness_z3(target):
    """Solver-generated witness: exact rationals from a z3 model over
    the polynomial fragment (nlsat decides it). Returns None when an
    atom falls outside the fragment or the region is unsat. Every
    returned point is still VERIFIED by substitution by the caller:
    the model is a candidate, never an oracle."""
    target = _unrolled(target)
    varmap = {}
    constraints = []
    for a in target:
        c = _to_z3(a, z3, varmap)
        if c is None:
            return None  # outside the fragment: fall back to sampling
        constraints.append(c)
    solver = z3.Solver()
    solver.set("timeout", 1000)
    solver.add(*constraints)
    if solver.check() != z3.sat:
        return None
    model = solver.model()
    subs = {}
    for sym, var in varmap.items():
        val = model.eval(var, model_completion=True)
        # exact rational out of z3 (decimals would round)
        frac = val.as_fraction()
        subs[sym] = sympy.Rational(frac.numerator, frac.denominator)
    return subs


def explore(fn, args, max_paths=MAX_PATHS, seed=0, time_budget=120.0):
    """Trace ``fn`` on ``args``, then keep finding inputs that take
    other branches until every branch is visited or proven infeasible.

    Returns an :class:`Exploration`; ``.complete`` is the covers()
    claim: the visited paths provably cover every input of this shape.
    ``time_budget`` (seconds of wall clock) ends exploration honestly:
    past it the result is marked capped and completeness is never
    claimed.
    """
    import time as _time

    deadline = _time.monotonic() + time_budget
    rng = np.random.default_rng(seed)
    result = Exploration()
    seen = set()
    worklist = [tuple(args)]
    while worklist:
        if len(result.paths) >= max_paths or _time.monotonic() > deadline:
            result.capped = True
            break
        cur = worklist.pop()
        try:
            out = to_sympy(fn, *[
                a.copy() if isinstance(a, np.ndarray) else a for a in cur
            ])
        except NotImplementedError as e:
            result.refusals.append(str(e)[:120])
            continue
        atoms = _atoms_of(getattr(out, "preconditions", sympy.true))
        sig = frozenset(atoms)
        if sig in seen:
            continue
        seen.add(sig)
        result.paths.append(Path(args=cur, condition=atoms, out=out))
        # DART step: negate each guard with the earlier ones held
        for k in range(len(atoms)):
            if _time.monotonic() > deadline:
                result.capped = True
                break
            target = list(atoms[:k]) + [_negate(atoms[k])]
            tsig = frozenset(target)
            if tsig in seen:
                continue
            wit = _witness_z3(target)
            if wit is not None and not all(
                _holds(a, wit) for a in _unrolled(target)
            ):
                wit = None  # model failed verification: never trust it
            if wit is None:
                wit = _witness(target, rng)
            if wit is not None:
                worklist.append(_rebuild_args(fn, cur, wit))
            elif _refuted(target):
                result.infeasible.append(sympy.And(*target))
                seen.add(tsig)
            else:
                result.undecided.append(sympy.And(*target))
                seen.add(tsig)
    return result


def covers(fn, args, **kw):
    """The one-word form: does exploration provably reach every branch?"""
    return explore(fn, args, **kw).complete
