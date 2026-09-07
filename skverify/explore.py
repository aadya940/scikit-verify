"""Explore every branch a function has, and say so honestly.

One trace covers one path: the verdict "matches" holds on that path
only. explore() closes the gap the way DART (Godefroid, Klarlund and
Sen, 2005) does: take the branch conditions the trace recorded, negate
one, find an input on the other side, trace again, repeat. When every
unexplored side is either visited or PROVEN infeasible, coverage is a
theorem, not a sample.

Witnesses come from constrained sampling (draw exact rationals, force
equalities constructively, reject points off the target region).
Infeasibility comes from sympy's LRA solver, used only for
refutations: a reported model is never trusted (the solver has a
known spurious-model mode), but False is a proof. Anything neither
witnessed nor refuted is an UNDECIDED region, named in the result --
covers() is then False and the caller knows exactly why.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy

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

    @property
    def complete(self):
        """True when the visited paths provably cover every input of
        this shape: no undecided regions, no refused paths, and every
        other region refuted."""
        return not self.undecided and not self.refusals and bool(self.paths)

    def summary(self):
        parts = [f"{len(self.paths)} path(s) explored"]
        if self.infeasible:
            parts.append(f"{len(self.infeasible)} region(s) proven infeasible")
        if self.undecided:
            parts.append(f"{len(self.undecided)} region(s) UNDECIDED")
        if self.refusals:
            parts.append(f"{len(self.refusals)} path(s) refused by the tracer")
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
    slots, syms = set(), set()
    axis_names = set("ijklm")
    for a in atoms:
        if not isinstance(a, sympy.Basic):
            continue
        for e in a.atoms(sympy.Indexed):
            if all(ix.is_Integer for ix in e.indices):
                slots.add(e)
        for s in a.free_symbols:
            if isinstance(s, sympy.Symbol) and s.name not in axis_names \
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


def _witness(target, rng):
    """An exact rational assignment satisfying every atom in target,
    or None. Equalities assign constructively (Eq(a, b) forces the
    slots equal); inequalities go by rejection sampling."""
    target = _unrolled(target)
    eqs = [a for a in target if isinstance(a, sympy.Eq)]
    rest = [a for a in target if not isinstance(a, sympy.Eq)]
    slots, syms = _slots_of(target)
    for _ in range(MAX_TRIES):
        subs = {
            e: sympy.Rational(int(rng.integers(-400, 400)), 100)
            for e in slots
        }
        subs.update({
            s: sympy.Rational(int(rng.integers(-400, 400)), 100)
            for s in syms
        })
        ok = True
        for eq in eqs:
            l, r = eq.lhs, eq.rhs
            if l in subs and not (isinstance(r, sympy.Basic) and r.free_symbols):
                subs[l] = sympy.nsimplify(r)
            elif r in subs and not (isinstance(l, sympy.Basic) and l.free_symbols):
                subs[r] = sympy.nsimplify(l)
            elif l in subs and r in subs:
                subs[l] = subs[r]
            elif not _holds(eq, subs):
                ok = False
                break
        if not ok:
            continue
        if all(_holds(a, subs) for a in eqs + rest):
            return subs
    return None


def _refuted(target):
    """True only when sympy PROVES the region empty. A model answer is
    not trusted (spurious-model mode); False is a proof. LRA does not
    treat Indexed as theory variables, so each distinct slot maps to a
    fresh real Symbol first -- sound for refutation (any real solution
    over slots is one over the symbols and vice versa)."""
    try:
        from sympy.logic.inference import satisfiable

        target = _unrolled(target)
        slots = {
            e
            for a in target
            if isinstance(a, sympy.Basic)
            for e in a.atoms(sympy.Indexed)
        }
        mapping = {
            e: sympy.Dummy(f"s{k}", real=True)
            for k, e in enumerate(sorted(slots, key=str))
        }
        expr = sympy.And(*[
            a.xreplace(mapping) if isinstance(a, sympy.Basic) else a
            for a in target
        ])
        # LRA takes rationals only; every binary float IS one exactly
        floats = {
            f: sympy.Rational(f) for f in expr.atoms(sympy.Float)
        }
        if floats:
            expr = expr.xreplace(floats)
        return satisfiable(expr, use_lra_theory=True) is False
    except Exception:
        return False


def _rebuild_args(base_args, subs):
    """Concrete arguments realizing a witness: copies of the originals
    with every assigned slot written in."""
    out = [np.array(a, dtype=float, copy=True) if isinstance(a, np.ndarray)
           else a for a in base_args]
    names = {}
    for k, a in enumerate(out):
        # positional parameter names were used to wrap: recover by order
        names[k] = a
    by_name = {}
    for e, val in subs.items():
        if isinstance(e, sympy.Indexed):
            by_name.setdefault(str(e.base.label), []).append(
                (tuple(int(ix) for ix in e.indices), float(val))
            )
    # match array args to bases by shape-compatible writes, in order
    import inspect as _unused  # names come from the trace wrap order
    bases = list(by_name)
    ai = 0
    for base in bases:
        while ai < len(out) and not isinstance(out[ai], np.ndarray):
            ai += 1
        if ai >= len(out):
            break
        for idx, val in by_name[base]:
            try:
                out[ai][idx] = val
            except Exception:
                pass
        ai += 1
    for e, val in subs.items():
        if isinstance(e, sympy.Symbol):
            for k, a in enumerate(out):
                if not isinstance(a, np.ndarray):
                    out[k] = float(val)
                    break
    return tuple(out)


def explore(fn, args, max_paths=MAX_PATHS, seed=0):
    """Trace ``fn`` on ``args``, then keep finding inputs that take
    other branches until every branch is visited or proven infeasible.

    Returns an :class:`Exploration`; ``.complete`` is the covers()
    claim: the visited paths provably cover every input of this shape.
    """
    rng = np.random.default_rng(seed)
    result = Exploration()
    seen = set()
    worklist = [tuple(args)]
    while worklist and len(result.paths) < max_paths:
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
            target = list(atoms[:k]) + [_negate(atoms[k])]
            tsig = frozenset(target)
            if tsig in seen:
                continue
            wit = _witness(target, rng)
            if wit is not None:
                worklist.append(_rebuild_args(cur, wit))
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
