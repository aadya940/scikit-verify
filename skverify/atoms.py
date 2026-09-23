"""Opaque atoms: compiled calls the trace cannot enter.

A compiled routine (LAPACK, Cython, f2py) runs on concrete values and
enters the formula as a NAMED term: ``svd_0_2[i, j]`` is output 2 of
the trace's svd call. The atom's defining call, with its operands'
formulas, is recorded in the session; contracts (``.contracts``)
verify what can be verified per call, and the honest remainder is
labeled unknown. The mutation snapshot guarantees no routine secretly
scribbled on traced inputs.
"""

from contextlib import contextmanager as _contextmanager

import numpy as np
import sympy

from .coercion import value_of
from .helpers import axis_idx
from .session import current as _session

_OPAQUE = _session.opaque  # historical alias; the session's list


def opaque_call(func, args, kwargs):
    """Seal one compiled call into a named atom.

    Parameters
    ----------
    func : callable
        The compiled routine (its resolved name labels the atom).
    args, kwargs :
        The call's arguments; Pairs contribute formulas as the atom's
        recorded operands and values for the actual run.

    Returns
    -------
    Pair
        The result carrying the atom's indexed formula (or tuple of
        Pairs for multi-output routines).
    """
    from .pair import Pair

    return _opaque_call_impl(Pair, func, args, kwargs)


# multi-output routines whose outputs have API-level names: an atom
# called svd_S is guessable by anyone who knows what an SVD is,
# svd_1_1 is not. Positions beyond the table fall back to indices.
_OUTPUT_ROLES = {
    "svd": ("U", "S", "Vh"),
    "eigh": ("w", "v"),
    "eig": ("w", "v"),
    "qr": ("Q", "R"),
    "slogdet": ("sign", "logabsdet"),
    "lstsq": ("x", "residuals", "rank", "s"),
}


def _atom_prefix(fname):
    """Group prefix for one multi-output call: role-named routines get
    svd / svd2 / ..; everything else keeps the record-index scheme the
    ancestry scoping matches on."""
    if fname in _OUTPUT_ROLES:
        prior = sum(
            1 for r in _OPAQUE if str(r[-1][0]).startswith(fname)
        )
        return fname if prior == 0 else f"{fname}{prior + 1}"
    return f"{fname}_{len(_OPAQUE)}"


def _out_name(prefix, fname, pos):
    roles = _OUTPUT_ROLES.get(fname)
    if roles and pos < len(roles):
        return f"{prefix}_{roles[pos]}"
    return f"{prefix}_{pos}"


# Generator/RandomState draw methods with an exact sympy.stats twin.
# Each entry: ordered (param, default) pairs matching numpy's call
# signature (size comes after). A default of None marks a required
# parameter. Draws outside this table stay concrete, which is also
# exact: the numbers drawn are the numbers used.
_RNG_PARAMS = {
    "normal": (("loc", 0), ("scale", 1)),
    "standard_normal": (),
    "uniform": (("low", 0), ("high", 1)),
    "random": (),
    "exponential": (("scale", 1),),
    "standard_exponential": (),
    "gamma": (("shape", None), ("scale", 1)),
    "standard_gamma": (("shape", None),),
    "beta": (("a", None), ("b", None)),
    "chisquare": (("df", None),),
    "laplace": (("loc", 0), ("scale", 1)),
    "logistic": (("loc", 0), ("scale", 1)),
    "lognormal": (("mean", 0), ("sigma", 1)),
    "rayleigh": (("scale", 1),),
    "standard_cauchy": (),
    "poisson": (("lam", 1),),
    "binomial": (("n", None), ("p", None)),
    "geometric": (("p", None),),
}

RNG_DISTS = set(_RNG_PARAMS)


def _rng_dist(name):
    """(sympy.stats constructor, numpy-params -> sympy-params map)."""
    import sympy.stats as st

    table = {
        "normal": (st.Normal, lambda p: (p[0], p[1])),
        "standard_normal": (st.Normal, lambda p: (0, 1)),
        "uniform": (st.Uniform, lambda p: (p[0], p[1])),
        "random": (st.Uniform, lambda p: (0, 1)),
        # numpy parameterizes by scale, sympy Exponential by rate
        "exponential": (st.Exponential, lambda p: (1 / p[0],)),
        "standard_exponential": (st.Exponential, lambda p: (1,)),
        "gamma": (st.Gamma, lambda p: (p[0], p[1])),
        "standard_gamma": (st.Gamma, lambda p: (p[0], 1)),
        "beta": (st.Beta, lambda p: (p[0], p[1])),
        "chisquare": (st.ChiSquared, lambda p: (p[0],)),
        "laplace": (st.Laplace, lambda p: (p[0], p[1])),
        "logistic": (st.Logistic, lambda p: (p[0], p[1])),
        "lognormal": (st.LogNormal, lambda p: (p[0], p[1])),
        "rayleigh": (st.Rayleigh, lambda p: (p[0],)),
        "standard_cauchy": (st.Cauchy, lambda p: (0, 1)),
        "poisson": (st.Poisson, lambda p: (p[0],)),
        "binomial": (st.Binomial, lambda p: (p[0], p[1])),
        "geometric": (st.Geometric, lambda p: (p[0],)),
    }
    return table[name]


def _user_code_draw():
    """True when the draw was requested by the user's own code. A
    library drawing internally (initializers, import-time examples)
    keeps plain numbers: that randomness is implementation detail,
    not the user's noise model, and foreign code cannot digest
    Pairs it never asked for."""
    import sys

    f = sys._getframe(2)
    while f is not None:
        mod = (f.f_globals.get("__name__", "") or "").split(".")[0]
        if mod in ("skverify", "numpy"):
            f = f.f_back
            continue
        fname = f.f_code.co_filename
        return not (
            "site-packages" in fname or fname.startswith(sys.base_prefix)
        )
    return False


def rng_draw(fn, args, kwargs):
    """Seal one random draw as a distribution-tagged atom.

    The concrete lane keeps the numbers actually drawn (the generator
    is consumed exactly as in an untraced run). The symbolic lane gets
    a sympy.stats random variable for a scalar draw, so E and variance
    of downstream formulas compute in closed form, or an IndexedBase
    recorded as iid draws for an array. Distribution parameters that
    are traced lift symbolically.
    """
    from .pair import Pair

    name = fn.__name__
    spec = _RNG_PARAMS[name]

    def concrete():
        return fn(
            *[value_of(a) for a in args],
            **{k: value_of(v) for k, v in kwargs.items()},
        )

    if not _user_code_draw():
        return concrete()
    extra = set(kwargs) - {p for p, _ in spec} - {"size"}
    if extra or len(args) > len(spec) + 1:
        # dtype/out and exotic call shapes: draw concretely, as before
        return concrete()
    params = []
    for i, (pname, default) in enumerate(spec):
        if i < len(args):
            params.append(args[i])
        elif pname in kwargs:
            params.append(kwargs[pname])
        elif default is None:
            return concrete()  # required parameter missing: numpy raises
        else:
            params.append(default)
    size = args[len(spec)] if len(args) > len(spec) else kwargs.get("size")
    call_kw = {"size": size} if size is not None else {}
    value = fn(*[value_of(p) for p in params], **call_kw)
    syms = [
        p.formula if isinstance(p, Pair) else sympy.sympify(p) for p in params
    ]
    ctor, xform = _rng_dist(name)
    try:
        dist_args = xform(syms)
    except Exception:
        return value
    desc = f"{ctor.__name__}({', '.join(str(a) for a in dist_args)})"
    label = f"{name}_{len(_OPAQUE)}"
    steps = Pair._steps_of(*[p for p in params if isinstance(p, Pair)])
    if np.shape(value) == ():
        try:
            rv = ctor(label, *dist_args)
        except Exception:
            return value
        _OPAQUE.append(
            (name, (("draw", "concrete"),), (label, f"{label} ~ {desc}"))
        )
        return Pair(value, rv, None, steps=steps)
    base = sympy.IndexedBase(label)
    letters = tuple(axis_idx(ax) for ax in range(value.ndim))
    _OPAQUE.append(
        (
            name,
            (("draw", "concrete"),),
            (f"{label}[...]", f"{label} entries iid ~ {desc}"),
        )
    )
    return Pair(
        value,
        base[letters],
        tuple((0, int(n)) for n in value.shape),
        steps=steps,
    )


def _base_draw(obj, name):
    """The numpy implementation of a draw method, bound to obj: never
    the traced subclass's override, so sealing cannot recurse."""
    cls = (
        np.random.Generator
        if isinstance(obj, np.random.Generator)
        else np.random.RandomState
    )
    return getattr(cls, name).__get__(obj)


def _make_traced_generator():
    def method(name):
        def draw(self, *args, **kwargs):
            return rng_draw(_base_draw(self, name), args, kwargs)

        return draw

    return type(
        "TracedGenerator",
        (np.random.Generator,),
        {nm: method(nm) for nm in RNG_DISTS},
    )


_TRACED_GENERATOR = _make_traced_generator()


@_contextmanager
def trace_rng():
    """While a trace runs, generators born inside the traced code seal
    their draws: default_rng returns a Generator subclass (isinstance
    holds) whose draw methods route through rng_draw, and the legacy
    np.random.* module functions are wrapped the same way. Everything
    is restored on exit, before any real-run arbitration."""
    orig_default = np.random.default_rng

    def default_rng(seed=None):
        return _TRACED_GENERATOR(orig_default(seed).bit_generator)

    def module_wrap(orig):
        def draw(*args, **kwargs):
            return rng_draw(orig, args, kwargs)

        return draw

    saved = {}
    np.random.default_rng = default_rng
    for nm in RNG_DISTS:
        orig = getattr(np.random, nm, None)
        if orig is not None:
            saved[nm] = orig
            setattr(np.random, nm, module_wrap(orig))
    try:
        yield
    finally:
        np.random.default_rng = orig_default
        for nm, orig in saved.items():
            setattr(np.random, nm, orig)


def _opaque_call_impl(Pair, func, args, kwargs):
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
    values = [Pair._numeric(value_of(a)) for a in args]
    kwvalues = {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in ((k, value_of(v)) for k, v in kwargs.items())
    }
    # contracts must judge the INPUTS, not overwrite_*-mutated buffers
    pristine = [
        np.array(v, copy=True) if isinstance(v, np.ndarray) else v
        for v in values
    ]
    try:
        result = func(*values, **kwvalues)
    except (ValueError, TypeError) as e:
        if "contiguous" not in str(e):
            raise
        # memory layout is bookkeeping, not mathematics: retry with
        # the layout the compiled signature demands
        values = [
            np.asfortranarray(v) if isinstance(v, np.ndarray) else v
            for v in values
        ]
        result = func(*values, **kwvalues)
    after = [
        np.asarray(a.value).tobytes()
        for a in pair_args
        if isinstance(a.value, np.ndarray)
    ]
    if snapshots != after:  # pragma: no cover
        # unreachable: the routine runs on copies, so the traced
        # values are never mutated and snapshots always equal after
        raise NotImplementedError(
            f"{func.__name__} mutated a traced input in place"
        )
    if result is None:
        fname_ = getattr(func, "__name__", "") or ""
        if fname_.startswith("__") and fname_.endswith("__"):
            # protocol dunders (a compiled __init__) return None by
            # design and callers discard it: pass the None through,
            # with the receipt on the record
            _OPAQUE.append(
                (
                    fname_,
                    (("state", "concrete"),),
                    (fname_, "compiled protocol call received traced operands"),
                )
            )
            return None
        # a state-setter: no output to name means no atom -- and its
        # effect lives in foreign internal state the trace cannot see
        raise NotImplementedError(
            f"{fname_ or 'a compiled call'} received "
            "traced data but returns nothing; its effect is internal "
            "state the trace cannot follow"
        )
    formulas = []
    notes = []
    n_const = 0
    for a in args:
        if isinstance(a, Pair):
            formulas.append(a.formula)
        elif np.isscalar(a):
            formulas.append(sympy.sympify(a))
        elif isinstance(a, np.ndarray):
            origin = _session.value_origins.get(id(a))
            if origin is not None and origin[0]() is a:
                # this exact buffer was extracted from a traced value:
                # the operand IS that formula (identity-verified, not
                # value-matched -- no guessing)
                formulas.append(origin[1])
            else:
                # a concrete operand: named, so the formula never hides
                # it. If it bitwise-equals a named input, DISCLOSE the
                # observed equality -- an alias through a foreign copy
                # is certificate content, but as an observation, never
                # as a silent substitution
                formulas.append(sympy.Symbol(f"const{n_const}"))
                for iname, ival in getattr(_session, "inputs", {}).items():
                    if a.shape == ival.shape and np.array_equal(a, ival):
                        notes.append(
                            f"const{n_const} == {iname} "
                            "(bitwise-equal at trace time)"
                        )
                        break
                n_const += 1
    # f2py fortran objects report __name__ as "function dgbsv":
    # keep the identifier part only
    fname = getattr(func, "__name__", "opaque").split()[-1].lstrip("_")
    call = sympy.Function(fname)(*formulas)
    if isinstance(result, tuple):
        # multi-output routine (LAPACK gbsv: lu, piv, x, info): each
        # float-array output becomes its own atom; integer bookkeeping
        # (pivots, status) passes through concrete
        outs = []
        prefix = _atom_prefix(fname)
        for pos, res in enumerate(result):
            if isinstance(res, np.ndarray) and res.dtype.kind in "fc":
                name = _out_name(prefix, fname, pos)
                if res.ndim == 0:
                    # a 0-d output (a residue, a rank): a scalar atom
                    outs.append(
                        Pair(
                            res,
                            sympy.Symbol(name, real=True),
                            None,
                            steps=Pair._steps_of(*args),
                        )
                    )
                    continue
                base = sympy.IndexedBase(name)
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
            + ((f"{prefix}_*", "; ".join([str(call)] + notes)),)
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
        check_call(fname, pristine, result)
        + ((str(formula), "; ".join([str(call)] + notes)),)
    )
    return Pair(result, formula, domain=domain, steps=Pair._steps_of(*args))

