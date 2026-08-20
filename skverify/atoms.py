"""Opaque atoms: compiled calls the trace cannot enter.

A compiled routine (LAPACK, Cython, f2py) runs on concrete values and
enters the formula as a NAMED term: ``svd_0_2[i, j]`` is output 2 of
the trace's svd call. The atom's defining call, with its operands'
formulas, is recorded in the session; contracts (``.contracts``)
verify what can be verified per call, and the honest remainder is
labeled unknown. The mutation snapshot guarantees no routine secretly
scribbled on traced inputs.
"""

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
    fname = getattr(func, "__name__", "opaque").split()[-1]
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

