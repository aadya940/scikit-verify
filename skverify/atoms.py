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
                name = f"{fname}_{len(_OPAQUE)}_{pos}"
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

