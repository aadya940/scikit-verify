"""The traced dialect and how to extend it.

skverify understands numpy through registration tables: ufuncs map to
sympy functions, array functions map to formula constructors, and
names at compiled or math-neutral boundaries carry declared roles.
This module is the public face of those tables. Library authors and
users extend the dialect here without touching skverify internals.

Extension points
----------------
``register_ufunc``
    An elementwise function gains an exact sympy form.
``register_function``
    A numpy-level function gains a formula constructor.
``register_opaque``
    A compiled callable (matched by its resolved ``__name__``) is
    sealed into a named atom instead of being entered.
``register_neutral``
    A call is declared math-neutral: traced operands pass through
    with formulas intact.
``register_contract``
    An opaque atom gains checks: requirements on its inputs and a
    residual law its output must satisfy, verified per call.

Every registration is a claim. The house rule applies: names propose,
runs dispose. Ship a differential test with any row you add (see
``tests/test_special_map.py`` for the pattern).
"""

from . import registry
from .contracts import CONTRACTS


def register_ufunc(np_ufunc, sympy_form):
    """Map an elementwise ufunc to its exact sympy form.

    Parameters
    ----------
    np_ufunc : numpy.ufunc
        The numpy-level ufunc object (``np.exp``, ``scipy.special.erf``).
    sympy_form : callable
        Receives the operands' formulas, returns the sympy expression.
        For same-arity sympy functions, pass the function itself.

    Examples
    --------
    >>> register_ufunc(scipy.special.ndtr,
    ...                lambda z: (1 + sympy.erf(z / sympy.sqrt(2))) / 2)
    """
    registry.UFUNC_TABLE[np_ufunc] = sympy_form


def register_function(np_function, constructor):
    """Map a numpy array function to a formula constructor.

    Parameters
    ----------
    np_function : callable
        The dispatching numpy function (``np.sum``, ``np.average``).
    constructor : callable
        Receives the original arguments (Pairs included) and returns
        the traced result. This is the tier for functions whose
        meaning is a named mathematical form (a ``Sum``, a
        contraction) rather than an elementwise map.
    """
    registry.FUNCTION_TABLE[np_function] = constructor


def register_opaque(name):
    """Seal a compiled callable into a named atom wherever it is called.

    Parameters
    ----------
    name : str
        The callable's resolved ``__name__``. Matching by resolved
        name means aliases and attribute lookups cannot dodge the
        boundary.
    """
    from .instrument import OPAQUE_CALLABLES

    OPAQUE_CALLABLES.add(name)


def register_neutral(name):
    """Declare a call math-neutral: traced operands pass through.

    Parameters
    ----------
    name : str
        The call-site name (``asarray`` and family). Reserve this for
        functions that are identity on valid input by construction;
        a human reads the source once, the registry remembers.
    """
    from .instrument import NEUTRAL

    NEUTRAL.add(name)


def register_contract(name, requires=(), law="", residual=None):
    """Attach checks to an opaque atom.

    Parameters
    ----------
    name : str
        The atom's callable name as it appears in ``.unchecked``.
    requires : sequence of (str, callable), optional
        Named input checks run on concrete values at trace time.
    law : str, optional
        Human-readable statement of what the output satisfies.
    residual : callable, optional
        ``residual(args, result) -> 'ok' | 'failed'``; the per-call
        verification that earns the atom its label.
    """
    CONTRACTS[name] = {
        "requires": list(requires),
        "law": law,
        "residual": residual,
    }
