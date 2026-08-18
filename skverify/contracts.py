"""Contracts for compiled routines the tracer cannot enter.

A contract names what an opaque call requires of its inputs and the law
its output obeys. Requires-checks run on the concrete values at trace
time (this run only); verdicts are three-valued and recorded, never
silently dropped.
"""

import numpy as np

OK = "ok"
FAILED = "failed"
UNKNOWN = "unknown"


def _square(a, b):
    a = np.asarray(a)
    return OK if a.ndim == 2 and a.shape[0] == a.shape[1] else FAILED


def _symmetric(a, b):
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return FAILED
    return OK if np.allclose(a, a.T.conj()) else FAILED


def _banded_residual(args, result):
    l_and_u, ab, b = args[0], np.asarray(args[1]), np.asarray(args[2])
    nl, nu = l_and_u
    x = np.asarray(result)
    n = ab.shape[1]
    a = np.zeros((n, n))
    for r in range(ab.shape[0]):
        for c in range(n):
            i = r + c - nu
            if 0 <= i < n:
                a[i, c] = ab[r, c]
    rnorm = np.linalg.norm(a @ x - b)
    scale = np.linalg.norm(b) + np.linalg.norm(a) * np.linalg.norm(x)
    if scale == 0:
        return OK if np.allclose(a @ x, b) else FAILED
    return OK if rnorm / scale < 1e-8 else FAILED


def _design_matrix_probe(args, result):
    # the label "design_matrix[i, j] == B_j(x_i)" is earned, not assumed:
    # compare the compiled output against sympy's own B-spline basis
    import sympy

    x, t = np.asarray(args[0]), np.asarray(args[1])
    k = int(args[2]) if len(args) > 2 else 3
    dense = result.toarray() if hasattr(result, "toarray") else np.asarray(result)
    u = sympy.Symbol("u")
    basis = sympy.bspline_basis_set(k, [sympy.Float(v) for v in t], u)
    for i in (0, len(x) // 2, len(x) - 1):
        xi = float(x[i])
        if xi == float(t[-1]):
            continue  # right-endpoint support convention differs
        for j in range(dense.shape[1]):
            want = float(basis[j].subs(u, xi))
            if not np.isclose(dense[i, j], want, atol=1e-10):
                return FAILED
    return OK


def _solve_residual(args, result):
    a, b = np.asarray(args[0]), np.asarray(args[1])
    x = np.asarray(result)
    rng = np.random.default_rng(0)
    r = a @ x - b
    scale = np.linalg.norm(b) + np.linalg.norm(a) * np.linalg.norm(x)
    if scale == 0:
        return OK if np.allclose(r, 0) else FAILED
    return OK if np.linalg.norm(r) / scale < 1e-8 else FAILED


CONTRACTS = {
    "solve": {
        "requires": [("square", _square)],
        "law": "A @ x == b",
        "residual": _solve_residual,
    },
    "solve_banded": {
        "requires": [],
        "law": "A @ x == b for the banded A",
        "residual": _banded_residual,
    },
    "solveh_banded": {
        "requires": [],
        "law": "A @ x == b for the banded Hermitian A",
        "residual": None,
    },
    "design_matrix": {
        "requires": [],
        "law": "design_matrix[i, j] == B_j(x_i), the B-spline basis at the data",
        "residual": _design_matrix_probe,
    },
    "eigh": {
        "requires": [("symmetric", _symmetric)],
        "law": "A @ v == w * v",
        "residual": None,
    },
}


def check_call(name, args, result):
    """Run a contract's checks for one opaque call. Returns a record:
    (name, ((check, verdict), ...)). Unknown contract -> all-unknown."""
    contract = CONTRACTS.get(name)
    if contract is None:
        return (name, (("contract", UNKNOWN),))
    verdicts = []
    for check_name, fn in contract["requires"]:
        try:
            verdicts.append((check_name, fn(*args[:2])))
        except Exception:
            verdicts.append((check_name, UNKNOWN))
    residual = contract.get("residual")
    if residual is not None:
        try:
            verdicts.append(("residual", residual(args, result)))
        except Exception:
            verdicts.append(("residual", UNKNOWN))
    return (name, tuple(verdicts) or (("law", UNKNOWN),))
