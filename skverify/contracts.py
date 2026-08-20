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


def _ab_to_dense(ab, nu):
    """Dense matrix from LAPACK band storage, THROUGH sympy.banded:
    row r of ab is diagonal nu - r. The layout is a declaration sympy
    executes, not loop code of our own to trust."""
    import sympy
    from sympy.matrices.sparsetools import banded

    ab = np.asarray(ab)
    n = ab.shape[1]
    diags = {}
    for r in range(ab.shape[0]):
        d = nu - r
        if d > 0:
            entries = [sympy.Float(v) for v in ab[r, d:]]
        elif d < 0:
            entries = [sympy.Float(v) for v in ab[r, : n + d]]
        else:
            entries = [sympy.Float(v) for v in ab[r]]
        if any(e != 0 for e in entries):
            diags[d] = entries
    dense = banded(n, n, diags) if diags else sympy.zeros(n, n)
    return np.array(dense.tolist(), dtype=float)


def _relative_residual(a, x, b):
    r = np.linalg.norm(a @ x - b)
    scale = np.linalg.norm(b) + np.linalg.norm(a) * np.linalg.norm(x)
    if scale == 0:
        return OK if np.allclose(a @ x, b) else FAILED
    return OK if r / scale < 1e-8 else FAILED


def _banded_residual(args, result):
    l_and_u, ab, b = args[0], np.asarray(args[1]), np.asarray(args[2])
    nl, nu = l_and_u
    a = _ab_to_dense(ab, nu)
    return _relative_residual(a, np.asarray(result), b)


def _hbanded_residual(args, result):
    # solveh_banded default: UPPER band storage, symmetric completion
    ab, b = np.asarray(args[0]), np.asarray(args[1])
    nu = ab.shape[0] - 1
    upper = _ab_to_dense(ab, nu)
    a = np.triu(upper) + np.triu(upper, 1).T
    return _relative_residual(a, np.asarray(result), b)


def _gbsv_residual(args, result):
    # dgbsv(kl, ku, ab, b): ab carries kl EXTRA fill rows on top; the
    # solution is output 2 of the (lu, piv, x, info) tuple
    kl, ku = int(args[0]), int(args[1])
    ab, b = np.asarray(args[2]), np.asarray(args[3])
    x = np.asarray(result[2] if isinstance(result, tuple) else result)
    a = _ab_to_dense(ab[kl:], ku)
    return _relative_residual(a, x, b)


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
        "residual": _hbanded_residual,
    },
    "gbsv": {
        "requires": [],
        "law": "A @ x == b for the banded A (LAPACK storage with fill rows)",
        "residual": _gbsv_residual,
    },
    "dgbsv": {
        "requires": [],
        "law": "A @ x == b for the banded A (LAPACK storage with fill rows)",
        "residual": _gbsv_residual,
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
    "inv": {
        "requires": [],
        "law": "A @ inv(A) == I",
        "residual": lambda args, out: (
            "ok"
            if np.allclose(np.asarray(args[0]) @ np.asarray(out), np.eye(len(out)))
            else "failed"
        ),
    },
    "eigvals": {
        "requires": [],
        "law": "det(A - w*I) == 0 for every eigenvalue w",
        "residual": lambda args, out: (
            "ok"
            if all(
                abs(np.linalg.det(np.asarray(args[0], dtype=complex) - w * np.eye(len(args[0]))))
                < 1e-6 * max(1.0, abs(np.linalg.det(np.asarray(args[0], dtype=complex))))
                for w in np.atleast_1d(out)
            )
            else "failed"
        ),
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
