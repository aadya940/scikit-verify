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
    "solveh_banded": {
        "requires": [],
        "law": "A @ x == b for the banded Hermitian A",
        "residual": None,
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
