"""Band storage through sympy.banded: the layout is a declaration."""

import numpy as np
from scipy.linalg import solve_banded, solveh_banded

from skverify import to_sympy
from skverify.contracts import _ab_to_dense, check_call


def test_ab_round_trip():
    # random banded matrix -> ab storage -> back through sympy.banded
    rng = np.random.default_rng(3)
    n, nl, nu = 6, 2, 1
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - nl), min(n, i + nu + 1)):
            a[i, j] = rng.standard_normal()
    ab = np.zeros((nl + nu + 1, n))
    for i in range(n):
        for j in range(max(0, i - nl), min(n, i + nu + 1)):
            ab[nu + i - j, j] = a[i, j]
    assert np.allclose(_ab_to_dense(ab, nu), a)


def test_solve_banded_contract_ok():
    ab = np.array([[0.0, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, 0.0]])
    b = np.arange(4.0) + 1
    x = solve_banded((1, 1), ab, b)
    name, verdicts = check_call("solve_banded", ((1, 1), ab, b), x)
    assert dict(verdicts)["residual"] == "ok"


def test_solveh_banded_contract_ok():
    ab = np.array([[0.0, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0]])
    b = np.arange(4.0) + 1
    x = solveh_banded(ab, b)
    name, verdicts = check_call("solveh_banded", (ab, b), x)
    assert dict(verdicts)["residual"] == "ok"


def test_solve_banded_contract_catches_wrong_answer():
    ab = np.array([[0.0, 1.0, 1.0, 1.0], [4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, 0.0]])
    b = np.arange(4.0) + 1
    name, verdicts = check_call("solve_banded", ((1, 1), ab, b), b * 9.0)
    assert dict(verdicts)["residual"] == "failed"


def test_gbsv_verdict_in_interp_gate():
    # gate 2's banded solve atom now carries a verified residual
    from scipy.interpolate import make_interp_spline

    from skverify.pair import _OPAQUE

    spl = to_sympy(_drive, np.linspace(0, 30), np.sin(np.linspace(0, 30)))
    records = {e[0]: dict(e[1]) for e in _OPAQUE if isinstance(e[1], tuple)}
    gbsv = [k for k in records if "gbsv" in k]
    assert gbsv and records[gbsv[0]].get("residual") == "ok"


def _drive(x, y):
    from scipy.interpolate import make_interp_spline

    return make_interp_spline(x, y)
