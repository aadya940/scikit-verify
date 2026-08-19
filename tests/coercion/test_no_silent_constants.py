"""The forbidden category, patrolled: a traced stats function whose
formula comes out CONSTANT has silently dropped its inputs. The
list of known offenders may only shrink."""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy

scipy_stats = pytest.importorskip("scipy.stats")

# open silent-loss bugs, tracked by name; fixing one means removing
# it here. ADDING a name requires a documented diagnosis, not a wave.
KNOWN_SILENT = {"sem", "trim_mean"}
# context-dependent (flag 27): symbolic standalone, constant/crash under
# other import orders -- the twin caches are order-dependent. These are
# NOT stable results; the flag-27 unification is the fix.
ORDER_DEPENDENT = {"skew", "kurtosis", "moment", "entropy"}

# module-level defs, not lambdas: instrumentation needs real source
def _gmean(y):
    return scipy_stats.gmean(y)


def _hmean(y):
    return scipy_stats.hmean(y)


def _sem(y):
    return scipy_stats.sem(y)


def _skew(y):
    return scipy_stats.skew(y)


def _kurtosis(y):
    return scipy_stats.kurtosis(y)


def _variation(y):
    return scipy_stats.variation(y)


def _moment(y):
    return scipy_stats.moment(y, order=2)


def _entropy(y):
    return scipy_stats.entropy(y)


def _trim_mean(y):
    return scipy_stats.trim_mean(y, 0.1)


def _gstd(y):
    return scipy_stats.gstd(y)


CASES = {
    "gmean": _gmean,
    "hmean": _hmean,
    "sem": _sem,
    "skew": _skew,
    "kurtosis": _kurtosis,
    "variation": _variation,
    "moment": _moment,
    "entropy": _entropy,
    "trim_mean": _trim_mean,
    "gstd": _gstd,
}

Y = np.linspace(1.0, 2.0, 8) ** 2


def _is_symbolic(formula):
    return bool(formula.atoms(sympy.Indexed)) or any(
        isinstance(s, sympy.Symbol) for s in formula.free_symbols
    )


@pytest.mark.parametrize("name", sorted(CASES), ids=str)
def test_formula_carries_the_inputs(name):
    try:
        r = to_sympy(CASES[name], Y)
    except Exception:
        return  # ANY loud failure is honest; silence is the crime
    if not isinstance(r, Pair):
        pytest.fail(f"{name}: non-Pair result")
    symbolic = _is_symbolic(r.formula)
    if name in ORDER_DEPENDENT and not symbolic:
        pytest.xfail("order-dependent twin caches (flag 27)")
    if name in KNOWN_SILENT:
        if symbolic:
            pytest.fail(
                f"{name} now carries its inputs -- remove it from KNOWN_SILENT"
            )
        return
    assert symbolic, f"{name}: SILENT CONSTANT {r.formula}"
