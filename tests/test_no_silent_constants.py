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

CASES = {
    "gmean": lambda y: scipy_stats.gmean(y),
    "hmean": lambda y: scipy_stats.hmean(y),
    "sem": lambda y: scipy_stats.sem(y),
    "skew": lambda y: scipy_stats.skew(y),
    "kurtosis": lambda y: scipy_stats.kurtosis(y),
    "variation": lambda y: scipy_stats.variation(y),
    "moment": lambda y: scipy_stats.moment(y, order=2),
    "entropy": lambda y: scipy_stats.entropy(y),
    "trim_mean": lambda y: scipy_stats.trim_mean(y, 0.1),
    "gstd": lambda y: scipy_stats.gstd(y),
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
    except NotImplementedError:
        return  # loud refusal is honest
    if not isinstance(r, Pair):
        pytest.fail(f"{name}: non-Pair result")
    symbolic = _is_symbolic(r.formula)
    if name in KNOWN_SILENT:
        if symbolic:
            pytest.fail(
                f"{name} now carries its inputs -- remove it from KNOWN_SILENT"
            )
        return
    assert symbolic, f"{name}: SILENT CONSTANT {r.formula}"
