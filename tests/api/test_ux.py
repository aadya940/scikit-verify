"""User-facing pleasantness: inputs people actually have, rendering
people actually want."""

import numpy as np
import pytest
import sympy

from skverify import Pair, to_sympy


def _centered(y):
    return (y - y.mean()).sum()


def test_pandas_series_accepted():
    pd = pytest.importorskip("pandas")
    s = pd.Series([51.0, 58.0, 62.0], name="score")
    r = to_sympy(_centered, s)
    assert r.formula.has(sympy.IndexedBase("y"))
    assert np.isclose(float(r.value), 0.0)


def test_pandas_dataframe_accepted():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    r = to_sympy(lambda d: d.sum(), df)
    assert np.isclose(float(r.value), 10.0)


def test_jupyter_latex_repr():
    u = Pair.array("u", np.arange(3.0))
    text = (u * 2)._repr_latex_()
    assert text.startswith("$") and "u" in text


def test_derivation_drops_pure_reference_lines():
    u = Pair.array("u", np.arange(4.0))
    r = np.exp(u) * 2.0 + np.exp(u)
    for line in r.derivation().splitlines():
        after = line.split(": ", 1)[-1]
        assert not after.startswith("step[")  # `step 6: step[5]` noise
