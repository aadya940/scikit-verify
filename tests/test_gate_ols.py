"""Gate 4: statsmodels OLS(y, X).fit().params lifts end to end.

The class mechanism carries a second library: the data-handling type
gate passes via the isinstance rewrite, constant detection runs on
formulas, pinv goes through svd atoms, and params carry the
estimator in indexed form."""

import numpy as np
import pytest

statsmodels = pytest.importorskip("statsmodels.api")

from skverify import Pair, to_sympy


def _fit(y, X):
    return statsmodels.OLS(y, X).fit().params


def test_ols_params_lift():
    y = np.array([1.0, 2.0, 2.9, 4.2])
    X = np.column_stack([np.ones(4), np.arange(4.0)])
    params = to_sympy(_fit, y, X)
    ref = statsmodels.OLS(y, X).fit().params
    vals = [getattr(e, "value", e) for e in np.atleast_1d(params)]
    assert np.allclose(np.asarray(vals, dtype=float), ref)
    p = np.atleast_1d(params)[1]
    assert isinstance(p, Pair)
    assert "svd" in str(p.formula)  # the pinv atoms carry the estimator
