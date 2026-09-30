import numpy as np
from skverify import to_sympy, latex

def test_latex_returns_a_string():
    out = to_sympy(lambda x: np.sum(x), np.array([1.0, 2.0, 3.0]))
    result = latex(out.formula)
    assert isinstance(result, str)
    assert len(result) > 0