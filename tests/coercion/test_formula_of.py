import numpy as np
import sympy
from skverify.coercion import formula_of

def test_formula_of_discloses_plain_array_as_named_constant_table():
    plain_array = np.array([1.0, 2.0, 3.0])  # not a Pair, no formula attached
    result = formula_of(plain_array)

    # it should NOT silently become a made-up formula —
    # it becomes an indexed symbol we can trace back to a disclosed table
    assert isinstance(result, sympy.Indexed)