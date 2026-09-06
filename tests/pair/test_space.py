import numpy as np
import pytest
import sympy

from skverify import Pair
from skverify.helpers import axis_idx


def assert_formula_matches(pair, inputs):
    substitutions = {}
    for name, values in inputs.items():
        indexed = sympy.IndexedBase(name)
        for position in np.ndindex(values.shape):
            substitutions[indexed[position]] = float(values[position])
    axes = tuple(axis_idx(axis) for axis in range(pair.value.ndim))
    for position in np.ndindex(pair.value.shape):
        expression = pair.formula.subs(
            dict(zip(axes, position, strict=True)), simultaneous=True
        ).doit()
        actual = float(sympy.N(expression.xreplace(substitutions)))
        assert actual == pytest.approx(pair.value[position])


@pytest.mark.parametrize(
    ("function", "start", "stop"),
    [
        (np.linspace, np.array([1.0, 2.0]), np.array([3.0, 6.0])),
        (np.logspace, np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        (np.geomspace, np.array([1.0, 4.0]), np.array([16.0, 64.0])),
        (np.geomspace, np.array([-1.0, -4.0]), np.array([-16.0, -64.0])),
    ],
)
def test_space_functions_support_array_endpoints(function, start, stop):
    result = function(Pair.array("a", start), Pair.array("b", stop), num=4)

    assert isinstance(result, Pair)
    assert result.value.shape == (4, 2)
    assert_formula_matches(result, {"a": start, "b": stop})


@pytest.mark.parametrize("function", [np.linspace, np.logspace, np.geomspace])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_space_functions_broadcast_endpoints_and_place_sample_axis(function, axis):
    start = np.array([[1.0], [2.0]])
    stop = np.array([3.0, 5.0, 9.0])

    result = function(
        Pair.array("a", start),
        Pair.array("b", stop),
        num=3,
        endpoint=False,
        axis=axis,
    )

    expected_shape = list(np.broadcast_shapes(start.shape, stop.shape))
    expected_shape.insert(axis % 3, 3)
    assert result.value.shape == tuple(expected_shape)
    assert_formula_matches(result, {"a": start, "b": stop})


@pytest.mark.parametrize("reduction", [np.sum, np.prod])
def test_space_axis_does_not_capture_endpoint_reduction_dummy(reduction):
    source = np.array([[1.0, 2.0], [4.0, 8.0]])
    start = reduction(Pair.array("a", source), axis=1)

    result = np.linspace(start, start + 6, num=3)

    assert result.value.shape == (3, 2)
    assert_formula_matches(result, {"a": source})
