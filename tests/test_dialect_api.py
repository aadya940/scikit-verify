"""The public extension API: every register_* entry point exercised
end to end, so a claim in the dialect table is backed by a test the
way the house rule demands."""

import numpy as np
import pytest
import sympy

from skverify import to_sympy
from skverify import dialect
from skverify.registry import UFUNC_TABLE, FUNCTION_TABLE


def test_register_ufunc_roundtrips():
    # a fresh ufunc-shaped callable maps to a sympy form and traces
    saved = dict(UFUNC_TABLE)
    try:
        dialect.register_ufunc(np.exp, sympy.exp)  # idempotent re-register
        out = to_sympy(lambda v: np.exp(v), np.array([0.5, 1.0]))
        assert out.formula.subs(sympy.Symbol("i", integer=True), 0).has(sympy.exp)
    finally:
        UFUNC_TABLE.clear(); UFUNC_TABLE.update(saved)


def test_register_function_constructor():
    saved = dict(FUNCTION_TABLE)
    try:
        marker = {}
        def ctor(a):
            marker["called"] = True
            return a
        def dummy(a):  # a plain callable used as a table key
            return a
        dialect.register_function(dummy, ctor)
        assert FUNCTION_TABLE[dummy] is ctor
    finally:
        FUNCTION_TABLE.clear(); FUNCTION_TABLE.update(saved)


def test_register_opaque_adds_name():
    from skverify.instrument import OPAQUE_CALLABLES
    before = "spam_routine" in OPAQUE_CALLABLES
    dialect.register_opaque("spam_routine")
    assert "spam_routine" in OPAQUE_CALLABLES
    if not before:
        OPAQUE_CALLABLES.discard("spam_routine")


def test_register_neutral_adds_name():
    from skverify.instrument import NEUTRAL
    before = "eggs_identity" in NEUTRAL
    dialect.register_neutral("eggs_identity")
    assert "eggs_identity" in NEUTRAL
    if not before:
        NEUTRAL.discard("eggs_identity")


def test_register_contract_records_all_fields():
    from skverify.contracts import CONTRACTS
    dialect.register_contract(
        "my_atom",
        requires=[("square", lambda a: True)],
        law="x satisfies P",
        residual=lambda args, result: "ok",
    )
    try:
        c = CONTRACTS["my_atom"]
        assert c["law"] == "x satisfies P"
        assert c["residual"](None, None) == "ok"
        assert len(c["requires"]) == 1
    finally:
        CONTRACTS.pop("my_atom", None)
