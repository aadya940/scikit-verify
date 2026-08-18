"""Smoke test for scikit-verify. Run: python smoke.py"""
import numpy as np
import sympy
from skverify import Pair, IDX

U = sympy.IndexedBase("u")
c_, dt_, dx_ = (sympy.Symbol(s, positive=True) for s in ("c", "dt", "dx"))


def ok(msg):
    print(f"  \u2713 {msg}")


def expect_raise(fn, exc, msg):
    try:
        fn()
    except exc:
        ok(msg)
    else:
        raise SystemExit(f"FAILED: {msg} (no {exc.__name__} raised)")


#  scalars 
print("── scalars ──")
x = Pair(3.0, sympy.Symbol("x"))
y = Pair(4.0, sympy.Symbol("y"))

r = 2 - x / 5 + y * x
assert abs(float(r.formula.subs({"x": 3.0, "y": 4.0})) - r.value) < 1e-12
ok("differential (formula == execution)")

assert (2 - x).value == -1.0
ok("__rsub__ operand order")

assert (2 ** x).formula == sympy.sympify("2**x")
ok("__rpow__")

#  arrays & slices 
print("── arrays & slices ──")
u = Pair.array("u", np.linspace(0.1, 0.9, 16))

d = u[1:] - u[:-1]
assert sympy.simplify(d.formula - (U[IDX + 1] - U[IDX])) == 0
assert d.domain == (0, 15)
assert np.allclose(d.value, np.diff(u.value))
ok("stencil lift + domain + value lane")

expect_raise(lambda: u[1:] - u[:-2], ValueError, "misalignment guard")

#  kernel 
print("── kernel ──")

def upwind(u, c, dt, dx):
    return u[1:] - (c * dt / dx) * (u[1:] - u[:-1])

out = upwind(u, Pair(0.9, c_), Pair(0.01, dt_), Pair(0.1, dx_))
assert sympy.simplify(out.formula - (U[IDX + 1] - c_ * dt_ / dx_ * (U[IDX + 1] - U[IDX]))) == 0
ok(f"upwind lift: {out.formula}")

#  ufuncs 
print("── ufuncs ──")
assert np.sin(u).formula == sympy.sin(U[IDX])
assert np.allclose(np.sin(u).value, np.sin(u.value))
ok("sin (formula + value lane)")

assert np.arcsin(u).formula == sympy.asin(U[IDX])
ok("arcsin (renamed)")

assert np.maximum(u[1:], u[:-1]).formula == sympy.Max(U[IDX + 1], U[IDX])
ok("maximum (binary ufunc + domains)")

assert (2.0 * np.exp(u)).formula == 2.0 * sympy.exp(U[IDX])
ok("array priority + arithmetic routing")

expect_raise(lambda: np.frexp(u), NotImplementedError, "unmapped ufunc refuses loudly")

# ── array functions 
print("── array functions ──")
w = np.where(u, u, -u)
assert w.formula == sympy.Piecewise((U[IDX], sympy.Ne(U[IDX], 0)), (-U[IDX], True))
assert np.allclose(w.value, np.where(u.value, u.value, -u.value))
assert w.domain == (0, 16)
ok(f"where -> {w.formula}")

assert abs(float(np.median(u).value) - np.median(u.value)) < 1e-12
ok("median traces through its guarded sort")

# refusal guards
print("── refusal guards ──")
expect_raise(lambda: bool(u), NotImplementedError, "__bool__ refuses (no silent branching)")

print("\nALL GREEN — engine verified.")


s = np.sum(u[1:] - u[:-1])
print(s.formula)                    # Sum(u[j+1] - u[j], (j, 0, 14))
assert s.domain is None
assert abs(s.value - (u.value[-1] - u.value[0])) < 1e-12   # telescoping! math checks itself
z = np.zeros(16) + u
assert z.formula == U[IDX] and z.domain == (0, 16)

# ── reductions ───────────────────────────────────────────────────
print("── reductions ──")
s = np.sum(u[1:] - u[:-1])
assert s.domain is None
assert abs(s.value - (u.value[-1] - u.value[0])) < 1e-12     # telescoping identity
ok(f"sum -> {s.formula}")

# ── constant fields ──────────────────────────────────────────────
print("── constant fields ──")
z = np.zeros(16) + u
assert z.formula == U[IDX] and z.domain == (0, 16)
ok("zeros(16) + u == u[i] (uniform ndarray → constant)")

expect_raise(lambda: np.array([1.0, 2.0, 3.0]) + u[:3],
             NotImplementedError, "non-uniform raw ndarray refuses (wrap it)")

assert np.zeros_like(u).formula == sympy.Integer(0)
assert np.zeros_like(u).domain == (0, 16)
