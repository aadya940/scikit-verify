# Architecture

`design_philosophy.md` explains *why* scikit-verify exists and the
contract it keeps. This page explains *how*: what happens, subsystem
by subsystem, between calling `to_sympy(fn, *args)` and getting a
certificate back.

We'll follow one call through the whole pipeline, then look at each
subsystem in isolation.

```python
import numpy as np
from skverify import to_sympy

def weighted_rms(x, w):
    return np.sqrt(np.sum(w * x**2) / np.sum(w))

out = to_sympy(weighted_rms, np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.3, 0.2]))
out.formula
# sqrt(Sum(w[j]*x[j]**2, (j, 0, 2))/Sum(w[j], (j, 0, 2)))
```

Nothing about `weighted_rms` was written for scikit-verify. It's
ordinary NumPy. The formula fell out of *running* it.

## 1. The core object: `Pair`

Every array or float argument is wrapped into a `Pair`
(`skverify/pair.py`) before the function ever sees it:

```python
class Pair:
    def __init__(self, value, formula, ...):
        self.value = value      # the real ndarray/scalar -- what executes
        self.formula = formula  # the sympy Expr -- what it means
```

`x = Pair(np.array([1.0, 2.0, 3.0]), IndexedBase("x")[i])` is both a
NumPy array *and* a symbolic formula at once, kept in lockstep. Every
arithmetic operator, every NumPy ufunc, every piece of indexing is
overridden on `Pair` to update **both lanes together**: the `.value`
lane keeps executing exactly as it would without scikit-verify, and
the `.formula` lane builds the matching SymPy expression alongside
it.

This is why your code is never modified or annotated — `Pair` behaves
enough like an ndarray that ordinary NumPy code runs on it unchanged,
and the formula is a side effect of that run rather than a static
analysis of your source.

A `Pair` also remembers its parents (`._parents`), so every value has
a full provenance DAG back to the function's arguments. That
provenance is what subsystem 8 (derivation) turns into readable
step-by-step prose.

## 2. The dialect: mapping tables

`Pair` doesn't know what `np.sqrt` means. That knowledge lives in
lookup tables (`skverify/registry.py`, populated by
`skverify/maps/numpy.py` and `skverify/maps/special.py`):

```python
UFUNC_TABLE[np.sqrt] = sympy.sqrt
UFUNC_TABLE[np.hypot] = lambda a, b: sympy.sqrt(a**2 + b**2)
UFUNC_TABLE[np.cbrt] = lambda x: sympy.real_root(x, 3)
```

When a `Pair` sees a NumPy ufunc called on it (via NumPy's
`__array_ufunc__` dispatch protocol — no monkey-patching of NumPy
itself), it looks the ufunc up in this table and calls the matching
SymPy constructor on its own `.formula`. `scipy.special` functions
route the same way through `maps/special.py` (`erf`, `gamma`,
Bessel functions, etc.).

This table-driven design is deliberate, not incidental —
`CONTRIBUTING.md` states the rule directly: *"generic mechanisms, not
per-library tables… we do not add 'if the function is called X, the
formula is Y' entries"* for anything that can instead be derived from
NumPy's actual semantics (broadcasting, reduction, indexing). The
tables in `maps/` exist only for the leaf-level elementwise functions
where there's genuinely nothing to derive — `sin` just *is* `sympy.sin`.

`skverify/dialect.py` is the public extension surface for this layer:
`register_ufunc`, `register_function`, `register_opaque`,
`register_neutral`, `register_contract`. Library authors can teach
scikit-verify a new function without touching internals.

## 3. Crossing the traced/raw boundary

`skverify/coercion.py` is, by its own docstring, *"the library's
highest-risk surface"*. Every place a `Pair` meets a plain NumPy value
(or vice versa) goes through one of three named conversions:

- `value_of` — anything to the concrete numeric lane. Never fails.
- `formula_of` — anything to the symbolic lane. **Can refuse** — a raw
  operand with no provable formula raises rather than silently
  guessing.
- `numeric` — object-dtype arrays of plain numbers to real float
  arrays, needed before handing data to compiled code.

Centralizing this in one file is a correctness decision: a conversion
that silently drops a formula produces a wrong certificate, so every
crossing point is reviewable in one place instead of scattered
throughout the codebase.

## 4. Branches become preconditions

`np.median` sorts its input and picks specific elements, so its
formula genuinely depends on the input's order:

```python
out = to_sympy(np.median, np.array([3.0, 1.0, 4.0, 1.5]))
print(out.pretty())
# formula    = a[0]/2 + a[3]/2
# assumes[0] = a[0] <= a[2]
# assumes[1] = a[1] <= a[3]
# assumes[2] = a[3] <= a[0]
```

Every `if`/comparison the trace actually took gets appended to the
active `TraceSession`'s `.guards` list (`skverify/session.py`). At the
end of the trace, `to_sympy` (`skverify/api.py`) harvests these guards
into `.preconditions` on the result. The formula is exact — but only
*given* the branch this particular run took, and that scope is stated
rather than hidden.

## 5. Loops: unroll, then fold

The simplest loops need nothing special — `np.sum` already builds a
`sympy.Sum` directly. Hand-written accumulator loops fold the same
way once traced. The harder case is a loop whose body threads *state*
across iterations, like a Newton solver:

```python
def newton_sqrt(a, x0, iters=5):
    x = x0
    for _ in range(iters):
        x = 0.5 * (x + a / x)
    return x
```

Unrolled naively, `iters` iterations produce a formula whose size
grows with `iters` — fine for `iters=5`, unusable for a 300-iteration
`BayesianRidge` fit. `skverify/recurrence.py` exists to fold that
history into a **constant-size held object**, `Iterate(step, init, k)`
— sympy's own tools (`subs`, `.doit()`, `free_symbols`) traverse and
unroll it lazily instead of a live giant expression sitting in memory.

The capture technique (its own docstring calls this out as the subtle
part) is *planting*: before a loop body runs, every carried value's
formula becomes a fresh dummy symbol, so the body's own execution
writes the step template directly in terms of those dummies — this
sidesteps float arithmetic re-associating differently across runs,
which would break naive template-matching. Verification is **by
path**: an iteration only folds into the shared template if it took
the same branches and called the same opaque routines as the
iteration the template was built from; a body that changes shape
partway through stays honestly unrolled instead of folding into a
wrong template. `LOOP_DOMAINS.md` is the original design note behind
this subsystem and is worth reading for the sympy-level details
(`RecursiveSeq`, `MatPow`, and why neither alone was sufficient).

`derivation.py`'s `pretty()` view renders the same information as
readable named steps rather than one large nested expression:

```python
out = to_sympy(newton_sqrt, 2.0, 1.5, iters=5)
print(out.pretty())
# t0      = a/x0
# t1      = a/(0.5*t0 + 0.5*x0)
# t2      = a/(0.25*t0 + 0.5*t1 + 0.25*x0)
# t3      = a/(0.125*t0 + 0.25*t1 + 0.5*t2 + 0.125*x0)
# formula = 0.5*a/(0.0625*t0 + 0.125*t1 + 0.25*t2 + 0.5*t3 + 0.0625*x0) +
#           0.03125*t0 + 0.0625*t1 + 0.125*t2 + 0.25*t3 + 0.03125*x0
```

## 6. Compiled code becomes named atoms

Some code the tracer simply cannot enter: LAPACK routines behind
`np.linalg.solve`, FFT, anything implemented in C, Cython, or Fortran.
`skverify/atoms.py` handles this boundary by **sealing** the call into
a named term instead of tracing through it:

```python
def solve_it(A, b):
    return np.linalg.solve(A, b)

out = to_sympy(solve_it, A, b)
out.formula
# solve_0[i]
out.unchecked
# (('solve', (('square', 'ok'), ('residual', 'ok')),
#   ('solve_0[i]', 'solve(A[i, j], b[i])')),)
```

`solve_0[i]` is an honestly-named opaque result, not a guess at what
LAPACK's internals compute. `skverify/contracts.py` pairs each opaque
callable with **requirements** on its inputs (`_square`, `_symmetric`,
etc., checked on the concrete values at trace time) and a **residual
law** its output must satisfy — `solve` is checked against
`A @ x == b`, `svd` against `U @ diag(S) @ Vh == A` with
orthonormality, `lstsq` against the normal equations. Every check
reports one of three verdicts (`OK`, `FAILED`, `UNKNOWN`) and is
recorded in `.unchecked` rather than silently passing or silently
failing — this is the "the certificate states its own scope" promise
from the README made concrete.

## 7. Verification: three-valued, never silent

`skverify/checks.py` provides the `against()` function used to compare
a traced formula against a reference (a textbook formula, another
implementation):

```python
Evidence = namedtuple("Evidence", "verdict method detail")
# verdict is one of: "proven", "refuted", "unknown"
```

The same three-valued discipline as `contracts.py`: SymPy's inability
to *decide* an equality maps to `"unknown"`, never silently to
`"proven"`. This is the same instinct as exact-or-refuse applied to
verification instead of translation — no plausible-looking answer is
allowed to stand in for a