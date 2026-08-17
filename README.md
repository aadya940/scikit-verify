<p align="center">
  <img src="doc/logos/scikit-verify-lockup.svg" alt="scikit-verify" width="380">
</p>

<p align="center">Convert Python + NumPy code to its symbolic mathematics</p>


![CI](https://github.com/aadya940/scikit-verify/actions/workflows/ci.yml/badge.svg)

<b> scikit-verify </b> recovers the symbolic mathematics implemented by Python and NumPy code.
It executes a function with tracing values: the numerical computation runs
normally, and each array operation additionally constructs the corresponding
SymPy expression. The recovered expressions can then be checked against a reference, against
the numerical execution itself, or handed to ordinary SymPy.

## Uses

- **Check code against the paper.** Recover the equation your code
  implements and compare it, term by term, with the one you published.
- **Catch silent math bugs.** Wrong-but-plausible numerics run fine and
  pass tests; the recovered formula makes the error visible.
- **Review generated code.** LLM-written numerical kernels look right
  more often than they are right; lift them and read the mathematics.
- **Understand inherited code.** See what a kernel actually computes
  without reverse-engineering it by hand.
- **Use SymPy on your code.** The recovered formula is an ordinary
  SymPy expression: differentiate it, simplify it, render it as LaTeX,
  substitute values.
- **Feed SciML frameworks.** Tools like NVIDIA PhysicsNeMo take SymPy
  equations as input; recover them from your existing NumPy code
  instead of retyping them.

## Example

```python
import numpy as np
from skverify import to_sympy

def upwind(u, c, dt, dx):
    return u[1:] - (c * dt / dx) * (u[1:] - u[:-1])

out = to_sympy(upwind, np.linspace(0, 1, 16), 0.9, 0.01, 0.1)

out.formula
# u[i + 1] - c*dt*(u[i + 1] - u[i])/dx
out.value
# array([...])  , the ordinary numerical result
out.domain
# (0, 15)
```

This works on real library code, unmodified. `scipy.integrate.trapezoid`,
called on a traced array, reveals itself:

```python
from scipy.integrate import trapezoid, simpson

y = np.linspace(0, 1, 8) ** 2
to_sympy(lambda y: trapezoid(y, dx=0.1), y).formula
# 0.05*y[0] + Sum(0.1*y[j + 1], (j, 0, 5)) + 0.05*y[7]
#   ^ the trapezoid rule: half-weight endpoints, uniform interior

to_sympy(lambda y: simpson(y, dx=0.125), np.linspace(0, 1, 9) ** 2).formula
# 0.0417*y[0] + Sum(0.1667*y[2*j + 1], ...) + Sum(0.0833*y[2*j + 2], ...) + 0.0417*y[8]
#   ^ composite Simpson: the alternating 4/3 and 2/3 weights, as two sums
```

A 2-D stencil, with broadcasting and strides:

```python
u = Pair.array("u", np.random.rand(4, 7))
v = Pair.array("v", np.random.rand(7))

(u[1:, :] - u[:-1, :]).formula   # u[i + 1, j] - u[i, j]
(u + v).formula                  # u[i, j] + v[j]      (v aligns to the last axis)
u[::2, ::-1].formula             # u[2*i, 6 - j]
u.T.formula                      # u[j, i]
u[2].formula                     # u[2, i]             (surviving axis is renamed)
u[2, 3].formula                  # u[2, 3]             (a scalar; domain is None)
np.where(u > 0, u, 0.0*u).formula   # Piecewise((u[i, j], u[i, j] > 0), (0, True))
np.sum(v > 0.5).formula          # Sum(Piecewise((1, v[j] > 0.5), (0, True)), (j, 0, 6))
```

Unmapped pure-Python NumPy functions are traced through their own
source: `np.diff(u, 2)` runs numpy's actual `diff` body on traced
values:

```python
to_sympy(np.diff, np.linspace(0, 9, 10), 2).formula
# a[i] - 2*a[i + 1] + a[i + 2]
```

`.formula` always takes one of three shapes, deterministically: an
indexed rule like the one above (array results whose pattern is
proven), a scalar expression, possibly containing `Sum` (reductions),
or a `sympy.Array` of per-element formulas when no general pattern can
be proven; the result is then exact for the given input shape. A
pattern is never guessed: general forms are emitted only when checking
them against every element succeeds.

## Supported

Current support covers N-dimensional vectorized NumPy code (up to 5-D):

- arithmetic operators, with per-axis index-domain tracking and
  alignment checking
- slicing with any start/stop/step (including flips and strides),
  integer indexing, `...`, and their composition
- rank broadcasting (a lower-rank operand aligns to the trailing axes;
  extent-1 stretching is not yet supported)
- transposition (`.T`, `np.transpose`, N-D axis permutations)
- comparisons and boolean masks: `u > 0` lifts to `u[i] > 0`, masks
  combine with `& | ~`, enter arithmetic as 0/1 (`(u > 0).sum()`
  counts), and reduce via `.all()` / `.any()`
- elementwise ufuncs (`np.sin`, `np.exp`, `np.maximum`, ...)
- `np.where` (lifted to `Piecewise`), `np.sum` (lifted to `Sum`,
  full reduction), `np.zeros_like` / `np.ones_like` / `np.full_like`
- uniform constant arrays as operands
- item assignment: `out[1:-1] = ...` boundary and stencil writes become
  `Piecewise` scatters; masked writes (`u[u > 0] = c`) included
- data-dependent branches: `if x > 0:` is decided by the real values and
  recorded, and the result carries `.preconditions`, the hypotheses under
  which its formula holds
- compiled routines become named opaque nodes (`solve(A[i, j], b[i])`)
  with per-call contract checks recorded in `.unchecked`; the trace
  continues around them
- unmapped pure-Python NumPy functions, traced through their own source
  (formulas come out per-element rather than indexed); real library code
  built on these lifts unmodified: `scipy.integrate.trapezoid`,
  `simpson`, `cumulative_trapezoid`, `np.diff`, `np.dot`

Every produced formula is exact, checked against the numerical
execution. Compiled routines appear as named opaque nodes and are
listed in `.unchecked` rather than silently absorbed. Everything else
that cannot be traced exactly raises `NotImplementedError` naming the
blocker: conversions that would discard the formula (`float()`,
dtype-forcing coercions) and operations whose semantics are not yet
implemented. There is no silent fallback.

Planned, in order: a broader contract library for compiled routines,
strided and view-aware assignment, and equation-level validators
(`skverify.checks`: `against`, `conserves_mass`, `centered` exist today).

## Installation

```bash
pip install git+https://github.com/aadya940/scikit-verify.git
```

Requires Python >= 3.11, `numpy`, and `sympy`. The import name is
`skverify`.

The project is pre-alpha and the API is subject to change.

## Related work

Conversion of NumPy functions to SymPy was proposed in
[sympy#2810](https://github.com/sympy/sympy/issues/2810) (2014). Verified
lifting of stencil computations to high-level summaries was developed by
Kamil et al. (PLDI 2016) for performance portability; scikit-verify applies
lifting to correctness checking instead.

## License

BSD-3-Clause. scikit-verify is an independent project and is not affiliated
with the SciPy developers.
