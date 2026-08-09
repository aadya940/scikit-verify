<p align="center">
  <img src="doc/logos/scikit-verify-lockup.svg" alt="scikit-verify" width="380">
</p>

<p align="center">Convert the Python + NumPy code to its symbolic mathematics </p>


![CI](https://github.com/aadya940/scikit-verify/actions/workflows/ci.yml/badge.svg)

<b> scikit-verify </b> recovers the symbolic mathematics implemented by Python and NumPy code.
It executes a function with tracing values: the numerical computation runs
normally, and each array operation additionally constructs the corresponding
SymPy expression. The recovered expressions can then be checked, against a
reference, against properties such as stability or conservation, or against
the numerical execution itself.

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

Symbol names are taken from the function's own signature. The function is
not modified, parsed, or recompiled; it runs under CPython and dispatch is
intercepted through the standard protocols (`__array_ufunc__`,
`__array_function__`, operator overloading).

A 2-D stencil, with broadcasting and strides:

```python
from skverify import Pair

u = Pair.array("u", np.random.rand(4, 7))
v = Pair.array("v", np.random.rand(7))

(u[1:, :] - u[:-1, :]).formula   # u[i + 1, j] - u[i, j]
(u + v).formula                  # u[i, j] + v[j]      (v aligns to the last axis)
u[::2, ::-1].formula             # u[2*i, 6 - j]
u[2].formula                     # u[2, j]
u[2, 3].formula                  # u[2, 3]             (a scalar; domain is None)
```

## Supported

Current support covers N-dimensional vectorized NumPy code (up to 5-D):

- arithmetic operators, with per-axis index-domain tracking and
  alignment checking
- slicing with any start/stop/step (including flips and strides),
  integer indexing, `...`, and their composition
- rank broadcasting (a lower-rank operand aligns to the trailing axes;
  extent-1 stretching is not yet supported)
- elementwise ufuncs (`np.sin`, `np.exp`, `np.maximum`, ...)
- `np.where` (lifted to `Piecewise`), `np.sum` (lifted to `Sum`),
  `np.zeros_like` / `np.ones_like` / `np.full_like`
- uniform constant arrays as operands

Unsupported operations raise `NotImplementedError`. Formulas are only
produced for operations whose semantics are implemented; there is no
best-effort fallback.

Planned, in order: reductions over a chosen axis and transposition,
in-place assignment, comparison and branch capture, and contract-based
handling of compiled routines (`scipy.linalg`, `scipy.sparse`).

## Installation

```bash
pip install scikit-verify
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
