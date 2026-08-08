# scikit-verify

<img src="doc/logos/scikit-verify-lockup.svg" alt="scikit-verify" width="500" align="right">

scikit-verify recovers the symbolic mathematics implemented by NumPy code.
It executes a function with tracing values: the numerical computation runs
normally, and each array operation additionally constructs the corresponding
SymPy expression. The recovered expressions can then be checked, against a
reference, against properties such as stability or conservation, or against
the numerical execution itself.

## Example

```python
import numpy as np
import sympy
from skverify import Pair, IDX

def upwind(u, c, dt, dx):
    return u[1:] - (c * dt / dx) * (u[1:] - u[:-1])

u = Pair.array("u", np.linspace(0, 1, 16))
c, dt, dx = (Pair(v, sympy.Symbol(s, positive=True))
             for v, s in [(0.9, "c"), (0.01, "dt"), (0.1, "dx")])

out = upwind(u, c, dt, dx)

out.formula
# u[i + 1] - c*dt*(u[i + 1] - u[i])/dx
out.value
# array([...])  , the ordinary numerical result
out.domain
# (0, 15)
```

The function is not modified, parsed, or recompiled; it runs under CPython
and dispatch is intercepted through the standard protocols
(`__array_ufunc__`, `__array_function__`, operator overloading).

## Supported

Current support covers 1-D vectorized NumPy code:

- arithmetic operators and step-1 slicing, with index-domain tracking and
  slice-alignment checking
- elementwise ufuncs (`np.sin`, `np.exp`, `np.maximum`, ...)
- `np.where` (lifted to `Piecewise`), `np.sum` (lifted to `Sum`),
  `np.zeros_like` / `np.ones_like` / `np.full_like`
- uniform constant arrays as operands

Unsupported operations raise `NotImplementedError`. Formulas are only
produced for operations whose semantics are implemented; there is no
best-effort fallback.

Planned, in order: test suite, N-dimensional arrays, in-place assignment,
comparison and branch capture, and contract-based handling of compiled
routines (`scipy.linalg`, `scipy.sparse`).

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
