<p align="center">
  <img src="doc/logos/scikit-verify-lockup.svg" alt="scikit-verify" width="380">
</p>

<p align="center">See the mathematics your code computes</p>

![CI](https://github.com/aadya940/scikit-verify/actions/workflows/ci.yml/badge.svg)

**scikit-verify** runs your NumPy function once and returns the symbolic
formula it computed, as an ordinary SymPy expression. Your code is not
modified, annotated, or rewritten. It just runs, and the mathematics
falls out.

```python
import numpy as np
from skverify import to_sympy

def step(u, dt, h):
    lap = (u[2:] - 2 * u[1:-1] + u[:-2]) / h**2
    unew = u.copy()
    unew[1:-1] = u[1:-1] + dt * lap
    return unew

out = to_sympy(step, np.sin(np.linspace(0, np.pi, 9)), 0.01, 0.1)

out.formula
# Piecewise((dt*(u[i+1] + u[i-1] - 2*u[i])/h**2 + u[i], (i >= 1) & (i < 8)),
#           (u[i], True))
```

The stencil, the boundary handling, and the parameters, all visible.
`dt` and `h` stayed symbolic because you passed floats. The concrete
result is still there in `out.value`.

## Why

- **Check code against the paper.** Recover the equation your code
  implements and prove it equal to the one you published, symbolically:

  ```python
  from skverify.checks import against
  against(out, textbook_rule)
  # Evidence(verdict='proven', method='canonical', detail=0)
  ```

- **Find bugs by subtracting formulas.** Two implementations that
  disagree numerically tell you *that* something is wrong. Subtracting
  their formulas tells you *where*:

  ```python
  (mine.formula - reference.formula).doit()
  # -y[1]/24 - y[3]/24 - y[5]/24 - y[7]/24
  #  ^ only odd indices survive: the bug is in the odd branch,
  #    and the weight is off by exactly one
  ```

- **Review generated code.** LLM-written numerical kernels look right
  more often than they are right. Lift them and read the mathematics.

- **Use SymPy on running code.** Differentiate a traced loss to get the
  gradient you never wrote. Render formulas as LaTeX for your paper.
  Simplify, substitute, `lambdify`.

## It works on real libraries

```python
from scipy.interpolate import make_smoothing_spline

def smooth(x, y):
    return make_smoothing_spline(x, y, lam=0.1)

sm = to_sympy(smooth, x, y)

sm.preconditions
# (x[1] - x[0] > 0) & (x[2] - x[1] > 0) & ...
#  ^ scipy's own input validation, recorded as formulas

sm.unchecked
# [('design_matrix', {'residual': 'ok'}), ('solve_banded', {'residual': 'ok'})]
#  ^ the compiled routines inside, each checked on this very call:
#    the basis builder against sympy's own B-splines, the solver
#    against A @ x == b
```

Traced code from **SciPy** (splines, quadrature, statistics, signal),
**statsmodels** (regression, autocovariance, diagnostics) and
**scikit-learn** (metrics, kernels, SVM decision functions) lifts
today. The walkthrough notebook shows all of it end to end:
[`examples/demo.ipynb`](examples/demo.ipynb).

## The contract

Three promises, kept everywhere:

1. **Exact or absent.** Every formula is verified against the numerical
   execution. If a pattern cannot be proven, it is not emitted.
2. **Loud refusal over plausible guessing.** Anything that cannot be
   traced exactly raises `NotImplementedError` naming the blocker.
   There is no silent fallback.
3. **The certificate states its own scope.** Data-dependent branches
   become `.preconditions`. Compiled results become named terms listed
   in `.unchecked`, checked where a check exists and honestly marked
   `unknown` where none does yet.

## Installation

```bash
pip install git+https://github.com/aadya940/scikit-verify.git
```

Requires Python >= 3.11, `numpy`, and `sympy`. The import name is
`skverify`. Pre-alpha; the API may change.

## Lineage

The ideas here are old and good. Pairing a concrete execution with a
symbolic one is King's symbolic execution (CACM 1976), run in the
concolic style of Cadar and Sen. Checking a compiled routine's answer
against its defining equation, instead of trusting its name, is
Blum and Kannan's result checking (1989). Folding a long trace back
into its loop structure follows Larus's whole-program paths (PLDI
1999), with templates recovered by Plotkin's anti-unification (1970).
The stance that code verification means checking code against the
mathematics it claims to implement is Oberkampf and Roy's (2010).
Verified lifting of stencils to summaries was developed by Kamil et
al. (PLDI 2016) for performance; scikit-verify lifts for correctness.
Converting NumPy to SymPy was wished for in
[sympy#2810](https://github.com/sympy/sympy/issues/2810) (2014).

## License

BSD-3-Clause. scikit-verify is an independent project and is not affiliated
with the SciPy developers.
