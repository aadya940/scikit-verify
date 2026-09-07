<p align="center">
  <img src="doc/logos/scikit-verify-lockup.svg" alt="scikit-verify" width="380">
</p>

<p align="center">Translate Python and NumPy programs to symbolic mathematics</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="doc/hero-dark.svg">
    <img src="doc/hero-light.svg" alt="weighted_rms in numpy maps to the square root of the ratio of weighted sums" width="640">
  </picture>
</p>

![CI](https://github.com/aadya940/scikit-verify/actions/workflows/ci.yml/badge.svg)

* [Source code](https://github.com/aadya940/scikit-verify)
* [Coverage](doc/coverage.md)
* [License](https://github.com/aadya940/scikit-verify/blob/master/LICENSE)
* [skverify-mcp](skverify-mcp/) - MCP for mathematical feedback for coding agents
* [Blog post](https://medium.com/@aadyachinubhai/scikit-verify-translate-python-numpy-programs-to-symbolic-mathematics-c664d41ba571) - the story and the mathematics, with examples
* [Demo](examples/penalty_matrix_check.ipynb) - a 20-page SciPy derivation, checked

scikit-verify is a tracer for numerical Python. It runs your NumPy
function once and returns the formula it computed, as an ordinary SymPy
expression you can read, simplify, compare against a paper, or evaluate
at any other input. Your code is not modified or annotated. For example:

```python
import numpy as np
from skverify import to_sympy

def weighted_rms(x, w):
    return np.sqrt(np.sum(w * x**2) / np.sum(w))

out = to_sympy(weighted_rms, np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.3, 0.2]))

out.formula
# sqrt(Sum(w[j]*x[j]**2, (j, 0, 2))/Sum(w[j], (j, 0, 2)))
```

Every formula comes as a certificate: the expression, plus the
assumptions it was derived under. When code branches on your data, the
branch taken becomes a stated hypothesis instead of a hidden one:

```python
out = to_sympy(np.median, np.array([3.0, 1.0, 4.0, 1.5]))
print(out.pretty())

# formula    = a[0]/2 + a[3]/2
# assumes[0] = a[0] <= a[2]
# assumes[1] = a[1] <= a[3]
# assumes[2] = a[3] <= a[0]
```

The contract raises an exception when it can't work something out. If an operation has no faithful
symbolic form, scikit-verify raises instead of guessing:

```python
to_sympy(lambda a: a.astype(int).mean(), np.array([1.4, 2.6]))
# NotImplementedError: astype to non-float would change the math
```

Tested against numpy, scipy, scikit-learn, statsmodels, cvxpy and
random research code from GitHub; the boards in [coverage](coverage/)
regenerate every number.

You can also state the formula you believe and let the trace check it,
as an ordinary pytest test:

```python
import sympy
from scipy.integrate import simpson
from skverify.testing import specifies

y = sympy.IndexedBase("y")

@specifies((y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4]) / 3)
def test_simpson_is_the_textbook_rule():
    return (lambda v: simpson(v)), (np.array([0.7, 1.2, 2.5, 0.3, 0.4]),)
```

The comparison is symbolic, a passing test means the code
computes that formula for every input of that shape, and a failing one
prints both formulas with a concrete counterexample. Specs come from
the paper or the docstring, never from the trace itself. 

One trace follows one path, so that claim holds for the branch your
input took. The `explore=True` flag visits the rest: every branch
condition is negated and handed to the Z3 solver, which either
produces an input for the other side or proves no such input exists.
A spec that is right on your branch and wrong on another now fails,
with the guilty branch printed:

```python
def f(v):
    if v.sum() > 0:
        return v * 2.0
    return v * 3.0

check_formula(f, (np.array([1.0, 2.0]),), 2 * v[i], indices=(i,), explore=True)
# verdict: differs
#   your spec:  2*v[0]
#   the code:   3.0*v[0]
#   on the path where: Sum(v[j], (j, 0, 1)) <= 0
```

And a passing one upgrades from "on the traced path" to a proof:
measured over every public numpy function the tracer lifts, 274 of
293 get full branch coverage proven, with every unvisited region
either refuted by the solver or named honestly.

In a nutshell, correctness of numerical programs is two questions:
1. Is the math itself correct?
2. Is the code numerically stable?

scikit-verify answers the first question!

## Installation

```bash
pip install scikit-verify
```

Requires Python >= 3.11, `numpy`, `sympy`, and `z3-solver` (a plain
pip wheel, nothing to install system-wide). The import name is
`skverify`. The companion layers install as extras:

```bash
pip install "scikit-verify[mcp]"          # MCP server for coding agents
```

Pre-alpha; the API may change. Iterative solvers at real sizes can be
slow to trace (minutes, not wrong); the boards in coverage/ carry
timings.

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

If this is useful to you, a star helps others find it ⭐
