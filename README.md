<p align="center">
  <img src="doc/logos/scikit-verify-lockup.svg" alt="scikit-verify" width="380">
</p>

<p align="center">See the mathematics your code computes</p>

![CI](https://github.com/aadya940/scikit-verify/actions/workflows/ci.yml/badge.svg)

* [Source code](https://github.com/aadya940/scikit-verify)
* [License](https://github.com/aadya940/scikit-verify/blob/master/LICENSE)

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

The contract is exact-or-refuse. If an operation has no faithful
symbolic form, scikit-verify raises instead of guessing:

```python
to_sympy(lambda a: np.round(a).mean(), np.array([1.4, 2.6]))
# NotImplementedError: rounding a traced value changes the math
```

This works on real library code, not just kernels: scikit-learn metrics
come back as their defining formulas (precision as its ratio of counting
sums), fitted estimators as their closed forms, iterative solvers as
held recurrences, and compiled routines (LAPACK, Cython) as named terms
that are checked against their defining equations on every call.

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
