# Sharp bits

scikit-verify traces ordinary NumPy code and either returns the exact
formula or refuses with one sentence. It never guesses. These are the
few places a refusal can happen, and what to do instead.

## 1. Discretization changes the math

Rounding is `floor(x*10^d + 1/2)/10^d` -- exact everywhere except
half-way ties, where numpy rounds half to EVEN. `round()` and
`np.round()` therefore lift with recorded tie-free assumptions, and
refuse only when a value actually sits on a tie:

```python
np.round(x, 1).sum()          # lifts; assumes Ne(Mod(10*x[k] + 1/2, 1), 0)
np.round(np.array([0.5]))
# NotImplementedError: rounding at an exact half-way tie ...
```

Integer casts of non-integral values still refuse: truncation has no
exact small formula and no tie-style guard rescues it.

## 2. Strings are not mathematics

Formatting a traced value into a string (`f"{x:.2f}"`,
`float(str(x))`) discards the formula by construction.

*Instead:* format `x.value` for display; the formula stays on `x`.

## 3. Writes into numeric buffers

Assigning traced values into a preallocated float array
(`out=`, `np.empty(...); buf[i] = traced`) would silently drop
formulas, so it refuses. Writes into object arrays, `np.zeros`-style
allocations made inside traced code, and ordinary assignment all work
-- the tracer rewrites them to keep both lanes.

*Instead:* build results functionally, or let the traced allocation
handle it (it usually does without any change to your code).

## 4. Loops fold when the body repeats

A loop whose body does the same mathematics each iteration folds into
a held recurrence -- constant-size, any iteration count. A body that
changes shape every iteration cannot fold; it stays exact but
unrolled, and past a growth budget it refuses rather than hang.

*Instead:* nothing, usually. Convergence branches flipping once are
fine (the fold resumes). Only genuinely shape-shifting loops refuse.

## 5. Data used as an address goes concrete

Indices, dictionary keys, bin positions, sort orders: when a traced
value chooses WHERE to read or write, the position it chose is
recorded as a fact about this run (an assumption in
`.preconditions`, or a disclosure in `.unchecked`), and the selection
happens concretely. The selected values keep their formulas.

*This is by design:* the certificate is path-scoped, like the branch
you actually took in an `if`.

## Compiled code becomes named atoms

LAPACK, Cython, and f2py routines cannot be entered. Each call is
sealed as a named term (`svd_S`, `solve_banded_0`) with its defining
call recorded and, where a contract exists, checked against its own
mathematics on this very call (`A @ x == b`, reconstruction, ...).
This is not a refusal -- it is the honest boundary, disclosed in
`.unchecked`.

## Everything else

Branches on data become recorded assumptions. Masks, gathers,
scatters, sets, dicts, classes, closures, generators, recursion,
broadcasting, empty arrays, pickling -- traced. If you find ordinary
NumPy code that dies with anything other than a one-sentence refusal,
that is a bug in scikit-verify: please report it.
