# Contributing

Thanks for being here. This page is short on purpose: the code has a
small number of load-bearing rules, and knowing them saves you time.

## Setup

```bash
git clone https://github.com/aadya940/scikit-verify.git
cd scikit-verify
pip install -e .[dev]
pytest -q
```

One warning from experience: if `import skverify` ever resolves to a
copied install instead of your checkout, everything will look
mysteriously stale. When in doubt:

```python
import skverify; print(skverify.__file__)   # must be your checkout
```

## The design in five lines

Every traced value is a Pair: the concrete value and a sympy
expression, kept in lockstep. NumPy calls are intercepted through the
dispatch protocol and mapped to symbolic form by a registry. Branches
taken on data are recorded as conditions on the result. Compiled calls
the trace cannot enter become named terms, checked against their
defining equation where one is known. When no exact form exists, we
refuse with one sentence.

## The rules that are not negotiable

1. **Exact or refuse.** Never return a formula that might be wrong.
   A refusal with a clear sentence is a feature; a plausible guess is
   the bug this project exists to prevent.
2. **Generic mechanisms, not per-library tables.** We support numpy
   semantics (allocation, reduction, indexing, masks, views). We do
   not add "if the function is called X, the formula is Y" entries
   for other libraries. Library code works because it is built from
   numpy pieces we handle.
3. **Values referee formulas.** Every change runs against the test
   suite and the coverage boards, which compare traced values with
   untraced runs. If your change makes a board's number move, the
   diff is part of your PR story.
4. **Every Pair owns its buffer.** In-place ops are supported, so a
   shared buffer between two Pairs is a silent-wrongness bug waiting
   to happen.

## Where things are

- `skverify/pair.py` - the two-lane value, indexing, in-place ops
- `skverify/maps/numpy.py` - the op-to-sympy registry
- `skverify/instrument/` - the tracer: call rewriting, class twins,
  dispatch policy
- `skverify/atoms.py` - compiled calls as named terms
- `skverify/contracts.py` - the defining-equation checks (solve
  against Ax = b, svd against U diag(S) Vh = A, ...)
- `skverify/dialect.py` - the extension API for new op meanings
- `coverage/` - the boards: rerunnable scripts measuring numpy,
  scipy, scikit-learn, statsmodels, cvxpy, and random GitHub code
- `skverify-mcp/` - the MCP server for coding agents

## Running the boards

```bash
python coverage/numpy_full.py
python coverage/skl_full.py       # needs scikit-learn
python coverage/wild_100.py       # needs the fetched corpus
python coverage/make_coverage.py  # regenerates doc/coverage.md
```

A board line reads: lifted-and-matched, refused, died. Died must stay
zero. Refusals are honest and listed by reason.

## Good first contributions

- A refused function whose reason names a missing numpy mechanism:
  add the mechanism, watch several boards move at once.
- New coverage boards for libraries we have not measured.
- **A torch backend.** Torch has `__torch_function__`, which works
  much like the numpy protocol we intercept. A Tensor subclass whose
  value lane stays a real tensor (GPU included) with the same sympy
  formula lane is the shape of it; elementwise + matmul +
  `torch.linalg.solve` would be a great first milestone. The hard
  parts we already know about: in-place ops and views, and autograd
  (trace forward-only at first). If you want to take this on, open an
  issue and we will map it out together.

## Testing

The test that matters is always the same one: trace the function,
evaluate the traced formula at the inputs, and compare against what
the untraced function returns. The equation checks itself against the
run that produced it. A typical test is four lines:

```python
def test_my_case():
    ref = fn(x)
    out = to_sympy(fn, x)
    assert np.allclose(float(out.value), ref)
```

For a new mechanism, add the case that used to fail, plus the edge
your mechanism claims to handle (the tie, the empty axis, the
in-place write). If the change touches branching, assert the
conditions too: the formula and its assumptions are one result.

To reproduce the coverage figure:

```bash
coverage run -m pytest
coverage report
```

The source and the report options are set in `pyproject.toml`, so no
flags are needed. Lines carrying `# pragma: no cover` are unreachable
defensive branches; each states why on the line above it.

## AI policy

Use AI for anything: code, tests, docs, this file. The one condition
is that you have manually reviewed every line you submit and can
explain it. You are the author; the AI was the typist. PRs that read
like nobody looked at them will be treated accordingly.

## Pull requests

Tests for the change, suite green, boards rerun if the tracer was
touched. Plain prose in comments and docs. If you are not sure whether
an idea fits rule 2, open an issue first and ask; that conversation is
cheap and welcome.
