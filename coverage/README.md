# Coverage boards

These scripts run real library code through `to_sympy` and report what
happened. They are how the numbers in the paper and in
[doc/coverage.md](../doc/coverage.md) are produced, and they are meant
to be rerun. Nothing here is hand-curated. Every function that was tried
is counted, including the ones that refuse.

## Run them

With the dev install (`pip install -e .[dev]`) you have numpy and scipy,
which is enough for most boards. To run everything at once and
regenerate the page:

```bash
./coverage/run_boards.sh
```

Each board writes its scoreboard to `coverage/<name>.out`, and
`make_coverage.py` folds those into `doc/coverage.md`. A board whose
optional dependency is not installed is skipped, not failed. To run a
single board and read it directly:

```bash
python coverage/numpy_full.py
```

## What a scoreboard says

Every board prints three sections.

- **LIFT+match** is the functions whose traced formula, evaluated at the
  inputs, equalled the untraced library result exactly.
- **REFUSED** is the functions that returned a one-sentence refusal
  instead of a formula, each listed with its reason.
- **DIED** is crashes. This section must stay empty. A non-empty DIED is
  a bug, not a refusal.

A refusal is a designed outcome, not a failure. The rule is exact or
refuse, so a function that cannot be lifted exactly says so.

## The boards

| script | measures | needs |
|---|---|---|
| `numpy_dialect.py` | every public numpy callable | numpy |
| `numpy_full.py` | the common-use numpy surface | numpy |
| `scipy_full.py` | scipy interpolate, integrate, signal, linalg | scipy |
| `skl_full.py` | scikit-learn metrics, preprocessing, linear models | scikit-learn |
| `sm_full.py` | statsmodels | statsmodels |
| `cvxpy_full.py` | cvxpy expressions and small solved problems | cvxpy |
| `wild_100.py` | a fixed random sample of research code from GitHub | scipy, a fetched corpus |
| `spec_board.py` | docstring formulas held to their implementations | scipy |
| `explore_board.py` | `explore()` branch coverage over lifted functions | scipy |
| `explore_sweep.py` | `explore()` over the full lifted numpy surface | numpy |
| `make_coverage.py` | regenerates `doc/coverage.md` from the `.out` files | none |

The wild-code boards draw from a corpus fetched by a fixed rule, kept
under `/tmp/wild` so nothing foreign is committed. Without the corpus
those boards skip.

## Test-suite coverage

The line-coverage figure for the library itself (the 96% in the paper)
is separate from these boards and comes from the test suite:

```bash
coverage run -m pytest
coverage report
```

The source and report options live in `pyproject.toml`, so no flags are
needed.
