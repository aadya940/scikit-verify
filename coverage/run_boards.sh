#!/usr/bin/env bash
# Run the coverage boards and regenerate doc/coverage.md.
#
# Each board prints a scoreboard to stdout, which is saved next to it as
# <name>.out. A board whose optional dependency is missing is skipped
# rather than failing the run, so the script works with only numpy and
# scipy installed (pip install -e .[dev]). Set PYTHON to choose an
# interpreter. Run from anywhere.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

boards=(
  numpy_dialect   # every public numpy callable
  numpy_full      # the common-use numpy surface
  scipy_full      # scipy interpolate, integrate, signal, linalg
  skl_full        # scikit-learn metrics, preprocessing, linear models
  sm_full         # statsmodels
  cvxpy_full      # cvxpy expressions and small solved problems
  wild_100        # a fixed random sample of research code from GitHub
)

for b in "${boards[@]}"; do
  echo "=== ${b} ==="
  if "$PY" "coverage/${b}.py" > "coverage/${b}.out" 2> "coverage/${b}.err"; then
    tail -n 4 "coverage/${b}.out"
    rm -f "coverage/${b}.err"
  else
    echo "skipped (missing dependency or corpus): $(tail -n 1 coverage/${b}.err)"
  fi
  echo
done

"$PY" coverage/make_coverage.py
echo "wrote doc/coverage.md"
