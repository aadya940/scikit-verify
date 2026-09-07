"""Exploration board: explore() over every function we lift.

For each function: how many paths exist at this input shape, were
they all visited, and what stopped us when they were not (cap,
undecided region, per-path refusal). This is the covers() story
measured against real library code instead of toy branches.
"""

import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.setrecursionlimit(20000)

import numpy as np
import sympy

from skverify.explore import explore

sys.path.insert(0, "coverage")
from spec_board import BOARD  # the 33 lifted functions with real args

import scipy.stats as st

EXTRA = [
    ("numpy.median", lambda v: np.median(v), (np.array([0.7, -1.2, 2.5]),)),
    ("numpy.sort[1]", lambda v: np.sort(v)[1], (np.array([0.7, -1.2, 2.5]),)),
    ("numpy.clip", lambda v: np.clip(v, 0.0, 1.0).sum(),
     (np.array([0.7, -1.2, 2.5]),)),
    ("numpy.maximum.pair", lambda u, w: np.maximum(u, w),
     (np.array([1.0, 5.0]), np.array([2.0, 3.0]))),
    ("stats.trim_mean", lambda v: st.trim_mean(v, 0.34),
     (np.array([0.7, -1.2, 2.5]),)),
]

def main():
    tally = Counter()
    rows = []
    for name, fn, args, *_ in [(n, f, a) for n, f, a, *r in BOARD] + EXTRA:
        t0 = time.time()
        try:
            r = explore(fn, tuple(
                a.copy() if isinstance(a, np.ndarray) else a for a in args
            ), max_paths=32)
            dt = time.time() - t0
            if r.complete:
                kind = "complete"
            elif r.capped:
                kind = "capped"
            elif r.undecided:
                kind = "undecided"
            elif r.refusals:
                kind = "path-refused"
            else:
                kind = "empty"
            rows.append((name, kind, len(r.paths), len(r.infeasible),
                         len(r.undecided), dt))
        except Exception as e:
            kind = f"BOARD-ERROR"
            rows.append((name, kind, 0, 0, 0, time.time() - t0))
            print(f"  {name:34s} ERROR {type(e).__name__}: {str(e)[:60]}")
            tally[kind] += 1
            continue
        tally[kind] += 1
        print(f"  {name:34s} {kind:12s} paths={len(r.paths):3d} "
              f"refuted={len(r.infeasible):3d} undecided={len(r.undecided):2d}"
              f"  {dt:6.1f}s")
    total = len(rows)
    print(f"\n== {total} lifted functions explored ==")
    for k, n in tally.most_common():
        print(f"  {k:14s} {n:3d}  ({100*n/total:.0f}%)")

main()
