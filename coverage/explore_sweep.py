"""Exploration sweep over the FULL lifted numpy surface.

Reuses the dialect board's machinery: walk every public callable in
np, np.linalg and np.fft, synthesize a call that lifts, then run
explore() on it. Tally: coverage proven / undecided / capped /
refused / timed out. The honest denominator is "functions that lift
with the synthesized recipe" -- the same universe the dialect board
measures.
"""

import signal
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.setrecursionlimit(20000)

import numpy as np

# reuse the dialect harness without running its board loop
src = open("coverage/numpy_dialect.py").read()
head = src[: src.index("skipped = {}")]
ns = {"__name__": "dialect_head"}
exec(compile(head, "coverage/numpy_dialect.py", "exec"), ns)
public_callables = ns["public_callables"]
synthesize = ns["synthesize"]
OUT_OF_SCOPE = ns["OUT_OF_SCOPE"]
TO = ns["TO"]

from skverify import to_sympy
from skverify.explore import explore

PER_FN_SECONDS = 90


def main():
    out_of_scope = set().union(*OUT_OF_SCOPE.values())
    tally = Counter()
    slow, findings = [], []
    n_seen = 0
    for qual, name, f in public_callables():
        if name in out_of_scope:
            continue
        probe, args, _ref = synthesize(qual, qual, f)
        if probe is None:
            continue  # no calling recipe: same bucket the board skips
        n_seen += 1
        t0 = time.time()
        try:
            signal.alarm(PER_FN_SECONDS)
            to_sympy(probe, *[np.copy(a) if isinstance(a, np.ndarray) else a
                              for a in args])  # must lift at all
        except TO:
            tally["trace-timeout"] += 1
            signal.alarm(0)
            continue
        except Exception:
            tally["does-not-lift"] += 1
            signal.alarm(0)
            continue
        try:
            signal.alarm(PER_FN_SECONDS)
            r = explore(probe, tuple(args), max_paths=16, time_budget=45)
            dt = time.time() - t0
            if r.complete:
                kind = "complete"
            elif r.capped:
                kind = "capped"
            elif r.undecided:
                kind = "undecided"
                findings.append((qual, str(r.undecided[0])[:100]))
            elif r.refusals:
                kind = "path-refused"
            else:
                kind = "empty"
            tally[kind] += 1
            npaths = len(r.paths)
            if dt > 10 or kind != "complete" or npaths > 1:
                print(f"  {qual:34s} {kind:12s} paths={npaths:3d} {dt:5.1f}s",
                      flush=True)
            if dt > 30:
                slow.append((qual, dt))
        except TO:
            tally["explore-timeout"] += 1
            print(f"  {qual:34s} explore-timeout", flush=True)
        except Exception as e:
            tally["error"] += 1
            print(f"  {qual:34s} ERROR {type(e).__name__}: {str(e)[:60]}",
                  flush=True)
        finally:
            signal.alarm(0)
    lifted = sum(v for k, v in tally.items()
                 if k not in ("does-not-lift", "trace-timeout"))
    print(f"\n== {n_seen} callable recipes, {lifted} lift and were explored ==")
    for k, n in tally.most_common():
        print(f"  {k:16s} {n:4d}")
    if findings:
        print("\nundecided examples:")
        for q, u in findings[:8]:
            print(f"  {q}: {u}")


main()
