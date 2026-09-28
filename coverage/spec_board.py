"""Spec board: docstring formulas held to their implementations.

Every spec below is transcribed from the function's OWN docstring (or
the textbook identity the docstring names), never from the trace.
The board measures the @specifies story end to end: how much shipped
numerical Python can be held to the mathematics it documents, and at
which verdict tier.

LOCAL ONLY until findings are adjudicated: a `differs` here is either
a transcription slip on our side or a real doc-vs-code divergence,
and only a human read decides which.
"""

import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.setrecursionlimit(20000)

import numpy as np
import sympy

from skverify.testing import check_formula

i = sympy.Symbol("i", integer=True)


def J():
    return sympy.Dummy("j", integer=True)


N = 5
V = sympy.IndexedBase("v")
W = sympy.IndexedBase("w")
U = sympy.IndexedBase("u")
POS = np.array([0.7, 1.2, 2.5, 0.3, 0.4])
MIX = np.array([0.7, -1.2, 2.5, 0.3, -0.4])
U3 = np.array([1.0, 2.0, 3.0])
W3 = np.array([2.0, 1.0, 5.0])
PROB = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

j = J()
MEAN = sympy.Sum(V[j], (j, 0, N - 1)) / N
j = J()
VAR = sympy.Sum((V[j] - MEAN) ** 2, (j, 0, N - 1)) / N
STD = sympy.sqrt(VAR)


def S(expr_fn):
    """Sum over a fresh dummy: S(lambda j: V[j]**2)."""
    jj = J()
    return sympy.Sum(expr_fn(jj), (jj, 0, N - 1))


def S3(expr_fn):
    jj = J()
    return sympy.Sum(expr_fn(jj), (jj, 0, 2))


BOARD = []


def entry(name, fn, args, spec, indices=(), assume=(), note=""):
    BOARD.append((name, fn, args, spec, indices, assume, note))


# numpy: docstring formulas ------------------------------------------
entry("numpy.average(w)", lambda v, w: np.average(v, weights=w),
      (POS, POS[::-1].copy()),
      S(lambda k: V[k] * W[k]) / S(lambda k: W[k]))
entry("numpy.var", lambda v: np.var(v), (MIX,), VAR)
entry("numpy.std", lambda v: np.std(v), (MIX,), STD)
entry("numpy.diff", lambda v: np.diff(v), (MIX,), V[i + 1] - V[i], (i,))
entry("numpy.trapezoid", lambda v: np.trapezoid(v), (MIX,),
      V[0] / 2 + V[1] + V[2] + V[3] + V[4] / 2)

# scipy.stats ---------------------------------------------------------
import scipy.stats as st

entry("stats.zscore", lambda v: st.zscore(v), (MIX,),
      (V[i] - MEAN) / STD, (i,))
entry("stats.gmean", lambda v: st.gmean(v), (POS,),
      sympy.exp(S(lambda k: sympy.log(V[k])) / N))
entry("stats.hmean", lambda v: st.hmean(v), (POS,),
      N / S(lambda k: 1 / V[k]),
      assume=[V[k] > 0 for k in range(N)])
entry("stats.pmean(p=3)", lambda v: st.pmean(v, 3), (POS,),
      (S(lambda k: V[k] ** 3) / N) ** sympy.Rational(1, 3),
      assume=[V[k] > 0 for k in range(N)])
entry("stats.sem", lambda v: st.sem(v), (MIX,),
      sympy.sqrt(S(lambda k: (V[k] - MEAN) ** 2) / (N - 1)) / sympy.sqrt(N))
entry("stats.variation", lambda v: st.variation(v), (POS,), STD / MEAN)
entry("stats.moment(2)", lambda v: st.moment(v, 2), (MIX,),
      S(lambda k: (V[k] - MEAN) ** 2) / N)
entry("stats.skew", lambda v: st.skew(v), (MIX,),
      (S(lambda k: (V[k] - MEAN) ** 3) / N) / VAR ** sympy.Rational(3, 2))
entry("stats.kurtosis", lambda v: st.kurtosis(v), (MIX,),
      (S(lambda k: (V[k] - MEAN) ** 4) / N) / VAR ** 2 - 3)
_tot = S(lambda k: V[k])
entry("stats.entropy", lambda v: st.entropy(v), (PROB,),
      -S(lambda k: (V[k] / _tot) * sympy.log(V[k] / _tot)),
      assume=[V[k] > 0 for k in range(N)],
      note="docstring: -sum(pk*log(pk)), pk normalized; positive domain")

# scipy.special -------------------------------------------------------
import scipy.special as sp

entry("special.softmax", lambda v: sp.softmax(v), (MIX,),
      sympy.exp(V[i] - V[2]) / S(lambda k: sympy.exp(V[k] - V[2])), (i,))
entry("special.logsumexp", lambda v: sp.logsumexp(v), (MIX,),
      sympy.log(S(lambda k: sympy.exp(V[k]))))
entry("special.log_softmax", lambda v: sp.log_softmax(v), (MIX,),
      V[i] - V[2] - sympy.log(S(lambda k: sympy.exp(V[k] - V[2]))), (i,))
entry("special.expit", lambda v: sp.expit(v), (MIX,),
      1 / (1 + sympy.exp(-V[i])), (i,))

# scipy.spatial.distance: docstrings all carry formulas ---------------
import scipy.spatial.distance as sd

entry("distance.euclidean", lambda u, w: sd.euclidean(u, w), (U3, W3),
      sympy.sqrt(S3(lambda k: (U[k] - W[k]) ** 2)))
entry("distance.sqeuclidean", lambda u, w: sd.sqeuclidean(u, w), (U3, W3),
      S3(lambda k: (U[k] - W[k]) ** 2))
entry("distance.cityblock", lambda u, w: sd.cityblock(u, w), (U3, W3),
      S3(lambda k: sympy.Abs(U[k] - W[k])))
entry("distance.chebyshev", lambda u, w: sd.chebyshev(u, w), (U3, W3),
      sympy.Max(*[sympy.Abs(U[k] - W[k]) for k in range(3)]))
entry("distance.minkowski(p=3)", lambda u, w: sd.minkowski(u, w, 3), (U3, W3),
      S3(lambda k: sympy.Abs(U[k] - W[k]) ** 3) ** sympy.Rational(1, 3))
entry("distance.cosine", lambda u, w: sd.cosine(u, w), (U3, W3),
      1 - S3(lambda k: U[k] * W[k])
      / (sympy.sqrt(S3(lambda k: U[k] ** 2)) * sympy.sqrt(S3(lambda k: W[k] ** 2))))
entry("distance.braycurtis", lambda u, w: sd.braycurtis(u, w), (U3, W3),
      S3(lambda k: sympy.Abs(U[k] - W[k])) / S3(lambda k: sympy.Abs(U[k] + W[k])))
entry("distance.canberra", lambda u, w: sd.canberra(u, w), (U3, W3),
      S3(lambda k: sympy.Abs(U[k] - W[k]) / (sympy.Abs(U[k]) + sympy.Abs(W[k]))))

# scipy.integrate / signal -------------------------------------------
import scipy.integrate as si
import scipy.signal as sg

entry("integrate.trapezoid", lambda v: si.trapezoid(v), (MIX,),
      V[0] / 2 + V[1] + V[2] + V[3] + V[4] / 2)
entry("integrate.simpson", lambda v: si.simpson(v), (MIX,),
      (V[0] + 4 * V[1] + 2 * V[2] + 4 * V[3] + V[4]) / 3)
entry("signal.detrend(const)", lambda v: sg.detrend(v, type="constant"),
      (MIX,), V[i] - MEAN, (i,))

# sklearn.metrics: docstring formulas --------------------------------
try:
    import sklearn.metrics as sm

    Y, P = sympy.IndexedBase("y_true"), sympy.IndexedBase("y_pred")
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.1, 1.9, 3.2, 3.7])

    def S4(expr_fn):
        jj = J()
        return sympy.Sum(expr_fn(jj), (jj, 0, 3))

    entry("metrics.mean_squared_error",
          lambda y_true, y_pred: sm.mean_squared_error(y_true, y_pred),
          (yt, yp), S4(lambda k: (Y[k] - P[k]) ** 2) / 4)
    entry("metrics.mean_absolute_error",
          lambda y_true, y_pred: sm.mean_absolute_error(y_true, y_pred),
          (yt, yp), S4(lambda k: sympy.Abs(Y[k] - P[k])) / 4)
    ym = S4(lambda k: Y[k]) / 4
    entry("metrics.r2_score",
          lambda y_true, y_pred: sm.r2_score(y_true, y_pred),
          (yt, yp),
          1 - S4(lambda k: (Y[k] - P[k]) ** 2) / S4(lambda k: (Y[k] - ym) ** 2))
except ImportError:
    pass


def main():
    tally = Counter()
    findings = []
    for name, fn, args, spec, indices, assume, note in BOARD:
        try:
            v = check_formula(fn, tuple(a.copy() for a in args), spec,
                              indices=indices, assume=assume)
            tier = v.tier
        except Exception as e:
            tier = f"BOARD-ERROR {type(e).__name__}"
            v = None
        tally[tier.split()[0]] += 1
        flag = "" if v and v.matches else "  <--"
        print(f"  {name:34s} {tier:16s}{flag}")
        if v is not None and not v.matches:
            findings.append((name, v, note))
    total = len(BOARD)
    print(f"\n== {total} docstring specs ==")
    for k, n in tally.most_common():
        print(f"  {k:16s} {n:3d}  ({100 * n / total:.0f}%)")
    if findings:
        print("\n== for adjudication (transcription slip vs real divergence) ==")
        for name, v, note in findings:
            print(f"\n-- {name}" + (f"   [{note}]" if note else ""))
            print("   " + v.message().replace("\n", "\n   ")[:500])


if __name__ == "__main__":
    main()
