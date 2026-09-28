"""The cvxpy sweep. It covers expression evaluation and small solved problems."""
import signal
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import cvxpy as cp
from skverify import to_sympy, Pair

rng = np.random.default_rng(7)
A = rng.standard_normal((6, 3))
b = rng.standard_normal(6)
v = rng.standard_normal(5)
vpos = np.abs(v) + 0.5
p = vpos / vpos.sum()
q = np.abs(rng.standard_normal(5)) + 0.5
q = q / q.sum()
S = A.T @ A + 0.5 * np.eye(3)
w = rng.standard_normal(3)

MENU = []

def add(name, fn, *args):
    MENU.append((name, fn, args))

def val(expr):
    return expr.value

# ---- atom evaluation. Evaluate the cvxpy expression on data and read the value
add("sum_squares", lambda a: val(cp.sum_squares(a)), v)
add("norm1", lambda a: val(cp.norm1(a)), v)
add("norm2", lambda a: val(cp.norm2(a)), v)
add("norm_inf", lambda a: val(cp.norm_inf(a)), v)
add("pnorm_3", lambda a: val(cp.pnorm(a, 3)), vpos)
add("abs_sum", lambda a: val(cp.sum(cp.abs(a))), v)
add("square_sum", lambda a: val(cp.sum(cp.square(a))), v)
add("sqrt_sum", lambda a: val(cp.sum(cp.sqrt(a))), vpos)
add("power_sum", lambda a: val(cp.sum(cp.power(a, 2))), v)
add("pos_sum", lambda a: val(cp.sum(cp.pos(a))), v)
add("neg_sum", lambda a: val(cp.sum(cp.neg(a))), v)
add("maximum_elem", lambda a, c: val(cp.sum(cp.maximum(a, c))), v, w[0])
add("max", lambda a: val(cp.max(a)), v)
add("min", lambda a: val(cp.min(a)), v)
add("logistic_sum", lambda a: val(cp.sum(cp.logistic(a))), v)
add("entr_sum", lambda a: val(cp.sum(cp.entr(a))), p)
add("kl_div_sum", lambda a, c: val(cp.sum(cp.kl_div(a, c))), p, q)
add("rel_entr_sum", lambda a, c: val(cp.sum(cp.rel_entr(a, c))), p, q)
add("log_sum_exp", lambda a: val(cp.log_sum_exp(a)), v)
add("geo_mean", lambda a: val(cp.geo_mean(a)), vpos)
add("harmonic_mean", lambda a: val(cp.harmonic_mean(a)), vpos)
add("huber_sum", lambda a: val(cp.sum(cp.huber(a, 1.0))), v)
add("quad_form", lambda a, s: val(cp.quad_form(a, s)), w, S)
add("quad_over_lin", lambda a, c: val(cp.quad_over_lin(a, c)), v, vpos[0])
add("matrix_frac", lambda a, s: val(cp.matrix_frac(a, s)), w, S)
add("trace", lambda s: val(cp.trace(s)), S)
add("lambda_max", lambda s: val(cp.lambda_max(s)), S)
add("lambda_min", lambda s: val(cp.lambda_min(s)), S)
add("sigma_max", lambda a: val(cp.sigma_max(a)), A)
add("normNuc", lambda a: val(cp.normNuc(a)), A)
add("log_det", lambda s: val(cp.log_det(s)), S)
add("tv_1d", lambda a: val(cp.tv(a)), v)
add("residual_norm", lambda a, c: val(cp.norm2(a @ np.ones(3) - c)), A, b)

# ---- solved problems. Data feeds the optimizer
def lstsq(a, c):
    x = cp.Variable(3)
    cp.Problem(cp.Minimize(cp.sum_squares(a @ x - c))).solve()
    return x.value

def ridge(a, c):
    x = cp.Variable(3)
    cp.Problem(cp.Minimize(cp.sum_squares(a @ x - c) + cp.sum_squares(x))).solve()
    return x.value

def lasso(a, c):
    x = cp.Variable(3)
    cp.Problem(cp.Minimize(cp.sum_squares(a @ x - c) + 0.1 * cp.norm1(x))).solve()
    return x.value

def nonneg_ls(a, c):
    x = cp.Variable(3)
    cp.Problem(cp.Minimize(cp.sum_squares(a @ x - c)), [x >= 0]).solve()
    return x.value

def chebyshev(a, c):
    x = cp.Variable(3)
    cp.Problem(cp.Minimize(cp.norm_inf(a @ x - c))).solve()
    return x.value

add("solve_lstsq", lstsq, A, b)
add("solve_ridge", ridge, A, b)
add("solve_lasso", lasso, A, b)
add("solve_nonneg_ls", nonneg_ls, A, b)
add("solve_chebyshev", chebyshev, A, b)

class TO(Exception):
    pass

signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))

lift_ok, lift_unverified, refused, died = [], [], [], []
for name, fn, args in MENU:
    signal.alarm(120)
    try:
        ref = fn(*args)
        r = to_sympy(fn, *args)
        got = r.value if isinstance(r, Pair) else r
        if isinstance(got, np.ndarray) and got.dtype == object:
            got = np.asarray(Pair._value_of(got), dtype=float)
        try:
            match = np.allclose(
                np.asarray(got, dtype=float), np.asarray(ref, dtype=float),
                rtol=1e-7, atol=1e-9, equal_nan=True,
            )
        except Exception:
            match = None
        (lift_ok if match else lift_unverified).append(name)
    except TO:
        died.append((name, "TIMEOUT"))
    except NotImplementedError as e:
        refused.append((name, str(e)[:44]))
    except Exception as e:
        died.append((name, f"{type(e).__name__} {str(e)[:44]}"))
    finally:
        signal.alarm(0)

total = len(MENU)
print(f"TOTAL {total} | LIFT+match {len(lift_ok)} | lift-unverified {len(lift_unverified)} | refused {len(refused)} | died {len(died)}")
print("\nLIFT+match:", ", ".join(lift_ok))
if lift_unverified:
    print("\nlift-unverified:", ", ".join(lift_unverified))
print("\nREFUSED:")
for n_, m in refused:
    print(f"  {n_:28s} {m}")
print("\nDIED:")
for n_, m in died:
    print(f"  {n_:28s} {m}")
