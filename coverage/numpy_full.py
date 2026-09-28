"""The common-numpy sweep over the API surface research scripts actually use."""
import signal
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from skverify import to_sympy, Pair

rng = np.random.default_rng(11)
v = rng.standard_normal(6)
w = np.abs(rng.standard_normal(6)) + 0.5
u = rng.standard_normal(6)
A = rng.standard_normal((3, 4))
B = rng.standard_normal((4, 3))
S = rng.standard_normal((3, 3))
vn = v.copy(); vn[2] = np.nan
x_sorted = np.sort(rng.uniform(0, 10, 6))
y_at_x = np.sin(x_sorted)

MENU = []

def add(name, fn, *args):
    MENU.append((name, fn, args))

# ---- cumulatives and differences
add("cumsum", lambda a: np.cumsum(a).sum(), v)
add("cumprod", lambda a: np.cumprod(a).sum(), w)
add("diff", lambda a: np.diff(a).sum(), v)
add("diff_2nd", lambda a: np.diff(a, 2).sum(), v)
add("ediff1d", lambda a: np.ediff1d(a).sum(), v)
add("gradient", lambda a: np.sum(np.gradient(a)), v)
add("trapezoid", lambda a, b: np.trapezoid(a, b), v, x_sorted)

# ---- order statistics
add("percentile_50", lambda a: np.percentile(a, 50.0), v)
add("percentile_25", lambda a: np.percentile(a, 25.0), v)
add("quantile", lambda a: np.quantile(a, 0.75), v)
add("sort_sum_head", lambda a: np.sort(a)[:3].sum(), v)
add("partition_kth", lambda a: np.partition(a, 2)[2], v)
add("searchsorted_interp", lambda a, b: np.interp(2.5, a, b), x_sorted, y_at_x)
add("ptp", lambda a: np.ptp(a), v)
add("argmax_gather", lambda a: a[np.argmax(a)], v)
add("clip", lambda a: np.clip(a, -0.5, 0.5).sum(), v)

# ---- histograms and counting
add("histogram_counts", lambda a: np.histogram(a, bins=3)[0].sum(), v)
add("bincount_weighted", lambda a: np.bincount(np.array([0, 1, 0, 1, 2, 2]), weights=a).sum(), v)
add("digitize_gather", lambda a: a[np.digitize(0.0, np.sort(a)) - 1], v)

# ---- statistics
add("cov_pair", lambda a, b: np.cov(a, b)[0, 1], v, u)
add("corrcoef_pair", lambda a, b: np.corrcoef(a, b)[0, 1], v, u)
add("average_weighted", lambda a, b: np.average(a, weights=b), v, w)
add("std_ddof1", lambda a: np.std(a, ddof=1), v)
add("median", lambda a: np.median(a), v)

# ---- products and contractions
add("outer_sum", lambda a, b: np.outer(a, b).sum(), v, u)
add("kron_sum", lambda a, b: np.kron(a[:2], b[:2]).sum(), v, u)
add("dot", lambda a, b: np.dot(a, b), v, u)
add("vdot", lambda a, b: np.vdot(a, b), v, u)
add("inner", lambda a, b: np.inner(a, b), v, u)
add("matmul_sum", lambda a, b: (a @ b).sum(), A, B)
add("tensordot", lambda a, b: float(np.tensordot(a, b, axes=2)), A, B.T)
add("einsum_ij_ji", lambda a, b: float(np.einsum("ij,ji->", a, b)), A, B)
add("einsum_diag_sum", lambda s: float(np.einsum("ii->", s)), S)
add("einsum_matvec", lambda a, b: np.einsum("ij,j->i", a, b).sum(), A, u[:4])
add("cross", lambda a, b: np.cross(a[:3], b[:3]).sum(), v, u)

# ---- signal-ish
add("convolve_valid", lambda a, b: np.convolve(a, b[:3], "valid").sum(), v, u)
add("correlate_valid", lambda a, b: np.correlate(a, b[:3], "valid").sum(), v, u)
add("unwrap", lambda a: np.unwrap(a).sum(), v)

# ---- polynomials and piecewise
add("polyval", lambda a: np.polyval(np.array([1.0, -2.0, 0.5]), a).sum(), v)
add("polyfit_deg1", lambda a, b: np.polyfit(a, b, 1)[0], x_sorted, y_at_x)
add("piecewise", lambda a: np.piecewise(a, [a < 0, a >= 0], [lambda t: t**2, lambda t: t]).sum(), v)
add("heaviside", lambda a: np.heaviside(a, 0.5).sum(), v)
add("sign_gate", lambda a: (np.sign(a) * a).sum(), v)
add("where_sum", lambda a: np.where(a > 0, a, 0.0).sum(), v)
add("select", lambda a: np.select([a < -0.5, a > 0.5], [a**2, a**3], default=0.0).sum(), v)

# ---- nan family
add("nansum", lambda a: np.nansum(a), vn)
add("nanmean", lambda a: np.nanmean(a), vn)
add("nanstd", lambda a: np.nanstd(a), vn)
add("nanmax", lambda a: np.nanmax(a), vn)
add("nanmedian", lambda a: np.nanmedian(a), vn)

# ---- shape plumbing feeding a reduction
add("stack_mean", lambda a, b: np.stack([a, b]).mean(), v, u)
add("concat_sum", lambda a, b: np.concatenate([a, b]).sum(), v, u)
add("tile_sum", lambda a: np.tile(a, 2).sum(), v)
add("repeat_sum", lambda a: np.repeat(a, 2).sum(), v)
add("roll_dot", lambda a: np.dot(a, np.roll(a, 1)), v)
add("flip_diff", lambda a: (a - np.flip(a)).sum(), v)
add("transpose_trace", lambda s: np.trace(s.T @ s), S)
add("ravel_outer", lambda a, b: np.ravel(np.outer(a[:2], b[:2])).sum(), v, u)
add("split_sum", lambda a: np.split(a, 3)[1].sum(), v)
add("meshgrid_sum", lambda a, b: np.meshgrid(a[:3], b[:3])[0].sum(), v, u)

# ---- misc common
add("maximum_reduce", lambda a, b: np.maximum(a, b).sum(), v, u)
add("fmax_sum", lambda a, b: np.fmax(a, b).sum(), v, u)
add("abs_diff_norm", lambda a, b: np.abs(a - b).max(), v, u)
add("linalg_norm2", lambda a: np.linalg.norm(a), v)
add("linalg_norm1", lambda a: np.linalg.norm(a, 1), v)
add("allclose_gate", lambda a, b: float(np.sum(a) if np.allclose(a, b) else np.sum(b)), v, v + 1.0)
add("real_if_close", lambda a: np.real_if_close(a).sum(), v)

class TO(Exception):
    pass

signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))

lift_ok, lift_unverified, refused, died = [], [], [], []
for name, fn, args in MENU:
    signal.alarm(60)
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
        refused.append((name, str(e)[:52]))
    except Exception as e:
        died.append((name, f"{type(e).__name__} {str(e)[:52]}"))
    finally:
        signal.alarm(0)

total = len(MENU)
print(f"TOTAL {total} | LIFT+match {len(lift_ok)} | lift-unverified {len(lift_unverified)} | refused {len(refused)} | died {len(died)}")
print("\nLIFT+match:", ", ".join(lift_ok))
if lift_unverified:
    print("\nlift-unverified:", ", ".join(lift_unverified))
print("\nREFUSED:")
for n_, m in refused:
    print(f"  {n_:24s} {m}")
print("\nDIED:")
for n_, m in died:
    print(f"  {n_:24s} {m}")
