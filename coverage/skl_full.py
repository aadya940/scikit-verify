"""The full sklearn sweep over every tractable numerical API surface."""
import signal
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import sklearn.metrics as M
import sklearn.metrics.pairwise as MP
import sklearn.preprocessing as PP
from sklearn import linear_model as LM
from skverify import to_sympy, Pair

rng = np.random.default_rng(7)
n = 14
X = np.column_stack([np.arange(float(n)), rng.standard_normal(n), rng.uniform(1, 2, n)])
Z = rng.standard_normal((4, 3))
y = X @ np.array([0.5, 1.0, -0.3]) + 0.1 * rng.standard_normal(n)
z = y + 0.05 * rng.standard_normal(n)
ypos = np.abs(y) + 1.0
zpos = np.abs(z) + 1.0
yb = (y > y.mean()).astype(float)          # binary labels
pb = 1 / (1 + np.exp(-z + z.mean()))       # probabilities
Xp = np.abs(X) + 0.5

MENU = []

def add(name, fn, *args):
    MENU.append((name, fn, args))

# ---- regression metrics
add("mean_squared_error", M.mean_squared_error, y, z)
add("root_mean_squared_error", M.root_mean_squared_error, y, z)
add("mean_absolute_error", M.mean_absolute_error, y, z)
add("median_absolute_error", M.median_absolute_error, y, z)
add("max_error", M.max_error, y, z)
add("mean_squared_log_error", M.mean_squared_log_error, ypos, zpos)
add("root_mean_squared_log_error", M.root_mean_squared_log_error, ypos, zpos)
add("mean_absolute_percentage_error", M.mean_absolute_percentage_error, ypos, zpos)
add("r2_score", M.r2_score, y, z)
add("explained_variance_score", M.explained_variance_score, y, z)
add("mean_poisson_deviance", M.mean_poisson_deviance, ypos, zpos)
add("mean_gamma_deviance", M.mean_gamma_deviance, ypos, zpos)
add("mean_tweedie_deviance", M.mean_tweedie_deviance, ypos, zpos)
add("mean_pinball_loss", M.mean_pinball_loss, y, z)
add("d2_absolute_error_score", M.d2_absolute_error_score, y, z)
add("d2_pinball_score", M.d2_pinball_score, y, z)
add("d2_tweedie_score", M.d2_tweedie_score, ypos, zpos)

# ---- classification metrics (binary labels / probabilities)
add("accuracy_score", M.accuracy_score, yb, (pb > 0.5).astype(float))
add("zero_one_loss", M.zero_one_loss, yb, (pb > 0.5).astype(float))
add("hamming_loss", M.hamming_loss, yb, (pb > 0.5).astype(float))
add("precision_score", M.precision_score, yb, (pb > 0.5).astype(float))
add("recall_score", M.recall_score, yb, (pb > 0.5).astype(float))
add("f1_score", M.f1_score, yb, (pb > 0.5).astype(float))
add("fbeta_score_2", lambda a, b: M.fbeta_score(a, b, beta=2.0), yb, (pb > 0.5).astype(float))
add("matthews_corrcoef", M.matthews_corrcoef, yb, (pb > 0.5).astype(float))
add("jaccard_score", M.jaccard_score, yb, (pb > 0.5).astype(float))
add("balanced_accuracy", M.balanced_accuracy_score, yb, (pb > 0.5).astype(float))
add("cohen_kappa", M.cohen_kappa_score, yb, (pb > 0.5).astype(float))
add("log_loss", M.log_loss, yb, pb)
add("brier_score_loss", M.brier_score_loss, yb, pb)
add("hinge_loss", M.hinge_loss, 2 * yb - 1, 2 * pb - 1)
add("roc_auc_score", M.roc_auc_score, yb, pb)
add("average_precision", M.average_precision_score, yb, pb)

# ---- pairwise
add("euclidean_distances", MP.euclidean_distances, X, Z)
add("manhattan_distances", MP.manhattan_distances, X, Z)
add("cosine_similarity", MP.cosine_similarity, X, Z)
add("cosine_distances", MP.cosine_distances, X, Z)
add("linear_kernel", MP.linear_kernel, X, Z)
add("polynomial_kernel", lambda a, b: MP.polynomial_kernel(a, b, degree=2), X, Z)
add("rbf_kernel", MP.rbf_kernel, X, Z)
add("laplacian_kernel", MP.laplacian_kernel, X, Z)
add("sigmoid_kernel", MP.sigmoid_kernel, X, Z)
add("chi2_kernel", MP.chi2_kernel, Xp, np.abs(Z) + 0.5)
add("additive_chi2_kernel", MP.additive_chi2_kernel, Xp, np.abs(Z) + 0.5)
add("haversine_distances", MP.haversine_distances, X[:, :2] / 10, Z[:, :2] / 10)
add("paired_euclidean", MP.paired_distances, X[:4], Z)
add("paired_cosine", lambda a, b: MP.paired_distances(a, b, metric="cosine"), X[:4], Z)

# ---- preprocessing (function forms + fit_transform)
add("scale", PP.scale, X)
add("minmax_scale_fn", PP.minmax_scale, X)
add("maxabs_scale_fn", PP.maxabs_scale, X)
add("robust_scale", PP.robust_scale, X)
add("normalize_l2", PP.normalize, X)
add("normalize_l1", lambda a: PP.normalize(a, norm="l1"), X)
add("normalize_max", lambda a: PP.normalize(a, norm="max"), X)
add("binarize", lambda a: PP.binarize(a, threshold=0.5), X)
add("quantile_transform", lambda a: PP.quantile_transform(a, n_quantiles=5), X)
add("power_transform", PP.power_transform, Xp)
add("StandardScaler", lambda a: PP.StandardScaler().fit_transform(a), X)
add("MinMaxScaler", lambda a: PP.MinMaxScaler().fit_transform(a), X)
add("MaxAbsScaler", lambda a: PP.MaxAbsScaler().fit_transform(a), X)
add("RobustScaler", lambda a: PP.RobustScaler().fit_transform(a), X)
add("Normalizer", lambda a: PP.Normalizer().fit_transform(a), X)
add("PolynomialFeatures", lambda a: PP.PolynomialFeatures(2).fit_transform(a), X[:5, :2])
add("KernelCenterer", lambda a: PP.KernelCenterer().fit_transform(a), MP.linear_kernel(X[:5], X[:5]))

# ---- linear models, coefficients from .fit
add("LinearRegression", lambda a, b: LM.LinearRegression().fit(a, b).coef_, X, y)
add("Ridge", lambda a, b: LM.Ridge(1.0).fit(a, b).coef_, X, y)
add("Lasso", lambda a, b: LM.Lasso(0.1, max_iter=200).fit(a, b).coef_, X, y)
add("ElasticNet", lambda a, b: LM.ElasticNet(0.1, max_iter=200).fit(a, b).coef_, X, y)
add("BayesianRidge", lambda a, b: LM.BayesianRidge().fit(a, b).coef_, X, y)
add("HuberRegressor", lambda a, b: LM.HuberRegressor(max_iter=50).fit(a, b).coef_, X, y)
add("SGDRegressor", lambda a, b: LM.SGDRegressor(max_iter=20, random_state=0).fit(a, b).coef_, X, y)
add("LinearRegression.predict", lambda a, b: LM.LinearRegression().fit(a, b).predict(a[:3]), X, y)
add("Ridge.predict", lambda a, b: LM.Ridge(1.0).fit(a, b).predict(a[:3]), X, y)

class TO(Exception):
    pass

signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))

lift_ok, lift_unverified, refused, died = [], [], [], []
for name, fn, args in MENU:
    signal.alarm(0)
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
