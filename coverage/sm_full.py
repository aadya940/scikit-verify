"""Full-surface statsmodels battery in skl_full's image. It reports
LIFT+match, refused, and died, with no time budget."""
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from skverify import to_sympy
from skverify.pair import Pair

rng = np.random.default_rng(11)
n = 12
Xr = np.column_stack([np.ones(n), np.arange(float(n)), rng.standard_normal(n)])
beta = np.array([0.5, 1.0, -0.4])
yr = Xr @ beta + 0.15 * rng.standard_normal(n)
resid = yr - Xr @ np.linalg.lstsq(Xr, yr, rcond=None)[0]
pos = np.abs(yr) + 0.5
cnt = rng.poisson(3.0, n).astype(float)
binv = (yr + 0.4 * rng.standard_normal(n) > yr.mean()).astype(float)  # not separable
w = rng.uniform(0.5, 2.0, n)
ser = np.sin(np.arange(30.0) / 3) + 0.1 * rng.standard_normal(30)

import statsmodels.api as sm
from statsmodels.stats import stattools as st
from statsmodels.robust.scale import mad
from statsmodels.robust import norms
from statsmodels.tools import tools as smtools
from statsmodels.tsa import stattools as tsa

MENU = []
def add(name, fn, *args):
    MENU.append((name, fn, args))

# stattools / diagnostics
add("durbin_watson", st.durbin_watson, resid)
add("jarque_bera", lambda x: st.jarque_bera(x)[0], resid)
add("omni_normtest", lambda x: st.omni_normtest(x)[0], resid)
add("robust_skewness", lambda x: st.robust_skewness(x)[0], resid)
add("robust_kurtosis", lambda x: st.robust_kurtosis(x)[0], resid)
add("medcouple", st.medcouple, resid)
add("mad", mad, resid)
# tools
add("add_constant", smtools.add_constant, yr)
add("pinv_extended", lambda a: smtools.pinv_extended(a)[0], Xr)
# OLS family
add("OLS.params", lambda X, y: sm.OLS(y, X).fit().params, Xr, yr)
add("OLS.resid", lambda X, y: sm.OLS(y, X).fit().resid, Xr, yr)
add("OLS.fittedvalues", lambda X, y: sm.OLS(y, X).fit().fittedvalues, Xr, yr)
add("OLS.rsquared", lambda X, y: sm.OLS(y, X).fit().rsquared, Xr, yr)
add("OLS.rsquared_adj", lambda X, y: sm.OLS(y, X).fit().rsquared_adj, Xr, yr)
add("OLS.bse", lambda X, y: sm.OLS(y, X).fit().bse, Xr, yr)
add("OLS.tvalues", lambda X, y: sm.OLS(y, X).fit().tvalues, Xr, yr)
add("OLS.pvalues", lambda X, y: sm.OLS(y, X).fit().pvalues, Xr, yr)
add("OLS.aic", lambda X, y: sm.OLS(y, X).fit().aic, Xr, yr)
add("OLS.bic", lambda X, y: sm.OLS(y, X).fit().bic, Xr, yr)
add("OLS.llf", lambda X, y: sm.OLS(y, X).fit().llf, Xr, yr)
add("OLS.predict", lambda X, y: sm.OLS(y, X).fit().predict(X[:3]), Xr, yr)
add("OLS.mse_resid", lambda X, y: sm.OLS(y, X).fit().mse_resid, Xr, yr)
add("OLS.ess", lambda X, y: sm.OLS(y, X).fit().ess, Xr, yr)
add("OLS.centered_tss", lambda X, y: sm.OLS(y, X).fit().centered_tss, Xr, yr)
add("OLS.fvalue", lambda X, y: sm.OLS(y, X).fit().fvalue, Xr, yr)
add("WLS.params", lambda X, y, w: sm.WLS(y, X, weights=w).fit().params, Xr, yr, w)
add("GLS.params", lambda X, y: sm.GLS(y, X).fit().params, Xr, yr)
add("OLS.cov_params", lambda X, y: sm.OLS(y, X).fit().cov_params(), Xr, yr)
add("OLS.HC0_se", lambda X, y: sm.OLS(y, X).fit().HC0_se, Xr, yr)
# GLM
add("GLM.gaussian.params", lambda X, y: sm.GLM(y, X).fit().params, Xr, yr)
add("GLM.poisson.params", lambda X, y: sm.GLM(y, X, family=sm.families.Poisson()).fit().params, Xr, cnt)
add("GLM.binomial.params", lambda X, y: sm.GLM(y, X, family=sm.families.Binomial()).fit().params, Xr, binv)
add("GLM.gaussian.mu", lambda X, y: sm.GLM(y, X).fit().mu, Xr, yr)
add("Logit.params", lambda X, y: sm.Logit(y, X).fit(disp=0).params, Xr, binv)
add("Poisson.params", lambda X, y: sm.Poisson(y, X).fit(disp=0).params, Xr, cnt)
# robust norms (pure math)
add("Huber.rho", lambda x: norms.HuberT().rho(x), resid)
add("Huber.psi", lambda x: norms.HuberT().psi(x), resid)
add("Tukey.rho", lambda x: norms.TukeyBiweight().rho(x), resid)
# tsa
add("acovf", lambda s: tsa.acovf(s, nlag=5, fft=False), ser)
add("acf", lambda s: tsa.acf(s, nlags=5, fft=False), ser)
add("pacf_yw", lambda s: tsa.pacf(s, nlags=4, method="yw"), ser)
add("adfuller_stat", lambda s: tsa.adfuller(s, maxlag=2)[0], ser)
# descriptive
add("zscore-ish", lambda x: (x - x.mean()) / x.std(), resid)
add("RLM.params", lambda X, y: sm.RLM(y, X).fit().params, Xr, yr)
add("quantile_reg", lambda X, y: sm.QuantReg(y, X).fit(q=0.5, max_iter=50).params, Xr, yr)

import signal, sys, time
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda a, b: (_ for _ in ()).throw(TO()))
lift_ok, lift_unverified, refused, died = [], [], [], []
for name, fn, args in MENU:
    print("::", name, flush=True, file=sys.stderr)
    t0 = time.time()
    signal.alarm(240)
    try:
        ref = fn(*args)
        r = to_sympy(fn, *args)
        got = r.value if isinstance(r, Pair) else r
        if isinstance(got, np.ndarray) and got.dtype == object:
            got = np.asarray([Pair._value_of(e) for e in np.ravel(got)], dtype=float).reshape(np.shape(got))
        try:
            match = np.allclose(np.asarray(got, dtype=float), np.asarray(ref, dtype=float),
                                rtol=1e-6, atol=1e-8, equal_nan=True)
        except Exception:
            match = None
        (lift_ok if match else lift_unverified).append(name)
    except TO:
        died.append((name, "TIMEOUT>240s"))
    except NotImplementedError as e:
        refused.append((name, str(e)[:52]))
    except Exception as e:
        died.append((name, f"{type(e).__name__} {str(e)[:52]}"))
    finally:
        signal.alarm(0)
        print("::", name, "done", round(time.time()-t0, 1), "s", flush=True, file=sys.stderr)

print(f"TOTAL {len(MENU)} | LIFT+match {len(lift_ok)} | unverified {len(lift_unverified)} | refused {len(refused)} | died {len(died)}")
print("LIFT:", ", ".join(lift_ok))
print("UNVERIFIED:", lift_unverified)
print("REFUSED:")
for n_, m in refused: print("  ", n_, "|", m)
print("DIED:")
for n_, m in died: print("  ", n_, "|", m)
