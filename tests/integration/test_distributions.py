"""Probability distributions through skverify: trace numpy density
code, then PROVE the lifted formulas against theory.

Float parameters lift as SYMBOLS (mu, sigma, lam), so the ladder ends
in exact symbolic identities, parameters and all:
  1. lift    -- the numpy implementation traces to a formula
  2. agree   -- formula evaluates to the value lane (differential)
  3. theory  -- the formula IS the textbook density: symbolically
                equal to sympy.stats' density, normalizes to 1 over
                the support, has the right moments, and satisfies
                pdf/logpdf consistency.
Steps 1-2 test skverify; step 3 is the verification layer: code
checked against mathematics, not against other code."""

import numpy as np
import sympy
from sympy.stats import Exponential, Logistic, Normal, density

from skverify import Pair, to_sympy
from skverify.helpers import axis_idx

I = axis_idx(0)
X = sympy.IndexedBase("x")
MU = sympy.Symbol("mu", real=True)
SIGMA = sympy.Symbol("sigma", real=True)
LAM = sympy.Symbol("lam", real=True)
S = sympy.Symbol("s", real=True)


# the traced kernels: plain numpy, the way research code writes them


def normal_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def normal_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def exponential_pdf(x, lam):
    return lam * np.exp(-lam * x)


def logistic_pdf(x, mu, s):
    z = np.exp(-(x - mu) / s)
    return z / (s * (1.0 + z) ** 2)


def gaussian_loglik(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * np.sum(z * z) - 8 * np.log(sigma) - 4.0 * np.log(2.0 * np.pi)


XS = np.linspace(-2.0, 3.0, 8)
POS = np.linspace(0.1, 4.0, 8)
PARAMS = {MU: 0.5, SIGMA: 1.3, LAM: 2.0, S: 1.0}


def _pointwise(formula, xs, params=None):
    mapping = {X[k]: sympy.Float(float(v)) for k, v in enumerate(xs)}
    mapping.update(PARAMS if params is None else params)
    out = []
    for k in range(len(xs)):
        e = formula.subs(I, k).doit().xreplace(mapping)
        out.append(float(sympy.N(e.doit())))
    return np.array(out)


def _exact(e):
    """Exactify float constants (0.3989... -> 1/sqrt(2 pi)), each
    replacement verified by round-trip -- the capability-map recipe."""
    for f in e.atoms(sympy.Float):
        cand = sympy.nsimplify(f, [sympy.pi])
        if abs(float(cand) - float(f)) < 1e-12:
            e = e.subs(f, cand)
    return e


def _scalarize(formula):
    """The lifted per-point rule as a function of a scalar t --
    parameters STAY symbolic."""
    t = sympy.Symbol("t", real=True)
    return formula.subs(X[I], t), t


class TestNormal:
    def _lift(self):
        return to_sympy(normal_pdf, XS, 0.5, 1.3)

    def test_lifts_and_agrees(self):
        r = self._lift()
        assert isinstance(r, Pair)
        got = _pointwise(r.formula, XS)
        assert np.allclose(got, np.asarray(r.value, dtype=float))
        assert np.allclose(got, normal_pdf(XS, 0.5, 1.3))

    def test_formula_is_the_textbook_density(self):
        # exact symbolic identity, parameters included
        expr, t = _scalarize(self._lift().formula)
        sig = sympy.Symbol("sigma", positive=True)
        theory = density(Normal("N", MU, sig))(t)
        diff = sympy.simplify(_exact(expr.subs(SIGMA, sig)) - theory)
        assert diff == 0

    def test_normalizes_to_one(self):
        expr, t = _scalarize(self._lift().formula)
        sig = sympy.Symbol("sigma", positive=True)
        total = sympy.integrate(
            _exact(expr.subs(SIGMA, sig)), (t, -sympy.oo, sympy.oo)
        )
        assert sympy.simplify(total - 1) == 0

    def test_mean_and_variance_identities(self):
        expr, t = _scalarize(self._lift().formula)
        sig = sympy.Symbol("sigma", positive=True)
        e = _exact(expr.subs(SIGMA, sig))
        mean = sympy.integrate(t * e, (t, -sympy.oo, sympy.oo))
        assert sympy.simplify(mean - MU) == 0
        var = sympy.integrate((t - MU) ** 2 * e, (t, -sympy.oo, sympy.oo))
        assert sympy.simplify(var - sig**2) == 0

    def test_logpdf_consistent_with_pdf(self):
        e_pdf, t = _scalarize(self._lift().formula)
        r_log = to_sympy(normal_logpdf, XS, 0.5, 1.3)
        e_log, t2 = _scalarize(r_log.formula)
        sig = sympy.Symbol("sigma", positive=True)
        diff = sympy.simplify(
            sympy.exp(e_log.subs(t2, t)).subs(SIGMA, sig)
            - e_pdf.subs(SIGMA, sig)
        )
        assert diff == 0


class TestExponential:
    def test_full_ladder(self):
        r = to_sympy(exponential_pdf, POS, 2.0)
        got = _pointwise(r.formula, POS)
        assert np.allclose(got, exponential_pdf(POS, 2.0))
        expr, t = _scalarize(r.formula)
        lam = sympy.Symbol("lam", positive=True)
        e = expr.subs(LAM, lam)
        theory = density(Exponential("E", lam))(t)
        assert sympy.simplify(e - theory) == 0
        tp = sympy.Symbol("tp", positive=True)
        assert sympy.simplify(
            sympy.integrate(e.subs(t, tp), (tp, 0, sympy.oo)) - 1
        ) == 0
        mean = sympy.integrate(tp * e.subs(t, tp), (tp, 0, sympy.oo))
        assert sympy.simplify(mean - 1 / lam) == 0


class TestLogistic:
    def test_lifts_and_matches_theory(self):
        r = to_sympy(logistic_pdf, XS, 0.0, 1.0)
        got = _pointwise(r.formula, XS, {MU: 0.0, S: 1.0})
        assert np.allclose(got, logistic_pdf(XS, 0.0, 1.0))
        expr, t = _scalarize(r.formula)
        spos = sympy.Symbol("s", positive=True)
        e = _exact(expr.subs(S, spos))
        theory = density(Logistic("L", MU, spos))(t)
        assert sympy.simplify(e - theory) == 0


class TestLogLikelihood:
    def test_gaussian_loglik_lifts_as_sum(self):
        r = to_sympy(gaussian_loglik, XS, 0.5, 1.3)
        assert r.formula.atoms(sympy.Sum)
        mapping = {X[k]: sympy.Float(float(v)) for k, v in enumerate(XS)}
        mapping.update(PARAMS)
        got = float(sympy.N(r.formula.doit().xreplace(mapping)))
        assert np.isclose(got, gaussian_loglik(XS, 0.5, 1.3))

    def test_score_equation_solves_to_sample_mean(self):
        # the MLE story end to end: differentiate the LIFTED loglik in
        # the SYMBOLIC mu, solve the score equation, land on x-bar
        r = to_sympy(gaussian_loglik, XS, 0.5, 1.3)
        score = sympy.diff(r.formula.doit(), MU)
        mapping = {X[k]: sympy.Float(float(v)) for k, v in enumerate(XS)}
        root = sympy.solve(score.xreplace(mapping).subs(SIGMA, 1.3), MU)
        assert np.isclose(float(root[0]), XS.mean())

    def test_fisher_information_sign(self):
        # second derivative of the loglik in mu is -n/sigma^2 < 0:
        # the estimator is a maximum, proven from the traced formula
        r = to_sympy(gaussian_loglik, XS, 0.5, 1.3)
        d2 = sympy.diff(r.formula.doit(), MU, 2)
        assert sympy.simplify(d2 + 8 / SIGMA**2) == 0
