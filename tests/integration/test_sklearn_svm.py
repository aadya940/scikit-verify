"""The RBF SVM decision function: traced from learned parameters
through sklearn's own kernel, verified against libsvm's C answer."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

from skverify import to_sympy


def test_svm_decision_function_certifies_libsvm():
    rng = np.random.default_rng(4)
    Xtr = rng.standard_normal((20, 2))
    ycls = (Xtr @ np.array([1.0, -0.5]) > 0).astype(int)
    svc = SVC(kernel="rbf").fit(Xtr, ycls)

    SV = svc.support_vectors_
    alpha = svc.dual_coef_.ravel()
    b = float(svc.intercept_[0])
    gamma = svc._gamma

    def decision(x):
        K = rbf_kernel(x, SV, gamma=gamma)
        return K @ alpha + b

    x = np.array([[0.5, -1.0], [1.5, 0.3]])
    r = to_sympy(decision, x)
    # the traced kernel expansion IS libsvm's answer: a result-check
    # of the C implementation by the textbook formula
    assert np.allclose(
        np.asarray(r.value, dtype=float), svc.decision_function(x)
    )
    assert "exp" in str(r.formula)
