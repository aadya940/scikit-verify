"Map scipy.special ufuncs to their exact sympy forms."

import numpy as np
import sympy

from ..registry import UFUNC_TABLE

try:
    import scipy.special as sp
except ImportError:  # pragma: no cover
    sp = None


def _register():
    same = "erf erfc gamma digamma zeta sinc".split()
    renamed = {
        "gammaln": "loggamma",
        "psi": "digamma",
        "erfinv": "erfinv",
        "erfcinv": "erfcinv",
        "eval_legendre": "legendre",
        "eval_chebyt": "chebyshevt",
        "eval_chebyu": "chebyshevu",
        "eval_hermite": "hermite",
        "eval_laguerre": "laguerre",
    }
    two_arg = {
        "beta": sympy.beta,
        "polygamma": sympy.polygamma,
        "jv": sympy.besselj,
        "yv": sympy.bessely,
        "iv": sympy.besseli,
        "kv": sympy.besselk,
    }
    composed = {
        "ndtr": lambda z: (1 + sympy.erf(z / sympy.sqrt(2))) / 2,
        "log_ndtr": lambda z: sympy.log((1 + sympy.erf(z / sympy.sqrt(2))) / 2),
        "expit": lambda z: 1 / (1 + sympy.exp(-z)),
        "logit": lambda p: sympy.log(p / (1 - p)),
        "xlogy": lambda x, y: sympy.Piecewise((0, sympy.Eq(x, 0)), (x * sympy.log(y), True)),
        "xlog1py": lambda x, y: sympy.Piecewise((0, sympy.Eq(x, 0)), (x * sympy.log(1 + y), True)),
        "log1p": lambda x: sympy.log(1 + x),
        "expm1": lambda x: sympy.exp(x) - 1,
        "cbrt": lambda x: sympy.cbrt(x),
        "rgamma": lambda x: 1 / sympy.gamma(x),
        "gammasgn": lambda x: sympy.sign(sympy.gamma(x)),
        "entr": lambda p: sympy.Piecewise(
            (-p * sympy.log(p), p > 0), (0, sympy.Eq(p, 0)), (-sympy.oo, True)
        ),
        "rel_entr": lambda p, q: sympy.Piecewise(
            (p * sympy.log(p / q), p > 0), (0, sympy.Eq(p, 0)), (sympy.oo, True)
        ),
        "huber": lambda d, r: sympy.Piecewise(
            (r**2 / 2, sympy.Abs(r) <= d), (d * (sympy.Abs(r) - d / 2), True)
        ),
    }
    for name in same:
        fn = getattr(sp, name, None)
        target = getattr(sympy, name, None)
        if isinstance(fn, np.ufunc) and target is not None:
            UFUNC_TABLE[fn] = target
    for sp_name, sy_name in renamed.items():
        fn = getattr(sp, sp_name, None)
        target = getattr(sympy, sy_name, None)
        if isinstance(fn, np.ufunc) and target is not None:
            UFUNC_TABLE[fn] = target
    for sp_name, target in {**two_arg, **composed}.items():
        fn = getattr(sp, sp_name, None)
        if isinstance(fn, np.ufunc):
            UFUNC_TABLE[fn] = target


if sp is not None:
    _register()
    from .numpy import _attach_ufunc_methods

    _attach_ufunc_methods()
