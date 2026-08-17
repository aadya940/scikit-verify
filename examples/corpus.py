"""The real-function corpus: every library kernel the tracer has been
pointed at, with its current verdict. Run: python examples/corpus.py

This is the refusal frequency table and the regression net for real-world
behavior. A DIES entry names the next piece of engine work; a verdict
CHANGING between runs is news either way.
"""

import warnings

import numpy as np

from skverify import to_sympy

warnings.filterwarnings("ignore")


def kdv_step(u, dt, dx):
    # Korteweg-de Vries u_t = -6*u*u_x - u_xxx: two stencil widths, realigned
    ux = (u[2:] - u[:-2]) / (2 * dx)
    uxxx = (u[4:] - 2 * u[3:-1] + 2 * u[1:-3] - u[:-4]) / (2 * dx**3)
    return u[2:-2] - dt * (6.0 * u[2:-2] * ux[1:-1] + uxxx)


def kdv_step_bad(u, dt, dx):
    # deliberate off-by-one (ux[:-2]): same shapes, wrong math. numpy runs
    # it silently; the lifted formula shows the off-center derivative.
    ux = (u[2:] - u[:-2]) / (2 * dx)
    uxxx = (u[4:] - 2 * u[3:-1] + 2 * u[1:-3] - u[:-4]) / (2 * dx**3)
    return u[2:-2] - dt * (6.0 * u[2:-2] * ux[:-2] + uxxx)


def birth_death_step(p, b, d):
    stay = 1.0 - b - d
    return b * p[:-2] + stay * p[1:-1] + d * p[2:]


def running_total(x):
    s = 0.0
    for k in range(len(x)):
        s = s + x[k]
    return s


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def two_pass_variance(x, n):
    m = np.sum(x) / n
    return np.sum((x - m) ** 2) / n


def mask_count(x):
    return np.sum(x > 0)


def horner(c, x):
    s = 0.0
    for k in range(len(c)):
        s = s * x + c[k]
    return s


def heat_step(u, r):
    dudt = np.zeros_like(u)
    dudt[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]
    dudt[0] = 0.0
    return u + r * dudt


def _entries():
    from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
    from scipy.stats import gmean, hmean

    y8 = np.linspace(0, 1, 8) ** 2
    y9 = np.linspace(0, 1, 9) ** 2
    Y2 = np.random.default_rng(0).random((3, 8))
    x10 = np.linspace(0, 9, 10)

    return {
        # hand-written kernels
        "kdv_step": lambda: to_sympy(
            kdv_step, np.cosh(np.linspace(-3, 3, 32)) ** -2, 1e-4, 0.1
        ),
        "kdv_step_bad (bug visible in formula)": lambda: to_sympy(
            kdv_step_bad, np.cosh(np.linspace(-3, 3, 32)) ** -2, 1e-4, 0.1
        ),
        "markov_birth_death": lambda: to_sympy(
            birth_death_step, np.full(16, 1 / 16), 0.3, 0.2
        ),
        "accumulator loop": lambda: to_sympy(running_total, np.arange(5.0)),
        "softmax": lambda: to_sympy(softmax, np.linspace(0, 1, 5)),
        "sigmoid": lambda: to_sympy(sigmoid, np.linspace(-1, 1, 6)),
        "two-pass variance": lambda: to_sympy(two_pass_variance, np.arange(6.0), 6),
        "mask count": lambda: to_sympy(mask_count, np.linspace(-1, 1, 6)),
        "horner loop": lambda: to_sympy(horner, np.array([2.0, 3.0, 5.0]), 1.5),
        "heat step (build-then-fill)": lambda: to_sympy(
            heat_step, np.linspace(0, 1, 8) ** 2, 0.1
        ),
        # scipy, unmodified
        "scipy trapezoid": lambda: to_sympy(lambda y: trapezoid(y, dx=0.1), y8),
        "scipy simpson": lambda: to_sympy(lambda y: simpson(y, dx=0.125), y9),
        "scipy cumulative_trapezoid": lambda: to_sympy(cumulative_trapezoid, y8),
        "scipy trapezoid 2-D axis=1": lambda: to_sympy(
            lambda y: trapezoid(y, dx=0.1, axis=1), Y2
        ),
        "scipy stats.gmean": lambda: to_sympy(gmean, np.linspace(1, 2, 6)),
        "scipy stats.hmean": lambda: to_sympy(hmean, np.linspace(1, 2, 6)),
        # numpy, through its own bodies
        "np.diff n=1": lambda: to_sympy(np.diff, x10),
        "np.diff n=3": lambda: to_sympy(np.diff, x10, 3),
        "np.dot": lambda: to_sympy(np.dot, np.arange(4.0), np.arange(4.0)),
        "np.trace": lambda: to_sympy(np.trace, np.arange(9.0).reshape(3, 3)),
        "np.average weighted": lambda: to_sympy(
            np.average, np.arange(5.0), None, np.arange(1.0, 6.0)
        ),
        "np.mean": lambda: to_sympy(np.mean, np.arange(5.0)),
        "np.var": lambda: to_sympy(np.var, np.arange(5.0)),
        "np.std": lambda: to_sympy(np.std, np.arange(5.0)),
        # trackers: expected to DIE until the named feature lands
        "np.gradient [tracker: guards/coercion]": lambda: to_sympy(
            np.gradient, 4 * np.linspace(0, 20, 8)
        ),
        "np.polyval [tracker: guards]": lambda: to_sympy(
            np.polyval, np.array([2.0, 3.0, 5.0]), np.arange(4.0)
        ),
        "np.median [tracker: guards]": lambda: to_sympy(np.median, np.arange(5.0)),
        "np.linalg.solve (opaque + contract)": lambda: to_sympy(
            lambda A, b: np.linalg.solve(A, b),
            np.array([[4.0, 1.0], [1.0, 3.0]]),
            np.array([1.0, 2.0]),
        ),
        "np.clip": lambda: to_sympy(
            np.clip, np.linspace(-1, 1, 6), 0.0, 0.5
        ),
    }


def main():
    lifted = died = 0
    for name, fn in _entries().items():
        try:
            r = fn()
            f = getattr(r, "formula", r)
            p = getattr(r, "preconditions", r)
            print(f"LIFTS  {name}\n       {str(f)[:100]}")
            print(f"PRECONDITIONS: {p}")
            unchecked = getattr(r, "unchecked", ())
            if unchecked:
                print(f"UNCHECKED: {unchecked}")
            print()
            lifted += 1
        except Exception as e:
            print(f"DIES   {name}\n       {type(e).__name__}: {str(e)[:80]}")
            died += 1
    print(f"\n{lifted} lift, {died} die")


if __name__ == "__main__":
    main()
