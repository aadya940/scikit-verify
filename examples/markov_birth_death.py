"""Birth - Death Markov chain on states 0..n-1, written without matrices.
Interior update:  p'[i] = b * p[i-1] + (1 - b - d) * p[i] + d * p[i+1]
(b = birth/right-hop probability, d = death/left-hop probability.)
This IS the transition-matrix product, the tridiagonal structure
written as slices.
"""

import numpy as np
from skverify import to_sympy


def birth_death_step(p, b, d):
    stay = 1.0 - b - d
    return b * p[:-2] + stay * p[1:-1] + d * p[2:]


def soft_absorbing_step(p, b, d):
    """Same chain, but zero probabilities stay exactly zero
    (numerical hygiene that is really a modeling change — the lifted
    Piecewise makes it visible)."""
    q = birth_death_step(p, b, d)
    return np.where(q, q, 0.0 * q)


def total_mass(p):
    return np.sum(p)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    p0 = rng.uniform(0, 1, 64)
    p0 = p0 / p0.sum()  # a probability distribution

    out = to_sympy(birth_death_step, p0, 0.3, 0.2)
    print("update rule:", out.formula)
    print("domain:     ", out.domain)

    soft = to_sympy(soft_absorbing_step, p0, 0.3, 0.2)
    print("soft rule:  ", soft.formula)
    print("soft domain:", soft.domain)

    mass = to_sympy(total_mass, p0)
    print("mass before:", mass.formula, "=", mass.value)
    print("mass after: ", out.value.sum(), " (interior update leaks boundary mass)")
