"""Random walk on a 2-D grid, written without matrices.

A walker on an n x m grid hops left/right/up/down or stays.
Interior update:

  p'[i,j] = stay * p[i,j]
          + r * p[i,j-1] + l * p[i,j+1]     (arrived from the left / right)
          + d * p[i-1,j] + u * p[i+1,j]     (arrived from above / below)

This IS the transition-matrix product; the five-point structure is
written as 2-D slices. The same boundary-mass leak as the 1-D chain
appears here, along all four edges at once.
"""

import numpy as np
from skverify import to_sympy


def walk_step(p, left, right, up, down):
    stay = 1.0 - left - right - up - down
    return (
        stay * p[1:-1, 1:-1]
        + right * p[1:-1, :-2]  # mass that hopped in from the left neighbour
        + left * p[1:-1, 2:]  # ... from the right neighbour
        + down * p[:-2, 1:-1]  # ... from above
        + up * p[2:, 1:-1]  # ... from below
    )


def total_mass(p):
    return np.sum(p)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    p0 = rng.uniform(0, 1, (8, 8))
    p0 = p0 / p0.sum()  # a probability distribution

    out = to_sympy(walk_step, p0, 0.1, 0.1, 0.1, 0.1)
    print("update rule:", out.formula)
    print("domain:     ", out.domain)

    mass = to_sympy(total_mass, p0)
    print("mass before:", mass.value, "  as formula:", mass.formula)
    print("mass after: ", out.value.sum())
    print(
        "leak:       ",
        mass.value - out.value.sum(),
        " (interior update drops all four edges)",
    )
