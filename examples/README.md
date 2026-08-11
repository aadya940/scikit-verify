# Examples
 
Small, self-contained numerical kernels traced with `to_sympy`. Each file
runs directly and prints the recovered formulas alongside the ordinary
numerical results.
 
```bash
python markov_birth_death.py
```
 
## markov_birth_death.py
 
A birth–death Markov chain on states `0..n-1`, written without matrices;
the tridiagonal transition structure is expressed as slices:
 
```python
def birth_death_step(p, b, d):
    stay = 1.0 - b - d
    return b * p[:-2] + stay * p[1:-1] + d * p[2:]
```
 
Recovered update rule:
 
```
b*p[i] + d*p[i + 2] + (1.0 - b - d)*p[i + 1]        domain (0, 62)
```

## markov_walk_2d.py

The same idea one dimension up: a random walk on an 8×8 grid. The
five-point transition structure (stay + four neighbours) is expressed
as 2-D slices:

```python
def walk_step(p, left, right, up, down):
    stay = 1.0 - left - right - up - down
    return (
        stay * p[1:-1, 1:-1]
        + right * p[1:-1, :-2]
        + left * p[1:-1, 2:]
        + down * p[:-2, 1:-1]
        + up * p[2:, 1:-1]
    )
```

Recovered update rule (one line describing all 36 interior cells;
a row of the 64×64 transition matrix, never built):

```
down*p[i, j+1] + left*p[i+1, j+2] + right*p[i+1, j]
+ up*p[i+2, j+1] + (1.0 - down - left - right - up)*p[i+1, j+1]
                                                    domain ((0, 6), (0, 6))
```

The mass check finds the modeling bug: the interior-only update drops
all four edges, and 44% of the probability leaks in a single step.
`np.sum` over the grid lifts to the double sum
`Sum(p[j0, j1], (j1, 0, 7), (j0, 0, 7))`, so the conservation claim is
checked symbolically, not just numerically.
