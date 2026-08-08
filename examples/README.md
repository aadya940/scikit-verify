# Examples
 
Small, self-contained numerical kernels traced with `to_sympy`. Each file
runs directly and prints the recovered formulas alongside the ordinary
numerical results.
 
```bash
python markov_birth_death.py
```
 
## markov_birth_death.py
 
A birth–death Markov chain on states `0..n-1`, written without matrices —
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
