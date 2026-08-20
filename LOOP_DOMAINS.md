# Loops as domains

A loop is an axis in time. Array axes are already domains: `x[i]` with
bounds, bound by `Sum(..., (i, 0, n-1))`. An iteration index `k` is the
same object, bound by a recurrence instead of a reduction. Today the
tracer unrolls loops into the live formula, so iterative solvers
(BayesianRidge: 300 iterations, each formula containing the last)
snowball and never finish. Folding exists (`_fold_runs`,
`derivation()`) but runs post-hoc on `.steps` -- downstream of the
blowup. This note moves the fold into the formula lane itself.

Everything below is explicitly sympy: `subs`, `doit`, `lambdify`,
`free_symbols` work on the certificate with no side-channel records.
All claims execution-verified on sympy 1.14.

## The three tiers

| loop shape | sympy object | verified behavior |
|---|---|---|
| accumulator `t += x[k]` | `Sum` / `Product` | already emitted for np.sum; hand loops should reach it |
| linear scalar recurrence | `rsolve` -> closed expression | `rsolve(C(k+1)-2C(k)-1, C(k), {C(0):1})` -> `2**(k+1)-1` |
| linear VECTOR recurrence | `MatPow`: `C[k] = A**k * C[0]` | `(A**k).doit()` diagonalizes symbolically; `subs(k,3)` exact |
| general scalar recurrence | `RecursiveSeq` | Basic; `subs` traverses; `.coeff(k)` unrolls lazily |
| general vector / coupled | custom `Iterate` Function (below) | held form, lazy unroll, Tuple state |

## Verified capabilities and limits

RecursiveSeq (sympy.series.sequences):
- IS `Basic`; `free_symbols` correct; symbolic parameters in the body
  survive (`coeff(3)` of `y=2y+a` -> `7*a + 8`); `subs` composes.
- Scalar ONLY: Matrix and MatrixSymbol terms fail (AttributeError /
  ShapeError). Tuple packing fails. A second Function in the body
  stays unresolved (`z(0), z(1)` free) -- no coupled systems.

MatPow:
- `A**k` with symbolic integer `k` is native, `.doit()` produces the
  closed form via diagonalization, exact under `subs(k, n)`.
- Covers every CONSTANT-coefficient linear vector recurrence.

Iterate (ours, ~10 lines):
```python
class Iterate(sympy.Function):
    """Iterate(step, init, k): k-fold application of step to init."""
    @classmethod
    def eval(cls, step, init, k):
        if k.is_Integer and k >= 0:
            out = init
            for _ in range(int(k)):
                out = step(*out) if isinstance(out, sympy.Tuple) else step(out)
            return out
```
- Held while `k` is symbolic: `Iterate(Lambda(c, 2c+a), 1, K)`,
  free symbols `{a, K}`. `subs(K, 3)` unrolls on demand -> `7*a + 8`;
  after full substitution it is a number (lambdify-able).
- Vector and COUPLED state as one `Tuple`:
  `Iterate(Lambda((c1,c2), Tuple(c1+c2, c1*a)), Tuple(1,2), K)`
  -> `subs(K,2)` -> `(a+3, 3*a)`. This is the BayesianRidge shape
  (coef and alpha updating each other).
- A subclass of `sympy.Function` IS sympy: printing, traversal,
  substitution all come from Basic. No metadata channel.

## Trace-time mechanics

The hooks exist. The rewriter already tags every loop
(`__skv_loop_iter__` / `__skv_loop_end__`, `session.loop_events`), so
Pairs are attributable to (loop, iteration) DURING the trace.

1. Run iteration 0 and 1 concretely, formulas eager (today's path).
2. At iteration 1's end, attempt the fold: do iteration 1's formulas
   equal iteration 0's under a template with the loop-carried state
   replaced by a state symbol? This is `_generalize`'s check, applied
   live. Multiple carried variables pack into one Tuple state.
3. Fold succeeds: replace the carried Pairs' formulas with the tier
   object (Sum/rsolve/MatPow/RecursiveSeq/Iterate) and run remaining
   iterations on the VALUE LANE ONLY, verifying each iteration's
   concrete values against the template (the per-iteration residual --
   same role contracts play for atoms). Formula size is now
   O(template), independent of iteration count.
4. Fold fails (body not one template -- data-dependent branch inside):
   keep today's unrolling, with a formula-size budget that refuses
   loudly instead of hanging.

Data-dependent exits (`while not converged`) are already guards: the
stopping condition's `__bool__` records the path. `K = 17` enters the
formula as a concrete bound with the convergence guard in
preconditions, exactly like searchsorted's counting bound.

## Verification story

- Tier objects evaluate: `.coeff(k)` / `subs(K, k)` reproduce the
  unrolled formula for any k, so the two-lane fuzzer applies directly.
- Per-iteration residual at trace time: template instantiated with
  iteration k's concrete state must equal iteration k+1's concrete
  state. 300 cheap numeric checks replace one 300-deep expression.

## Refusals (loud, by construction)

- Loop body changes shape between iterations (branch taken differently)
  and sizes exceed the budget -> "iterative body is not one template".
- `rsolve` fails and the recurrence is nonlinear scalar -> RecursiveSeq
  (still exact); vector -> Iterate (still exact). Refusal is only for
  non-foldable bodies, never for "no closed form" -- a held recurrence
  IS an exact formula.
