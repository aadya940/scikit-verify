# Coercion break-pass — failure list (lifting only, no fixes)
~200 adversarial cases, 6 rounds. Verified formulas track perturbed inputs in 24/25 spot checks.
Scripts: /tmp/break/round{1..6}.py (repro: run any with the scipy-dev python)

## A. SILENT-WRONG (value or formula lies — worst class)
1. np.add.at(y, [0,0], 1.0)   -> silently NO-OPS on a traced array (got 1.5, expect 3.5)
2. np.place(y, y<0, [0.0])    -> silently NO-OPS (got 3.25, expect 5.25)
3. np.nan_to_num(x/0.0)       -> inf NOT clamped: value lane -inf vs numpy's ±1.798e308
4. pandas Series(x).sum()     -> returns the full array instead of the scalar sum
5. int scalar input: to_sympy(lambda a: a+1, 3) -> formula "4", input symbol dropped
   (float scalar input correctly gives a+1; only python ints collapse)
6. nditer over object array   -> runs but formula constant

## B. DIED (should refuse loudly or work)
7.  round(pair, n) / round(pair)      TypeError: no __round__
8.  divmod(pair, 2.0)                 TypeError
9.  f"{pair:.2f}" / format(pair)      TypeError: unsupported format string
10. float(str(pair))                  ValueError (repr leaks "Pair(x[0])")
11. np.interp(t, xs, ys) traced xs    TypeError: cannot cast dtype('O')
12. np.corrcoef(x, x[::-1])           TypeError: scalar Pair not subscriptable
13. np.cov(x)                         ValueError: unpack
14. pair // 2.0 (floor div)           TypeError: no __floordiv__
15. np.select([conds],[choices])      TypeError: condlist must be bool ndarray
16. np.linalg.inv / eigvals on Pair   AttributeError: no __array_wrap__
17. np.putmask(pair, ...)             TypeError: first arg must be array
18. np.copyto(y, pair)                TypeError
19. pair.fill(2.0)                    AttributeError: no fill
20. pair.tolist()                     AttributeError: no tolist
21. scipy.stats.iqr(x)                ValueError: broadcast remap
22. varargs signature def f(*arrs)    TypeError: takes 1 arguments, got 2 (api wrapper)

## C. Loud refusals — correct behavior, listed for coverage review
- np.round (math-changing: right), np.piecewise (bool coercion), np.histogram(raw operand msg misleads: input WAS wrapped)
- ufunc.accumulate / .outer / .reduceat / out= / frexp / modf (2-output)
- astype(str), structured arrays (fancy indexing msg misleading)
- pandas DataFrame ops (data-dependent branch)
- chained float() in user code (right; message good)

## D. Notes / oddities (not failures, worth eyes)
- weighted_mean formula contains disclosed const_0 table -> substitution needs table values (by design)
- np.sort(x)[1] etc: all ordering-guard paths verified correct incl. preconditions gating
- empty array: Sum(a[j], (j,0,-1)) — correct empty-sum convention, prints oddly
- f(x, x) same array twice -> named a and b independently (defensible; loses aliasing info)
- kwarg scalars become symbols (scale*a[i]) — nice, but means kwargs are never constant-folded
