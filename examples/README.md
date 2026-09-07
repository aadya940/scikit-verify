# Examples

Executed notebooks first, small runnable scripts after. The notebooks
render every formula in LaTeX; open them on GitHub and the math shows
inline.

## Notebooks

**[penalty_matrix_check.ipynb](penalty_matrix_check.ipynb)** is the
headline: a 20-page scipy derivation checked mechanically. The
implementation is traced, its entries come back as formulas in the
knots, and the identity with the defining integral is proved for every
knot vector of that shape at once.

**[branch_coverage_check.ipynb](branch_coverage_check.ipynb)** shows
the blind spot every test suite has and the fix: a spec that is right
on the branch your input took and wrong on the other one. The
explorer negates the branch conditions, asks Z3 for inputs on the
other side, and either visits every branch or proves the leftovers
unreachable. Ends with np.sinc's eight zero-pattern paths, found by
solving equalities no random input ever hits.

**[sklearn_bug_hunting.ipynb](sklearn_bug_hunting.ipynb)** traces
BayesianRidge's uncertainty on sklearn 1.8 and 1.9: the bug the 1.9
release fixed appears as the difference between two formulas.

**[demo.ipynb](demo.ipynb)** is the tour: trace a function, read the
formula, see the assumptions, watch a refusal.

**[ridgecv_decision.ipynb](ridgecv_decision.ipynb)** asks what
cross-validation actually decides. Tracing sklearn's RidgeClassifierCV
returns the alpha choice as two inequalities you can check against
sklearn's own error numbers, and shows the coefficients jump when a
sample sits exactly on the class boundary.

**[sabotage_audit.ipynb](sabotage_audit.ipynb)** takes sabotaged
research code (a precision function quietly counting the wrong thing,
an entropy formula from a published sabotage benchmark) and shows the
corruption in the formulas even where the outputs are identical.
