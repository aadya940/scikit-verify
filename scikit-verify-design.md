# scikit-verify: Design Document

`scikit-verify` executes NumPy functions to extract their computed formulas as SymPy expressions. It supports complex control flow, loop accumulation, compiled library calls (e.g., LAPACK), and data-dependent branching by intercepting standard NumPy execution. 

This document details the core architectural subsystems enabling this capability, the mechanisms used to bypass standard limitations, and the extension interfaces.

## 1. Dual-State Tracing (The Two-Lane Value)
Extracting both a numerical result and a symbolic formula requires parallel evaluation. 

* **Mechanism:** Every traced value is a `Pair` containing a concrete NumPy array (or scalar) and a SymPy expression. Operations leveraging NumPy’s `__array_ufunc__` and `__array_function__` protocols are intercepted. The numerical operation executes normally while simultaneously building the matching symbolic formula.
* **Routing:** Operations are mapped via `UFUNC_TABLE` (element-wise mappings like `np.exp` → `sympy.exp`) and `FUNCTION_TABLE` (structural mappings mapping dimensions and reductions).
* **Extension:** Element-wise functions are registered via `register_ufunc`. Structural functions require custom constructors via `register_function`.
```python
from skverify.dialect import register_ufunc
register_ufunc(scipy.special.ndtr, lambda z: (1 + sympy.erf(z / sympy.sqrt(2))) / 2)
```

## 2. Control Flow and Preconditions
Data-dependent branches (e.g., `np.median`) must record the conditions under which a specific formula was chosen.

* **Mechanism:** The `Pair.__bool__` method is overridden. When evaluated in control flow (`if`, `while`), it returns the concrete truth value to execute the correct branch, while appending the symbolic relation (e.g., `a[0] <= a[2]`) to the session's guard list.
* **Output:** The final formula includes `.preconditions`, explicitly defining the domain constraints under which the generated formula is valid. Ambiguous array-wide comparisons trigger an immediate refusal.

## 3. AST Rewriting for Undispatchable Calls
Standard dispatch fails for initialization functions lacking traced operands (e.g., `np.zeros()`).

* **Mechanism:** If standard execution fails, the system parses the function's AST, rewrites targeted call sites, and re-executes.
* **Registries:** 
  * `ALLOC`: Replaces allocators with traced equivalents that seed values and formulas.
  * `NEUTRAL`: Passes `Pair` objects through unchanged.
* **Extension:** Register new intercept targets to define execution behavior for undispatchable calls.
```python
from skverify.dialect import register_neutral, register_opaque
register_neutral("as_strided")   
register_opaque("dgesdd")        
```

## 4. Opaque Call Isolation and Contract Checking
Compiled extensions (Cython/Fortran/LAPACK) lack Python source code for AST rewriting.

* **Mechanism:** Unreachable routines are sealed into fresh indexed SymPy symbols derived from their operands, ensuring the result is explicitly disclosed as an opaque routine rather than silently ignored.
* **Verification:** `contracts.py` defines numerical checks for known mathematical routines. For example, `solve_banded` is numerically verified against `A @ x == b` on a copy of the inputs. Results are flagged in `.unchecked` as `ok`, `failed`, or `unknown`.
* **Extension:** Implement numerical verifications for compiled functions via `register_contract`.
```python
from skverify.dialect import register_contract
register_contract(
    "cho_solve",
    law="A @ x == b for the Cholesky-factored A",
    residual=lambda args, result: "ok" if good_enough(args, result) else "failed",
)
```

## 5. Symbolic Loop Folding
Unrolling long loops (e.g., iterative optimization) symbolically results in non-termination and formula explosion.

* **Mechanism:** For loops exceeding `FOLD_START` iterations, a recurrence state machine is engaged:
  1. **Plant:** Inserts symbolic dummies to track state carried between iterations.
  2. **Probe:** Anti-unifies subsequent iterations into a single parameterized template.
  3. **Verify:** Checks loop body signatures (operation sequence, guards, opaque calls) at every step to ensure matching dataflow.
* **Output:** Repetitive execution is collapsed into a SymPy `Iterate` function, representing bounded nonlinear recurrences checkable via substitution. If convergence checks diverge, the loop safely drops back to exact eager tracing.

## 6. Mask and Gather Provenance
Boolean masks collapse symbolic conditions into plain integers during reductions, destroying trace history.

* **Mechanism:** Array comparisons generate concrete boolean arrays tagged as `MaskedElems`, retaining the per-position SymPy condition. Indexing and gathering operations carry this provenance forward.
* **Output:** Reductions over masked data fuse directly into a SymPy `Sum` over a `Piecewise` function. This preserves exact conditional logic without generating unscalable per-element preconditions. 

## 7. Runtime Validation of Constant Outputs
Array-API compatibility layers may inadvertently strip traced arrays into plain floats mid-computation, yielding an incorrect constant formula.

* **Mechanism:** The `_refuse_lost_influence` check runs prior to returning output. If the derived formula is constant but the inputs were symbolic, the system reruns the untraced function on perturbed numerical inputs.
* **Enforcement:** If the numerical output diverges while the formula remains constant, tracing is aborted. A constant is only certified if the untraced rerun also confirms data independence.

## 8. External Specification and Verification
Verifying a traced function against its own trace produces false positives.

* **Mechanism:** `check_formula` evaluates the trace against an independent specification using a graduated verification ladder:
  1. Exact symbolic expansion.
  2. Cancellation and algebraic combination.
  3. Time-bounded simplification (prevents CI hangs).
  4. Numeric arbitration at exact rational sample points.
* **Extension:** Use `@specifies` for closed-form mathematical matching or `@specifies.property` for behavioral invariants.
```python
from skverify.testing import specifies

@specifies.property(lambda F: sympy.Eq(sum(F.subs(i, k) for k in range(N)), 0))
def test_zero_sum_property():
    pass
```

## 9. Session State Management
Global state persistence causes test pollution and nondeterministic evaluation. 

* **Mechanism:** `TraceSession` encapsulates all trace-scoped state—including active guards, opaque records, loop progress, and AST caches. A fresh session object is initialized for every `to_sympy` call, ensuring strict isolation.
