"""Declared roles for call sites: the instrumentation registries.

Each set or mapping is a curated declaration, one line per fact, each
reviewable on its own: which names allocate, which are math-neutral,
which compiled boundaries seal into atoms, which return through
out-parameters. ``skverify.dialect`` is the public API for extending
these.
"""

ALLOC = {"zeros", "empty", "ones", "full"}
NEUTRAL = {
    "asarray",
    "asanyarray",
    "ascontiguousarray",
    "asfortranarray",
    "asarray_chkfinite",
}
OPAQUE_CALLABLES = {
    "solve_banded",
    "solveh_banded",
    "cho_solve",
    "design_matrix",
    "gbsv",
    "data_matrix",
    "fpback",
    "evaluate_all_bspl",
    "solve",
    "lstsq",
    "_lstsq",
    "svd",
    "pinv",
    "r2c",
    "c2c",
    "c2r",
    "rfft",
    "irfft",
    "fft",
    "ifft",
    "rfftn",
    "fftn",
}
NEUTRAL_METHODS = {"toarray", "astype", "copy", "view", "type"}
CONCRETE = {"isfinite", "isnan", "isinf"}  # validation checks, not math
SCALARIZE = {"float", "int"}  # scalar coercion at a compiled boundary
# compiled lookups whose result is bookkeeping (an interval index),
# not mathematics: run on values, return the plain result
CONCRETE_CALLABLES = {"find_interval"}
# compiled routines that RETURN through array out-parameters (scipy's
# Cython convention); value = argument positions of the out arrays
OPAQUE_OUT = {"_coloc": (3,), "qr_reduce": (0, 3)}

_SITES = []

