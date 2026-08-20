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
    # identity on valid input like the asarray family; the dtype/order
    # requirements are memory bookkeeping, not math
    "require",
}

# neutral ONLY as an attribute call (np.array, xp.array): "array" is a
# common VARIABLE name, so bare-name matching or reference swapping
# would hijack locals. Adding "array" makes inputs survive sklearn's
# validation (y stops becoming an anonymous array_0 atom) but opens a
# cascade of numpy-internals walls on the deeper-traced values
# (average with out=, umath reductions on bags) -- its own chase.
# Empty until that chase happens.
NEUTRAL_ATTR_ONLY = set()
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
# numpy set-routines: which values exist / where they appear are facts
# about THIS trace (label inventories), not mathematics -- run on
# concrete values wherever the call resolves, body or dispatcher
CONCRETE_BY_NAME = {"in1d", "_in1d", "setdiff1d", "union1d", "intersect1d", "isin", "unique"}
# compiled routines that RETURN through array out-parameters (scipy's
# Cython convention); value = argument positions of the out arrays
OPAQUE_OUT = {"_coloc": (3,), "qr_reduce": (0, 3)}

_SITES = []

