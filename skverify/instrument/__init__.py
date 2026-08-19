"""Trace-time instrumentation, as a package.

The subsystems:

``registries``
    The declared roles: allocations, math-neutral calls, opaque
    boundaries, out-parameter conventions.
``runtime``
    The traced twins that replace math-neutral calls at run time.
``triage``
    The runtime dispatch policy: every call in instrumented code is
    classified (pass through, twin, seal as atom, concretize) by one
    ordered rule list.
``rewriter``
    The AST visitor that produces a semantically identical copy of a
    function with rewritten call sites.
``twins``
    Building instrumented copies of functions, classes and bound
    methods, with caches owned by the active TraceSession.

This ``__init__`` re-exports every name the historical flat module
exposed, so ``from skverify.instrument import X`` keeps working for
all X.
"""

from .registries import *  # noqa: F401,F403
from .runtime import *  # noqa: F401,F403
from .triage import *  # noqa: F401,F403
from .rewriter import *  # noqa: F401,F403
from .twins import *  # noqa: F401,F403

from .registries import (  # noqa: F401
    ALLOC,
    NEUTRAL,
    OPAQUE_CALLABLES,
    NEUTRAL_METHODS,
    CONCRETE,
    SCALARIZE,
    CONCRETE_CALLABLES,
    OPAQUE_OUT,
)
from .twins import _CLASS_TWINS, _FN_MEMO, _instrument, _instrument_class, instrument  # noqa: F401
from .triage import _skv_maybe, _twinnable, runtime_twin  # noqa: F401
from .rewriter import _Rewriter  # noqa: F401
