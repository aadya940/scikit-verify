"""Per-trace state: one object owns everything a trace accumulates.

Historically this state lived as module globals scattered across
``pair.py`` and ``instrument.py``, which made trace results depend on
what had been traced earlier in the process (the order-dependence bug
class). A :class:`TraceSession` gathers all of it behind one reset
boundary: ``to_sympy`` resets the active session before running, so
every trace starts from the same blank state by construction.

The session's collections are module-lifetime *objects* (the lists are
cleared in place, never rebound). Long-standing imports like
``from skverify.pair import _GUARDS`` therefore keep observing live
state through the same list they always did.
"""


class TraceSession:
    """All state accumulated while tracing one function call.

    Attributes
    ----------
    guards : list of sympy.Expr
        Branch conditions taken during the trace, in order. Harvested
        by ``to_sympy`` into ``.preconditions``.
    opaque : list of tuple
        One record per opaque compiled call: contract verdicts plus the
        atom's defining expression. Harvested into ``.unchecked``.
    loop_events : list of tuple
        ``(sequence_watermark, context_stack)`` snapshots emitted by
        instrumented for/while loops; ``derivation()`` reads these to
        group steps by the program's own loop structure.
    loop_stack : list of list
        The live loop-context stack while the trace runs.
    fn_twins : dict
        Instrumented-function cache, keyed by the original function
        object. Session-scoped so a twin built under one trace can
        never leak assumptions into another.
    class_twins : dict
        Instrumented-class cache, keyed by the original class.
    seq : int
        Monotone Pair-creation counter; loop events are positioned
        against it.
    """

    def __init__(self):
        self.guards = []
        self.opaque = []
        self.loop_events = []
        self.loop_stack = []
        self.fn_twins = {}
        self.class_twins = {}
        self.hashed = set()
        self.pending_mask_guards = {}
        self.loop_new = []
        self.loop_fold = {}
        self.seq = 0

    def reset(self):
        """Return the session to blank pre-trace state.

        Collections are cleared *in place*: external references to the
        lists (the historical module-global aliases) stay valid and
        observe the cleared state.
        """
        self.guards.clear()
        self.opaque.clear()
        self.loop_events.clear()
        self.loop_stack.clear()
        self.fn_twins.clear()
        self.class_twins.clear()
        self.hashed.clear()
        self.pending_mask_guards.clear()
        self.loop_new.clear()
        self.loop_fold.clear()
        self.seq = 0

    def next_seq(self):
        """Advance and return the Pair-creation counter."""
        self.seq += 1
        return self.seq


# The process-wide active session. A contextvar would allow parallel
# traces later; today a single module-level instance matches the
# library's actual concurrency story (one trace at a time) while
# giving every consumer one named owner for the state.
current = TraceSession()
