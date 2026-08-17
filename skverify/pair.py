import numpy as np
import sympy

from .registry import (
    UFUNC_TABLE,
    FUNCTION_TABLE,
)
from .helpers import (
    axis_idx,
    _AXIS_SYMBOLS,
    normalize_key,
)

IDX = axis_idx(0)  # `i`

_GUARDS = []  # branch conditions taken during a trace; harvested by to_sympy


class Pair:
    """Convert math and array style operations to SymPy
    expressions.
    """

    def __init__(self, value, formula, domain=None, steps=None):
        self.value = value  # the real ndarray/scalar, what executes
        self.formula = formula  # the sympy Expr, what it means
        # the derivation: every intermediate formula that led here, in
        # execution order, ending with this one. Two runs taking different
        # branches produce different steps, because different ops ran.
        self.steps = (steps or []) + [formula]

        if domain is not None and not isinstance(domain[0], tuple):
            domain = (domain,)

        self._axis_bounds = domain  # for ndarray

    @staticmethod
    def _steps_of(*operands):
        collected = []
        for x in operands:
            if isinstance(x, Pair):
                collected.extend(x.steps)
        return collected

    @staticmethod
    def _formula_of(x):
        if isinstance(x, Pair):
            return x.formula
        if isinstance(x, np.ndarray):
            vals = np.unique(x)
            if len(vals) == 1:  # uniform: zeros, ones, full
                return sympy.sympify(vals.item())  # constant field, clean
            raise NotImplementedError(
                "raw non-uniform ndarray operand, wrap it: Pair.array(name, x)"
            )
        return sympy.sympify(x)

    @staticmethod
    def _domain_of(x):
        if isinstance(x, Pair):
            return x._axis_bounds  # canonical, always
        if isinstance(x, np.ndarray):
            return tuple((0, s) for s in x.shape)  # ndim > 1 handled too
        return None

    @property
    def domain(self):
        if self._axis_bounds is None:
            return None
        if len(self._axis_bounds) == 1:
            return self._axis_bounds[0]  # the 60's flat face
        return self._axis_bounds

    @staticmethod
    def _merge_domains(*domains):
        """Merge canonical axis tuples. None = scalar. Non-None domains
        must agree per axis; differing ranks are refused."""
        result = None
        for d in domains:
            if d is None:
                continue
            if result is None:
                result = d
                continue
            if len(d) != len(result):
                # (4, 7) meeting (7,): line the short one up against the
                # END of the long one and check those axes agree
                short, long = sorted((d, result), key=len)
                if short != long[len(long) - len(short) :]:
                    raise ValueError(f"cannot broadcast {short} with {long}")
                result = long
                continue
            for ax, (result_item, d_item) in enumerate(zip(result, d)):
                if d_item != result_item:
                    raise ValueError(
                        f"domain mismatch at axis {ax}: {result_item} vs {d_item}"
                    )
        return result

    @staticmethod
    def _binary(inputs, fwd, rev, self):
        a, b = inputs
        return fwd(a, b) if a is self else rev(b, a)

    @staticmethod
    def _value_of(x):
        return x.value if isinstance(x, Pair) else x

    @staticmethod
    def _shift_axes(formula, bounds, ndim):
        """Rename a lower-rank operand's letters to the trailing axes.

        v[i] (1 axis) meeting a 2-axis result: v runs along the last
        axis, so v[i] -> v[j].
        """
        if bounds is None:
            return formula  # scalar: no letters to move
        offset = ndim - len(bounds)
        if offset == 0:
            return formula
        index_map = {axis_idx(a): axis_idx(a + offset) for a in range(len(bounds))}
        return formula.subs(index_map, simultaneous=True)

    @staticmethod
    def _is_condition(formula):
        # NOT isinstance(..., Boolean): sympy Symbols subclass Boolean
        return isinstance(
            formula,
            (
                sympy.core.relational.Relational,
                sympy.logic.boolalg.BooleanFunction,
                sympy.logic.boolalg.BooleanAtom,
            ),
        )

    @staticmethod
    def _bridge_numeric(formula):
        """Boolean entering arithmetic means 0/1, as in numpy:
        (u > 0).sum() counts, mask * u selects."""
        if Pair._is_condition(formula):
            if _piecewise_under_sum(formula):
                # sympy's Piecewise ctor hoists conditions through Sum
                # bounds (even with evaluate=False), leaking the bound
                # index -- a wrong formula. Refuse until fixed upstream.
                raise NotImplementedError(
                    "condition over a Sum of Piecewise: sympy rewrites "
                    "this incorrectly; restructure or use .value"
                )
            return sympy.Piecewise((1, formula), (0, True))
        return formula

    @staticmethod
    def _broadcast(a, b, bridge=True):
        """Formulas + merged bounds for a binary op.

        u (4x7) + v (7):  u[i, j] + v[j],  bounds ((0, 4), (0, 7))
        bridge=False (mask algebra, comparisons) keeps Booleans raw.
        """
        bounds_a, bounds_b = Pair._domain_of(a), Pair._domain_of(b)
        merged = Pair._merge_domains(bounds_a, bounds_b)
        formula_a = Pair._formula_of(a)
        formula_b = Pair._formula_of(b)
        if merged is not None:
            formula_a = Pair._shift_axes(formula_a, bounds_a, len(merged))
            formula_b = Pair._shift_axes(formula_b, bounds_b, len(merged))
        if bridge:
            formula_a = Pair._bridge_numeric(formula_a)
            formula_b = Pair._bridge_numeric(formula_b)
        return formula_a, formula_b, merged

    def _remap(self, value, index_map, axis_bounds):
        """Implements infrastructure for array methods where
        indexes are remapped from old to new using `index_map`.

        For example:
        index_map = {i: j, j: i} for a transpose operation.

        _axis_bounds are reversed in order the transpose operation for example.
        """
        for old_sym, expr in index_map.items():
            expr = sympy.sympify(expr)
            if not expr.free_symbols <= set(_AXIS_SYMBOLS):
                raise NotImplementedError(
                    f"index map {old_sym} -> {expr} is not an index expression"
                )
            if not expr.is_integer and not all(
                expr.diff(s).is_Integer for s in expr.free_symbols
            ):
                raise NotImplementedError(
                    f"index map {old_sym} -> {expr} is not affine"
                )
        formula = self.formula.subs(index_map, simultaneous=True)
        return Pair(value, formula, domain=axis_bounds or None, steps=self.steps)

    def _getitem_nd(self, key):
        # u[1:, 2] on a 4x7 array -> entries ((1, 4), 2)
        entries = normalize_key(key, tuple(hi - lo for lo, hi in self._axis_bounds))
        index_map, new_bounds = {}, []
        for ax, entry in enumerate(entries):
            sym = axis_idx(ax)
            if isinstance(entry, tuple):
                # u[1:]   : u[i] -> u[i + 1]      4 rows -> 3 rows
                # u[::2]  : u[i] -> u[2*i]        5 -> 3
                # u[::-1] : u[i] -> u[4 - i]      5 -> 5
                # Survivors take the letter of their RESULT position, so that
                # letter k always means axis k:  u[2, 1:] -> u[2, i + 1], not j
                start, stop, step = entry
                new_sym = axis_idx(len(new_bounds))
                index_map[sym] = step * new_sym + start
                count = max(0, -(-(stop - start) // step))  # ceil for +/- step
                new_bounds.append((0, count))
            else:
                # u[2] : u[i, j] -> u[2, i], row axis gone, survivor renamed
                index_map[sym] = sympy.Integer(entry)
        value = self.value[key]
        if isinstance(value, np.ndarray):
            value = value.copy()  # slices are value-semantic: no view aliasing
        return self._remap(
            value=value,  # raw key: numpy interprets it independently
            index_map=index_map,
            axis_bounds=tuple(new_bounds),  # u[2, 3] -> (), remap makes it scalar
        )

    def __setitem__(self, key, val):
        # u[2:5] = v: the formula becomes a scatter, old rule outside the
        # written region, the value's rule (re-indexed) inside it
        if self._axis_bounds is None:
            raise TypeError("scalar Pair does not support item assignment")
        val_formula = Pair._formula_of(val)
        if isinstance(key, Pair):
            if Pair._is_condition(key.formula):
                if isinstance(val, (Pair, np.ndarray)) and np.ndim(
                    Pair._value_of(val)
                ):
                    raise NotImplementedError(
                        "masked assignment with an array value is data-dependent"
                    )
                self.value[key.value] = Pair._value_of(val)
                self.formula = sympy.Piecewise(
                    (val_formula, key.formula), (self.formula, True)
                )
                self.steps = Pair._steps_of(self, key, val) + [self.formula]
                return
            raise NotImplementedError("only boolean Pair keys are supported")

        lengths = tuple(hi - lo for lo, hi in self._axis_bounds)
        entries = normalize_key(key, lengths)
        condition = []
        val_map = {}
        val_rank = 0
        for ax, entry in enumerate(entries):
            sym = axis_idx(ax)
            if isinstance(entry, tuple):
                start, stop, step = entry
                if step != 1:
                    raise NotImplementedError(
                        "strided assignment not supported yet"
                    )
                if not (start == 0 and stop == lengths[ax]):
                    condition.append(sympy.Ge(sym, start))
                    condition.append(sympy.Lt(sym, stop))
                val_map[axis_idx(val_rank)] = sym - start
                val_rank += 1
            else:
                condition.append(sympy.Eq(sym, entry))
        if isinstance(val, Pair) and val._axis_bounds is not None:
            if len(val._axis_bounds) != val_rank:
                raise NotImplementedError(
                    "assigned value rank must match the sliced region"
                )
            val_formula = val_formula.subs(
                {k: v for k, v in val_map.items()}, simultaneous=True
            )
        self.value[key] = Pair._value_of(val)
        if condition:
            self.formula = sympy.Piecewise(
                (val_formula, sympy.And(*condition)), (self.formula, True)
            )
        else:
            self.formula = val_formula
        self.steps = Pair._steps_of(self, val) + [self.formula]

    def transpose(self, axes=None):
        # u (4x7), u.T: u[i, j] -> u[j, i], bounds ((0,4),(0,7)) -> ((0,7),(0,4))
        # axes=(2,0,1) on 3-D: result position k reads old axis axes[k]
        if self._axis_bounds is None or len(self._axis_bounds) == 1:
            return self  # scalars and 1-D: transpose is a no-op, like numpy
        ndim = len(self._axis_bounds)
        if axes is None:
            axes = tuple(reversed(range(ndim)))
        index_map = {
            axis_idx(old_ax): axis_idx(new_pos) for new_pos, old_ax in enumerate(axes)
        }
        return self._remap(
            value=self.value.transpose(axes),
            index_map=index_map,
            axis_bounds=tuple(self._axis_bounds[old_ax] for old_ax in axes),
        )

    @property
    def T(self):
        return self.transpose()

    def __add__(self, other):
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value + Pair._value_of(other),
            formula=mine + theirs,  # sympy dunder does the rest
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __radd__(self, other):  # handles  2 + u
        return self.__add__(other)

    def __sub__(self, other):
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value - Pair._value_of(other),
            formula=mine - theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __rsub__(self, other):  # handles  2 - u   (order matters!)
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=Pair._value_of(other) - self.value,
            formula=theirs - mine,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __mul__(self, other):
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value * Pair._value_of(other),
            formula=mine * theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    __rmul__ = __mul__

    def __abs__(self):
        return Pair(
            abs(self.value),
            sympy.Abs(self.formula),
            domain=self._axis_bounds,
            steps=self.steps,
        )

    def __bool__(self):
        # Branch capture: a scalar condition (if x > 0:) is decided by the
        # concrete lane and RECORDED -- the branch taken becomes a hypothesis
        # on the certificate. to_sympy harvests _GUARDS into .preconditions.
        if Pair._is_condition(self.formula) and np.ndim(self.value) == 0:
            outcome = bool(self.value)
            _GUARDS.append(self.formula if outcome else sympy.Not(self.formula))
            return outcome
        # arrays: ambiguous, like numpy's own error -- use .all()/.any().
        # non-conditions: silent truthiness on math is never meaningful.
        raise NotImplementedError(
            "data-dependent branch on a traced value; "
            "for masks use .all()/.any(), for combining use & | ~"
        )

    # bare-number conversions discard the formula: the value keeps computing
    # but the trace silently dies. Use .value for a deliberate exit.
    def __float__(self):
        raise NotImplementedError(
            "float() on a traced value discards the formula; use .value"
        )

    def __int__(self):
        raise NotImplementedError(
            "int() on a traced value discards the formula; use .value"
        )

    def __complex__(self):
        raise NotImplementedError(
            "complex() on a traced value discards the formula; use .value"
        )

    # facts about the concrete lane; some library bodies read these
    # before doing any math
    @property
    def ndim(self):
        return np.ndim(self.value)

    @property
    def shape(self):
        return np.shape(self.value)

    @property
    def dtype(self):
        # object, deliberately: numpy cast branches like ret.dtype.type(x)
        # become passthroughs instead of float(Pair) deaths. The concrete
        # lane's real dtype is np.asarray(self.value).dtype if ever needed.
        return np.dtype(object)

    def __truediv__(self, other):  # self / other
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value / Pair._value_of(other),
            formula=mine / theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __rtruediv__(self, other):  # other / self
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=Pair._value_of(other) / self.value,
            formula=theirs / mine,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __pow__(self, other):  # self ** other
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=self.value ** Pair._value_of(other),
            formula=mine**theirs,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __rpow__(self, other):  # other ** self
        mine, theirs, merged = Pair._broadcast(self, other)
        return Pair(
            value=Pair._value_of(other) ** self.value,
            formula=theirs**mine,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    def __neg__(self):  # -self
        return Pair(
            -self.value, -self.formula, domain=self._axis_bounds, steps=self.steps
        )

    @classmethod
    def array(cls, name, value):
        value = np.asarray(value)
        if value.ndim > 5:
            raise NotImplementedError(
                "arrays beyond 5D are not supported.",
            )
        idxs = tuple([axis_idx(idx) for idx in range(value.ndim)])
        formula = sympy.IndexedBase(name)[idxs]
        return cls(value, formula, domain=tuple((0, s) for s in value.shape))

    def __len__(self):
        n = self.value.shape[0]  # truth: the real array
        lo, hi = self._axis_bounds[0]
        assert n == hi - lo, "domain drifted from value"
        return n

    def __getitem__(self, key):
        """Slicing and integer indexing; 1-D is just the N=1 case."""
        if self._axis_bounds is None:
            raise TypeError("scalar Pair is not subscriptable")
        return self._getitem_nd(key)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if out is not None:
            raise NotImplementedError("out= is not supported (mutation)")
        if method != "__call__" or kwargs.get("out") is not None:
            raise NotImplementedError(f"{ufunc.__name__}.{method} not supported")

        if ufunc is np.add:
            return Pair._binary(inputs, Pair.__add__, Pair.__radd__, self)
        if ufunc is np.subtract:
            return Pair._binary(inputs, Pair.__sub__, Pair.__rsub__, self)
        if ufunc is np.multiply:
            return Pair._binary(inputs, Pair.__mul__, Pair.__rmul__, self)
        if ufunc is np.true_divide:
            return Pair._binary(inputs, Pair.__truediv__, Pair.__rtruediv__, self)
        if ufunc is np.power:
            return Pair._binary(inputs, Pair.__pow__, Pair.__rpow__, self)
        if ufunc is np.negative:
            return -inputs[0]

        target = UFUNC_TABLE.get(ufunc)
        if target is None:
            raise NotImplementedError(f"ufunc {ufunc.__name__} not mapped")

        values = [Pair._value_of(x) for x in inputs]
        domain = Pair._merge_domains(*(Pair._domain_of(x) for x in inputs))
        formulas = [
            Pair._shift_axes(
                Pair._formula_of(x),
                Pair._domain_of(x),
                0 if domain is None else len(domain),
            )
            for x in inputs
        ]

        return Pair(
            ufunc(*values),
            target(*formulas),
            domain,
            steps=Pair._steps_of(*inputs),
        )

    def __array_function__(self, func, types, args, kwargs):
        fn = FUNCTION_TABLE.get(func)
        if fn is not None:
            return fn(*args, **kwargs)  # curated: indexed formulas
        inner = getattr(func, "__wrapped__", None)
        if inner is not None:
            # pure-Python numpy: run its real body on the Pairs; slices and
            # arithmetic inside dispatch back here, formulas unrolled per element
            return inner(*args, **kwargs)
        raise NotImplementedError(
            f"np.{func.__name__} is compiled; needs a contract",
        )


# relational & mask layer
# u > 0        -> Pair([True,...], u[i] > 0)     one rule, letters/domains reused
# (a) & (b)    -> And(a, b)                       raw Booleans, no bridge
# (u > 0).sum()-> Sum(Piecewise((1, cond), (0, True)))   via the bridge

_RELATIONALS = {
    "__lt__": (np.less, sympy.Lt),
    "__le__": (np.less_equal, sympy.Le),
    "__gt__": (np.greater, sympy.Gt),
    "__ge__": (np.greater_equal, sympy.Ge),
    "__eq__": (np.equal, sympy.Eq),
    "__ne__": (np.not_equal, sympy.Ne),
}

_MASK_OPS = {
    "__and__": (np.bitwise_and, sympy.And),
    "__or__": (np.bitwise_or, sympy.Or),
    "__xor__": (np.bitwise_xor, sympy.Xor),
}


def _piecewise_under_sum(expr):
    return isinstance(expr, sympy.Basic) and any(
        s.function.has(sympy.Piecewise) for s in expr.atoms(sympy.Sum)
    )


def _make_binary(np_op, sy_op, bridge):
    def op(self, other):
        mine, theirs, merged = Pair._broadcast(self, other, bridge=bridge)
        # sympy's relational constructor hoists Piecewise conditions OUT
        # of a Sum, leaking the bound index. Build unevaluated when that
        # hazard is present; the formula is the honest raw relation.
        if bridge and (_piecewise_under_sum(mine) or _piecewise_under_sum(theirs)):
            formula = sy_op(mine, theirs, evaluate=False)
        else:
            formula = sy_op(mine, theirs)
        return Pair(
            np_op(self.value, Pair._value_of(other)),
            formula,
            domain=merged,
            steps=Pair._steps_of(self, other),
        )

    return op


for _name, (_np_op, _sy_op) in _RELATIONALS.items():
    setattr(Pair, _name, _make_binary(_np_op, _sy_op, bridge=True))
for _name, (_np_op, _sy_op) in _MASK_OPS.items():
    setattr(Pair, _name, _make_binary(_np_op, _sy_op, bridge=False))
    setattr(Pair, "__r" + _name[2:], _make_binary(_np_op, _sy_op, bridge=False))


def _invert(self):
    return Pair(
        np.bitwise_not(self.value),
        sympy.Not(self.formula),
        domain=self._axis_bounds,
        steps=self.steps,
    )


Pair.__invert__ = _invert
Pair.all = lambda self, axis=None: np.all(self, axis=axis)
Pair.any = lambda self, axis=None: np.any(self, axis=axis)
Pair.__hash__ = None  # elementwise __eq__ (numpy semantics): unhashable, like ndarray
