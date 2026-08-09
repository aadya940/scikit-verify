import numpy as np
import sympy

from .registry import (
    UFUNC_TABLE,
    FUNCTION_TABLE,
)
from .helpers import (
    axis_idx,
    normalize_slice,
    _AXIS_SYMBOLS,
    normalize_key,
)

IDX = axis_idx(0)  # `i`


class Pair:
    """Convert math and array style operations to SymPy
    expressions.
    """

    def __init__(self, value, formula, domain=None):
        self.value = value  # the real ndarray/scalar, what executes
        self.formula = formula  # the sympy Expr, what it means

        if domain is not None and not isinstance(domain[0], tuple):
            domain = (domain,)

        self._axis_bounds = domain  # for ndarray

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
                raise ValueError(
                    f"rank mismatch: {len(result)}D vs {len(d)}D "
                    "(broadcast not yet supported).",
                )
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
                expr.diff(s) in (0, 1, -1) for s in expr.free_symbols
            ):
                raise NotImplementedError(
                    f"index map {old_sym} -> {expr} is not affine (step-1)"
                )
        formula = self.formula.subs(index_map, simultaneous=True)
        return Pair(value, formula, domain=axis_bounds or None)

    def _getitem_nd(self, key):
        # u[1:, 2] on a 4x7 array -> entries ((1, 4), 2)
        entries = normalize_key(key, tuple(hi - lo for lo, hi in self._axis_bounds))
        index_map, new_bounds = {}, []
        for ax, entry in enumerate(entries):
            sym = axis_idx(ax)
            if isinstance(entry, tuple):
                # u[1:] : u[i] -> u[i + 1], 4 rows -> 3 rows
                start, stop = entry
                index_map[sym] = sym + start
                new_bounds.append((0, stop - start))
            else:
                # u[2] : u[i, j] -> u[2, j], row axis gone, `j` survives as-is
                index_map[sym] = sympy.Integer(entry)
        return self._remap(
            value=self.value[key],  # raw key: numpy interprets it independently
            index_map=index_map,
            axis_bounds=tuple(new_bounds),  # u[2, 3] -> (), remap makes it scalar
        )

    def __add__(self, other):
        return Pair(
            value=self.value + Pair._value_of(other),
            formula=self.formula
            + Pair._formula_of(other),  # sympy dunder does the rest
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    def __radd__(self, other):  # handles  2 + u
        return self.__add__(other)

    def __sub__(self, other):
        return Pair(
            value=self.value - Pair._value_of(other),
            formula=self.formula - Pair._formula_of(other),
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    def __rsub__(self, other):  # handles  2 - u   (order matters!)
        return Pair(
            value=Pair._value_of(other) - self.value,
            formula=Pair._formula_of(other) - self.formula,
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    def __mul__(self, other):
        return Pair(
            value=self.value * Pair._value_of(other),
            formula=self.formula * Pair._formula_of(other),
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    __rmul__ = __mul__

    def __abs__(self):
        return Pair(abs(self.value), sympy.Abs(self.formula))

    def __bool__(self):
        raise NotImplementedError(
            "data-dependent branch on a traced value"  # guard-logging comes later
        )

    def __truediv__(self, other):  # self / other  (if not added yet)
        return Pair(
            value=self.value / Pair._value_of(other),
            formula=self.formula / Pair._formula_of(other),
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    def __rtruediv__(self, other):  # other / self
        return Pair(
            value=Pair._value_of(other) / self.value,
            formula=Pair._formula_of(other) / self.formula,
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    def __pow__(self, other):  # self ** other
        return Pair(
            value=self.value ** Pair._value_of(other),
            formula=self.formula ** Pair._formula_of(other),
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    def __rpow__(self, other):  # other ** self
        return Pair(
            value=Pair._value_of(other) ** self.value,
            formula=Pair._formula_of(other) ** self.formula,
            domain=Pair._merge_domains(self._axis_bounds, Pair._domain_of(other)),
        )

    def __neg__(self):  # -self
        return Pair(-self.value, -self.formula)

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
        for input in inputs:
            if isinstance(input, np.ndarray):
                if input.ndim > 1:
                    raise NotImplementedError("")
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
        formulas = [Pair._formula_of(x) for x in inputs]
        domain = Pair._merge_domains(*(Pair._domain_of(x) for x in inputs))

        return Pair(ufunc(*values), target(*formulas), domain)

    def __array_function__(self, func, types, args, kwargs):
        fn = FUNCTION_TABLE.get(func)
        if fn is None:
            raise NotImplementedError(
                "The function is not mapped in `skverify`.",
            )
        return fn(*args, **kwargs)
