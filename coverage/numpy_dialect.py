"""The dialect battery measures every public numpy callable.

Enumerates np, np.linalg and np.fft public callables. Each one is
probed with synthesized float inputs (brute force over a small shape
menu, wrappers generated as real source so the instrumented retry can
read them), traced, and verified against the untraced run.

The classes are LIFT+match, refused, died, unverified (value compare
failed), uncallable (no synthesized input worked), and out-of-scope
(non-mathematical by rule, every rule name counted). The headline is
lift over (lift + refused + died + unverified), the measured dialect.
"""
import inspect as _inspect
import itertools
import linecache
import signal
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from skverify import to_sympy, Pair

rng = np.random.default_rng(0)

# non-mathematical by rule. Every name lands in a COUNTED bucket
OUT_OF_SCOPE = {
    "io": {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
           "genfromtxt", "fromfile", "frombuffer", "fromstring", "fromregex",
           "fromfunction", "fromiter", "memmap", "lib", "DataSource"},
    "printing": {"array2string", "array_repr", "array_str", "set_printoptions",
                 "get_printoptions", "printoptions", "binary_repr", "base_repr",
                 "format_float_positional", "format_float_scientific",
                 "get_include", "info", "show_config", "show_runtime",
                 "typename"},
    "dtype-and-type-system": {"dtype", "can_cast", "promote_types", "result_type",
                  "min_scalar_type", "common_type", "mintypecode", "issubdtype",
                  "iinfo", "finfo", "isdtype", "set_typeDict", "maskna",
                  "astype", "bool", "bool_", "byte", "bytes_", "cdouble",
                  "character", "clongdouble", "complex128", "complex64",
                  "complexfloating", "csingle", "datetime64", "double",
                  "flexible", "float16", "float32", "float64", "floating",
                  "generic", "half", "inexact", "int16", "int32", "int64",
                  "int8", "int_", "intc", "integer", "intp", "long",
                  "longdouble", "longlong", "number", "object_", "short",
                  "signedinteger", "single", "str_", "timedelta64", "ubyte",
                  "uint", "uint16", "uint32", "uint64", "uint8", "uintc",
                  "uintp", "ulong", "ulonglong", "unsignedinteger", "ushort",
                  "void", "float128", "complex256", "uint128", "int128"},
    "predicates-and-introspection": {"isscalar", "iterable", "ndim", "size",
                  "shape", "may_share_memory", "shares_memory", "is_busday",
                  "isfortran", "isrealobj", "iscomplexobj", "issubclass_",
                  "isnat", "typecodes", "ScalarType"},
    "datetime-strings-bits": {"busday_count", "busday_offset", "busdaycalendar",
                  "datetime_as_string", "datetime_data", "char", "strings",
                  "packbits", "unpackbits", "invert", "bitwise_and",
                  "bitwise_count", "bitwise_invert", "bitwise_left_shift",
                  "bitwise_not", "bitwise_or", "bitwise_right_shift",
                  "bitwise_xor", "left_shift", "right_shift", "gcd", "lcm"},
    "random-and-state": {"random", "seed", "get_state", "set_state"},
    "infrastructure": {"errstate", "seterr", "geterr", "seterrcall",
                  "geterrcall", "setbufsize", "getbufsize", "ndenumerate",
                  "ndindex", "nditer", "nested_iters", "broadcast",
                  "ndarray", "flatiter", "vectorize", "frompyfunc",
                  "apply_along_axis", "apply_over_axes", "piecewise",
                  "ufunc", "testing", "test", "matrix", "asmatrix", "bmat",
                  "ma", "ctypeslib", "emath", "exceptions", "f2py", "fft",
                  "linalg", "polynomial", "rec", "core", "matlib",
                  "version", "distutils", "dtypes", "typing",
                  "iterable", "require", "einsum_path", "get_array_wrap",
                  "who", "safe_eval", "disp", "deprecate",
                  "deprecate_with_doc", "byte_bounds", "add_docstring",
                  "add_newdoc", "add_newdoc_ufunc", "compare_chararrays",
                  "setdiff1d", "pv", "fv", "pmt", "ppmt", "ipmt", "irr",
                  "mirr", "nper", "npv", "rate", "copyto", "place",
                  "putmask", "put", "put_along_axis", "fill_diagonal",
                  "shape_base", "resize", "delete", "insert", "append",
                  "pad", "block", "print_function"},
}
OOS_LOOKUP = {n: cat for cat, names in OUT_OF_SCOPE.items() for n in names}
# module names (fft, linalg, random) must not shadow FUNCTIONS of the
# same name inside submodules. Qualified names win
OOS_QUALIFIED_EXEMPT = {"np.fft.fft", "np.fft.ifft", "np.linalg.solve"}
# copyto/place/putmask/fill_diagonal are supported as MUTATORS of traced
# targets. As standalone f(concrete) probes they are not formula-producing.
# pad/delete/insert/append/block/resize are shape editors worth entries
# someday. They are counted out-of-scope=infrastructure TODAY, disclosed here.

V6 = lambda: rng.uniform(0.5, 2.0, 6)
M43 = lambda: rng.uniform(0.5, 2.0, (4, 3))
SQ3 = lambda: rng.uniform(0.5, 2.0, (3, 3))
SPD = lambda: (lambda a: a @ a.T + 0.5 * np.eye(3))(rng.uniform(0.5, 2.0, (3, 3)))
SC = lambda: float(rng.uniform(0.5, 2.0))
SORTED6 = lambda: np.sort(rng.uniform(0.5, 2.0, 6))
MENU = [V6, M43, SC, SQ3, SPD, SORTED6]

EXTRA = {  # concrete non-traced extras tried as trailing args
    (): [()],
    (1,): [(2,), (0,), (3,)],
}


def public_callables():
    for modname, mod in (("np", np), ("np.linalg", np.linalg), ("np.fft", np.fft)):
        names = getattr(mod, "__all__", None) or dir(mod)
        for n in sorted(set(names)):
            if n.startswith("_"):
                continue
            f = getattr(mod, n, None)
            if not callable(f):
                continue
            yield f"{modname}.{n}", n, f


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def make_probe(qual, expr_src, n_args):
    params = ", ".join(f"a{i}" for i in range(n_args))
    src = f"def probe({params}):\n    return {expr_src}\n"
    fname = f"<dialect {qual}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = {"np": np}
    exec(compile(src, fname, "exec"), ns)
    return ns["probe"]


def synthesize(qual, callname, f):
    """Find (probe, args, ref) from traced-arg forms f(A), f(A,B), f(A, k)."""
    forms = []
    for n_tr in (1, 2):
        for extra in ([], [2], [0.5]):
            extra_src = "".join(f", {e!r}" for e in extra)
            args_src = ", ".join(f"a{i}" for i in range(n_tr)) + extra_src
            forms.append((n_tr, f"{callname}({args_src})"))
    for n_tr, expr in forms:
        for combo in itertools.product(range(len(MENU)), repeat=n_tr):
            args = [MENU[k]() for k in combo]
            try:
                signal.alarm(6)
                probe = make_probe(qual, expr, n_tr)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    ref = probe(*[np.copy(a) if isinstance(a, np.ndarray) else a
                                  for a in args])
                flat = ref if isinstance(ref, (tuple, list)) else (ref,)
                arrs = [np.asarray(x, dtype=float) for x in flat]
                if all(a.size and np.all(np.isfinite(a)) for a in arrs):
                    return probe, args, ref
            except Exception:
                continue
            finally:
                signal.alarm(0)
    return None, None, None


def concrete(x):
    if isinstance(x, Pair):
        return np.asarray(x.value, dtype=float)
    if isinstance(x, np.ndarray) and x.dtype == object:
        return np.asarray(Pair._value_of(x), dtype=float)
    return np.asarray(x, dtype=float)


skipped = {}
lift_ok, unverified, refused, died, uncallable = [], [], [], [], []
for qual, name, f in public_callables():
    if name in OOS_LOOKUP and qual not in OOS_QUALIFIED_EXEMPT:
        skipped.setdefault(OOS_LOOKUP[name], []).append(qual)
        continue
    callname = qual.replace("np.linalg.", "np.linalg.").replace("np.fft.", "np.fft.")
    probe, args, ref = synthesize(qual, callname if "." in qual[3:] else f"np.{name}", f)
    if probe is None:
        uncallable.append(qual)
        continue
    signal.alarm(45)
    try:
        r = to_sympy(probe, *args)
        try:
            if isinstance(ref, (tuple, list)):
                ok = isinstance(r, (tuple, list)) and len(r) == len(ref) and all(
                    np.allclose(concrete(g), np.asarray(w, dtype=float),
                                rtol=1e-7, atol=1e-9, equal_nan=True)
                    for g, w in zip(r, ref)
                )
            else:
                ok = np.allclose(concrete(r), np.asarray(ref, dtype=float),
                                 rtol=1e-7, atol=1e-9, equal_nan=True)
        except Exception:
            ok = False
        (lift_ok if ok else unverified).append(qual)
    except TO:
        died.append((qual, "TIMEOUT"))
    except NotImplementedError as e:
        refused.append((qual, str(e)[:48].replace("\n", " ")))
    except Exception as e:
        died.append((qual, f"{type(e).__name__} {str(e)[:48]}".replace("\n", " ")))
    finally:
        signal.alarm(0)

measured = len(lift_ok) + len(unverified) + len(refused) + len(died)
print(f"MEASURED {measured} | LIFT+match {len(lift_ok)} | unverified {len(unverified)} "
      f"| refused {len(refused)} | died {len(died)}")
print(f"uncallable {len(uncallable)} | out-of-scope "
      + ", ".join(f"{k}:{len(v)}" for k, v in sorted(skipped.items())))
print("\nLIFT+match:", ", ".join(lift_ok))
print("\nUNVERIFIED:", ", ".join(unverified))
print("\nREFUSED:")
for n_, m in refused:
    print(f"  {n_:34s} {m}")
print("\nDIED:")
for n_, m in died:
    print(f"  {n_:34s} {m}")
print("\nUNCALLABLE:", ", ".join(uncallable))
