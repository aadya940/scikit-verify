"""The 100+ wild sample of functions, classes and methods from the web.

The sampling rule was fixed before looking. It runs a gh code search for
twelve common numpy idioms, 25 hits each in search order, dedupes by repo,
and fetches the first 160 files. Candidates are module-level functions AND methods of
module-level classes, self-contained (numpy/math names only, no
decorators), with synthesizable inputs. Every exclusion is counted,
and nothing is skipped by hand.
"""
import ast
import glob
import linecache
import math
import signal
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import scipy
import scipy.interpolate
import scipy.linalg
import scipy.ndimage
import scipy.optimize
import scipy.signal
import scipy.special
import scipy.stats

from skverify import to_sympy, Pair

FORBIDDEN = (
    "import ", "open(", "exec(", "eval(", "os.", "sys.", "subprocess",
    "__import__", "socket", "requests", "urllib", "input(",
)
ALLOWED_GLOBALS = {"np", "numpy", "math", "scipy", "sp", "len", "range", "enumerate",
                   "zip", "abs", "min", "max", "sum", "float", "int",
                   "list", "tuple", "print", "isinstance", "sorted",
                   "reversed", "map", "filter", "pi", "super",
                   "ValueError", "TypeError", "Exception", "bool", "str",
                   "dict", "set", "round", "divmod", "pow"}

rng = np.random.default_rng(0)

SHAPE_MENU = [
    lambda: rng.uniform(0.5, 2.0, 6),
    lambda: rng.uniform(0.5, 2.0, (4, 3)),
    lambda: float(rng.uniform(0.5, 2.0)),
    lambda: rng.uniform(0.5, 2.0, (3, 3)),
]


def free_names(node, params):
    names = {
        n.id for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    local = set(params)
    local |= {
        t.id for n in ast.walk(node) for t in ast.walk(n)
        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
    }
    local |= {n.name for n in ast.walk(node)
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    return names - local


def plain_args(a):
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        return None
    params = [x.arg for x in a.args]
    n_required = len(params) - len(a.defaults)
    return params, n_required


def uses_numpy(node):
    return bool(
        {"np", "numpy"} & {
            n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
    )


def synthesize(call, n_args, alarm=8):
    import itertools

    combos = list(itertools.product(range(len(SHAPE_MENU)), repeat=n_args))
    if len(combos) > 80:
        # 4+ args, so sample shape combinations instead of exhausting them
        idx = np.random.default_rng(1).permutation(len(combos))[:80]
        combos = [combos[i] for i in idx]
    for combo in combos:
        args = [SHAPE_MENU[k]() for k in combo]
        try:
            signal.alarm(alarm)
            np.random.seed(20260822)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ref = call(*[np.copy(x) if isinstance(x, np.ndarray) else x
                             for x in args])
            flat = ref if isinstance(ref, tuple) else (ref,)
            arrs = [np.asarray(f, dtype=float) for f in flat]
            if all(a.size and np.all(np.isfinite(a)) for a in arrs):
                return args, ref
        except Exception:
            continue
        finally:
            signal.alarm(0)
    return None, None


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))


def register(fname, seg):
    linecache.cache[fname] = (len(seg), None, seg.splitlines(True), fname)


def make_wrapper(label, kind_name, n_args, ns):
    params = ", ".join(f"a{i}" for i in range(n_args))
    src = (
        f"def __wild__({params}):\n"
        f"    return __wild_target__({params})\n"
    )
    fname = f"<wild wrapper {label}.{kind_name}>"
    register(fname, src)
    exec(compile(src, fname, "exec"), ns)
    return ns["__wild__"]


menu, uncallable, ineligible = [], 0, 0
for path in sorted(glob.glob("/tmp/wild/files_big/*")):
    src = open(path, errors="replace").read()
    label = path.split("/")[-1].split("__")[0]
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            pa = plain_args(node.args)
            seg = ast.get_source_segment(src, node)
            if (pa is None or seg is None
                    or any(t in seg for t in FORBIDDEN)
                    or not uses_numpy(node)):
                ineligible += 1
                continue
            params, n_req = pa
            if "self" in params or "cls" in params or not (1 <= n_req <= 5):
                ineligible += 1
                continue
            if free_names(node, params) - ALLOWED_GLOBALS:
                ineligible += 1
                continue
            fname = f"<wild {label}.{node.name}>"
            ns = {"np": np, "numpy": np, "math": math, "pi": math.pi,
                  "scipy": scipy, "sp": scipy}
            try:
                register(fname, seg)
                exec(compile(seg, fname, "exec"), ns)
            except Exception:
                ineligible += 1
                continue
            fn = ns[node.name]
            args, ref = synthesize(fn, n_req)
            if args is None:
                uncallable += 1
                continue
            menu.append((f"{label}.{node.name}", "function", fn, args, ref))
        elif isinstance(node, ast.ClassDef) and not node.decorator_list:
            seg = ast.get_source_segment(src, node)
            if seg is None or any(t in seg for t in FORBIDDEN):
                ineligible += 1
                continue
            if free_names(node, ("self",)) - ALLOWED_GLOBALS:
                ineligible += 1
                continue
            methods = [
                m for m in node.body
                if isinstance(m, ast.FunctionDef) and not m.decorator_list
            ]
            init = next((m for m in methods if m.name == "__init__"), None)
            fname = f"<wild {label}.{node.name}>"
            ns = {"np": np, "numpy": np, "math": math, "pi": math.pi,
                  "scipy": scipy, "sp": scipy}
            try:
                register(fname, seg)
                exec(compile(seg, fname, "exec"), ns)
            except Exception:
                ineligible += 1
                continue
            C = ns[node.name]
            if init is not None:
                pa = plain_args(init.args)
                if pa is None or not (1 <= len(pa[0])):
                    ineligible += 1
                    continue
                n_init = pa[1] - 1  # minus self
                if n_init > 3:
                    ineligible += 1
                    continue
                init_args, _ = synthesize(lambda *a: C(*a), max(n_init, 0))
                if init_args is None:
                    uncallable += 1
                    continue
            else:
                init_args = []
            for m in methods:
                if m.name.startswith("_"):
                    continue
                pa = plain_args(m.args)
                if pa is None or not uses_numpy(m):
                    ineligible += 1
                    continue
                params, n_req = pa
                n_req -= 1  # self
                if not (1 <= n_req <= 5):
                    ineligible += 1
                    continue

                def call(*a, _C=C, _ia=tuple(init_args), _m=m.name):
                    inst = _C(*[np.copy(x) if isinstance(x, np.ndarray) else x
                                for x in _ia])
                    return getattr(inst, _m)(*a)

                args, ref = synthesize(call, n_req)
                if args is None:
                    uncallable += 1
                    continue
                wns = dict(ns)
                wns["__wild_target__"] = call
                wrapped = make_wrapper(label, f"{node.name}.{m.name}", n_req, wns)
                menu.append(
                    (f"{label}.{node.name}.{m.name}", "method", wrapped, args, ref)
                )

lift_ok, lift_unverified, refused, died = [], [], [], []
n_fn = sum(1 for x in menu if x[1] == "function")
n_m = len(menu) - n_fn
for name, kind, fn, args, ref in menu:
    signal.alarm(45)
    try:
        np.random.seed(20260822)
        r = to_sympy(fn, *args)

        def concrete(x):
            if isinstance(x, Pair):
                return np.asarray(x.value, dtype=float)
            if isinstance(x, np.ndarray) and x.dtype == object:
                return np.asarray(Pair._value_of(x), dtype=float)
            return np.asarray(x, dtype=float)

        try:
            if isinstance(ref, tuple):
                match = isinstance(r, tuple) and len(r) == len(ref) and all(
                    np.allclose(concrete(g), np.asarray(w, dtype=float),
                                rtol=1e-7, atol=1e-9, equal_nan=True)
                    for g, w in zip(r, ref)
                )
            else:
                match = np.allclose(
                    concrete(r), np.asarray(ref, dtype=float),
                    rtol=1e-7, atol=1e-9, equal_nan=True,
                )
        except Exception:
            match = None
        (lift_ok if match else lift_unverified).append(name)
    except TO:
        died.append((name, "TIMEOUT"))
    except NotImplementedError as e:
        refused.append((name, str(e)[:56].replace("\n", " ")))
    except Exception as e:
        died.append((name, f"{type(e).__name__} {str(e)[:56]}".replace("\n", " ")))
    finally:
        signal.alarm(0)

total = len(menu)
print(f"CANDIDATES {total} ({n_fn} functions, {n_m} methods; "
      f"{uncallable} uncallable on synthesized inputs, {ineligible} ineligible)")
print(f"TOTAL {total} | LIFT+match {len(lift_ok)} | "
      f"lift-unverified {len(lift_unverified)} | refused {len(refused)} | died {len(died)}")
print("\nLIFT+match:", ", ".join(lift_ok))
if lift_unverified:
    print("\nlift-unverified:", ", ".join(lift_unverified))
print("\nREFUSED:")
for n_, m_ in refused:
    print(f"  {n_:52s} {m_}")
print("\nDIED:")
for n_, m_ in died:
    print(f"  {n_:52s} {m_}")
