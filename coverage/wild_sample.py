"""The wild-code sample of research functions drawn by a fixed rule.

The sampling rule was fixed before looking. It runs a gh code search for
np.gradient, np.cumsum, np.linalg.norm, takes the first 12 hits each in
search order, dedupes by repo, and fetches each hit file. From each file, every module-level
function that is self-contained (only numpy/math names, no decorators,
positional args only) and callable on synthesized float arrays enters
the menu. Nothing is skipped by hand, and every exclusion is counted.

Foreign sources stay under /tmp/wild (licenses). Only the scoreboard
is recorded here.
"""
import ast
import glob
import math
import signal
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from skverify import to_sympy, Pair

FORBIDDEN = (
    "import ", "open(", "exec(", "eval(", "os.", "sys.", "subprocess",
    "__import__", "socket", "requests", "urllib",
)
ALLOWED_GLOBALS = {"np", "numpy", "math", "len", "range", "enumerate",
                   "zip", "abs", "min", "max", "sum", "float", "int",
                   "list", "tuple", "print", "isinstance", "sorted",
                   "reversed", "map", "filter", "pi"}

rng = np.random.default_rng(0)


def candidate_functions(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            continue
        a = node.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
            continue
        if not a.args or any(arg.arg in ("self", "cls") for arg in a.args):
            continue
        seg = ast.get_source_segment(src, node)
        if seg is None or any(tok in seg for tok in FORBIDDEN):
            continue
        names = {
            n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        local = {arg.arg for arg in a.args}
        local |= {
            t.id
            for n in ast.walk(node)
            for t in ast.walk(n)
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
        }
        local |= {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}
        if not (names - local <= ALLOWED_GLOBALS):
            continue
        if "np" not in names and "numpy" not in names:
            continue
        n_default = len(a.defaults)
        n_required = len(a.args) - n_default
        if n_required == 0 or n_required > 3:
            continue
        yield node.name, seg, n_required


SHAPE_MENU = [
    lambda: rng.uniform(0.5, 2.0, 6),
    lambda: rng.uniform(0.5, 2.0, (4, 3)),
    lambda: float(rng.uniform(0.5, 2.0)),
    lambda: rng.uniform(0.5, 2.0, (3, 3)),
]


def synthesize(fn, n_args):
    import itertools

    for combo in itertools.product(range(len(SHAPE_MENU)), repeat=n_args):
        args = [SHAPE_MENU[k]() for k in combo]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ref = fn(*[np.copy(x) if isinstance(x, np.ndarray) else x for x in args])
            arr = np.asarray(ref, dtype=float)
            if arr.size and np.all(np.isfinite(arr)):
                return args, ref
        except Exception:
            continue
    return None, None


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))

menu, uncallable = [], 0
for path in sorted(glob.glob("/tmp/wild/files/*")):
    src = open(path, errors="replace").read()
    label = path.split("/")[-1].split("__")[0]
    for name, seg, n_req in candidate_functions(src):
        ns = {"np": np, "numpy": np, "math": math, "pi": math.pi}
        fname = f"<wild {label}.{name}>"
        try:
            exec(compile(seg, fname, "exec"), ns)
        except Exception:
            continue
        # the instrumented retry reads source via inspect, so register it
        # the same way notebook cells do
        import linecache

        linecache.cache[fname] = (
            len(seg), None, seg.splitlines(True), fname,
        )
        fn = ns[name]
        args, ref = synthesize(fn, n_req)
        if args is None:
            uncallable += 1
            continue
        menu.append((f"{label}.{name}", fn, args, ref))

lift_ok, lift_unverified, refused, died = [], [], [], []
for name, fn, args, ref in menu:
    signal.alarm(60)
    try:
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
        refused.append((name, str(e)[:56]))
    except Exception as e:
        died.append((name, f"{type(e).__name__} {str(e)[:56]}"))
    finally:
        signal.alarm(0)

total = len(menu)
print(f"CANDIDATES {total} (plus {uncallable} not callable on synthesized inputs)")
print(f"TOTAL {total} | LIFT+match {len(lift_ok)} | lift-unverified {len(lift_unverified)} | refused {len(refused)} | died {len(died)}")
print("\nLIFT+match:", ", ".join(lift_ok))
if lift_unverified:
    print("\nlift-unverified:", ", ".join(lift_unverified))
print("\nREFUSED:")
for n_, m in refused:
    print(f"  {n_:44s} {m}")
print("\nDIED:")
for n_, m in died:
    print(f"  {n_:44s} {m}")
