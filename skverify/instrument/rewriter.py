"""The AST rewriter: a semantically identical copy with traced call sites.

One class walks a function's tree and rewrites exactly the calls the
registries name: allocations become traced twins, neutral calls
preserve Pairs, opaque boundaries seal, loops gain provenance
markers, ``super()`` becomes explicit for re-executed methods. Every
rewrite is recorded as a site and disclosed on the traced result.
"""

import ast

from .registries import (
    ALLOC,
    CONCRETE,
    CONCRETE_CALLABLES,
    NEUTRAL,
    NEUTRAL_METHODS,
    OPAQUE_CALLABLES,
    OPAQUE_OUT,
    SCALARIZE,
)

class _Rewriter(ast.NodeTransformer):
    def __init__(self, fn_globals, tag=""):
        self.fn_globals = fn_globals
        self.tag = tag
        self.sites = []
        self.callees = set()
        self.wrapped = 0
        self.first_arg = None

    def _tag_loop(self, node):
        self.generic_visit(node)
        loop_id = f"{self.tag}:{node.lineno}"
        marker = ast.Expr(
            ast.Call(
                func=ast.Name(id="__skv_loop_iter__", ctx=ast.Load()),
                args=[ast.Constant(value=loop_id)],
                keywords=[],
            )
        )
        end = ast.Expr(
            ast.Call(
                func=ast.Name(id="__skv_loop_end__", ctx=ast.Load()),
                args=[ast.Constant(value=loop_id)],
                keywords=[],
            )
        )
        node.body.insert(0, marker)
        self.sites.append(f"loop {loop_id} tagged")
        return [node, end]

    def visit_Expr(self, node):
        # `fn(x, t, k, ab.T, n)` filling ab in place -> `ab = twin(...)`;
        # multiple out-params become tuple assignments. Out-parameter
        # Cython calls turn into assignments of opaque atoms
        call = node.value if isinstance(node.value, ast.Call) else None
        name = self._target_name(call.func) if call else None
        if name in OPAQUE_OUT and not call.keywords:
            idxs = OPAQUE_OUT[name]
            targets, flags, ok = [], [], max(idxs) < len(call.args)
            if ok:
                for idx in idxs:
                    arg = call.args[idx]
                    transposed = (
                        isinstance(arg, ast.Attribute)
                        and arg.attr == "T"
                        and isinstance(arg.value, ast.Name)
                    )
                    base = (
                        arg.value
                        if transposed
                        else arg if isinstance(arg, ast.Name) else None
                    )
                    if base is None:
                        ok = False
                        break
                    targets.append(base.id)
                    flags.append(transposed)
            if ok:
                # visit args only: generic_visit would let the doorman
                # wrap the callee and rename the atom to 'wrapper'
                args = [self.visit(a) for a in call.args]
                for idx, tid in zip(idxs, targets):
                    args[idx] = ast.Name(id=tid, ctx=ast.Load())
                self.sites.append(f"{name} -> opaque out-parameter call")
                names = [ast.Name(id=t, ctx=ast.Store()) for t in targets]
                target = (
                    names[0]
                    if len(names) == 1
                    else ast.Tuple(elts=names, ctx=ast.Store())
                )
                return ast.Assign(
                    targets=[target],
                    value=ast.Call(
                        func=ast.Name(id="__skv_opaque_out__", ctx=ast.Load()),
                        args=[
                            call.func,
                            ast.Constant(value=tuple(idxs)),
                            ast.Constant(value=tuple(flags)),
                        ]
                        + args,
                        keywords=[],
                    ),
                )
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        return self._tag_loop(node)

    def visit_While(self, node):
        return self._tag_loop(node)

    def _target_name(self, func):
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return None

    def visit_Call(self, node):
        self.generic_visit(node)
        # a neutral function passed by REFERENCE (map(np.asarray_chkfinite,
        # ...)) must be swapped too, or Pairs decompress inside it
        for pos, arg in enumerate(node.args):
            ref = self._target_name(arg)
            if ref in NEUTRAL and isinstance(arg, (ast.Attribute, ast.Name)):
                self.sites.append(f"{ref} (reference) -> pair-preserving")
                node.args[pos] = ast.Name(id="__skv_neutral__", ctx=ast.Load())
        name = self._target_name(node.func)
        if name in ALLOC:
            self.sites.append(f"{name} -> traced allocation")
            node.func = ast.Name(id=f"__skv_{name}__", ctx=ast.Load())
        elif name in NEUTRAL:
            self.sites.append(f"{name} -> pair-preserving")
            node.func = ast.Name(id="__skv_neutral__", ctx=ast.Load())
        elif name in CONCRETE:
            self.sites.append(f"{name} -> concrete-lane check")
            node = ast.Call(
                func=ast.Name(id="__skv_concrete__", ctx=ast.Load()),
                args=[ast.Constant(value=name)] + node.args,
                keywords=node.keywords,
            )
        elif name == "isinstance" and isinstance(node.func, ast.Name):
            self.sites.append("isinstance -> Pair counts as ndarray")
            node.func = ast.Name(id="__skv_isinstance__", ctx=ast.Load())
        elif name in SCALARIZE and isinstance(node.func, ast.Name):
            self.sites.append(f"{name} -> concrete scalar")
            node = ast.Call(
                func=ast.Name(id="__skv_scalarize__", ctx=ast.Load()),
                args=[node.func] + node.args,
                keywords=[],
            )
        elif name == "super" and not node.args:
            # re-exec'd methods lose the __class__ cell zero-arg super()
            # needs; the twin class is injected as __skv_class__
            node.args = [
                ast.Name(id="__skv_class__", ctx=ast.Load()),
                ast.Name(id=self.first_arg or "self", ctx=ast.Load()),
            ]
            self.sites.append("super() -> explicit twin super")
        elif name == "at":
            self.sites.append("at -> traced functional update")
            node = ast.Call(
                func=ast.Name(id="__skv_at__", ctx=ast.Load()),
                args=[node.func] + node.args,
                keywords=node.keywords,
            )
        elif name == "array_namespace":
            self.sites.append("array_namespace -> numpy (compat layer skipped)")
            node.func = ast.Name(id="__skv_namespace__", ctx=ast.Load())
        elif name == "finfo":
            self.sites.append("finfo -> concrete-lane precision")
            node.func = ast.Call(
                func=ast.Name(id="__skv_finfo__", ctx=ast.Load()),
                args=[node.func],
                keywords=[],
            )
        elif name in CONCRETE_CALLABLES:
            self.sites.append(f"{name} -> concrete lookup")
            node.func = ast.Call(
                func=ast.Name(id="__skv_concrete_call__", ctx=ast.Load()),
                args=[node.func],
                keywords=[],
            )
        elif name in OPAQUE_CALLABLES:
            self.sites.append(f"{name} -> opaque contract call")
            node.func = ast.Call(
                func=ast.Name(id="__skv_opaque__", ctx=ast.Load()),
                args=[node.func],
                keywords=[],
            )
        elif name in NEUTRAL_METHODS and isinstance(node.func, ast.Attribute):
            self.sites.append(f".{name}() -> pair-preserving")
            node = ast.Call(
                func=ast.Name(id="__skv_method__", ctx=ast.Load()),
                args=[ast.Constant(value=name), node.func.value] + node.args,
                keywords=node.keywords,
            )
        elif isinstance(node.func, ast.Name):
            self.callees.add(node.func.id)
            node = self._maybe_opaque(node)
        elif isinstance(node.func, ast.Attribute):
            node = self._maybe_opaque(node)
        return node

    def visit_Name(self, node):
        if node.id == "__class__":
            node.id = "__skv_class__"  # re-exec'd methods lose the cell
        return node

    def _maybe_opaque(self, node):
        # the doorman: unregistered COMPILED callables auto-seal into
        # opaque atoms at runtime; Python functions, ufuncs, builtins
        # and classes pass through untouched
        if isinstance(node.func, ast.Name) and node.func.id.startswith("__skv"):
            return node
        self.wrapped += 1
        node.func = ast.Call(
            func=ast.Name(id="__skv_maybe__", ctx=ast.Load()),
            args=[node.func],
            keywords=[],
        )
        return node


