"""Python control-flow lowering helpers.

FLIR executes Python code to build MLIR IR. A plain Python loop like:

  for i in range(10):
      ...

is executed by the Python interpreter, effectively *unrolling* IR construction.

Sometimes we want control-flow written with Python syntax to become IR
control-flow (`scf.for`, `scf.if`) instead of Python unrolling / compile-time
branch selection. This module provides an opt-in AST rewrite that transforms:

  for i in range(bound):
      body(i)

into:

      with scf.range_(...) as i:   # emits scf.for
          body(i)

To explicitly request Python unrolling (compile-time loop expansion), use
`range_constexpr(...)` in the loop iterable:

  for i in range_constexpr(10):
      ...

Limitations (current):
  - `break` / `continue` inside rewritten loops are not supported.
  - `for ... else:` is not supported.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class LoweringOptions:
    scf_alias: str = "__flir_scf"
    value_alias: str = "__flir_Value"
    is_value_fn: str = "__flir_is_value"
    lower_builtin_range: bool = True


class _BreakContinueChecker(ast.NodeVisitor):
    def __init__(self):
        self.has_break_or_continue = False

    def visit_Break(self, node):  # noqa: N802
        self.has_break_or_continue = True

    def visit_Continue(self, node):  # noqa: N802
        self.has_break_or_continue = True


class _RangeForLowerer(ast.NodeTransformer):
    def __init__(self, opts: LoweringOptions):
        super().__init__()
        self.opts = opts
        self._tmp_id = 0

    def _fresh(self, prefix: str) -> str:
        self._tmp_id += 1
        return f"__flir_{prefix}_{self._tmp_id}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        node = self.generic_visit(node)
        # Drop decorators: caller will re-wrap.
        node.decorator_list = []

        # Inject imports and helpers at function entry so the transformed code is self-contained.
        injected = [
            ast.ImportFrom(
                module="_mlir.ir",
                names=[ast.alias(name="Value", asname=self.opts.value_alias)],
                level=0,
            ),
            ast.ImportFrom(
                module="pyflir.dialects.ext",
                names=[ast.alias(name="scf", asname=self.opts.scf_alias)],
                level=0,
            ),
            ast.ImportFrom(
                module="pyflir.dialects.ext.python_control_flow",
                names=[ast.alias(name="range_constexpr", asname="range_constexpr")],
                level=0,
            ),
        ]

        # def __flir_is_value(x):
        #   if isinstance(x, Value): return True
        #   if hasattr(x, "_value") and isinstance(x._value, Value): return True
        #   if hasattr(x, "value") and isinstance(x.value, Value): return True
        #   return False
        is_value_def = ast.FunctionDef(
            name=self.opts.is_value_fn,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="x")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=[
                ast.If(
                    test=ast.Call(
                        func=ast.Name(id="isinstance", ctx=ast.Load()),
                        args=[
                            ast.Name(id="x", ctx=ast.Load()),
                            ast.Name(id=self.opts.value_alias, ctx=ast.Load()),
                        ],
                        keywords=[],
                    ),
                    body=[ast.Return(value=ast.Constant(True))],
                    orelse=[],
                ),
                ast.If(
                    test=ast.BoolOp(
                        op=ast.And(),
                        values=[
                            ast.Call(func=ast.Name(id="hasattr", ctx=ast.Load()), args=[ast.Name(id="x", ctx=ast.Load()), ast.Constant("_value")], keywords=[]),
                            ast.Call(
                                func=ast.Name(id="isinstance", ctx=ast.Load()),
                                args=[
                                    ast.Attribute(value=ast.Name(id="x", ctx=ast.Load()), attr="_value", ctx=ast.Load()),
                                    ast.Name(id=self.opts.value_alias, ctx=ast.Load()),
                                ],
                                keywords=[],
                            ),
                        ],
                    ),
                    body=[ast.Return(value=ast.Constant(True))],
                    orelse=[],
                ),
                ast.If(
                    test=ast.BoolOp(
                        op=ast.And(),
                        values=[
                            ast.Call(func=ast.Name(id="hasattr", ctx=ast.Load()), args=[ast.Name(id="x", ctx=ast.Load()), ast.Constant("value")], keywords=[]),
                            ast.Call(
                                func=ast.Name(id="isinstance", ctx=ast.Load()),
                                args=[
                                    ast.Attribute(value=ast.Name(id="x", ctx=ast.Load()), attr="value", ctx=ast.Load()),
                                    ast.Name(id=self.opts.value_alias, ctx=ast.Load()),
                                ],
                                keywords=[],
                            ),
                        ],
                    ),
                    body=[ast.Return(value=ast.Constant(True))],
                    orelse=[],
                ),
                ast.Return(value=ast.Constant(False)),
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        is_value_def = ast.fix_missing_locations(is_value_def)

        # def __flir_unwrap(x):
        #   if hasattr(x, "value"): return x.value
        #   if hasattr(x, "_value"): return x._value
        #   return x
        unwrap_def = ast.FunctionDef(
            name="__flir_unwrap",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="x")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=[
                ast.If(
                    test=ast.Call(
                        func=ast.Name(id="hasattr", ctx=ast.Load()),
                        args=[ast.Name(id="x", ctx=ast.Load()), ast.Constant("value")],
                        keywords=[],
                    ),
                    body=[ast.Return(value=ast.Attribute(value=ast.Name(id="x", ctx=ast.Load()), attr="value", ctx=ast.Load()))],
                    orelse=[],
                ),
                ast.If(
                    test=ast.Call(
                        func=ast.Name(id="hasattr", ctx=ast.Load()),
                        args=[ast.Name(id="x", ctx=ast.Load()), ast.Constant("_value")],
                        keywords=[],
                    ),
                    body=[ast.Return(value=ast.Attribute(value=ast.Name(id="x", ctx=ast.Load()), attr="_value", ctx=ast.Load()))],
                    orelse=[],
                ),
                ast.Return(value=ast.Name(id="x", ctx=ast.Load())),
            ],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        unwrap_def = ast.fix_missing_locations(unwrap_def)

        # Minimal "pytree" helpers for loop-carried state:
        # - Support lists/tuples nesting with leaf nodes as MLIR Values (or wrappers).
        # - We carry flattened MLIR Values through scf.for_ iter_args/results.
        tree_helpers_src = """
def __flir_tree_flatten(x):
    if isinstance(x, (list, tuple)):
        flat = []
        spec = []
        for e in x:
            f, s = __flir_tree_flatten(e)
            flat.extend(f)
            spec.append(s)
        return flat, ("seq", isinstance(x, tuple), spec)
    v = __flir_unwrap(x)
    if not __flir_is_value(v):
        raise TypeError(f"loop-carried values must be MLIR Values; got {type(v)}")
    return [v], ("leaf",)


def __flir_tree_unflatten(spec, flat):
    # Returns (obj, rest_flat)
    if spec[0] == "leaf":
        return flat[0], flat[1:]
    _, is_tuple, subs = spec
    out = []
    rest = flat
    for s in subs:
        v, rest = __flir_tree_unflatten(s, rest)
        out.append(v)
    return (tuple(out) if is_tuple else out), rest
"""
        tree_nodes = ast.parse(textwrap.dedent(tree_helpers_src)).body
        tree_nodes = [ast.fix_missing_locations(n) for n in tree_nodes]

        node.body = injected + [is_value_def, unwrap_def] + tree_nodes + node.body
        return ast.fix_missing_locations(node)

    def visit_If(self, node: ast.If):  # noqa: N802
        node = self.generic_visit(node)

        # Evaluate condition once.
        cond_tmp = self._fresh("if_cond")
        assign_cond = ast.Assign(
            targets=[ast.Name(id=cond_tmp, ctx=ast.Store())],
            value=node.test,
        )

        # if __flir_is_value(cond_tmp):  -> emit scf.IfOp
        dyn_test = ast.Call(
            func=ast.Name(id=self.opts.is_value_fn, ctx=ast.Load()),
            args=[ast.Name(id=cond_tmp, ctx=ast.Load())],
            keywords=[],
        )

        # Detect simple assignments to names in the top-level of the branches.
        # These are candidates for phi-style lowering (scf.if results) so that
        # Python variables assigned in a dynamic if behave like SSA values.
        def _assigned_simple_names(stmts: list[ast.stmt]) -> set[str]:
            out: set[str] = set()
            for s in stmts:
                if isinstance(s, ast.Assign):
                    for t in s.targets:
                        if isinstance(t, ast.Name):
                            out.add(t.id)
                elif isinstance(s, ast.AnnAssign):
                    if isinstance(s.target, ast.Name):
                        out.add(s.target.id)
                elif isinstance(s, ast.AugAssign):
                    if isinstance(s.target, ast.Name):
                        out.add(s.target.id)
            return out

        carry_vars = sorted(_assigned_simple_names(list(node.body)) | _assigned_simple_names(list(node.orelse)))
        has_else = bool(node.orelse)
        ifop_tmp = self._fresh("ifop")

        dyn_body: list[ast.stmt] = []

        if carry_vars:
            # Save pre-if values so we can:
            # - compute result types from the pre-state
            # - reset Python variables before building the else-region (since both
            #   regions are built sequentially in Python)
            pre_names: dict[str, str] = {}
            spec_names: dict[str, str] = {}
            types_names: dict[str, str] = {}
            flag_names: dict[str, str] = {}
            had_names: dict[str, str] = {}

            for v in carry_vars:
                pre_n = self._fresh(f"if_pre_{v}")
                pre_names[v] = pre_n
                spec_n = self._fresh(f"if_spec_{v}")
                spec_names[v] = spec_n
                types_n = self._fresh(f"if_types_{v}")
                types_names[v] = types_n
                flag_n = self._fresh(f"if_carry_{v}")
                flag_names[v] = flag_n
                had_n = self._fresh(f"if_had_{v}")
                had_names[v] = had_n
                flat_n = self._fresh(f"if_flat_{v}")

                # Default: not carried.
                dyn_body.append(ast.Assign(targets=[ast.Name(id=flag_n, ctx=ast.Store())], value=ast.Constant(False)))
                dyn_body.append(ast.Assign(targets=[ast.Name(id=types_n, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())))
                dyn_body.append(ast.Assign(targets=[ast.Name(id=had_n, ctx=ast.Store())], value=ast.Constant(False)))
                dyn_body.append(ast.Assign(targets=[ast.Name(id=pre_n, ctx=ast.Store())], value=ast.Constant(None)))

                # Try to capture/flatten the pre-state. If the variable is not yet bound
                # (NameError/UnboundLocalError) or not a tree of MLIR Values (TypeError),
                # we simply don't treat it as a phi variable.
                try_body = [
                    ast.Assign(
                        targets=[ast.Name(id=pre_n, ctx=ast.Store())],
                        value=ast.Name(id=v, ctx=ast.Load()),
                    ),
                    ast.Assign(targets=[ast.Name(id=had_n, ctx=ast.Store())], value=ast.Constant(True)),
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=flat_n, ctx=ast.Store()),
                                    ast.Name(id=spec_n, ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Call(
                            func=ast.Name(id="__flir_tree_flatten", ctx=ast.Load()),
                            args=[ast.Name(id=pre_n, ctx=ast.Load())],
                            keywords=[],
                        ),
                    ),
                    ast.Assign(
                        targets=[ast.Name(id=types_n, ctx=ast.Store())],
                        value=ast.ListComp(
                            elt=ast.Attribute(
                                value=ast.Call(
                                    func=ast.Name(id="__flir_unwrap", ctx=ast.Load()),
                                    args=[ast.Name(id="__flir_x", ctx=ast.Load())],
                                    keywords=[],
                                ),
                                attr="type",
                                ctx=ast.Load(),
                            ),
                            generators=[
                                ast.comprehension(
                                    target=ast.Name(id="__flir_x", ctx=ast.Store()),
                                    iter=ast.Name(id=flat_n, ctx=ast.Load()),
                                    ifs=[],
                                    is_async=0,
                                )
                            ],
                        ),
                    ),
                    ast.Assign(targets=[ast.Name(id=flag_n, ctx=ast.Store())], value=ast.Constant(True)),
                ]
                exc_tuple = ast.Tuple(
                    elts=[ast.Name(id="NameError", ctx=ast.Load()), ast.Name(id="UnboundLocalError", ctx=ast.Load()), ast.Name(id="TypeError", ctx=ast.Load())],
                    ctx=ast.Load(),
                )
                dyn_body.append(
                    ast.Try(
                        body=try_body,
                        handlers=[ast.ExceptHandler(type=exc_tuple, name=None, body=[ast.Pass()])],
                        orelse=[],
                        finalbody=[],
                    )
                )

            # Build results type list dynamically from successfully-carried vars.
            results_types_name = self._fresh("if_result_types")
            dyn_body.append(ast.Assign(targets=[ast.Name(id=results_types_name, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load())))
            for v in carry_vars:
                dyn_body.append(
                    ast.If(
                        test=ast.Name(id=flag_names[v], ctx=ast.Load()),
                        body=[
                            ast.AugAssign(
                                target=ast.Name(id=results_types_name, ctx=ast.Store()),
                                op=ast.Add(),
                                value=ast.Name(id=types_names[v], ctx=ast.Load()),
                            )
                        ],
                        orelse=[],
                    )
                )

            # ifop = scf.IfOp(cond, results_types, hasElse=True)
            dyn_body.append(
                ast.Assign(
                    targets=[ast.Name(id=ifop_tmp, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                            attr="IfOp",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Name(id=cond_tmp, ctx=ast.Load()),
                            ast.Name(id=results_types_name, ctx=ast.Load()),
                        ],
                        keywords=[ast.keyword(arg="hasElse", value=ast.Constant(True))],
                    ),
                )
            )

            # then:
            yield_vals_name = self._fresh("if_yield_vals")
            then_yield_build: list[ast.stmt] = [
                ast.Assign(targets=[ast.Name(id=yield_vals_name, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))
            ]
            for v in carry_vars:
                flatcur = self._fresh(f"if_flatcur_{v}")
                speccur = self._fresh(f"if_speccur_{v}")
                then_yield_build.append(
                    ast.If(
                        test=ast.Name(id=flag_names[v], ctx=ast.Load()),
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Tuple(
                                        elts=[ast.Name(id=flatcur, ctx=ast.Store()), ast.Name(id=speccur, ctx=ast.Store())],
                                        ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Call(
                                    func=ast.Name(id="__flir_tree_flatten", ctx=ast.Load()),
                                    args=[ast.Name(id=v, ctx=ast.Load())],
                                    keywords=[],
                                ),
                            ),
                            ast.AugAssign(
                                target=ast.Name(id=yield_vals_name, ctx=ast.Store()),
                                op=ast.Add(),
                                value=ast.Name(id=flatcur, ctx=ast.Load()),
                            ),
                        ],
                        orelse=[],
                    )
                )
            then_yield_build.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                            attr="yield_",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=yield_vals_name, ctx=ast.Load())],
                        keywords=[],
                    )
                )
            )

            with_then = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=ifop_tmp, ctx=ast.Load()), attr="then", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        optional_vars=None,
                    )
                ],
                body=list(node.body) + then_yield_build,
                type_comment=None,
            )

            # else: reset python vars back to pre-state, then build else body, then yield.
            else_reset: list[ast.stmt] = []
            for v in carry_vars:
                else_reset.append(
                    ast.If(
                        test=ast.Name(id=had_names[v], ctx=ast.Load()),
                        body=[ast.Assign(targets=[ast.Name(id=v, ctx=ast.Store())], value=ast.Name(id=pre_names[v], ctx=ast.Load()))],
                        orelse=[ast.Assign(targets=[ast.Name(id=v, ctx=ast.Store())], value=ast.Constant(None))],
                    )
                )

            else_yield_build: list[ast.stmt] = [
                ast.Assign(targets=[ast.Name(id=yield_vals_name, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))
            ]
            for v in carry_vars:
                flatcur = self._fresh(f"if_flatcur_else_{v}")
                speccur = self._fresh(f"if_speccur_else_{v}")
                else_yield_build.append(
                    ast.If(
                        test=ast.Name(id=flag_names[v], ctx=ast.Load()),
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Tuple(
                                        elts=[ast.Name(id=flatcur, ctx=ast.Store()), ast.Name(id=speccur, ctx=ast.Store())],
                                        ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Call(
                                    func=ast.Name(id="__flir_tree_flatten", ctx=ast.Load()),
                                    args=[ast.Name(id=v, ctx=ast.Load())],
                                    keywords=[],
                                ),
                            ),
                            ast.AugAssign(
                                target=ast.Name(id=yield_vals_name, ctx=ast.Store()),
                                op=ast.Add(),
                                value=ast.Name(id=flatcur, ctx=ast.Load()),
                            ),
                        ],
                        orelse=[],
                    )
                )
            else_yield_build.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                            attr="yield_",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=yield_vals_name, ctx=ast.Load())],
                        keywords=[],
                    )
                )
            )

            with_else = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=ifop_tmp, ctx=ast.Load()), attr="else_", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        optional_vars=None,
                    )
                ],
                body=else_reset + list(node.orelse) + else_yield_build,
                type_comment=None,
            )

            dyn_body.extend([with_then, with_else])

            # After building both regions, restore all branch-assigned Python names to
            # their pre-state (or None) to avoid leaking Values defined in one region
            # into later codegen (which can cause dominance issues).
            for v in carry_vars:
                dyn_body.append(
                    ast.If(
                        test=ast.Name(id=had_names[v], ctx=ast.Load()),
                        body=[ast.Assign(targets=[ast.Name(id=v, ctx=ast.Store())], value=ast.Name(id=pre_names[v], ctx=ast.Load()))],
                        orelse=[ast.Assign(targets=[ast.Name(id=v, ctx=ast.Store())], value=ast.Constant(None))],
                    )
                )

            # Unflatten results back into variables.
            post_rest = self._fresh("if_post_rest")
            dyn_body.append(
                ast.Assign(
                    targets=[ast.Name(id=post_rest, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="list", ctx=ast.Load()),
                        args=[
                            ast.Attribute(value=ast.Name(id=ifop_tmp, ctx=ast.Load()), attr="results", ctx=ast.Load())
                        ],
                        keywords=[],
                    ),
                )
            )
            for v in carry_vars:
                obj_n = self._fresh(f"if_post_obj_{v}")
                dyn_body.append(
                    ast.If(
                        test=ast.Name(id=flag_names[v], ctx=ast.Load()),
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Tuple(
                                        elts=[ast.Name(id=obj_n, ctx=ast.Store()), ast.Name(id=post_rest, ctx=ast.Store())],
                                        ctx=ast.Store(),
                                    )
                                ],
                                value=ast.Call(
                                    func=ast.Name(id="__flir_tree_unflatten", ctx=ast.Load()),
                                    args=[ast.Name(id=spec_names[v], ctx=ast.Load()), ast.Name(id=post_rest, ctx=ast.Load())],
                                    keywords=[],
                                ),
                            ),
                            ast.Assign(
                                targets=[ast.Name(id=v, ctx=ast.Store())],
                                value=ast.Name(id=obj_n, ctx=ast.Load()),
                            ),
                        ],
                        orelse=[],
                    )
                )
        else:
            # No tracked assignments: just emit scf.if for side effects / control.
            create_ifop = ast.Assign(
                targets=[ast.Name(id=ifop_tmp, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                        attr="IfOp",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=cond_tmp, ctx=ast.Load())],
                    keywords=[ast.keyword(arg="hasElse", value=ast.Constant(True))] if has_else else [],
                ),
            )
            with_then = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=ifop_tmp, ctx=ast.Load()), attr="then", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        optional_vars=None,
                    )
                ],
                body=node.body,
                type_comment=None,
            )
            dyn_body = [create_ifop, with_then]
            if has_else:
                with_else = ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Call(
                                func=ast.Attribute(value=ast.Name(id=ifop_tmp, ctx=ast.Load()), attr="else_", ctx=ast.Load()),
                                args=[],
                                keywords=[],
                            ),
                            optional_vars=None,
                        )
                    ],
                    body=node.orelse,
                    type_comment=None,
                )
                dyn_body.append(with_else)

        # else: keep python if semantics (compile-time selection)
        py_if = ast.If(
            test=ast.Name(id=cond_tmp, ctx=ast.Load()),
            body=node.body,
            orelse=node.orelse,
        )

        lowered = [assign_cond, ast.If(test=dyn_test, body=dyn_body, orelse=[py_if])]
        for n in lowered:
            ast.copy_location(n, node)
        return lowered

    def visit_For(self, node: ast.For):  # noqa: N802
        node = self.generic_visit(node)

        if node.orelse:
            raise NotImplementedError("Lowering does not support `for ... else:`")

        # Only handle builtin `range(...)` for now.
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)):
            return node
        func_id = node.iter.func.id
        if func_id == "range_constexpr":
            # Explicit opt-out: keep Python unrolling semantics.
            node.iter.func.id = "range"
            return node
        if func_id != "range" or not self.opts.lower_builtin_range:
            return node

        # Reject break/continue (they won't map to scf semantics in this simple rewrite).
        chk = _BreakContinueChecker()
        for stmt in node.body:
            chk.visit(stmt)
        if chk.has_break_or_continue:
            raise NotImplementedError("break/continue inside lowered range-loops is not supported yet")

        # Evaluate range args once into temps.
        args = list(node.iter.args)
        if not (1 <= len(args) <= 3):
            return node

        tmp_names = [self._fresh(f"range_arg{i}") for i in range(len(args))]
        assigns = [
            ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=a)
            for n, a in zip(tmp_names, args)
        ]

        # Infer loop-carried variables for common "Python for + reassignment" patterns:
        #   acc = init
        #   for i in range(...):
        #     acc = acc + ...
        #
        # If we detect such variables, lower to scf.for_ with iter_args and automatically
        # add scf.yield_ plus bind results after the loop.
        def _load_names(n: ast.AST) -> set[str]:
            out: set[str] = set()
            for nn in ast.walk(n):
                if isinstance(nn, ast.Name) and isinstance(nn.ctx, ast.Load):
                    out.add(nn.id)
            return out

        def _store_names(n: ast.AST) -> set[str]:
            out: set[str] = set()
            for nn in ast.walk(n):
                if isinstance(nn, ast.Name) and isinstance(nn.ctx, ast.Store):
                    out.add(nn.id)
            return out

        def _analyze_stmt_list(stmts: list[ast.stmt], defs_in: set[str]) -> tuple[set[str], set[str], set[str]]:
            """Return (use_before_def, defs_definite, defs_any)."""
            ubd: set[str] = set()
            defs_def = set(defs_in)
            defs_any: set[str] = set(defs_in)

            for s in stmts:
                u_s, defs_def, defs_any = _analyze_stmt(s, defs_def, defs_any)
                ubd |= u_s
            return ubd, defs_def, defs_any

        def _analyze_stmt(s: ast.stmt, defs_def: set[str], defs_any: set[str]) -> tuple[set[str], set[str], set[str]]:
            """Analyze one statement.

            - defs_def: names definitely defined before this statement (flow-sensitive)
            - defs_any: names possibly defined before this statement (for bookkeeping)
            """
            ubd: set[str] = set()

            # Helpers
            def use_expr(expr):
                nonlocal ubd
                if expr is None:
                    return
                for v in _load_names(expr):
                    if v not in defs_def:
                        ubd.add(v)

            if isinstance(s, ast.Assign):
                use_expr(s.value)
                defs = set()
                for t in s.targets:
                    defs |= _store_names(t)
                defs_def = defs_def | defs
                defs_any = defs_any | defs
                return ubd, defs_def, defs_any

            if isinstance(s, ast.AugAssign):
                # target is read then written
                if isinstance(s.target, ast.Name) and s.target.id not in defs_def:
                    ubd.add(s.target.id)
                use_expr(s.value)
                defs = _store_names(s.target)
                defs_def = defs_def | defs
                defs_any = defs_any | defs
                return ubd, defs_def, defs_any

            if isinstance(s, ast.Expr):
                use_expr(s.value)
                return ubd, defs_def, defs_any

            if isinstance(s, ast.If):
                use_expr(s.test)
                u_then, defs_then_def, defs_then_any = _analyze_stmt_list(list(s.body), set(defs_def))
                u_else, defs_else_def, defs_else_any = _analyze_stmt_list(list(s.orelse), set(defs_def))
                ubd |= (u_then | u_else)
                # definite defs after if: intersection of branch definite defs
                defs_def_out = defs_def | (defs_then_def & defs_else_def)
                defs_any_out = defs_any | defs_then_any | defs_else_any
                return ubd, defs_def_out, defs_any_out

            if isinstance(s, ast.For):
                # iter expression uses
                use_expr(s.iter)
                # loop target is defined within body
                loop_defs = set(defs_def)
                if isinstance(s.target, ast.Name):
                    loop_defs.add(s.target.id)
                u_body, _, defs_body_any = _analyze_stmt_list(list(s.body), loop_defs)
                ubd |= u_body
                # Anything assigned in the loop is "possibly defined" after it (Python semantics),
                # but not necessarily definitely defined.
                defs_any = defs_any | defs_body_any
                return ubd, defs_def, defs_any

            if isinstance(s, ast.With):
                for item in s.items:
                    use_expr(item.context_expr)
                    if item.optional_vars is not None:
                        defs = _store_names(item.optional_vars)
                        defs_def |= defs
                        defs_any |= defs
                u_body, defs_def, defs_any = _analyze_stmt_list(list(s.body), defs_def)
                ubd |= u_body
                return ubd, defs_def, defs_any

            # Fallback: treat as using all loads and defining all stores in a conservative, non-flow-sensitive way.
            for v in _load_names(s):
                if v not in defs_def:
                    ubd.add(v)
            defs = _store_names(s)
            defs_def |= defs
            defs_any |= defs
            return ubd, defs_def, defs_any

        # Compute use-before-def and defs for the loop body with the induction variable pre-defined.
        predefs: set[str] = set()
        target_name = node.target.id if isinstance(node.target, ast.Name) else None
        if target_name is not None:
            predefs.add(target_name)
        ubd_body, _, defs_any_body = _analyze_stmt_list(list(node.body), predefs)
        defs_any_body.discard(target_name)
        carry_vars = sorted((ubd_body & defs_any_body))

        if carry_vars:
            forop_name = self._fresh("forop")
            # Precompute flattened iter_args + tree specs for each carried var.
            pre = []
            flat_names = []
            spec_names = []
            for v in carry_vars:
                flat_n = self._fresh(f"flat_{v}")
                spec_n = self._fresh(f"spec_{v}")
                pre.append(
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=flat_n, ctx=ast.Store()),
                                    ast.Name(id=spec_n, ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Call(
                            func=ast.Name(id="__flir_tree_flatten", ctx=ast.Load()),
                            args=[ast.Name(id=v, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                )
                flat_names.append(flat_n)
                spec_names.append(spec_n)

            # iter_args = flat0 + flat1 + ...
            if flat_names:
                iter_args_expr = ast.Name(id=flat_names[0], ctx=ast.Load())
                for fn in flat_names[1:]:
                    iter_args_expr = ast.BinOp(
                        left=iter_args_expr,
                        op=ast.Add(),
                        right=ast.Name(id=fn, ctx=ast.Load()),
                    )
            else:
                iter_args_expr = ast.List(elts=[], ctx=ast.Load())

            iter_args_name = self._fresh("iter_args")
            pre.append(
                ast.Assign(
                    targets=[ast.Name(id=iter_args_name, ctx=ast.Store())],
                    value=iter_args_expr,
                )
            )

            with_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                    attr="for_",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=n, ctx=ast.Load()) for n in tmp_names],
                keywords=[
                    ast.keyword(
                        arg="iter_args",
                        value=ast.Name(id=iter_args_name, ctx=ast.Load()),
                    )
                ],
            )

            # Prologue in loop body: bind induction var + iter args.
            prologue = [
                ast.Assign(
                    targets=[node.target],
                    value=ast.Attribute(
                        value=ast.Name(id=forop_name, ctx=ast.Load()),
                        attr="induction_variable",
                        ctx=ast.Load(),
                    ),
                )
            ]
            # Unflatten inner_iter_args into each carried variable, consuming from a rest list.
            rest_name = self._fresh("rest")
            prologue.append(
                ast.Assign(
                    targets=[ast.Name(id=rest_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="list", ctx=ast.Load()),
                        args=[
                            ast.Attribute(
                                value=ast.Name(id=forop_name, ctx=ast.Load()),
                                attr="inner_iter_args",
                                ctx=ast.Load(),
                            )
                        ],
                keywords=[],
                    ),
                )
            )

            for v, spec_n in zip(carry_vars, spec_names):
                obj_n = self._fresh(f"obj_{v}")
                prologue.append(
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=obj_n, ctx=ast.Store()),
                                    ast.Name(id=rest_name, ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Call(
                            func=ast.Name(id="__flir_tree_unflatten", ctx=ast.Load()),
                            args=[ast.Name(id=spec_n, ctx=ast.Load()), ast.Name(id=rest_name, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                )
                prologue.append(
                    ast.Assign(
                        targets=[ast.Name(id=v, ctx=ast.Store())],
                        value=ast.Name(id=obj_n, ctx=ast.Load()),
                    )
                )

            # Build yield values by flattening each carried var and concatenating.
            yield_vals_name = self._fresh("yield_vals")
            yield_build = [
                ast.Assign(targets=[ast.Name(id=yield_vals_name, ctx=ast.Store())], value=ast.List(elts=[], ctx=ast.Load()))
            ]
            for v in carry_vars:
                flat_tmp = self._fresh(f"flatcur_{v}")
                spec_tmp = self._fresh(f"speccur_{v}")
                yield_build.append(
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[ast.Name(id=flat_tmp, ctx=ast.Store()), ast.Name(id=spec_tmp, ctx=ast.Store())],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Call(
                            func=ast.Name(id="__flir_tree_flatten", ctx=ast.Load()),
                            args=[ast.Name(id=v, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                )
                yield_build.append(
                    ast.AugAssign(
                        target=ast.Name(id=yield_vals_name, ctx=ast.Store()),
                        op=ast.Add(),
                        value=ast.Name(id=flat_tmp, ctx=ast.Load()),
                    )
                )

            yield_stmt = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                        attr="yield_",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=yield_vals_name, ctx=ast.Load())],
                    keywords=[],
                )
            )

            with_stmt = ast.With(
                items=[
                    ast.withitem(
                        context_expr=with_call,
                        optional_vars=ast.Name(id=forop_name, ctx=ast.Store()),
                    )
                ],
                body=prologue + node.body + yield_build + [yield_stmt],
                type_comment=None,
            )

            post = []
            # Unflatten results back into carried vars.
            post_rest = self._fresh("post_rest")
            post.append(
                ast.Assign(
                    targets=[ast.Name(id=post_rest, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="list", ctx=ast.Load()),
                        args=[
                            ast.Attribute(
                                value=ast.Name(id=forop_name, ctx=ast.Load()),
                                attr="results",
                                ctx=ast.Load(),
                            )
                        ],
                        keywords=[],
                    ),
                )
            )
            for v, spec_n in zip(carry_vars, spec_names):
                obj_n = self._fresh(f"post_obj_{v}")
                post.append(
                    ast.Assign(
                        targets=[
                            ast.Tuple(
                                elts=[
                                    ast.Name(id=obj_n, ctx=ast.Store()),
                                    ast.Name(id=post_rest, ctx=ast.Store()),
                                ],
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Call(
                            func=ast.Name(id="__flir_tree_unflatten", ctx=ast.Load()),
                            args=[ast.Name(id=spec_n, ctx=ast.Load()), ast.Name(id=post_rest, ctx=ast.Load())],
                            keywords=[],
                        ),
                    )
                )
                post.append(
                    ast.Assign(
                        targets=[ast.Name(id=v, ctx=ast.Store())],
                        value=ast.Name(id=obj_n, ctx=ast.Load()),
                    )
                )

            lowered = assigns + pre + [with_stmt] + post
        else:
            # No loop-carried vars: use simple scf.range_ lowering.
            with_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.opts.scf_alias, ctx=ast.Load()),
                    attr="range_",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=n, ctx=ast.Load()) for n in tmp_names],
                keywords=[],
            )
            with_stmt = ast.With(
                items=[ast.withitem(context_expr=with_call, optional_vars=node.target)],
                body=node.body,
                type_comment=None,
            )
            lowered = assigns + [with_stmt]

        for n in lowered:
            ast.copy_location(n, node)
        return lowered


def range_constexpr(*args):
    """Marker iterable to keep Python unrolling semantics for a `for` loop.

    This is intended to be used in code that otherwise enables range-loop lowering:

      for i in range_constexpr(4):
          ...
    """
    return range(*args)


def lower_range_for_loops(fn: Callable, *, options: Optional[LoweringOptions] = None) -> Callable:
    """Return a new function with range-loop lowering applied."""
    if options is None:
        options = LoweringOptions()

    try:
        src = inspect.getsource(fn)
    except OSError as e:  # pragma: no cover
        raise RuntimeError(f"Cannot get source for function {fn.__name__}: {e}") from e

    src = textwrap.dedent(src)
    mod = ast.parse(src)

    func_def = None
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            func_def = node
            break
    if func_def is None:
        raise RuntimeError(f"Could not find function definition for {fn.__name__} in source")

    transformer = _RangeForLowerer(options)
    new_def = transformer.visit(func_def)
    new_mod = ast.Module(body=[new_def], type_ignores=[])
    new_mod = ast.fix_missing_locations(new_mod)

    # Build globals: keep original globals and materialize closure vars as globals.
    g: Dict = dict(fn.__globals__)
    if fn.__closure__ and fn.__code__.co_freevars:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                g[name] = cell.cell_contents
            except ValueError:
                pass

    compiled = compile(new_mod, filename=fn.__code__.co_filename, mode="exec")
    l: Dict = {}
    exec(compiled, g, l)  # noqa: S102 (intentional)
    new_fn = l[fn.__name__]

    # Preserve a few useful attributes.
    for attr in ("__defaults__", "__kwdefaults__", "__annotations__", "__doc__", "__module__", "__qualname__"):
        try:
            setattr(new_fn, attr, getattr(fn, attr))
        except Exception:
            pass
    if hasattr(fn, "__type_params__"):
        setattr(new_fn, "__type_params__", getattr(fn, "__type_params__"))

    return new_fn


